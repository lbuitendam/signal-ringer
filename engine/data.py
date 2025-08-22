from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf


# --------------------- Base ---------------------
class BaseAdapter:
    name: str = "base"

    def fetch_history(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        """
        Return a UTC-indexed DataFrame with columns:
        ['open','high','low','close','volume'] and a tz-aware DatetimeIndex (UTC).
        `lookback` is the number of bars desired (approximate).
        """
        raise NotImplementedError


# --------------------- Helpers ---------------------
_INTERVAL_TO_SECONDS: Dict[str, int] = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "90m": 5400,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "5d": 432000,
    "1wk": 604800,
    "1mo": 2629800,
}

def timeframe_seconds(tf: str) -> int:
    tf = tf.lower()
    if tf in _INTERVAL_TO_SECONDS:
        return _INTERVAL_TO_SECONDS[tf]
    # allow variants like "3min", "2h"
    if tf.endswith("min"):
        return int(tf[:-3]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    raise ValueError(f"Unsupported timeframe: {tf}")


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    # flatten possible MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    # lower-map names
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index.copy())
    for src, dst in (
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("adj close", "close"),
        ("volume", "volume"),
    ):
        if src in cols:
            if dst in out.columns:
                continue
            out[dst] = pd.to_numeric(df[cols[src]], errors="coerce")
    # tz → UTC
    if getattr(out.index, "tz", None) is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    # ensure columns exist & order
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            out[c] = np.nan
    out = out[["open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"]
    )
    return out


# --------------------- YFinance ---------------------
class YFinanceAdapter(BaseAdapter):
    name = "yfinance"

    def _yf_period(self, timeframe: str, lookback: int) -> str:
        """
        Pick a period string large enough for lookback bars while staying within Yahoo limits.
        """
        sec = timeframe_seconds(timeframe)
        intraday = sec < 86400
        total_sec = (lookback + 50) * sec  # small buffer
        days = max(1, math.ceil(total_sec / 86400))

        if intraday:
            # yahoo caps intraday to ~60d; yfinance auto-chunks now, keep <=60d
            return f"{min(days, 60)}d"
        # daily+ can go long
        if days <= 365:
            return f"{days}d"
        years = math.ceil(days / 365)
        return f"{min(years, 10)}y"

    def _yf_interval(self, tf: str) -> str:
        tf = tf.lower()
        # yfinance supports "1h" directly; no native "4h" — fetch 1h (callers can resample if needed)
        if tf == "4h":
            return "1h"
        return tf

    def fetch_history(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        interval = self._yf_interval(timeframe)
        period = self._yf_period(timeframe, lookback)
        # Explicit auto_adjust to avoid FutureWarning noise
        hist = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False,
        )
        return _normalize_cols(hist).tail(lookback)


# --------------------- CSV ---------------------
@dataclass
class CSVAdapter(BaseAdapter):
    """
    CSV path can include {symbol} placeholder.
    Expected columns (case-insensitive): time/timestamp, open, high, low, close[, volume]
    """
    path: str = "data/{symbol}.csv"
    time_col: Optional[str] = None

    def fetch_history(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        fpath = self.path.format(symbol=symbol)
        try:
            df = pd.read_csv(fpath)
        except FileNotFoundError:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # determine time col
        tcol = self.time_col
        if tcol is None:
            for c in df.columns:
                cl = str(c).lower()
                if cl in ("time", "timestamp", "date", "datetime"):
                    tcol = c
                    break
        if tcol is None:
            parsed = df
            if not isinstance(parsed.index, pd.DatetimeIndex):
                # try common 'Date' column implicitly
                for c in ("Date", "Datetime", "Time"):
                    if c in parsed.columns:
                        parsed = parsed.set_index(c)
                        break
                parsed.index = pd.to_datetime(parsed.index, utc=True, errors="coerce")
        else:
            parsed = df.copy()
            parsed[tcol] = pd.to_datetime(parsed[tcol], utc=True, errors="coerce")
            parsed = parsed.set_index(tcol)

        out = _normalize_cols(parsed)
        return out.tail(lookback)


# --------------------- Synthetic (for tests/diagnose) ---------------------
class SyntheticAdapter(BaseAdapter):
    name = "synthetic"

    def fetch_history(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        n = max(lookback, 600)
        sec = timeframe_seconds(timeframe)
        end = datetime.now(timezone.utc).replace(microsecond=0)
        idx = pd.date_range(end=end, periods=n, freq=pd.Timedelta(seconds=sec))
        # create a smooth series with noise
        t = np.linspace(0, 6 * math.pi, n)
        base = 100 + 5 * np.sin(t)
        noise = np.random.normal(0, 0.3, n)
        close = base + noise
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) + np.abs(np.random.normal(0.2, 0.1, n))
        low = np.minimum(open_, close) - np.abs(np.random.normal(0.2, 0.1, n))
        vol = np.random.randint(1000, 5000, n)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=idx,
        )
        df.index = df.index.tz_convert("UTC")
        return df.tail(lookback)


# --------------------- Factory ---------------------
def get_adapter(name: str, **kwargs) -> BaseAdapter:
    key = (name or "").lower()
    if key in ("yfinance", "yf", "yahoo"):
        return YFinanceAdapter()
    if key == "csv":
        return CSVAdapter(**kwargs)
    if key in ("synthetic", "test", "demo"):
        return SyntheticAdapter()
    raise ValueError(f"Unknown adapter: {name}")
