# data/provider.py
from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass
from typing import Literal, Dict, Any, List
from datetime import datetime, timezone
import math

import pandas as pd
import yfinance as yf

Timeframe = Literal["1m","2m","5m","15m","30m","60m","90m","1h","1H","1d","1D","1wk","1mo"]

# ------------ symbol normalization ------------

_NORMALIZE_MAP = {
    "=XAU": "XAUUSD=X",
    "=xau": "XAUUSD=X",
    "XAU": "XAUUSD=X",
    "XAUUSD": "XAUUSD=X",
    "=XAG": "XAGUSD=X",
    "=xag": "XAGUSD=X",
    "XAG": "XAGUSD=X",
    "XAGUSD": "XAGUSD=X",
    # Futures already OK:
    "GC=F": "GC=F",
    "SI=F": "SI=F",
}

def normalize_symbol(sym: str) -> str:
    s = (sym or "").strip()
    return _NORMALIZE_MAP.get(s, _NORMALIZE_MAP.get(s.upper(), s))

def normalize_watchlist(wl: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in wl or []:
        d = dict(t)
        d["symbol"] = normalize_symbol(d.get("symbol", ""))
        out.append(d)
    return out

# ------------ yfinance adapter ------------

_YF_INTERVALS = {
    "1m":"1m","2m":"2m","5m":"5m","15m":"15m","30m":"30m","60m":"60m","90m":"90m",
    "1h":"60m","1H":"60m","1d":"1d","1D":"1d","1wk":"1wk","1mo":"1mo",
}

def _bars_per_day(tf: str) -> float:
    tf = tf.lower()
    if tf.endswith("m"):
        mins = int(tf[:-1])
        return math.floor(24*60/mins)
    if tf in ("1h","1H"):
        return 24
    if tf in ("1d","1D"):
        return 1
    if tf == "1wk":
        return 1/7
    if tf == "1mo":
        return 1/30
    return 300.0

def _period_for(tf: str, lookback_bars: int) -> str:
    tfy = _YF_INTERVALS.get(tf, "1d")
    bars_per_day = max(1.0, _bars_per_day(tf))
    days_needed = max(1, int(math.ceil(lookback_bars / bars_per_day)))
    # yfinance intraday caps: 1m ~7d, 2m-90m ~60d
    if tfy == "1m":
        return f"{min(days_needed, 7)}d"
    if tfy in {"2m","5m","15m","30m","60m","90m"}:
        return f"{min(days_needed, 60)}d"
    # daily+ can use months/years
    if days_needed < 30:  return f"{days_needed}d"
    if days_needed < 365: return f"{max(1, days_needed//30)}mo"
    return f"{max(1, days_needed//365)}y"

@dataclass
class Provider:
    name: str = "yfinance"

    @lru_cache(maxsize=256)
    def get_ohlcv(self, symbol: str, timeframe: Timeframe, lookback_bars: int = 600) -> pd.DataFrame:
        """
        Returns DF with columns: open, high, low, close, volume (all float),
        UTC tz-aware index.
        """
        sym = normalize_symbol(symbol)
        interval = _YF_INTERVALS.get(timeframe, "1d")
        period = _period_for(timeframe, lookback_bars)

        df = yf.download(sym, interval=interval, period=period, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open","high","low","close","volume"])

        # flatten multicol if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]

        # map typical yfinance names
        cols = {c.lower(): c for c in df.columns}
        out = pd.DataFrame(index=df.index.copy())
        for tgt, src in (("open","open"),("high","high"),("low","low"),("close","close")):
            if src in cols:
                out[tgt] = pd.to_numeric(df[cols[src]], errors="coerce").astype(float)
        vol_col = cols.get("volume")
        out["volume"] = pd.to_numeric(df[vol_col], errors="coerce").astype(float) if vol_col else 0.0

        out = out.dropna(subset=["open","high","low","close"])
        # ensure UTC tz
        idx = out.index
        if getattr(idx, "tz", None) is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        return out

# singleton accessor
_provider: Provider | None = None
def get_provider() -> Provider:
    global _provider
    if _provider is None:
        _provider = Provider()
    return _provider
