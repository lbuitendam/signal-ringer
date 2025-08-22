from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List
import numpy as np
import pandas as pd

__all__ = [
    "Signal",
    "OrderSuggestion",
    "rolling_atr",
    "ensure_ohlcv_lower",
]


# ---------- Data models ----------
@dataclass
class Signal:
    name: str
    side: Literal["long", "short"]
    index: int                 # bar index in the df
    time: pd.Timestamp
    confidence: float
    reasons: List[str]
    price: float


@dataclass
class OrderSuggestion:
    symbol: str
    timeframe: str
    side: Literal["buy", "sell"]
    type: Literal["market", "limit", "stop"]
    qty: float
    entry: float
    sl: float
    tp: list[float]
    time_in_force: str
    strategy: str
    confidence: float
    reason: str
    paper: bool = True


# ---------- Helpers ----------
def ensure_ohlcv_lower(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with lowercase O/H/L/C/V columns (if present).
    Index is preserved.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.DataFrame(index=df.index)
    cols = {c.lower(): c for c in df.columns}
    for k in ("open", "high", "low", "close", "volume"):
        if k in cols:
            out[k] = pd.to_numeric(df[cols[k]], errors="coerce")
        elif k in df.columns:
            out[k] = pd.to_numeric(df[k], errors="coerce")
        else:
            out[k] = np.nan
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def rolling_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Wilder-style ATR using simple moving average of True Range over n periods.
    Expects lowercase columns: open, high, low, close.
    """
    p = ensure_ohlcv_lower(df)
    h = p["high"]
    l = p["low"]
    c = p["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    # SMA is OK here; if you want Wilder’s RMA use ewm(alpha=1/n, adjust=False)
    return tr.rolling(int(max(1, n)), min_periods=1).mean()
 