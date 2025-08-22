from __future__ import annotations

import numpy as np
import pandas as pd

from engine.utils import Signal
from strategies.base import BaseStrategy


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=int(max(1, span)), adjust=False, min_periods=1).mean()


def _rising_edges(mask: np.ndarray) -> np.ndarray:
    """
    Return indices where mask flips False -> True.
    Uses numpy to avoid pandas fillna/downcasting warnings.
    """
    if mask.size == 0:
        return np.array([], dtype=int)
    prev = np.r_[False, mask[:-1]]
    return np.flatnonzero(mask & ~prev)


def _falling_edges(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=int)
    prev = np.r_[False, mask[:-1]]
    return np.flatnonzero(~mask & prev)


class MacdTrend(BaseStrategy):
    name = "MACD Trend"

    def __init__(self, params: dict | None = None):
        super().__init__(params or {})
        p = self.params
        self.fast = int(p.get("fast", 12))
        self.slow = int(p.get("slow", 26))
        self.signal = int(p.get("signal", 9))
        self.trend_len = int(p.get("trend_len", 50))  # EMA trend filter

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        p = df.copy()
        # expect lowercase columns
        close = pd.to_numeric(p["close"], errors="coerce").astype(float)
        ema_fast = _ema(close, self.fast)
        ema_slow = _ema(close, self.slow)
        macd = ema_fast - ema_slow
        macd_sig = _ema(macd, self.signal)
        trend = _ema(close, self.trend_len)

        bull_mask = (macd.values > macd_sig.values) & (close.values > trend.values)
        bear_mask = (macd.values < macd_sig.values) & (close.values < trend.values)

        out: list[Signal] = []

        # Longs at rising edges of bull_mask
        for i in _rising_edges(bull_mask):
            ts = p.index[i]
            price = float(close.iat[i])
            out.append(Signal(self.name, "long", int(i), ts, 0.65, ["macd cross + trend"], price))

        # Shorts at rising edges of bear_mask (i.e., flip to bearish)
        for i in _rising_edges(bear_mask):
            ts = p.index[i]
            price = float(close.iat[i])
            out.append(Signal(self.name, "short", int(i), ts, 0.65, ["macd cross + trend"], price))

        return out
