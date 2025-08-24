from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from engine.utils import Signal

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

class EmaPullback(BaseStrategy):
    name = "EMA20/50 Pullback"
    CATEGORY = "Trend"
    PARAMS_SCHEMA = {
        "fast": {"type": "int", "min": 2, "max": 200, "step": 1, "default": 20, "label": "Fast EMA"},
        "slow": {"type": "int", "min": 5, "max": 400, "step": 1, "default": 50, "label": "Slow EMA"},
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params or {})
        p = self.params
        self.fast = int(p.get("fast", 20))
        self.slow = int(p.get("slow", 50))

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema20"] = ema(out["close"], self.fast)
        out["ema50"] = ema(out["close"], self.slow)
        out["trend_up"] = out["ema20"] > out["ema50"]
        out["trend_dn"] = out["ema20"] < out["ema50"]
        return out

    def signals(self, df: pd.DataFrame):
        p = self.prepare(df)
        out: List[Signal] = []
        cross_up = (p["close"] > p["ema20"]) & (p["close"].shift(1) <= p["ema20"].shift(1)) & p["trend_up"]
        cross_dn = (p["close"] < p["ema20"]) & (p["close"].shift(1) >= p["ema20"].shift(1)) & p["trend_dn"]
        for ts in p.index[cross_up]:
            out.append(Signal(self.name, "long", p.index.get_loc(ts), ts, 0.7, ["ema pullback"], float(p.loc[ts,"close"])))
        for ts in p.index[cross_dn]:
            out.append(Signal(self.name, "short", p.index.get_loc(ts), ts, 0.7, ["ema pullback"], float(p.loc[ts,"close"])))
        return out
