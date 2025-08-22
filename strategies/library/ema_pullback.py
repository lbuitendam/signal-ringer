# strategies/library/ema_pullback.py
from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from engine.utils import Signal

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

class EmaPullback(BaseStrategy):
    name = "EMA20/50 Pullback"
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema20"] = ema(out["close"], 20)
        out["ema50"] = ema(out["close"], 50)
        out["trend_up"] = out["ema20"] > out["ema50"]
        out["trend_dn"] = out["ema20"] < out["ema50"]
        return out

    def signals(self, df: pd.DataFrame):
        p = self.prepare(df)
        out: List[Signal] = []
        # long setup: trend_up and close crosses back above ema20 after being below
        cross_up = (p["close"] > p["ema20"]) & (p["close"].shift(1) <= p["ema20"].shift(1)) & p["trend_up"]
        # short setup: mirror
        cross_dn = (p["close"] < p["ema20"]) & (p["close"].shift(1) >= p["ema20"].shift(1)) & p["trend_dn"]
        for ts in p.index[cross_up]:
            out.append(Signal(self.name, "long", p.index.get_loc(ts), ts, 0.7, ["ema pullback"], float(p.loc[ts,"close"])))
        for ts in p.index[cross_dn]:
            out.append(Signal(self.name, "short", p.index.get_loc(ts), ts, 0.7, ["ema pullback"], float(p.loc[ts,"close"])))
        return out
