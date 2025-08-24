from __future__ import annotations
import pandas as pd, numpy as np
from strategies.base import BaseStrategy
from engine.utils import Signal

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _pivots(x: pd.Series, lb: int = 5):
    # return local minima/maxima indices (very simple)
    idx = x.index
    mins, maxs = [], []
    for i in range(lb, len(x)-lb):
        window = x.iloc[i-lb:i+lb+1]
        if x.iat[i] == window.min(): mins.append(idx[i])
        if x.iat[i] == window.max(): maxs.append(idx[i])
    return mins, maxs

class RsiDivergenceAtLevel(BaseStrategy):
    name = "RSI Divergence at Level"
    CATEGORY = "Reversal"
    PARAMS_SCHEMA = {"period":{"type":"int","min":5,"max":50,"step":1,"default":14,"label":"RSI"},
                     "level_lookback":{"type":"int","min":20,"max":400,"step":5,"default":100,"label":"S/R lookback"}}

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        p = df.copy()
        c = pd.to_numeric(p["close"], errors="coerce").astype(float)
        r = rsi(c, int(self.params.get("period",14)))
        lb = int(self.params.get("level_lookback",100))

        mins, maxs = _pivots(c, lb=min(5, max(2, int(lb/20))))
        out: list[Signal] = []

        # Bullish divergence: price lower low, RSI higher low
        for a, b in zip(mins[:-1], mins[1:]):
            if c.loc[b] < c.loc[a] and r.loc[b] > r.loc[a]:
                i = p.index.get_loc(b)
                out.append(Signal(self.name, "long", i, b, 0.6, ["bullish div"], float(c.loc[b])))

        # Bearish divergence: price higher high, RSI lower high
        for a, b in zip(maxs[:-1], maxs[1:]):
            if c.loc[b] > c.loc[a] and r.loc[b] < r.loc[a]:
                i = p.index.get_loc(b)
                out.append(Signal(self.name, "short", i, b, 0.6, ["bearish div"], float(c.loc[b])))
        return out
