from __future__ import annotations
import pandas as pd
from strategies.base import BaseStrategy
from engine.utils import Signal

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(max(1,n)), min_periods=1).mean()

class BollingerMeanReversion(BaseStrategy):
    name = "Bollinger Mean Reversion"
    CATEGORY = "Mean Reversion"
    PARAMS_SCHEMA = {"period":{"type":"int","min":5,"max":200,"step":1,"default":20,"label":"Period"},
                     "stddev":{"type":"float","min":0.5,"max":6.0,"step":0.1,"default":2.0,"label":"StdDev"}}

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        p = df.copy()
        c = pd.to_numeric(p["close"], errors="coerce").astype(float)
        n = int(self.params.get("period",20))
        k = float(self.params.get("stddev",2.0))
        m = sma(c, n)
        sd = c.rolling(n, min_periods=1).std(ddof=0)
        upper, lower = m + k*sd, m - k*sd

        # Fade extremes when price closes back inside bands
        out: list[Signal] = []
        reenter_long = (c.shift(1) < lower.shift(1)) & (c >= lower)
        reenter_short = (c.shift(1) > upper.shift(1)) & (c <= upper)
        for ts in p.index[reenter_long]:
            i = p.index.get_loc(ts); out.append(Signal(self.name, "long", i, ts, 0.58, ["bb re-entry"], float(c.iat[i])))
        for ts in p.index[reenter_short]:
            i = p.index.get_loc(ts); out.append(Signal(self.name, "short", i, ts, 0.58, ["bb re-entry"], float(c.iat[i])))
        return out
