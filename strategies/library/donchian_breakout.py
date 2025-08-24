from __future__ import annotations
import numpy as np, pandas as pd
from strategies.base import BaseStrategy
from engine.utils import Signal

class DonchianBreakout(BaseStrategy):
    name = "Donchian 20 Breakout"
    CATEGORY = "Breakout"
    PARAMS_SCHEMA = {"lookback":{"type":"int","min":5,"max":200,"step":1,"default":20,"label":"Lookback"},
                     "retest":{"type":"bool","default":True,"label":"Require retest?"}}

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        p = df.copy()
        hi = pd.to_numeric(p["high"], errors="coerce").astype(float).values
        lo = pd.to_numeric(p["low"], errors="coerce").astype(float).values
        cl = pd.to_numeric(p["close"], errors="coerce").astype(float).values
        lb = int(self.params.get("lookback",20))
        want_retest = bool(self.params.get("retest", True))
        out: list[Signal] = []
        n = len(p)
        for i in range(lb, n):
            up = float(np.max(hi[i-lb:i])); dn = float(np.min(lo[i-lb:i])); c = float(cl[i])
            if c > up:
                ok = True
                if want_retest and i+1<n:
                    ok = (cl[i+1] >= up)
                if ok: out.append(Signal(self.name, "long", i, p.index[i], 0.62, ["donchian high break"], c))
            if c < dn:
                ok = True
                if want_retest and i+1<n:
                    ok = (cl[i+1] <= dn)
                if ok: out.append(Signal(self.name, "short", i, p.index[i], 0.62, ["donchian low break"], c))
        return out
