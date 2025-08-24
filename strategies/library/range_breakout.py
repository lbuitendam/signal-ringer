from __future__ import annotations

import numpy as np
import pandas as pd

from engine.utils import Signal
from strategies.base import BaseStrategy


class RangeBreakout(BaseStrategy):
    name = "Range Breakout"

    # Optional: show it under “Breakout” in the catalog
    CATEGORY = "Breakout"

    # Optional: drives the UI controls (fallback: inferred from __init__)
    PARAMS_SCHEMA = {
        "lookback": {"type": "int", "min": 5, "max": 300, "step": 1, "default": 20, "label": "Lookback bars"},
        "retest":   {"type": "int", "min": 0, "max": 20, "step": 1, "default": 5,  "label": "Retest window"},
    }

    def __init__(self, lookback: int = 20, retest: int = 5, params: dict | None = None):
        super().__init__(params or {})
        self.lb = int(lookback)
        self.retest = int(retest)

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        # Accept either lower-case or cap-case columns; normalize to lower-case
        p = df.copy()
        p.columns = [c.lower() for c in p.columns]

        high = pd.to_numeric(p["high"], errors="coerce").astype(float).values
        low = pd.to_numeric(p["low"], errors="coerce").astype(float).values
        close = pd.to_numeric(p["close"], errors="coerce").astype(float).values

        out: list[Signal] = []
        n = len(p)
        if n <= self.lb:
            return out

        for i in range(self.lb, n):
            hh = float(np.max(high[i - self.lb : i]))
            ll = float(np.min(low[i - self.lb : i]))
            c = float(close[i])

            if c > hh:
                ok = True
                if self.retest > 0 and i + 1 < n:
                    end = min(n, i + 1 + self.retest)
                    dips_below = (close[i:end] < hh).any()
                    if dips_below:
                        ok = False
                if ok:
                    ts = p.index[i]
                    out.append(Signal(self.name, "long", int(i), ts, 0.6, ["range breakout up"], c))

            if c < ll:
                ok = True
                if self.retest > 0 and i + 1 < n:
                    end = min(n, i + 1 + self.retest)
                    pops_above = (close[i:end] > ll).any()
                    if pops_above:
                        ok = False
                if ok:
                    ts = p.index[i]
                    out.append(Signal(self.name, "short", int(i), ts, 0.6, ["range breakout down"], c))

        return out
