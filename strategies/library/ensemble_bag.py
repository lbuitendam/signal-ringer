# strategies/library/ensemble_bag.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import pandas as pd
from engine.utils import Signal

class EnsembleBAG:
    """
    Not a BaseStrategy; this merges recent signals from enabled strategies.
    Conditions:
      - At least K strategies agree on side within last Kbars
      - Price proximity tolerance
    """
    def __init__(self, k_bars:int=3, tol:float=0.003) -> None:
        self.k_bars = k_bars
        self.tol = tol

    def combine(self, df: pd.DataFrame, signals: List[Signal]) -> List[Signal]:
        if not signals: return []
        out: List[Signal] = []
        # group by last k bars
        last_idx = len(df) - 1
        recent_idx = set(range(max(0, last_idx - self.k_bars + 1), last_idx + 1))
        bucket = {"long": [], "short": []}
        for s in signals:
            if s.index in recent_idx:
                bucket[s.side].append(s)
        for side in ("long","short"):
            arr = bucket[side]
            if len(arr) >= 2:
                prices = [s.price for s in arr]
                pmin, pmax = min(prices), max(prices)
                if (pmax - pmin) / max(1e-9, pmax) <= self.tol:
                    # merge into one signal at last bar
                    conf = min(0.95, sum(s.confidence for s in arr)/len(arr) + 0.05*len(arr))
                    reasons = list({r for s in arr for r in s.reasons})
                    price = float(df["close"].iloc[-1])
                    out.append(Signal("Big Boss (B.A.G.)", side, last_idx, df.index[-1], conf, reasons, price))
        return out
