from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

from engine.utils import OrderSuggestion, Signal, rolling_atr


@dataclass
class RiskOptions:
    equity: float = 10000.0
    risk_pct: float = 0.01
    atr_mult_sl: float = 1.5
    rr: float = 2.0
    tp_count: int = 2
    cooldown_min: int = 15
    max_positions: int = 6


class RiskManager:
    def __init__(self, opts: RiskOptions):
        self.opts = opts
        # cooldown key -> last signal timestamp (pd.Timestamp, tz-aware)
        self._cool: Dict[str, pd.Timestamp] = {}

    # ---------- cooldown ----------
    def in_cooldown(self, key: str, now: pd.Timestamp) -> bool:
        """Return True if (key) is still cooling down."""
        last = self._cool.get(key)
        if last is None:
            return False
        # both `now` and `last` should be tz-aware; engine passes index[-1]
        delta = now - last
        return delta < pd.Timedelta(minutes=float(self.opts.cooldown_min))

    def arm_cooldown(self, key: str, now: pd.Timestamp) -> None:
        self._cool[key] = now

    # ---------- SL/TP ----------
    def build_sl_tp(self, df: pd.DataFrame, idx: int, side: str, entry: float) -> Tuple[float, list]:
        """
        Compute stop-loss using ATR and a simple TP ladder using R-multiples.
        df must have lowercase ohlcv.
        """
        n = max(14, int(self.opts.atr_mult_sl * 10))  # enough bars for a smooth ATR
        atr = rolling_atr(df, n=n)
        # use the *current bar's* ATR; safe with .iat for scalars
        atr_i = float(atr.iat[int(idx)])
        k = float(self.opts.atr_mult_sl)
        rr = float(self.opts.rr)
        tps = []

        if side == "long":
            sl = entry - k * atr_i
            step = rr * (entry - sl)  # 1R
            for r in range(1, int(self.opts.tp_count) + 1):
                tps.append(entry + r * step)
        else:
            sl = entry + k * atr_i
            step = rr * (sl - entry)  # 1R
            for r in range(1, int(self.opts.tp_count) + 1):
                tps.append(entry - r * step)

        return float(sl), [float(x) for x in tps]
