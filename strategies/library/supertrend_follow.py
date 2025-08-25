# strategies/library/supertrend_follow.py
from __future__ import annotations
import pandas as pd
from strategies.base import BaseStrategy
from engine.utils import Signal

class SupertrendFollow(BaseStrategy):
    name = "Supertrend Trend Following"
    CATEGORY = "Systems"
    desc = "ATR-based overlay; follow flips."
    PARAMS_SCHEMA = {
        "atr_period": {"type": "int","min":5,"max":50,"step":1,"default":10,"label":"ATR Period"},
        "mult": {"type": "float","min":1.0,"max":5.0,"step":0.1,"default":3.0,"label":"Multiplier"},
    }
    def signals(self, df: pd.DataFrame):
        # compute supertrend -> emit List[Signal]
        return []
