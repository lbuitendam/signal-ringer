# sizing/position.py
from __future__ import annotations
import math
from engine.utils import OrderSuggestion

def position_size(equity: float, risk_pct: float, entry: float, sl: float, lot_size: float = 1.0) -> float:
    risk_amt = equity * risk_pct
    per_unit_risk = abs(entry - sl)
    if per_unit_risk <= 0:
        return 0.0
    qty = risk_amt / per_unit_risk
    # round down to lot size
    return math.floor(qty / lot_size) * lot_size
