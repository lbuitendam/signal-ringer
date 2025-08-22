# strategies/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from engine.utils import Signal, OrderSuggestion

class BaseStrategy:
    name: str = "BaseStrategy"
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.state: Dict[str, Any] = {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute required indicators, add to df, return df."""
        return df

    def signals(self, df: pd.DataFrame) -> List[Signal]:
        """Return list of Signal (index/time must refer to df.index)."""
        return []

    def propose_orders(self, df: pd.DataFrame, sig: Signal, context: Dict[str, Any]) -> List[OrderSuggestion]:
        """Map a signal to specific OrderSuggestion(s). Risk manager fills SL/TP/qty."""
        return []
