from __future__ import annotations

"""
strategies.library subpackage

Exports a curated set of strategy classes implemented in this folder.
"""

from .ema_pullback import EmaPullback
from .macd_trend import MacdTrend
from .range_breakout import RangeBreakout
from .candlesticks import CandlesStrategy
from .ensemble_bag import EnsembleBAG

__all__ = [
    "EmaPullback",
    "MacdTrend",
    "RangeBreakout",
    "CandlesStrategy",
    "EnsembleBAG",
]
