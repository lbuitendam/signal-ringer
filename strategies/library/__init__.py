"""
strategies.library subpackage

Exports a curated set of strategy classes implemented in this folder.
"""
from __future__ import annotations

from .ema_pullback import EmaPullback        # noqa: F401
from .macd_trend import MacdTrend            # noqa: F401
from .range_breakout import RangeBreakout    # noqa: F401
from .candlesticks import CandlesStrategy    # noqa: F401
from .ensemble_bag import EnsembleBAG        # noqa: F401

__all__ = [
    "EmaPullback",
    "MacdTrend",
    "RangeBreakout",
    "CandlesStrategy",
    "EnsembleBAG",
]
