"""
strategies package

Exports:
- BaseStrategy
- Library strategies: EmaPullback, MacdTrend, RangeBreakout, CandlesStrategy, EnsembleBAG
"""
from __future__ import annotations

from .base import BaseStrategy  # noqa: F401
from .library import (          # noqa: F401
    EmaPullback,
    MacdTrend,
    RangeBreakout,
    CandlesStrategy,
    EnsembleBAG,
)

__all__ = [
    "BaseStrategy",
    "EmaPullback",
    "MacdTrend",
    "RangeBreakout",
    "CandlesStrategy",
    "EnsembleBAG",
]
