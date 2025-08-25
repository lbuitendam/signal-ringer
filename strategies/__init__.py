from __future__ import annotations

"""
strategies package

Exports:
- BaseStrategy
- (optionally) library strategies: EmaPullback, MacdTrend, RangeBreakout, CandlesStrategy, EnsembleBAG
"""

from .base import BaseStrategy

__all__ = ["BaseStrategy"]

# Re-export library strategies if available (don’t fail if library not yet present)
try:
    from .library import (
        EmaPullback,
        MacdTrend,
        RangeBreakout,
        CandlesStrategy,
        EnsembleBAG,
    )

    __all__ += [
        "EmaPullback",
        "MacdTrend",
        "RangeBreakout",
        "CandlesStrategy",
        "EnsembleBAG",
    ]
except Exception:
    pass
