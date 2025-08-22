"""
patterns package

Exports:
- DEFAULT_CONFIG
- PatternHit dataclass
- detect_all (dispatcher)
- hits_to_markers (convert to chart markers)
"""
from __future__ import annotations

from .engine import (                 # noqa: F401
    DEFAULT_CONFIG,
    PatternHit,
    detect_all,
    hits_to_markers,
)

__all__ = ["DEFAULT_CONFIG", "PatternHit", "detect_all", "hits_to_markers"]
