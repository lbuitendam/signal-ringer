"""
sizing package

Exports:
- position_size (compute qty from equity and SL distance)
"""
from __future__ import annotations

from .position import position_size  # noqa: F401

__all__ = ["position_size"]
