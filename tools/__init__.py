"""
tools package

Exports:
- diagnose_main: entrypoint for the self-diagnosis tool (tools/diagnose.py)
"""
from __future__ import annotations

from .diagnose import main as diagnose_main  # noqa: F401

__all__ = ["diagnose_main"]
