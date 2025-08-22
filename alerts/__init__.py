"""
alerts package

Exports:
- format_entry: format an OrderSuggestion into a human-friendly line
- maybe_toast: Streamlit toast helper (no-op outside Streamlit)
- csv_log: append suggestions to CSV
"""
from __future__ import annotations

from .messages import format_entry, maybe_toast, csv_log  # noqa: F401

__all__ = ["format_entry", "maybe_toast", "csv_log"]
