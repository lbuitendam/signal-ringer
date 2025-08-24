# strategies/catalog.py
from __future__ import annotations

"""
Catalog wrapper around dynamic strategy discovery.

- Tries absolute import first ("strategies.registry") so running the app from project
  root works (e.g., `streamlit run app.py`).
- Falls back to relative import (".registry") so opening the 'strategies' package
  in isolation also works.
- Caches discovery so the UI can call get_catalog() frequently without re-importing
  every module.
"""

from functools import lru_cache
from typing import Any, Dict, List

# Prefer absolute import for VS Code/Pylance; fall back to relative when needed.
try:
    from strategies.registry import discover_strategies as _discover  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    from .registry import discover_strategies as _discover  # type: ignore


@lru_cache(maxsize=1)
def _discovered() -> Dict[str, Dict[str, Any]]:
    """
    Returns the raw discovery map:
      { sid: {"name","category","schema","defaults","class", ...}, ... }
    """
    return _discover()


def refresh_catalog() -> None:
    """Clear the discovery cache (e.g., after adding new strategy files)."""
    _discovered.cache_clear()


def get_catalog() -> Dict[str, Dict[str, Any]]:
    """
    Shape the discovery output to what pages/1_Scanner.py expects:
      { sid: {"name","category","params","desc","class"} }
    """
    cat = _discovered()
    out: Dict[str, Dict[str, Any]] = {}
    for sid, meta in cat.items():
        out[sid] = {
            "name": meta.get("name", sid.rsplit(".", 1)[-1]),
            "category": meta.get("category", "Uncategorized"),
            "params": meta.get("schema", {}),   # Scanner expects key "params"
            "desc": meta.get("desc", ""),
            "class": meta.get("class"),
        }
    return out


def find_by_name(q: str, category: str | None = None) -> List[str]:
    """
    Search by name/module id/category. Returns list of strategy ids (sid).
    """
    cat = get_catalog()
    ql = (q or "").strip().lower()
    ids = list(cat.keys())

    if ql:
        ids = [
            sid
            for sid in ids
            if ql in cat[sid]["name"].lower()
            or ql in sid.lower()
            or ql in cat[sid]["category"].lower()
        ]

    if category and category != "All":
        ids = [sid for sid in ids if cat[sid]["category"] == category]

    return ids


__all__ = ["get_catalog", "find_by_name", "refresh_catalog"]
