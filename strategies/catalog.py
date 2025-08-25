from __future__ import annotations

"""
Catalog facade used by the Scanner page.

Primary path:
  - Use strategies.registry.discover_strategies() for dynamic discovery.

Fallback path (if registry cannot be imported for any reason):
  - Reflect over strategies.library exports (EmaPullback, MacdTrend, ...).

get_catalog() -> {id: {name, category, params, desc, class}}
find_by_name(q, category?) -> list of ids
refresh_catalog() -> clear cache (when adding/removing files at runtime)
"""

from functools import lru_cache
from typing import Dict, Any, List, Callable
import importlib

# ---------- dynamic import with safe fallback ----------
def _import_discover() -> Callable[[], Dict[str, Dict[str, Any]]]:
    try:
        return importlib.import_module("strategies.registry").discover_strategies  # type: ignore
    except Exception:
        # Fallback: build a tiny in-memory catalog from strategies.library exports
        try:
            from strategies.base import BaseStrategy
            from strategies import library as lib  # type: ignore

            def _fallback() -> Dict[str, Dict[str, Any]]:
                out: Dict[str, Dict[str, Any]] = {}
                for name in getattr(lib, "__all__", []):
                    cls = getattr(lib, name, None)
                    if cls and isinstance(cls, type) and issubclass(cls, BaseStrategy):
                        key = f"strategies.library.{name}.{name}"
                        nm = getattr(cls, "name", name)
                        cat = getattr(cls, "CATEGORY", "Uncategorized")
                        schema = getattr(cls, "PARAMS_SCHEMA", {})
                        desc = getattr(cls, "DESC", "") or (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
                        if not isinstance(schema, dict) or not schema:
                            # crude inference (minimal)
                            schema = {}
                        out[key] = {
                            "name": nm,
                            "category": cat,
                            "schema": schema,
                            "defaults": {k: v.get("default") for k, v in schema.items()},
                            "desc": desc,
                            "class": cls,
                        }
                return out

            return _fallback
        except Exception:
            # Last-resort: empty
            return lambda: {}

_discover = _import_discover()


@lru_cache(maxsize=1)
def _discovered() -> Dict[str, Dict[str, Any]]:
    return _discover()  # {sid: {"name","category","schema","defaults","desc","class"}}


def refresh_catalog() -> None:
    _discovered.cache_clear()


def get_catalog() -> Dict[str, Dict[str, Any]]:
    cat = _discovered()
    return {
        sid: {
            "name": m["name"],
            "category": m.get("category", "Uncategorized"),
            "params": m.get("schema", {}),
            "desc": m.get("desc", ""),
            "class": m["class"],
        }
        for sid, m in cat.items()
    }


def find_by_name(q: str, category: str | None = None) -> List[str]:
    cat = get_catalog()
    ql = (q or "").strip().lower()
    ids = [
        k
        for k in cat
        if not ql
        or ql in k.lower()
        or ql in cat[k]["name"].lower()
        or ql in cat[k]["category"].lower()
    ]
    if category and category != "All":
        ids = [i for i in ids if cat[i]["category"] == category]
    return ids
