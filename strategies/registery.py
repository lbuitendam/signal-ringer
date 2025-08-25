from __future__ import annotations

"""
Auto-discovers strategy classes in the `strategies` package (incl. strategies.library.*).
Each discovered class must subclass BaseStrategy.

Returns a dict keyed by "module.Class", with:
  {
    "name": display name,
    "category": CATEGORY or "Uncategorized",
    "schema": PARAMS_SCHEMA or inferred from __init__,
    "defaults": dict of default param values,
    "desc": short description,
    "class": class object,
  }
"""

import inspect
import importlib
import pkgutil
from typing import Dict, Any, Type

from strategies.base import BaseStrategy

_IGNORE_MODULES = {
    "strategies.base",
    "strategies.registry",
    "strategies.__init__",
    "strategies.catalog",
    "strategies.catalog_overrides",
}


def _infer_schema_from_init(cls: Type[BaseStrategy]) -> Dict[str, Any]:
    """
    Infer UI param schema from __init__ signature if PARAMS_SCHEMA not provided.
    """
    schema: Dict[str, Any] = {}
    sig = inspect.signature(cls.__init__)
    for name, p in sig.parameters.items():
        if name in ("self", "params"):
            continue

        if p.default is inspect._empty:
            schema[name] = {"type": "int", "min": 1, "max": 5000, "step": 1, "default": 20, "label": name}
            continue

        dv = p.default
        if isinstance(dv, bool):
            schema[name] = {"type": "bool", "default": dv, "label": name}
        elif isinstance(dv, int):
            schema[name] = {"type": "int", "min": 1, "max": 5000, "step": 1, "default": dv, "label": name}
        elif isinstance(dv, float):
            schema[name] = {"type": "float", "min": 0.0, "max": 1e9, "step": 0.1, "default": dv, "label": name}
        else:
            schema[name] = {"type": "text", "default": str(dv), "label": name}
    return schema


def _defaults_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v.get("default") for k, v in schema.items()}


def _load_overrides() -> Dict[str, Dict[str, Any]]:
    """
    Optional overrides file: strategies/catalog_overrides.py

    OVERRIDES = {
      "strategies.library.range_breakout.RangeBreakout": {
          "name": "Range Breakout",
          "category": "Breakout",
          "desc": "Breakout of N-bar range with optional retest.",
          "schema": { ... PARAMS_SCHEMA ... },
      },
      ...
    }
    """
    try:
        mod = importlib.import_module("strategies.catalog_overrides")
        return getattr(mod, "OVERRIDES", {}) or {}
    except Exception:
        return {}


def discover_strategies() -> Dict[str, Dict[str, Any]]:
    import strategies as pkg  # ensure package importable (strategies/__init__.py exists)

    overrides = _load_overrides()
    results: Dict[str, Dict[str, Any]] = {}

    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if modname in _IGNORE_MODULES:
            continue
        try:
            module = importlib.import_module(modname)
        except Exception:
            # skip modules that fail to import
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseStrategy) or obj is BaseStrategy:
                continue
            if obj.__module__ != modname:
                continue

            class_key = f"{modname}.{obj.__name__}"

            name = getattr(obj, "name", obj.__name__)
            category = getattr(obj, "CATEGORY", "Uncategorized")
            schema = getattr(obj, "PARAMS_SCHEMA", None)
            desc = getattr(obj, "DESC", None)
            if not desc:
                doc = (obj.__doc__ or "").strip()
                desc = doc.splitlines()[0] if doc else ""

            if not isinstance(schema, dict):
                schema = _infer_schema_from_init(obj)

            # apply overrides
            ov = overrides.get(class_key, {})
            if "name" in ov:
                name = ov["name"]
            if "category" in ov:
                category = ov["category"]
            if "desc" in ov:
                desc = ov["desc"]
            if isinstance(ov.get("schema"), dict):
                schema = ov["schema"]

            defaults = _defaults_from_schema(schema)

            results[class_key] = {
                "name": name,
                "category": category,
                "schema": schema,
                "defaults": defaults,
                "desc": desc,
                "class": obj,
            }

    return results


def search_ids(q: str, catalog: Dict[str, Dict[str, Any]], category: str | None = None) -> list[str]:
    ql = (q or "").strip().lower()
    ids = list(catalog.keys())
    if ql:
        ids = [
            i
            for i in ids
            if ql in catalog[i]["name"].lower()
            or ql in catalog[i]["category"].lower()
            or ql in i.lower()
        ]
    if category and category != "All":
        ids = [i for i in ids if catalog[i]["category"] == category]
    return ids
