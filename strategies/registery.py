# strategies/registry.py
from __future__ import annotations

import inspect
import importlib
import pkgutil
from typing import Dict, Any, Type

from strategies.base import BaseStrategy

_IGNORE_MODULES = {
    "strategies.base",
    "strategies.registry",
    "strategies.__init__",
    "strategies.catalog",           # wrapper
    "strategies.catalog_overrides", # optional user overrides
}

def _infer_schema_from_init(cls: Type[BaseStrategy]) -> Dict[str, Any]:
    """
    If a strategy doesn't define PARAMS_SCHEMA, infer sliders/inputs from __init__ defaults.
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
    Optional file strategies/catalog_overrides.py with:
      OVERRIDES = {
        "strategies.library.range_breakout.RangeBreakout": {
            "name": "Range Breakout",
            "category": "Breakout",
            "desc": "Breakout of N-bar range with optional retest filter.",
            "schema": {
                "lookback": {"type":"int","min":5,"max":300,"step":1,"default":20,"label":"Lookback"},
                "retest":   {"type":"int","min":0,"max":20,"step":1,"default":5,"label":"Retest bars"},
            },
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
    """
    Walk 'strategies' (including 'strategies.library.*'), import modules,
    and collect subclasses of BaseStrategy.

    Returns dict keyed by "module.Class", with:
      {
        "name": display name,
        "category": CATEGORY or "Uncategorized",
        "schema": PARAMS_SCHEMA or inferred,
        "defaults": dict of default param values,
        "desc": text description (DESC or docstring first line),
        "class": class object,
      }
    """
    import strategies as pkg
    overrides = _load_overrides()
    results: Dict[str, Dict[str, Any]] = {}

    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if modname in _IGNORE_MODULES:
            continue
        try:
            module = importlib.import_module(modname)
        except Exception:
            # skip broken imports without killing discovery
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseStrategy) or obj is BaseStrategy:
                continue
            if obj.__module__ != modname:
                continue

            class_key = f"{modname}.{obj.__name__}"

            # Base metadata from class
            name = getattr(obj, "name", obj.__name__)
            category = getattr(obj, "CATEGORY", "Uncategorized")
            schema = getattr(obj, "PARAMS_SCHEMA", None)
            desc = getattr(obj, "DESC", None)
            if not desc:
                doc = (obj.__doc__ or "").strip()
                desc = doc.splitlines()[0] if doc else ""

            # Fallback schema inference
            if not isinstance(schema, dict):
                schema = _infer_schema_from_init(obj)

            # Apply overrides if present
            ov = overrides.get(class_key, {})
            if "name" in ov: name = ov["name"]
            if "category" in ov: category = ov["category"]
            if "desc" in ov: desc = ov["desc"]
            if "schema" in ov and isinstance(ov["schema"], dict):
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
        ids = [i for i in ids
               if ql in catalog[i]["name"].lower()
               or ql in catalog[i]["category"].lower()
               or ql in i.lower()]
    if category and category != "All":
        ids = [i for i in ids if catalog[i]["category"] == category]
    return ids
