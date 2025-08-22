"""
Engine package: data adapters + runner.
"""
from .data import get_adapter, BaseAdapter  # noqa: F401
from .runner import run_engine_loop, scan_once  # noqa: F401

__all__ = ["get_adapter", "BaseAdapter", "run_engine_loop", "scan_once"]
