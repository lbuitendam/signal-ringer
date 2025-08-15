from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

# Point to the built frontend
_BUILD_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")
_component_func = components.declare_component("st_trading_draw", path=_BUILD_DIR)

def st_trading_draw(
    *,
    ohlcv: List[Dict[str, Any]],
    symbol: str,
    timeframe: str,
    initial_drawings: Optional[Dict[str, Any]] = None,
    magnet: bool = True,
    toolbar_default: str = "docked-right",  # "floating" | "docked-left" | "docked-right"
    key: Optional[str] = None,
) -> Dict[str, Any]:
    """Render the trading draw component and return current drawings state."""
    payload = {
        "ohlcv": ohlcv,
        "symbol": symbol,
        "timeframe": timeframe,
        "initial_drawings": initial_drawings or {},
        "magnet": magnet,
        "toolbar_default": toolbar_default,
    }
    result = _component_func(default=initial_drawings or {}, key=key, **payload)
    # result is drawings+ui state (dict)
    if not isinstance(result, dict):
        return initial_drawings or {}
    return result
