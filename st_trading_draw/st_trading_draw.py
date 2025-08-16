# st_trading_draw/st_trading_draw.py
import os
import streamlit as st
import streamlit.components.v1 as components

_COMPONENT_FUNC = components.declare_component(
    "st_trading_draw",
    path=os.path.join(os.path.dirname(__file__), "frontend", "dist"),
)

def trading_draw(
    data,
    key=None,
    symbol:str="SYMBOL",
    timeframe:str="1m",
    pattern_markers=None,
    pattern_settings=None,
):
    """
    data: list[dict] with fields {time, open, high, low, close, volume}
    pattern_markers: list[dict] -> [{time, name, direction, confidence, index}]
    pattern_settings: dict persisted to localStorage by the component (optional)
    """
    return _COMPONENT_FUNC(
        data=data,
        symbol=symbol,
        timeframe=timeframe,
        patternMarkers=pattern_markers or [],
        patternSettings=pattern_settings or {},
        key=key,
        default=None,
    )
