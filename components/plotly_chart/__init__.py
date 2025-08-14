import os
import pathlib
import streamlit.components.v1 as components

_COMPONENT_PATH = os.path.join(os.path.dirname(__file__), "frontend")
print("plotly_chart frontend:", pathlib.Path(_COMPONENT_PATH).resolve(), "exists:", os.path.exists(_COMPONENT_PATH))

_plotly_chart = components.declare_component("plotly_chart", path=_COMPONENT_PATH)

def plotly_chart(uid, candles, shapes, layout=None, config=None, height=640, key=None):
    """
    Plotly candlestick with drawing tools. Receives tool changes via postMessage.
    Returns {'dirty': True, 'shapes': [...]} when shapes change.
    """
    return _plotly_chart(
        uid=uid,
        candles=candles,
        shapes=shapes,
        layout=layout or {},
        config=config or {},
        height=height,
        key=key,
        default=None,
    )

