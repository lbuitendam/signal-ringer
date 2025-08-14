import os
import pathlib
import streamlit.components.v1 as components

_COMPONENT_PATH = os.path.join(os.path.dirname(__file__), "frontend")
print("plotly_toolbar frontend:", pathlib.Path(_COMPONENT_PATH).resolve(), "exists:", os.path.exists(_COMPONENT_PATH))

_plotly_toolbar = components.declare_component("plotly_toolbar", path=_COMPONENT_PATH)

def plotly_toolbar(uid, key=None, height=260, theme="dark"):
    """
    Renders a lightweight toolbar that posts tool changes to the chart via window.postMessage.
    Does not trigger Streamlit reruns.
    """
    return _plotly_toolbar(uid=uid, theme=theme, height=height, key=key, default=None)
