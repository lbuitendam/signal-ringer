# st_trading_draw/__init__.py
from __future__ import annotations

from pathlib import Path
import streamlit.components.v1 as components


def st_trading_draw(
    *,
    ohlcv,
    symbol=None,
    timeframe=None,
    initial_drawings=None,
    magnet=False,
    toolbar_default: str = "docked-right",
    overlay_indicators=None,
    pane_indicators=None,
    markers=None,
    key=None,
):
    """
    Declare the component at call-time to avoid Streamlit's
    'module is None' edge case when importing from certain contexts.
    """

    dist_dir = (Path(__file__).parent / "frontend" / "dist").resolve()
    if not dist_dir.exists():
        raise RuntimeError(
            "st_trading_draw frontend bundle not found.\n"
            f"Expected: {dist_dir}\n"
            "Build it (e.g., in st_trading_draw/frontend run: npm install && npm run build)\n"
            "Or set ST_COMPONENT_DEV_URL to use a dev server."
        )

    comp = components.declare_component("st_trading_draw", path=str(dist_dir))
    return comp(
        ohlcv=ohlcv,
        symbol=symbol,
        timeframe=timeframe,
        initial_drawings=initial_drawings or {},
        magnet=magnet,
        toolbar_default=toolbar_default,
        overlay_indicators=overlay_indicators or [],
        pane_indicators=pane_indicators or [],
        markers=markers or [],
        key=key,
        default=None,
    )
