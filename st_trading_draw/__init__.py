# st_trading_draw/__init__.py
from pathlib import Path
import os
import streamlit.components.v1 as components

# Set to False to use a dev server (see below)
_RELEASE = True

base_dir = (Path(__file__).parent / "frontend").resolve()
dist_dir = (base_dir / "dist").resolve()    # Vite default
build_dir = (base_dir / "build").resolve()  # CRA default

if _RELEASE:
    # Accept either Vite's "dist" or CRA's "build"
    if dist_dir.exists():
        _COMP = components.declare_component("st_trading_draw", path=str(dist_dir))
    elif build_dir.exists():
        _COMP = components.declare_component("st_trading_draw", path=str(build_dir))
    else:
        raise RuntimeError(
            f"No frontend bundle found.\n"
            f"Looked for:\n  {dist_dir}\n  {build_dir}\n"
            f"Run 'npm run build' inside: {base_dir}"
        )
else:
    # Use a dev server (e.g., Vite) when developing the component
    dev_url = os.environ.get("ST_COMPONENT_DEV_URL", "http://localhost:3001")
    _COMP = components.declare_component("st_trading_draw", url=dev_url)

def st_trading_draw(
    *,
    ohlcv,
    symbol=None,
    timeframe=None,
    initial_drawings=None,
    magnet=False,
    toolbar_default="docked-right",
    overlay_indicators=None,
    pane_indicators=None,   # optional panes
    markers=None,           # pattern markers
    key=None,
):
    return _COMP(
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
