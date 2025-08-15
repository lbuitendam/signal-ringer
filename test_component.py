import streamlit as st
from st_trading_draw import st_trading_draw

ohlcv = [
    {"time": 1716249600, "open": 100, "high": 110, "low": 95, "close": 105, "volume": 1000},
    {"time": 1716336000, "open": 105, "high": 112, "low": 101, "close": 108, "volume": 1200},
    {"time": 1716422400, "open": 108, "high": 115, "low": 104, "close": 112, "volume": 900},
]

state = st_trading_draw(
    ohlcv=ohlcv,
    symbol="TEST",
    timeframe="1d",
    initial_drawings={},
    magnet=True,
    toolbar_default="docked-right",
    key="demo",
)
st.write("Component returned:", state)
