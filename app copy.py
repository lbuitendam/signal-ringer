# app.py — Gapless Candles + Modebar Drawing (in-chart) + Datetime-anchored overlays
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ---------- Page ----------
st.set_page_config(layout="wide", page_title="Professional Trading Chart")

# ---------- Logging ----------
def log_debug(message: str):
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {message}\n")
        
# ---- Drawing defaults (Plus500-style)
DEFAULT_LINE_COLOR = "rgba(0,229,255,0.9)"
DEFAULT_FILL_COLOR = "rgba(0,229,255,0.25)"
DEFAULT_WIDTH = 2
DEFAULT_OPACITY = 0.25

# ---------- Maps & State ----------
RESOLUTION_MAP = {
    "1 Minute": "1m", "2 Minutes": "2m", "5 Minutes": "5m", "15 Minutes": "15m",
    "30 Minutes": "30m", "1 Hour": "1h", "1 Day": "1d", "1 Week": "1wk", "1 Month": "1mo"
}
RES_LADDER = ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
RES_TO_LABEL = {v: k for k, v in RESOLUTION_MAP.items()}
PERIOD_MAP = {
    "1 Day": timedelta(days=1), "5 Days": timedelta(days=5), "1 Month": timedelta(days=30),
    "3 Months": timedelta(days=90), "6 Months": timedelta(days=180), "1 Year": timedelta(days=365),
    "2 Years": timedelta(days=730), "5 Years": timedelta(days=1825), "Max": None
}

if "selected_period_label" not in st.session_state:
    st.session_state.selected_period_label = "1 Day"
if "interval_code" not in st.session_state:
    st.session_state.interval_code = RESOLUTION_MAP["1 Minute"]
if "drawings" not in st.session_state:
    st.session_state.drawings = []  # persistent, datetime-anchored overlays

# ---------- Sidebar: data controls ----------
st.sidebar.title("Chart Controls")
st.sidebar.markdown("---")
st.sidebar.header("Instrument Selection")
asset_class = st.sidebar.selectbox("Asset Class (Examples)", ["Commodities (Futures)", "Stocks", "Forex", "Crypto"])
ticker_examples = {
    "Stocks": "AAPL, GOOGL, MSFT",
    "Forex": "EURUSD=X, GBPJPY=X, XAUUSD=X",
    "Crypto": "BTC-USD, ETH-USD",
    "Commodities (Futures)": "CL=F (Crude Oil), GC=F (Gold), SI=F (Silver)"
}
st.sidebar.info(f"Examples: {ticker_examples[asset_class]}")
symbol = st.sidebar.text_input("Enter Ticker Symbol", "CL=F").upper()

st.sidebar.markdown("---")
st.sidebar.header("Timeframe")
c1, c2 = st.sidebar.columns(2)
with c1:
    st.session_state.selected_period_label = st.selectbox(
        "Period", list(PERIOD_MAP.keys()),
        index=list(PERIOD_MAP.keys()).index(st.session_state.selected_period_label)
    )
with c2:
    current_res_label = RES_TO_LABEL.get(st.session_state.interval_code, "1 Minute")
    sel_label = st.selectbox(
        "Resolution", list(RESOLUTION_MAP.keys()),
        index=list(RESOLUTION_MAP.keys()).index(current_res_label)
    )
    st.session_state.interval_code = RESOLUTION_MAP[sel_label]

cc1, cc2, _ = st.sidebar.columns([1,1,3])
with cc1:
    if st.button("◀︎ Finer"):
        try:
            i = RES_LADDER.index(st.session_state.interval_code)
            if i > 0:
                st.session_state.interval_code = RES_LADDER[i-1]
                st.rerun()
        except ValueError:
            pass
with cc2:
    if st.button("Coarser ▶︎"):
        try:
            i = RES_LADDER.index(st.session_state.interval_code)
            if i < len(RES_LADDER) - 1:
                st.session_state.interval_code = RES_LADDER[i+1]
                st.rerun()
        except ValueError:
            pass

st.sidebar.markdown("---")


# ---------- Helpers ----------
def hex_to_rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a:.3f})"

def prepare_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    expected = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        lower = {c.lower(): c for c in df.columns}
        for need in expected:
            if need not in df.columns and need.lower() in lower:
                df.rename(columns={lower[need.lower()]: need}, inplace=True)
        if not all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            return pd.DataFrame()
    for c in expected:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    if "Volume" in df.columns and (df["Volume"].fillna(0) > 0).any():
        df = df[df["Volume"].fillna(0) > 0]
    df = df[~df.index.duplicated(keep="first")]
    return df

def build_compressed_x_and_ticks(dt_index: pd.DatetimeIndex, max_ticks: int = 8):
    n = len(dt_index)
    x = list(range(n))
    if n == 0:
        return x, [], []
    step = max(1, n // max_ticks)
    tickvals = list(range(0, n, step))
    if tickvals[-1] != n - 1:
        tickvals.append(n - 1)
    span_days = (dt_index.max() - dt_index.min()).days if n > 1 else 0
    if span_days >= 365 * 2: fmt = "%Y-%m"
    elif span_days >= 60:    fmt = "%Y-%m-%d"
    else:                    fmt = "%Y-%m-%d %H:%M"
    ticktext = [dt_index[i].strftime(fmt) for i in tickvals]
    return x, tickvals, ticktext

def nearest_idx_from_dt(dt: pd.Timestamp, index: pd.DatetimeIndex) -> int:
    pos = index.searchsorted(dt)
    if pos <= 0: return 0
    if pos >= len(index): return len(index)-1
    before = index[pos-1]; after = index[pos]
    return pos-1 if (dt - before) <= (after - dt) else pos

# ---------- Data ----------
@st.cache_data(ttl=60)
def fetch_data(ticker, period_label: str, interval_code: str):
    end_date = datetime.now()
    period_delta = PERIOD_MAP[period_label]
    if interval_code == '1m' and period_delta and period_delta > timedelta(days=7):
        period_delta = timedelta(days=7)
    elif interval_code in ['2m', '5m', '15m', '30m', '1h'] and period_delta and period_delta > timedelta(days=60):
        period_delta = timedelta(days=60)
    if period_delta:
        start_date = end_date - period_delta
        data = yf.download(tickers=ticker, start=start_date, end=end_date,
                           interval=interval_code, auto_adjust=False, progress=False, group_by="column")
    else:
        data = yf.download(tickers=ticker, period="max",
                           interval=interval_code, auto_adjust=False, progress=False, group_by="column")
    return data

if st.sidebar.button("🔄 Refresh data cache"):
    try:
        fetch_data.clear()
        st.rerun()
    except Exception as e:
        log_debug(f"Cache clear failed: {e}")

# ---------- Main ----------
raw = fetch_data(symbol, st.session_state.selected_period_label, st.session_state.interval_code)
if raw.empty:
    st.error(f"No data found for '{symbol}'."); st.stop()
data = prepare_ohlcv_frame(raw)
if data.empty:
    st.warning("No valid OHLC rows for this timeframe."); st.stop()

x_idx, tickvals, ticktext = build_compressed_x_and_ticks(data.index)
time_strings = data.index.strftime("%Y-%m-%d %H:%M:%S")

# default times for overlay inputs (only if user left blank)
def _default_dt(i):
    if len(time_strings) == 0: return ""
    i = max(0, min(len(time_strings)-1, i))
    return time_strings[i]
for d in st.session_state.drawings:
    p = d.get("params", {})
    if "dtA" in p and (not p["dtA"]):
        p["dtA"] = _default_dt(len(time_strings)//4)
    if "dtB" in p and (not p["dtB"]):
        p["dtB"] = _default_dt(3*len(time_strings)//4)

# ---------- Chart ----------
# Build base figure (candles + volume) first
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Candles
candle_hover = [
    f"Time: {t}<br>Open: {o:.4f}<br>High: {h:.4f}<br>Low: {l:.4f}<br>Close: {c:.4f}"
    for t, o, h, l, c in zip(
        time_strings,
        data["Open"].astype(float),
        data["High"].astype(float),
        data["Low"].astype(float),
        data["Close"].astype(float),
    )
]
fig.add_trace(
    go.Candlestick(
        x=list(range(len(data))),
        open=data["Open"].astype(float),
        high=data["High"].astype(float),
        low=data["Low"].astype(float),
        close=data["Close"].astype(float),
        name="Candlestick",
        hovertext=candle_hover,
        hoverinfo="text",
    ),
    secondary_y=False,
)

# Volume (secondary y)
if "Volume" in data.columns and not data["Volume"].isna().all():
    fig.add_trace(
        go.Bar(
            x=list(range(len(data))),
            y=data["Volume"].astype(float).fillna(0),
            name="Volume",
            hoverinfo="skip",
            marker_color="rgba(128,128,128,0.5)",
        ),
        secondary_y=True,
    )

# Minimal trading-style defaults (spikes/rulers + shape defaults)
if "drawmode" not in st.session_state:
    st.session_state.drawmode = "zoom"   # default tool on load
if "uirev_token" not in st.session_state:
    st.session_state.uirev_token = "drawings"  # used to preserve/clear client shapes

fig.update_layout(
    template="plotly_dark",
    height=650,
    title=f"{symbol} Price and Volume",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    uirevision=st.session_state.uirev_token,
    newshape=dict(  # defaults for drawline/drawrect/drawopenpath/drawtext
        line_color="rgba(0,229,255,0.95)",
        line_width=2,
        fillcolor="rgba(0,229,255,0.20)",
        opacity=0.20,
    ),
)

fig.update_xaxes(
    title_text="Date/Time",
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    rangeslider_visible=False,
    fixedrange=False,
    showspikes=True,
    spikemode="across",
    spikesnap="cursor",
    spikethickness=1,
    spikedash="dot",
)
fig.update_yaxes(
    fixedrange=False,
    showspikes=True,
    spikemode="across",
    spikethickness=1,
    spikedash="dot",
)

# --- LEFT TOOLBAR + CHART LAYOUT ---
toolbar_col, chart_col = st.columns([0.12, 0.88])

with toolbar_col:
    st.markdown("**Tools**")
    if st.button("🔎 Zoom", use_container_width=True):
        st.session_state.drawmode = "zoom"
    if st.button("✋ Pan", use_container_width=True):
        st.session_state.drawmode = "pan"
    st.markdown("---")
    if st.button("／ Line", use_container_width=True):
        st.session_state.drawmode = "drawline"
    if st.button("▭ Rect", use_container_width=True):
        st.session_state.drawmode = "drawrect"
    if st.button("✏️ Path", use_container_width=True):
        st.session_state.drawmode = "drawopenpath"
    if st.button("🅣 Text", use_container_width=True):
        st.session_state.drawmode = "drawtext"
    if st.button("🧽 Erase", use_container_width=True):
        st.session_state.drawmode = "eraseshape"
    st.markdown("---")
    add_hline = st.button("— H-Line", use_container_width=True)
    clear_all = st.button("🗑 Clear", use_container_width=True)

# Apply current tool to the figure
fig.update_layout(dragmode=st.session_state.drawmode)

# Programmatic actions
if add_hline:
    last_price = float(data["Close"].iloc[-1])
    fig.add_shape(
        type="line", xref="x", yref="y",
        x0=0, y0=last_price, x1=len(data)-1, y1=last_price,
        line=dict(color="rgba(255,255,255,0.9)", width=2, dash="dash"),
        layer="above",
    )

if clear_all:
    # Force a fresh UI state so client-side drawings disappear
    st.session_state.uirev_token = f"clear-{datetime.now().timestamp()}"
    # Also clear any server-side overlays you might add later
    st.session_state.drawings = []

with chart_col:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,   # no top-right bar; we use the left toolbar
            "displaylogo": False,
            "scrollZoom": True,
            "editable": True,          # allow moving/resizing shapes/text after drawing
            "edits": {
                "shapePosition": True,
                "annotationPosition": True,
                "shapeLayer": True,
            },
        },
        key="main_chart",
    )

with st.expander("Show Raw Data Table"):
    st.dataframe(data.tail(200).style.format(precision=4))
