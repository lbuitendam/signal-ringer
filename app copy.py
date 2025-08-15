# app.py
# Streamlit + Plotly trading chart with GAPLESS + Options popover (Gapless + Indicators UI)
# Stack: streamlit, plotly.graph_objects, yfinance, pandas, numpy
# Notes:
# - Sidebar defines ticker/interval FIRST
# - Then we render a sticky top navbar with a single "Options" popover
# - Timestamps stored UTC, displayed in Europe/Berlin
# - Gapless removes closed-market gaps by mapping x to 0..N-1

import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ------------- Config -------------
LOCAL_TZ = "Europe/Berlin"
st.set_page_config(page_title="Trading Chart (Gapless + Options)", layout="wide")

# ------------- Sidebar (define ticker/interval FIRST) -------------
ASSET_CLASSES = ["Stocks","ETFs","Forex","Crypto","Commodities (Futures)"]
DEFAULT_BY_CLASS = {
    "Stocks": "AAPL",
    "ETFs": "SPY",
    "Forex": "EURUSD=X",
    "Crypto": "BTC-USD",
    "Commodities (Futures)": "GC=F",
}
EXAMPLES_BY_CLASS = {
    "Stocks": "Examples: AAPL, MSFT, NVDA, TSLA",
    "ETFs": "Examples: SPY, QQQ, IWM, EEM",
    "Forex": "Examples: EURUSD=X, USDJPY=X, GBPUSD=X, AUDUSD=X",
    "Crypto": "Examples: BTC-USD, ETH-USD, SOL-USD",
    "Commodities (Futures)": "Examples: CL=F, NG=F, SI=F, ZC=F",
}
PERIOD_OPTIONS = {
    "1D":"1d","5D":"5d","1M":"1mo","3M":"3mo","6M":"6mo","YTD":"ytd",
    "1Y":"1y","2Y":"2y","5Y":"5y","10Y":"10y","Max":"max",
}
INTERVAL_OPTIONS = ["1m","2m","5m","15m","30m","60m","90m","1d","5d","1wk","1mo"]

with st.sidebar:
    st.header("Chart Settings")
    asset_class = st.selectbox("Asset Class", ASSET_CLASSES, index=0)
    default_symbol = DEFAULT_BY_CLASS[asset_class]
    st.caption(EXAMPLES_BY_CLASS[asset_class])
    ticker = st.text_input("Ticker / Symbol", value=default_symbol).strip()
    period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()), index=0)
    interval = st.selectbox("Resolution", INTERVAL_OPTIONS, index=0)
    show_volume = st.toggle("Show volume", value=True)

if not ticker:
    st.stop()

# ------------- Options state (Indicators scaffold) -------------
if "gapless" not in st.session_state:
    st.session_state["gapless"] = True

OVERLAY_TYPES = {"SMA","EMA","BB","VWAP"}
SUBPANE_TYPES = {"RSI","MACD"}
DEFAULT_PARAMS = {
    "SMA":{"period":20,"source":"close"},
    "EMA":{"period":50,"source":"close"},
    "RSI":{"period":14},
    "MACD":{"fast":12,"slow":26,"signal":9},
    "BB":{"period":20,"stddev":2.0,"source":"close"},
    "VWAP":{"session":"daily"},
}
SOURCE_CHOICES = ["close","open","hlc3","ohlc4"]
LINE_STYLES = {"Solid":None,"Dash":"dash","Dot":"dot","DashDot":"dashdot"}

def symbol_key(t: str, iv: str) -> str:
    return f"{t}@{iv}"

if "indicators_store" not in st.session_state:
    st.session_state["indicators_store"] = {}  # key -> list of indicator dicts

def _ensure_profile(key: str):
    if key not in st.session_state["indicators_store"]:
        st.session_state["indicators_store"][key] = []

prof_key = symbol_key(ticker, interval)
_ensure_profile(prof_key)
ind_list = st.session_state["indicators_store"][prof_key]

def add_indicator(kind: str):
    inst = {
        "id": str(uuid4()),
        "type": kind,
        "params": DEFAULT_PARAMS[kind].copy(),
        "color": "#33cccc" if kind in {"SMA","EMA","VWAP"} else ("#ffa600" if kind=="BB" else "#1f77b4"),
        "style": "Solid",
        "visible": True,
        "pane": "overlay" if kind in OVERLAY_TYPES else "sub",
        "scope": {"ticker": ticker, "interval": interval},
    }
    ind_list.append(inst)

def remove_indicator(ind_id: str):
    st.session_state["indicators_store"][prof_key] = [i for i in ind_list if i["id"] != ind_id]

# ------------- Sticky Top Navbar + Options popover -------------
st.markdown("""
<style>
  .topbar { position: sticky; top: 0; z-index: 9999;
            background: #0e1117; border-bottom: 1px solid #222; padding: 8px 12px; }
  .topbar-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .topbar-title { font-weight: 600; font-size: 16px; color: #e5e7eb; margin: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="topbar"><div class="topbar-row">', unsafe_allow_html=True)
col_left, col_right = st.columns([1,1])
with col_left:
    st.markdown('<p class="topbar-title">Trading Chart</p>', unsafe_allow_html=True)
with col_right:
    pop = st.popover("Options")
    with pop:
        # Section: Gapless
        st.checkbox("Gapless axis (remove non-trading gaps)",
                    key="gapless",
                    help="Plot candles on an index-based x-axis to hide closed-market gaps.")

        st.divider()

        # Section: Indicators (UI scaffold; renderers can be wired later)
        st.markdown("#### Indicators")
        cols = st.columns([2,1])
        with cols[0]:
            new_kind = st.selectbox("Add Indicator", ["SMA","EMA","RSI","MACD","BB","VWAP"], key="add_kind")
        with cols[1]:
            if st.button("Add", type="primary", use_container_width=True):
                add_indicator(new_kind)

        st.markdown("##### Current Indicators")
        if not ind_list:
            st.caption("No indicators yet. Add one above.")
        else:
            for ind in list(ind_list):
                with st.container(border=True):
                    top = st.columns([1.2,0.9,0.7,0.8,0.7,0.5])
                    with top[0]:
                        st.markdown(f"**{ind['type']}** · `{ind['pane']}`")
                    with top[1]:
                        ind["visible"] = st.checkbox("Show", value=ind["visible"], key=f"vis_{ind['id']}")
                    with top[2]:
                        ind["color"] = st.color_picker("Color", value=ind["color"], key=f"col_{ind['id']}")
                    with top[3]:
                        ind["style"] = st.selectbox("Line", list(LINE_STYLES.keys()), index=0, key=f"sty_{ind['id']}")
                    with top[4]:
                        allowed = ["overlay","sub"] if ind["type"] not in {"RSI","MACD"} else ["sub"]
                        ind["pane"] = st.selectbox("Pane", allowed, index=allowed.index(ind["pane"]), key=f"pane_{ind['id']}")
                    with top[5]:
                        if st.button("Delete", key=f"del_{ind['id']}", use_container_width=True):
                            remove_indicator(ind["id"])
                            st.experimental_rerun()

                    if ind["type"] in {"SMA","EMA","BB"}:
                        c = st.columns([1,1,1])
                        ind["params"]["period"] = int(c[0].number_input("Period", 1, 5000, int(ind["params"]["period"]),
                                                                        key=f"per_{ind['id']}"))
                        ind["params"]["source"] = c[1].selectbox("Source", SOURCE_CHOICES,
                                                                 index=SOURCE_CHOICES.index(ind["params"].get("source","close")),
                                                                 key=f"src_{ind['id']}")
                        if ind["type"] == "BB":
                            ind["params"]["stddev"] = float(c[2].number_input("Std Dev", 0.1, 10.0, float(ind["params"]["stddev"]), 0.1,
                                                                              key=f"std_{ind['id']}"))
                    elif ind["type"] == "RSI":
                        ind["params"]["period"] = int(st.number_input("Period", 2, 5000, int(ind["params"]["period"]),
                                                                      key=f"per_{ind['id']}"))
                    elif ind["type"] == "MACD":
                        c = st.columns(3)
                        ind["params"]["fast"] = int(c[0].number_input("Fast", 1, 1000, int(ind["params"]["fast"]), key=f"fast_{ind['id']}"))
                        ind["params"]["slow"] = int(c[1].number_input("Slow", 2, 2000, int(ind["params"]["slow"]), key=f"slow_{ind['id']}"))
                        ind["params"]["signal"] = int(c[2].number_input("Signal", 1, 1000, int(ind["params"]["signal"]), key=f"sig_{ind['id']}"))
                        if ind["params"]["fast"] >= ind["params"]["slow"]:
                            st.warning("Fast must be < Slow.")
st.markdown("</div></div>", unsafe_allow_html=True)

gapless = bool(st.session_state.get("gapless", True))

# ------------- Helpers -------------
def _days_since_jan1_now() -> int:
    now = datetime.now(timezone.utc)
    jan1 = datetime(now.year, 1, 1, tzinfo=timezone.utc)
    return max(1, (now - jan1).days)

def _period_days(yf_period: str) -> float:
    return {
        "1d":1,"5d":5,"1mo":30,"3mo":90,"6mo":182,"ytd":float(_days_since_jan1_now()),
        "1y":365,"2y":730,"5y":1825,"10y":3650
    }.get(yf_period, math.inf)

def clamp_period_for_interval(period_key: str, interval: str) -> Tuple[str, Optional[str]]:
    yf_period = PERIOD_OPTIONS.get(period_key, "1d")
    days = _period_days(yf_period)
    if interval == "1m" and days > 7:
        return "7d", "1m data limited to last 7 days; period clamped to 7d."
    if interval in {"2m","5m","15m","30m","60m","90m"} and days > 60:
        return "60d", "Intraday data (2m–90m) limited to ~60 days; period clamped to 60d."
    return yf_period, None

@st.cache_data(show_spinner=False, ttl=120)
def fetch_ohlcv(symbol: str, yf_period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=yf_period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    col_map = {}
    for key in ["open","high","low","close","adj close","volume"]:
        if key in cols: col_map[key] = cols[key]
        else:
            for c in df.columns:
                if c.lower().startswith(key): col_map[key] = c; break
    out = pd.DataFrame(index=df.index.copy())
    for pretty, raw in (("Open","open"),("High","high"),("Low","low"),("Close","close")):
        if raw in col_map: out[pretty] = pd.to_numeric(df[col_map[raw]], errors="coerce")
    if "volume" in col_map: out["Volume"] = pd.to_numeric(df[col_map["volume"]], errors="coerce")
    out = out.dropna(subset=[c for c in ["Open","High","Low","Close"] if c in out.columns])
    # UTC tz-aware index
    if getattr(out.index, "tz", None) is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out

def to_local_ts(ts: pd.Timestamp, local_tz: str = LOCAL_TZ) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ZoneInfo(local_tz))

def _format_timestamp_index(idx: pd.DatetimeIndex, local_tz: str = LOCAL_TZ) -> List[str]:
    if len(idx) == 0: return []
    if idx.tz is None:
        local_idx = idx.tz_localize("UTC").tz_convert(ZoneInfo(local_tz))
    else:
        local_idx = idx.tz_convert(ZoneInfo(local_tz))
    span_days = (local_idx.max() - local_idx.min()).days if len(local_idx) > 1 else 0
    fmt = "%Y-%m-%d %H:%M" if span_days <= 7 else ("%Y-%m-%d" if span_days <= 365 else "%Y-%m")
    return [ts.strftime(fmt) for ts in local_idx]

def _gapless_ticks(n: int, idx: pd.DatetimeIndex, max_ticks: int = 8) -> Tuple[List[int], List[str]]:
    if n == 0: return [], []
    positions = sorted({int(round(v)) for v in np.linspace(0, n - 1, num=min(max_ticks, n))})
    labels_all = _format_timestamp_index(idx, LOCAL_TZ)
    labels = [labels_all[p] for p in positions]
    return positions, labels

# ------------- Data & chart -------------
clamped_period, clamp_msg = clamp_period_for_interval(period_label, interval)
if clamp_msg: st.info(clamp_msg)

df = fetch_ohlcv(ticker, clamped_period, interval)
if df.empty or {"Open","High","Low","Close"}.difference(df.columns):
    st.warning("No data returned for this symbol/period/interval combination. Try another selection.")
    st.stop()
# === Indicator calculations (NEW) ===
import numpy as np
import pandas as pd
# === Indicators state for current symbol@interval (NEW) ===
def _profile_key(t: str, iv: str) -> str:
    return f"{t}@{iv}"

if "indicators_store" not in st.session_state:
    st.session_state["indicators_store"] = {}

prof_key = _profile_key(ticker, interval)
ind_list = st.session_state["indicators_store"].get(prof_key, [])

# Only draw currently visible indicators
visible_inds = [i for i in ind_list if i.get("visible", True)]

# Decide if we need subpanes
need_rsi = any(i["type"] == "RSI" and i.get("visible", True) for i in visible_inds)
need_macd = any(i["type"] == "MACD" and i.get("visible", True) for i in visible_inds)
rows = 1 + int(need_rsi) + int(need_macd)

SOURCE_CHOICES = ["close", "open", "hlc3", "ohlc4"]

def get_source_series(df: pd.DataFrame, source: str) -> pd.Series:
    s = (source or "close").lower()
    if s == "close":
        return df["Close"].astype(float)
    if s == "open":
        return df["Open"].astype(float)
    if s == "hlc3":
        return (df["High"] + df["Low"] + df["Close"]) / 3.0
    if s == "ohlc4":
        return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    return df["Close"].astype(float)

def sma(series: pd.Series, period: int) -> pd.Series:
    p = max(1, int(period))
    return series.rolling(p, min_periods=1).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    p = max(1, int(period))
    return series.ewm(span=p, adjust=False, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    p = max(2, int(period))
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast = max(1, int(fast))
    slow = max(fast + 1, int(slow))  # ensure slow > fast
    signal = max(1, int(signal))
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def bollinger(series: pd.Series, period: int = 20, stddev: float = 2.0):
    p = max(1, int(period))
    m = sma(series, p)
    sd = series.rolling(p, min_periods=1).std(ddof=0)
    u = m + float(stddev) * sd
    l = m - float(stddev) * sd
    return m, u, l

def vwap(df: pd.DataFrame) -> pd.Series:
    # daily reset VWAP using typical price (H+L+C)/3
    if "Volume" not in df or df["Volume"].isna().all() or (df["Volume"] == 0).all():
        return pd.Series([np.nan]*len(df), index=df.index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].astype(float)
    by_day = pd.Index(df.index.tz_convert("UTC").date, name="day")
    cum_pv = pd.Series(tp.values * vol.values, index=df.index).groupby(by_day).cumsum()
    cum_v = vol.groupby(by_day).cumsum()
    return (cum_pv / cum_v).astype(float)

LINE_STYLES = {"Solid": None, "Dash": "dash", "Dot": "dot", "DashDot": "dashdot"}
def _dash(style_name: str | None) -> str | None:
    return LINE_STYLES.get(style_name or "Solid")

last_ts_local = to_local_ts(df.index[-1], LOCAL_TZ)
last_close = df["Close"].iloc[-1]
st.markdown(
    f"### {ticker}  "
    f"<span style='opacity:0.8'>Last:</span> **{last_close:,.4f}**  "
    f"<span style='opacity:0.6'>@ {last_ts_local.strftime('%Y-%m-%d %H:%M:%S %Z')} (local)</span>",
    unsafe_allow_html=True,
)

if st.session_state.get("gapless", True):
    x_vals = list(range(len(df)))
    tickvals, ticktext = _gapless_ticks(len(df), df.index)
    xaxis_kwargs = dict(type="linear", tickmode="array", tickvals=tickvals, ticktext=ticktext)
else:
    x_vals = df.index
    xaxis_kwargs = dict()

labels = _format_timestamp_index(df.index, LOCAL_TZ)
if rows == 1:
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
else:
    heights = [0.62] + ([0.19] if need_rsi else []) + ([0.19] if need_macd else [])
    specs = [[{"secondary_y": True}]] + [[{}] for _ in range(rows - 1)]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=heights, specs=specs)

row_price = 1
row_rsi = 2 if need_rsi else None
row_macd = (3 if (need_rsi and need_macd) else (2 if need_macd else None))

# Candles
fig.add_trace(
    go.Candlestick(
        x=x_vals, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        hovertext=[f"<b>{t}</b><br>Open {o:.4f}<br>High {h:.4f}<br>Low {l:.4f}<br>Close {c:.4f}"
                   for t,o,h,l,c in zip(labels, df["Open"], df["High"], df["Low"], df["Close"])],
        hoverinfo="text",
    ),
    row=row_price, col=1, secondary_y=False
)

# Volume (secondary y on price row)
if show_volume and "Volume" in df and not df["Volume"].isna().all() and not (df["Volume"] == 0).all():
    fig.add_trace(
        go.Bar(
            x=x_vals, y=df["Volume"], name="Volume", opacity=0.3,
            hovertext=[f"<b>{t}</b><br>Volume {int(v):,}" for t,v in zip(labels, df["Volume"])],
            hoverinfo="text",
        ),
        row=row_price, col=1, secondary_y=True
    )

# Prepare cache for indicator source series
_src_cache: dict[str, pd.Series] = {}
def _src(s: str) -> pd.Series:
    if s not in _src_cache:
        _src_cache[s] = get_source_series(df, s)
    return _src_cache[s]

# Overlays: SMA / EMA / BB / VWAP on price
for ind in visible_inds:
    if ind["type"] in {"SMA","EMA","BB","VWAP"} and ind.get("pane","overlay") == "overlay":
        col = ind.get("color", "#33cccc")
        dash = _dash(ind.get("style"))
        if ind["type"] == "SMA":
            p = int(ind["params"].get("period", 20))
            src = ind["params"].get("source", "close")
            y = sma(_src(src), p)
            fig.add_trace(go.Scatter(x=x_vals, y=y, name=f"SMA({p})",
                                     mode="lines", line=dict(width=1.6, color=col, dash=dash)),
                          row=row_price, col=1, secondary_y=False)
        elif ind["type"] == "EMA":
            p = int(ind["params"].get("period", 50))
            src = ind["params"].get("source", "close")
            y = ema(_src(src), p)
            fig.add_trace(go.Scatter(x=x_vals, y=y, name=f"EMA({p})",
                                     mode="lines", line=dict(width=1.6, color=col, dash=dash)),
                          row=row_price, col=1, secondary_y=False)
        elif ind["type"] == "BB":
            p = int(ind["params"].get("period", 20))
            sd = float(ind["params"].get("stddev", 2.0))
            src = ind["params"].get("source", "close")
            m,u,l = bollinger(_src(src), p, sd)
            fig.add_trace(go.Scatter(x=x_vals, y=l, name=f"BB Lower",
                                     mode="lines", line=dict(width=1, color=col, dash=dash)),
                          row=row_price, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=x_vals, y=u, name=f"BB Upper",
                                     mode="lines", line=dict(width=1, color=col, dash=dash),
                                     fill="tonexty", fillcolor="rgba(255,165,0,0.08)"),
                          row=row_price, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=x_vals, y=m, name=f"BB Mid",
                                     mode="lines", line=dict(width=1, color=col, dash=dash)),
                          row=row_price, col=1, secondary_y=False)
        elif ind["type"] == "VWAP":
            y = vwap(df)
            fig.add_trace(go.Scatter(x=x_vals, y=y, name="VWAP",
                                     mode="lines", line=dict(width=2, color=col, dash=dash)),
                          row=row_price, col=1, secondary_y=False)

# Subpane: RSI
if need_rsi and row_rsi is not None:
    # If multiple RSI indicators exist, draw all; default source close
    rsi_inds = [i for i in visible_inds if i["type"] == "RSI"]
    for ind in rsi_inds:
        p = int(ind["params"].get("period", 14))
        y = rsi(get_source_series(df, "Close"), p)
        fig.add_trace(go.Scatter(x=x_vals, y=y, name=f"RSI({p})",
                                 mode="lines", line=dict(width=1.6, color=ind.get("color","#7f7f7f"))),
                      row=row_rsi, col=1)
    # Guides
    fig.add_hline(y=70, line_width=1, line_dash="dot", line_color="#888", row=row_rsi, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dot", line_color="#888", row=row_rsi, col=1)
    fig.update_yaxes(range=[0,100], row=row_rsi, col=1, showgrid=True)

# Subpane: MACD
if need_macd and row_macd is not None:
    macd_inds = [i for i in visible_inds if i["type"] == "MACD"]
    for ind in macd_inds:
        f = int(ind["params"].get("fast", 12))
        s = int(ind["params"].get("slow", 26))
        sig = int(ind["params"].get("signal", 9))
        line, signal_line, hist = macd(get_source_series(df, "Close"), f, s, sig)
        col = ind.get("color","#1f77b4"); dash = _dash(ind.get("style"))
        fig.add_trace(go.Bar(x=x_vals, y=hist, name="MACD Hist", opacity=0.6),
                      row=row_macd, col=1)
        fig.add_trace(go.Scatter(x=x_vals, y=line, name=f"MACD({f},{s},{sig})",
                                 mode="lines", line=dict(width=1.6, color=col, dash=dash)),
                      row=row_macd, col=1)
        fig.add_trace(go.Scatter(x=x_vals, y=signal_line, name="Signal",
                                 mode="lines", line=dict(width=1.2, color="#aaaaaa", dash="dot")),
                      row=row_macd, col=1)
    fig.update_yaxes(showgrid=True, row=row_macd, col=1)

# Layout (keep your existing xaxis_kwargs / spikes etc.)
layout_xaxis = dict(
    rangeslider=dict(visible=False), showspikes=True, spikemode="across",
    spikesnap="cursor", spikethickness=1, spikedash="dot", showgrid=False,
)
layout_xaxis.update(xaxis_kwargs)

fig.update_layout(
    template="plotly_dark",
    height=650 if rows == 1 else (780 if rows == 2 else 920),
    margin=dict(l=10, r=10, t=10, b=10),
    uirevision="ohlc_ind",
    dragmode="pan",
    showlegend=False,
    xaxis=layout_xaxis,
    yaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor",
               spikethickness=1, spikedash="dot", zeroline=False),
)

# Volume axis if present
if any(getattr(tr, "type", "") == "bar" and tr.name == "Volume" for tr in fig.data):
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False, rangemode="tozero"))

st.plotly_chart(
    fig,
    use_container_width=True,
    config=dict(
        scrollZoom=True,
        displaylogo=False,
        modeBarButtonsToRemove=[
            "lasso2d",
            "select2d",
            "zoom2d",
            "autoScale2d",
            "resetScale2d",
            "toggleSpikelines",
            "toImage",
            "pan2d",
            "zoomIn2d",
            "zoomOut2d",
        ],
    ),
)
