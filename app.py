# app.py
# Streamlit + Lightweight-Charts component with Indicators, Drawing Tools, and Pattern Detection

import math
import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from st_trading_draw import st_trading_draw

# ---- Pattern engine ----
from patterns.engine import (
    DEFAULT_CONFIG as PAT_DEFAULTS,
    detect_all,
    hits_to_markers,
)

# -------------------- Config --------------------
LOCAL_TZ = "Europe/Berlin"
st.set_page_config(page_title="Signal Ringer • Chart", layout="wide")

# -------------------- Sidebar (choose data first) --------------------
ASSET_CLASSES = ["Stocks", "ETFs", "Forex", "Crypto", "Commodities (Futures)"]
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
    "1D": "1d",
    "5D": "5d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "10Y": "10y",
    "Max": "max",
}
INTERVAL_OPTIONS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo"]

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

# -------------------- Indicator state scaffolding --------------------
if "gapless" not in st.session_state:
    st.session_state["gapless"] = True  # placeholder; LWC uses time-indexed data

OVERLAY_TYPES = {"SMA", "EMA", "BB", "VWAP"}
SUBPANE_TYPES = {"RSI", "MACD"}
DEFAULT_PARAMS = {
    "SMA": {"period": 20, "source": "close"},
    "EMA": {"period": 50, "source": "close"},
    "RSI": {"period": 14},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "BB": {"period": 20, "stddev": 2.0, "source": "close"},
    "VWAP": {"session": "daily"},
}
SOURCE_CHOICES = ["close", "open", "hlc3", "ohlc4"]
LINE_STYLES = {"Solid": None, "Dash": "dash", "Dot": "dot", "DashDot": "dashdot"}
DASH_MAP = {"Solid": "solid", "Dash": "dash", "Dot": "dot", "DashDot": "dash"}

def _profile_key(t: str, iv: str) -> str:
    return f"{t}@{iv}"

if "indicators_store" not in st.session_state:
    st.session_state["indicators_store"] = {}

prof_key = _profile_key(ticker, interval)
if prof_key not in st.session_state["indicators_store"]:
    st.session_state["indicators_store"][prof_key] = []
ind_list = st.session_state["indicators_store"][prof_key]

def add_indicator(kind: str):
    inst = {
        "id": str(uuid4()),
        "type": kind,
        "params": DEFAULT_PARAMS[kind].copy(),
        "color": "#33cccc" if kind in {"SMA", "EMA", "VWAP"} else ("#ffa600" if kind == "BB" else "#1f77b4"),
        "style": "Solid",
        "visible": True,
        "pane": "overlay" if kind in OVERLAY_TYPES else "sub",
        "scope": {"ticker": ticker, "interval": interval},
    }
    ind_list.append(inst)

def remove_indicator(ind_id: str):
    st.session_state["indicators_store"][prof_key] = [i for i in ind_list if i["id"] != ind_id]

# -------------------- Top bar + Options --------------------
st.markdown(
    """
<style>
  .topbar { position: sticky; top: 0; z-index: 9999;
            background: #0e1117; border-bottom: 1px solid #222; padding: 8px 12px; }
  .topbar-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .topbar-title { font-weight: 600; font-size: 16px; color: #e5e7eb; margin: 0; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="topbar"><div class="topbar-row">', unsafe_allow_html=True)
col_left, col_right = st.columns([1, 1])
with col_left:
    st.markdown('<p class="topbar-title">Signal Ringer — Chart</p>', unsafe_allow_html=True)
with col_right:
    pop = st.popover("Options")
    with pop:
        # Drawing options for the component
        st.markdown("#### Drawing Tools")
        enable_draw = st.toggle("Enable chart component", value=True, help="Uses the React lightweight-charts component.")
        magnet = st.toggle("Magnet snap to candles", value=True)
        st.caption("Export / Import drawings (JSON) are below the chart.")

        st.divider()

        # Indicators UI
        st.markdown("#### Indicators")
        cols = st.columns([2, 1])
        with cols[0]:
            new_kind = st.selectbox("Add Indicator", ["SMA", "EMA", "RSI", "MACD", "BB", "VWAP"], key="add_kind")
        with cols[1]:
            if st.button("Add", type="primary", use_container_width=True):
                add_indicator(new_kind)

        st.markdown("##### Current Indicators")
        if not ind_list:
            st.caption("No indicators yet. Add one above.")
        else:
            for ind in list(ind_list):
                with st.container(border=True):
                    top = st.columns([1.2, 0.9, 0.7, 0.8, 0.7, 0.5])
                    with top[0]:
                        st.markdown(f"**{ind['type']}** · `{ind['pane']}`")
                    with top[1]:
                        ind["visible"] = st.checkbox("Show", value=ind["visible"], key=f"vis_{ind['id']}")
                    with top[2]:
                        ind["color"] = st.color_picker("Color", value=ind["color"], key=f"col_{ind['id']}")
                    with top[3]:
                        ind["style"] = st.selectbox("Line", list(LINE_STYLES.keys()), index=0, key=f"sty_{ind['id']}")
                    with top[4]:
                        allowed = ["overlay", "sub"] if ind["type"] not in {"RSI", "MACD"} else ["sub"]
                        ind["pane"] = st.selectbox(
                            "Pane", allowed, index=allowed.index(ind["pane"]), key=f"pane_{ind['id']}"
                        )
                    with top[5]:
                        if st.button("Delete", key=f"del_{ind['id']}", use_container_width=True):
                            remove_indicator(ind["id"])
                            st.experimental_rerun()

                    # parameter editors
                    if ind["type"] in {"SMA", "EMA", "BB"}:
                        c = st.columns([1, 1, 1])
                        ind["params"]["period"] = int(
                            c[0].number_input("Period", 1, 5000, int(ind["params"]["period"]), key=f"per_{ind['id']}")
                        )
                        ind["params"]["source"] = c[1].selectbox(
                            "Source",
                            SOURCE_CHOICES,
                            index=SOURCE_CHOICES.index(ind["params"].get("source", "close")),
                            key=f"src_{ind['id']}",
                        )
                        if ind["type"] == "BB":
                            ind["params"]["stddev"] = float(
                                c[2].number_input(
                                    "Std Dev", 0.1, 10.0, float(ind["params"]["stddev"]), 0.1, key=f"std_{ind['id']}"
                                )
                            )
                    elif ind["type"] == "RSI":
                        ind["params"]["period"] = int(
                            st.number_input("Period", 2, 5000, int(ind["params"]["period"]), key=f"per_{ind['id']}")
                        )
                    elif ind["type"] == "MACD":
                        c = st.columns(3)
                        ind["params"]["fast"] = int(
                            c[0].number_input("Fast", 1, 1000, int(ind["params"]["fast"]), key=f"fast_{ind['id']}")
                        )
                        ind["params"]["slow"] = int(
                            c[1].number_input("Slow", 2, 2000, int(ind["params"]["slow"]), key=f"slow_{ind['id']}")
                        )
                        ind["params"]["signal"] = int(
                            c[2].number_input("Signal", 1, 1000, int(ind["params"]["signal"]), key=f"sig_{ind['id']}")
                        )
                        if ind["params"]["fast"] >= ind["params"]["slow"]:
                            st.warning("Fast must be < Slow.")

        st.divider()
        # ------------- Patterns (NEW) -------------
        st.markdown("#### Patterns")
        pkey = f"{ticker}@{interval}"
        if "patterns_state" not in st.session_state:
            st.session_state["patterns_state"] = {}
        pst = st.session_state["patterns_state"].setdefault(
            pkey,
            {
                "enabled": True,
                "min_conf": PAT_DEFAULTS["min_confidence"],
                "enabled_names": [
                    "Hammer",
                    "Inverted Hammer",
                    "Bullish Engulfing",
                    "Bearish Engulfing",
                    "Doji",
                    "Morning Star",
                    "Evening Star",
                    "Bullish Harami",
                    "Bearish Harami",
                    "Tweezer Top",
                    "Tweezer Bottom",
                ],
                "cfg": PAT_DEFAULTS.copy(),
                "last_alert_idx": {},
            },
        )

        colsP = st.columns([1, 1])
        with colsP[0]:
            pst["enabled"] = st.toggle("Enable pattern engine", value=bool(pst["enabled"]))
        with colsP[1]:
            pst["min_conf"] = float(
                st.slider("Only alert if confidence ≥", 0.0, 1.0, float(pst["min_conf"]), 0.05)
            )

        if st.button("Restore default thresholds"):
            pst["cfg"] = PAT_DEFAULTS.copy()
            st.success("Pattern thresholds restored.")

        with st.expander("Select patterns"):
            all_names = [
                "Hammer",
                "Inverted Hammer",
                "Bullish Engulfing",
                "Bearish Engulfing",
                "Doji",
                "Morning Star",
                "Evening Star",
                "Bullish Harami",
                "Bearish Harami",
                "Tweezer Top",
                "Tweezer Bottom",
                # v2 stubs (off by default)
                "Piercing Line",
                "Dark Cloud Cover",
                "Three White Soldiers",
                "Three Black Crows",
                "Three Inside Up",
                "Three Inside Down",
                "Three Outside Up",
                "Three Outside Down",
                "Marubozu",
                "Rising Window",
                "Falling Window",
                "Tasuki Up",
                "Tasuki Down",
                "Kicker Bull",
                "Kicker Bear",
                "Rising Three Methods",
                "Falling Three Methods",
                "Mat Hold",
                # Head & Shoulders will appear here once added to engine.RULES
            ]
            chosen = set(pst["enabled_names"])
            new_selected = []
            for nm in all_names:
                val = st.checkbox(nm, value=(nm in chosen))
                if val:
                    new_selected.append(nm)
            pst["enabled_names"] = new_selected

st.markdown("</div></div>", unsafe_allow_html=True)

# -------------------- Time & Period helpers --------------------
def _days_since_jan1_now() -> int:
    now = datetime.now(timezone.utc)
    jan1 = datetime(now.year, 1, 1, tzinfo=timezone.utc)
    return max(1, (now - jan1).days)

def _period_days(yf_period: str) -> float:
    return {
        "1d": 1,
        "5d": 5,
        "1mo": 30,
        "3mo": 90,
        "6mo": 182,
        "ytd": float(_days_since_jan1_now()),
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
    }.get(yf_period, math.inf)

def clamp_period_for_interval(period_key: str, interval: str) -> Tuple[str, Optional[str]]:
    yf_period = PERIOD_OPTIONS.get(period_key, "1d")
    days = _period_days(yf_period)
    if interval == "1m" and days > 7:
        return "7d", "1m data limited to last 7 days; period clamped to 7d."
    if interval in {"2m", "5m", "15m", "30m", "60m", "90m"} and days > 60:
        return "60d", "Intraday data (2m–90m) limited to ~60 days; period clamped to 60d."
    return yf_period, None

# -------------------- Data fetch --------------------
@st.cache_data(show_spinner=False, ttl=120)
def fetch_ohlcv(symbol: str, yf_period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=yf_period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # flatten columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    col_map = {}
    for key in ["open", "high", "low", "close", "adj close", "volume"]:
        if key in cols:
            col_map[key] = cols[key]
        else:
            for c in df.columns:
                if c.lower().startswith(key):
                    col_map[key] = c
                    break
    out = pd.DataFrame(index=df.index.copy())
    for pretty, raw in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")):
        if raw in col_map:
            out[pretty] = pd.to_numeric(df[col_map[raw]], errors="coerce")
    if "volume" in col_map:
        out["Volume"] = pd.to_numeric(df[col_map["volume"]], errors="coerce")
    out = out.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in out.columns])
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

# -------------------- Indicator math --------------------
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
    up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
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
    if "Volume" not in df or df["Volume"].isna().all() or (df["Volume"] == 0).all():
        return pd.Series([np.nan] * len(df), index=df.index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].astype(float)
    by_day = pd.Index(df.index.tz_convert("UTC").date, name="day")
    cum_pv = pd.Series(tp.values * vol.values, index=df.index).groupby(by_day).cumsum()
    cum_v = vol.groupby(by_day).cumsum()
    return (cum_pv / cum_v).astype(float)

# -------------------- Fetch + prepare data --------------------
clamped_period, clamp_msg = clamp_period_for_interval(period_label, interval)
if clamp_msg:
    st.info(clamp_msg)

df = fetch_ohlcv(ticker, clamped_period, interval)
if df.empty or {"Open", "High", "Low", "Close"}.difference(df.columns):
    st.warning("No data returned for this symbol/period/interval combination. Try another selection.")
    st.stop()

last_ts_local = to_local_ts(df.index[-1], LOCAL_TZ)
last_close = df["Close"].iloc[-1]
st.markdown(
    (
        f"### {ticker}  "
        f"<span style='opacity:0.8'>Last:</span> <strong>{last_close:,.4f}</strong>  "
        f"<span style='opacity:0.6'>@ {last_ts_local.strftime('%Y-%m-%d %H:%M:%S %Z')} (local)</span>"
    ),
    unsafe_allow_html=True,
)

# payload for the component (time in seconds)
ohlcv_payload = [
    dict(
        time=int(ts.timestamp()),
        open=float(o),
        high=float(h),
        low=float(l),
        close=float(c),
        volume=float(v) if ("Volume" in df and show_volume) else None,
    )
    for ts, o, h, l, c, v in zip(
        df.index,
        df["Open"],
        df["High"],
        df["Low"],
        df["Close"],
        (df["Volume"] if "Volume" in df else [0] * len(df)),
    )
]

# -------------------- Build indicator payloads --------------------
visible_inds = [i for i in ind_list if i.get("visible", True)]

def _line_series_from_pd(idx: pd.DatetimeIndex, series: pd.Series):
    rows = []
    for ts, v in series.dropna().items():
        try:
            rows.append({"time": int(ts.timestamp()), "value": float(v)})
        except Exception:
            continue
    return rows

overlay_indicators = []
for ind in visible_inds:
    if ind["type"] in {"SMA", "EMA", "BB", "VWAP"} and ind.get("pane", "overlay") == "overlay":
        color = ind.get("color", "#33cccc")
        dash = DASH_MAP.get(ind.get("style", "Solid"), "solid")
        if ind["type"] == "SMA":
            p = int(ind["params"].get("period", 20))
            src = ind["params"].get("source", "close")
            y = sma(get_source_series(df, src), p)
            overlay_indicators.append(
                {"id": f"SMA_{p}", "name": f"SMA({p})", "color": color, "width": 1.6, "dash": dash,
                 "data": _line_series_from_pd(df.index, y)}
            )
        elif ind["type"] == "EMA":
            p = int(ind["params"].get("period", 50))
            src = ind["params"].get("source", "close")
            y = ema(get_source_series(df, src), p)
            overlay_indicators.append(
                {"id": f"EMA_{p}", "name": f"EMA({p})", "color": color, "width": 1.6, "dash": dash,
                 "data": _line_series_from_pd(df.index, y)}
            )
        elif ind["type"] == "BB":
            p = int(ind["params"].get("period", 20))
            sd = float(ind["params"].get("stddev", 2.0))
            src = ind["params"].get("source", "close")
            m, u, l = bollinger(get_source_series(df, src), p, sd)
            overlay_indicators += [
                {"id": f"BB_LO_{p}", "name": "BB Lower", "color": color, "width": 1.0, "dash": "dash",
                 "data": _line_series_from_pd(df.index, l)},
                {"id": f"BB_UP_{p}", "name": "BB Upper", "color": color, "width": 1.0, "dash": "dash",
                 "data": _line_series_from_pd(df.index, u)},
                {"id": f"BB_MID_{p}", "name": "BB Mid", "color": color, "width": 1.0, "dash": "dot",
                 "data": _line_series_from_pd(df.index, m)},
            ]
        elif ind["type"] == "VWAP":
            y = vwap(df)
            overlay_indicators.append(
                {"id": "VWAP", "name": "VWAP", "color": color, "width": 2.0, "dash": dash,
                 "data": _line_series_from_pd(df.index, y)}
            )

pane_indicators = []

# RSI (can be multiple)
rsi_inds = [i for i in visible_inds if i["type"] == "RSI"]
if rsi_inds:
    lines = []
    for ind in rsi_inds:
        p = int(ind["params"].get("period", 14))
        y = rsi(get_source_series(df, "Close"), p)
        lines.append(
            {"id": f"RSI_{p}", "name": f"RSI({p})", "color": ind.get("color", "#7f7f7f"),
             "width": 1.6, "dash": "solid", "data": _line_series_from_pd(df.index, y)}
        )
    pane_indicators.append({
        "id": "RSI",
        "height": 140,
        "yRange": {"min": 0, "max": 100},
        "lines": lines,
        "hlines": [{"y": 70, "color": "#888", "dash": "dot"}, {"y": 30, "color": "#888", "dash": "dot"}],
    })

# MACD (one pane per MACD config)
for ind in [i for i in visible_inds if i["type"] == "MACD"]:
    f = int(ind["params"].get("fast", 12))
    s = int(ind["params"].get("slow", 26))
    sig = int(ind["params"].get("signal", 9))
    line, signal_line, hist = macd(get_source_series(df, "Close"), f, s, sig)
    pane_indicators.append({
        "id": f"MACD_{f}_{s}_{sig}",
        "height": 160,
        "lines": [
            {"id": f"MACD_L_{f}_{s}_{sig}", "name": f"MACD({f},{s})", "color": ind.get("color", "#1f77b4"),
             "width": 1.6, "dash": "solid", "data": _line_series_from_pd(df.index, line)},
            {"id": f"MACD_S_{f}_{s}_{sig}", "name": "Signal", "color": "#aaaaaa",
             "width": 1.2, "dash": "dot", "data": _line_series_from_pd(df.index, signal_line)},
        ],
        "hist": [{"id": f"MACD_H_{f}_{s}_{sig}", "name": "Hist", "color": "#60a5fa",
                  "data": _line_series_from_pd(df.index, hist)}],
    })

# -------------------- Pattern detection + markers --------------------
pattern_markers = []
profile_key = _profile_key(ticker, interval)

if "patterns_state" in st.session_state and profile_key in st.session_state["patterns_state"]:
    pst = st.session_state["patterns_state"][profile_key]
    if pst.get("enabled", True):
        # evaluate a tail slice for speed
        N = min(len(df), 400)
        df_slice = df.iloc[-N:].copy()
        hits = detect_all(df_slice, pst["enabled_names"], pst["cfg"])

        # offset to original df index
        offset = len(df) - N
        for h in hits:
            h.index += offset
            h.bars = [b + offset for b in h.bars]

        # alerts + throttle
        if "signals_log" not in st.session_state:
            st.session_state["signals_log"] = []
        last_idx = pst.setdefault("last_alert_idx", {})
        new_hits = []
        for h in hits:
            if h.confidence < pst["min_conf"]:
                continue
            li = last_idx.get(h.name, -10**9)
            if h.index - li >= pst["cfg"]["min_bars_between_alerts"]:
                last_idx[h.name] = h.index
                new_hits.append(h)
                ts_local = df.index[h.index].astimezone(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d %H:%M")
                try:
                    st.toast(f"[{ticker} {interval}] {h.name} @ {ts_local}  (conf {h.confidence:.2f})")
                except Exception:
                    st.info(f"[{ticker} {interval}] {h.name} @ {ts_local}  (conf {h.confidence:.2f})")
                st.session_state["signals_log"].append({
                    "time": df.index[h.index].isoformat(),
                    "symbol": ticker,
                    "tf": interval,
                    "pattern": h.name,
                    "dir": h.direction,
                    "conf": round(h.confidence, 3),
                    "price": float(df['Close'].iat[h.index]),
                })

        pattern_markers = hits_to_markers(hits, df)

# -------------------- Drawings persistence + component render --------------------
if "drawings" not in st.session_state:
    st.session_state["drawings"] = {}
initial_drawings = st.session_state["drawings"].get(profile_key, {})

if enable_draw:
    draw_state = st_trading_draw(
        ohlcv=ohlcv_payload,
        symbol=ticker,
        timeframe=interval,
        initial_drawings=initial_drawings,
        magnet=magnet,
        toolbar_default="docked-right",
        overlay_indicators=overlay_indicators,
        pane_indicators=pane_indicators,
        markers=pattern_markers,   # pattern markers to the component
        key=f"draw_{profile_key}",
    )
    if isinstance(draw_state, dict) and "drawings" in draw_state:
        st.session_state["drawings"][profile_key] = draw_state["drawings"]
else:
    st.info("Chart component disabled. Enable it in Options → Drawing Tools.")

# -------------------- Export / Import drawings --------------------
with st.expander("Drawings: Export / Import", expanded=False):
    colx, coly = st.columns(2)
    with colx:
        st.download_button(
            "Export drawings (JSON)",
            data=json.dumps(st.session_state["drawings"].get(profile_key, {}), indent=2),
            file_name=f"drawings_{profile_key}.json",
            mime="application/json",
            use_container_width=True,
        )
    with coly:
        up = st.file_uploader("Import drawings (JSON)", type=["json"])
        if up:
            try:
                payload = json.loads(up.read().decode("utf-8"))
                st.session_state["drawings"][profile_key] = payload
                st.success("Imported drawings")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

# -------------------- Signals log --------------------
with st.expander("Pattern Signals Log", expanded=False):
    if st.session_state.get("signals_log"):
        df_log = pd.DataFrame(st.session_state["signals_log"])[-200:]
        st.dataframe(df_log, use_container_width=True, height=240)
    else:
        st.caption("No signals yet.")
