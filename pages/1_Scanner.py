# pages/1_Scanner.py
from __future__ import annotations
import math
from datetime import datetime, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from engine.singleton import get_engine
from ui.sidebar import load_settings
from st_trading_draw import st_trading_draw

from patterns.engine import (
    DEFAULT_CONFIG as PAT_DEFAULTS,
    detect_all,
    hits_to_markers,
)

LOCAL_TZ = "Europe/Berlin"

st.set_page_config(layout="wide", page_title="Scanner — Signal Ringer")

eng = get_engine()
st.title("🧠 Scanner — Patterns & Strategies")

# ---- Helpers
PERIOD_OPTIONS = {"1D": "1d","5D": "5d","1M": "1mo","3M": "3mo","6M": "6mo","YTD": "ytd","1Y": "1y","2Y": "2y","5Y": "5y"}
INTERVAL_OPTIONS = ["1m","2m","5m","15m","30m","60m","90m","1d","1wk","1mo"]

def _days_since_jan1_now() -> int:
    now = datetime.now(timezone.utc)
    jan1 = datetime(now.year, 1, 1, tzinfo=timezone.utc)
    return max(1, (now - jan1).days)

def _period_days(yf_period: str) -> float:
    return {
        "1d": 1, "5d": 5, "1w": 7, "1mo": 30, "3mo": 91, "6mo": 182,
        "ytd": float(_days_since_jan1_now()), "1y": 365, "2y": 730, "5y": 1825,
    }.get(yf_period, math.inf)

def clamp_period_for_interval(period_key: str, interval: str) -> tuple[str, str|None]:
    yf_period = PERIOD_OPTIONS.get(period_key, "1d")
    days = _period_days(yf_period)
    if interval == "1m" and days > 7:
        return "7d", "1m limited to last 7 days → clamped to 7d."
    if interval in {"2m","5m","15m","30m","60m","90m"} and days > 60:
        return "60d", "Intraday (2m–90m) limited to ~60 days → clamped to 60d."
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
                if c.lower().startswith(key):
                    col_map[key] = c; break
    out = pd.DataFrame(index=df.index.copy())
    for pretty, raw in (("Open","open"),("High","high"),("Low","low"),("Close","close")):
        if raw in col_map:
            out[pretty] = pd.to_numeric(df[col_map[raw]], errors="coerce")
    if "volume" in col_map:
        out["Volume"] = pd.to_numeric(df[col_map["volume"]], errors="coerce")
    out = out.dropna(subset=[c for c in ["Open","High","Low","Close"] if c in out.columns])
    if getattr(out.index, "tz", None) is None: out.index = out.index.tz_localize("UTC")
    else: out.index = out.index.tz_convert("UTC")
    return out

def to_local_ts(ts: pd.Timestamp, local_tz: str = LOCAL_TZ) -> pd.Timestamp:
    if ts.tzinfo is None: ts = ts.tz_localize("UTC")
    return ts.tz_convert(ZoneInfo(local_tz))

# ---- UI
s = load_settings()
wl = s.get("watchlist", [])
symbols = [w["symbol"] for w in wl] or ["AAPL"]
left, right = st.columns([1,1])

with left:
    ticker = st.selectbox("Symbol", symbols, index=0)
    period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()), index=2)  # 1M default
    interval = st.selectbox("Resolution", INTERVAL_OPTIONS, index=2)  # 5m default
    show_volume = st.toggle("Show volume", value=True)
with right:
    st.markdown("**Pattern Controls**")
    if "patterns_state" not in st.session_state:
        st.session_state["patterns_state"] = {}
    pkey = f"{ticker}@{interval}"
    pst = st.session_state["patterns_state"].setdefault(
        pkey,
        {"enabled": True, "min_conf": PAT_DEFAULTS["min_confidence"],
         "enabled_names": [], "cfg": PAT_DEFAULTS.copy(), "last_alert_idx": {}}
    )
    pst["enabled"] = st.toggle("Enable pattern engine", value=bool(pst["enabled"]))
    pst["min_conf"] = float(st.slider("Only alert if confidence ≥", 0.0, 1.0, float(pst["min_conf"]), 0.05))
    if st.button("Restore default thresholds"):
        pst["cfg"] = PAT_DEFAULTS.copy()
        st.success("Pattern thresholds restored.")

with st.expander("Select patterns"):
    all_names = [
        "Hammer","Inverted Hammer","Bullish Engulfing","Bearish Engulfing","Doji",
        "Morning Star","Evening Star","Bullish Harami","Bearish Harami","Tweezer Top","Tweezer Bottom",
        "Head & Shoulders","Inverse Head & Shoulders","Piercing Line","Dark Cloud Cover",
        "Three White Soldiers","Three Black Crows","Three Inside Up","Three Inside Down",
        "Three Outside Up","Three Outside Down","Marubozu","Rising Window","Falling Window",
        "Tasuki Up","Tasuki Down","Kicker Bull","Kicker Bear","Rising Three Methods","Falling Three Methods","Mat Hold",
    ]
    chosen = set(pst["enabled_names"])
    sel = []
    cols = st.columns(3)
    for i, nm in enumerate(all_names):
        with cols[i % 3]:
            if st.checkbox(nm, value=(nm in chosen), key=f"pat_{pkey}_{nm}"):
                sel.append(nm)
    pst["enabled_names"] = sel

yf_period, clamp_msg = clamp_period_for_interval(period_label, interval)
if clamp_msg: st.info(clamp_msg)

df = fetch_ohlcv(ticker, yf_period, interval)
if df.empty:
    st.warning("No data returned for this selection.")
    st.stop()

# ---- Detect on recent window
N = min(len(df), 400)
df_slice = df.iloc[-N:].copy()

hits = []
if pst.get("enabled", True) and pst["enabled_names"]:
    try:
        hits = detect_all(df_slice, pst["enabled_names"], pst["cfg"])
        hits = [h for h in hits if h.confidence >= pst["min_conf"]]
    except Exception as e:
        st.error(f"Pattern detection failed: {e}")
        hits = []

# Shift hit indexes to full DF
offset = len(df) - len(df_slice)
for h in hits:
    h.index += offset
    h.bars = [b + offset for b in h.bars]

# markers
markers = hits_to_markers(hits, df)

# ---- Render chart
ohlcv_payload = [
    dict(time=int(ts.timestamp()), open=float(o), high=float(h), low=float(l),
         close=float(c), volume=float(v) if ("Volume" in df and show_volume) else None)
    for ts, o, h, l, c, v in zip(
        df.index, df["Open"], df["High"], df["Low"], df["Close"],
        (df["Volume"] if "Volume" in df else [0]*len(df))
    )
]
draw_state = st_trading_draw(
    ohlcv=ohlcv_payload,
    symbol=ticker,
    timeframe=interval,
    initial_drawings={},
    magnet=True,
    toolbar_default="docked-right",
    overlay_indicators=[],
    pane_indicators=[],
    markers=markers,
    key=f"scanner_draw_{ticker}_{interval}",
)

# ---- Hits table
st.markdown("### Detected signals")
if hits:
    rows = []
    for h in hits[-300:]:
        rows.append({
            "time": df.index[h.index].isoformat(),
            "symbol": ticker,
            "tf": interval,
            "pattern": h.name,
            "dir": h.direction,
            "conf": round(h.confidence, 3),
            "price": float(df["Close"].iat[h.index]),
        })
    df_hits = pd.DataFrame(rows)
    st.dataframe(df_hits, use_container_width=True, height=360)
    st.download_button("Export hits CSV", df_hits.to_csv(index=False), file_name=f"scanner_hits_{ticker}_{interval}.csv")
else:
    st.caption("No patterns at current settings.")
