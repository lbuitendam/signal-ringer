# pages/2_Backtesting.py
from __future__ import annotations
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from patterns.engine import DEFAULT_CONFIG as PAT_DEFAULTS, detect_all

st.set_page_config(layout="wide", page_title="Backtesting — Signal Ringer")
st.title("🔁 Backtesting — Quick Pattern Study")

PERIOD_OPTIONS = {"1M":"1mo","3M":"3mo","6M":"6mo","YTD":"ytd","1Y":"1y","2Y":"2y","5Y":"5y"}
INTERVAL_OPTIONS = ["5m","15m","30m","60m","1d"]

def _period_days(yf_period: str) -> float:
    now = datetime.now(timezone.utc)
    jan1 = datetime(now.year, 1, 1, tzinfo=timezone.utc)
    ytd = max(1, (now - jan1).days)
    return {"1mo":30,"3mo":91,"6mo":182,"ytd":float(ytd),"1y":365,"2y":730,"5y":1825}.get(yf_period, math.inf)

def clamp_period_for_interval(yf_period: str, interval: str) -> tuple[str, str|None]:
    days = _period_days(yf_period)
    if interval in {"5m","15m","30m","60m"} and days > 60:
        return "60d", "Intraday limited to ~60 days → clamped to 60d."
    return yf_period, None

@st.cache_data(show_spinner=False, ttl=120)
def fetch_ohlcv(symbol: str, yf_period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=yf_period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    df = df.rename(columns={c: c.title() for c in df.columns})
    for col in ["Open","High","Low","Close"]: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" in df: df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"])
    if getattr(df.index, "tz", None) is None: df.index = df.index.tz_localize("UTC")
    else: df.index = df.index.tz_convert("UTC")
    return df

# ---- UI
left, right = st.columns([1,1])
with left:
    ticker = st.text_input("Symbol", value="AAPL").strip()
    period_label = st.selectbox("History", list(PERIOD_OPTIONS.keys()), index=2)  # 6M
    interval = st.selectbox("Resolution", INTERVAL_OPTIONS, index=3)  # 60m
with right:
    horizon = st.number_input("Forward horizon (bars)", min_value=1, max_value=200, value=10)
    min_conf = st.slider("Use signals with confidence ≥", 0.0, 1.0, 0.5, 0.05)

with st.expander("Choose patterns"):
    all_names = [
        "Hammer","Inverted Hammer","Bullish Engulfing","Bearish Engulfing","Doji",
        "Morning Star","Evening Star","Bullish Harami","Bearish Harami","Tweezer Top","Tweezer Bottom",
        "Head & Shoulders","Inverse Head & Shoulders","Piercing Line","Dark Cloud Cover",
        "Three White Soldiers","Three Black Crows","Three Inside Up","Three Inside Down",
        "Three Outside Up","Three Outside Down","Marubozu","Rising Window","Falling Window",
        "Tasuki Up","Tasuki Down","Kicker Bull","Kicker Bear","Rising Three Methods","Falling Three Methods","Mat Hold",
    ]
    selected = []
    cols = st.columns(3)
    for i, nm in enumerate(all_names):
        with cols[i % 3]:
            if st.checkbox(nm, value=(i < 6), key=f"bt_{nm}"):
                selected.append(nm)

yf_period, clamp_msg = clamp_period_for_interval(PERIOD_OPTIONS[period_label], interval)
if clamp_msg: st.info(clamp_msg)
df = fetch_ohlcv(ticker, yf_period, interval)
if df.empty:
    st.warning("No data.")
    st.stop()

st.markdown("### Run")
if st.button("Run backtest", type="primary"):
    cfg = PAT_DEFAULTS.copy()
    try:
        hits = detect_all(df, selected, cfg)
    except Exception as e:
        st.error(f"Detection failed: {e}")
        hits = []

    hits = [h for h in hits if h.confidence >= min_conf]
    if not hits:
        st.info("No signals at this threshold.")
    else:
        recs = []
        close = df["Close"].values
        for h in hits:
            t = h.index
            t2 = min(len(df) - 1, t + int(horizon))
            fwd = (close[t2] - close[t]) / close[t]
            signed = fwd if (h.direction.lower() in ("bull","long","buy")) else -fwd
            recs.append({
                "time": df.index[t].isoformat(),
                "pattern": h.name,
                "dir": h.direction,
                "conf": float(h.confidence),
                f"ret_{horizon}b": float(signed),
            })
        out = pd.DataFrame(recs)
        grp = out.groupby(["pattern","dir"]).agg(
            n=("conf","count"),
            win_rate=(f"ret_{horizon}b", lambda s: float((s > 0).mean())),
            avg_ret=(f"ret_{horizon}b","mean"),
            median_ret=(f"ret_{horizon}b","median"),
            avg_conf=("conf","mean"),
        ).reset_index().sort_values("win_rate", ascending=False)
        st.markdown("#### Results (by pattern & direction)")
        st.dataframe(grp, use_container_width=True, height=400)
        st.download_button("Export results CSV", grp.to_csv(index=False), file_name=f"bt_summary_{ticker}_{interval}.csv")

        with st.expander("Raw signals"):
            st.dataframe(out, use_container_width=True, height=300)
            st.download_button("Export raw CSV", out.to_csv(index=False), file_name=f"bt_raw_{ticker}_{interval}.csv")
