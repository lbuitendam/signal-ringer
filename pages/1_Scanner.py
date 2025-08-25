# pages/1_Scanner.py
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from ui.sidebar import load_settings, save_settings
from st_trading_draw import st_trading_draw

# Patterns
from patterns.engine import (
    DEFAULT_CONFIG as PAT_DEFAULTS,
    detect_all,
    hits_to_markers,
)

# Strategies
from strategies.catalog import get_catalog
from engine.runner import build_strategy  # reuse factory


# ---------------- Consts / Page ----------------
LOCAL_TZ = "Europe/Berlin"
st.set_page_config(layout="wide", page_title="Scanner — Signal Ringer")
st.title("🧠 Scanner — Patterns & Strategies")

PERIOD_OPTIONS = {
    "1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo",
    "YTD": "ytd", "1Y": "1y", "2Y": "2y", "5Y": "5y"
}
INTERVAL_OPTIONS = ["1m","2m","5m","15m","30m","60m","90m","1d","1wk","1mo"]


# ---------------- Helpers ----------------
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
    col_map: Dict[str,str] = {}
    for key in ["open","high","low","close","adj close","volume"]:
        if key in cols:
            col_map[key] = cols[key]
        else:
            for c in df.columns:
                if c.lower().startswith(key):
                    col_map[key] = c
                    break
    out = pd.DataFrame(index=df.index.copy())
    for pretty, raw in (("Open","open"),("High","high"),("Low","low"),("Close","close")):
        if raw in col_map:
            out[pretty] = pd.to_numeric(df[col_map[raw]], errors="coerce")
    if "volume" in col_map:
        out["Volume"] = pd.to_numeric(df[col_map["volume"]], errors="coerce")
    out = out.dropna(subset=[c for c in ["Open","High","Low","Close"] if c in out.columns])
    if getattr(out.index, "tz", None) is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    o = pd.to_numeric(df["Open"], errors="coerce")
    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")
    prev = c.shift(1).bfill()
    tr = np.maximum.reduce([(h - l).values, np.abs(h - prev).values, np.abs(l - prev).values])
    s = pd.Series(tr, index=df.index)
    return s.ewm(span=n, adjust=False, min_periods=1).mean()

def strat_signals_to_markers(sigs: List[Dict[str,Any]], df: pd.DataFrame) -> List[Dict[str,Any]]:
    out = []
    for s in sigs:
        i = int(s["idx"])
        ts = df.index[i]
        up = s["side"] == "long"
        out.append({
            "id": f"strat-{i}-{s['name']}",
            "time": int(ts.timestamp()),
            "position": "belowBar" if up else "aboveBar",
            "shape": "arrowUp" if up else "arrowDown",
            "color": "#10b981" if up else "#ef4444",
            "text": f"{s['name']} ({'L' if up else 'S'})"
        })
    return out

def preview_markers(ts: pd.Timestamp, entry: float, sl: float, tps: List[float]) -> List[Dict[str,Any]]:
    base = int(ts.timestamp())
    m = [
        {"id": f"pv-entry-{base}", "time": base, "position": "inBar", "shape": "circle", "color": "#2563eb", "text": f"ENTRY {entry:.2f}"},
        {"id": f"pv-sl-{base}", "time": base, "position": "inBar", "shape": "circle", "color": "#ef4444", "text": f"SL {sl:.2f}"},
    ]
    for k, tp in enumerate(tps, start=1):
        m.append({"id": f"pv-tp{k}-{base}", "time": base, "position": "inBar", "shape": "circle", "color": "#10b981", "text": f"TP{k} {tp:.2f}"})
    return m


# ---------------- Settings / Watchlist (normalize before use) ----------------
def _normalize_watchlist(wl_raw: List[Any]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for item in wl_raw or []:
        if isinstance(item, dict):
            d = dict(item)
            d.setdefault("symbol", str(d.get("symbol", d.get("ticker",""))).upper())
            d.setdefault("enabled", True)
        else:
            d = {"symbol": str(item).upper(), "enabled": True}
        if d.get("symbol"):
            norm.append(d)
    # de-dupe by symbol, keep first
    seen = set()
    out = []
    for d in norm:
        s = d["symbol"].upper()
        if s not in seen:
            seen.add(s); out.append(d)
    return out

s = load_settings()
wl = _normalize_watchlist(s.get("watchlist", []))
watch_syms = [d["symbol"] for d in wl if d.get("enabled", True)]
examples = ["AAPL","MSFT","NVDA","SPY","QQQ","BTC-USD","ETH-USD","XAUUSD=X","XAGUSD=X"]

if "scanner_recent" not in st.session_state:
    st.session_state["scanner_recent"] = []
if "scanner_symbol" not in st.session_state:
    st.session_state["scanner_symbol"] = (watch_syms[0] if watch_syms else "AAPL")

def _pick(sym: str):
    sym = (sym or "").strip().upper()
    if not sym:
        return
    st.session_state["scanner_symbol"] = sym
    rec = list(st.session_state["scanner_recent"])
    if sym in rec:
        rec.remove(sym)
    st.session_state["scanner_recent"] = [sym] + rec[:9]
    st.rerun()


# ---------------- UI: Symbol picker (quick picks + recent + examples) ----------------
with st.container():
    st.write("**Symbol**")
    c1, c2, c3 = st.columns([3,1,1])
    with c1:
        st.text_input("Type any Yahoo ticker", key="scanner_input",
                      value=st.session_state["scanner_symbol"],
                      placeholder="e.g. AAPL, BTC-USD, XAUUSD=X …", label_visibility="collapsed")
    with c2:
        st.button("Scan", use_container_width=True, on_click=_pick,
                  args=(st.session_state.get("scanner_input",""),),
                  key="scanner_btn_scan")
    with c3:
        if st.button("📌 Pin", use_container_width=True, key="scanner_btn_pin"):
            sym = st.session_state.get("scanner_input","").strip().upper()
            if sym and sym not in watch_syms:
                wl.append({"symbol": sym, "asset": "stock", "timeframe": "5m", "adapter": "yfinance", "enabled": True})
                s["watchlist"] = wl
                save_settings(s)
                st.success(f"Added {sym} to watchlist")
                # refresh local cache
                watch_syms = [d["symbol"] for d in _normalize_watchlist(s.get("watchlist", [])) if d.get("enabled", True)]
    if watch_syms:
        st.caption("Quick picks — Watchlist")
        cols = st.columns(min(8, len(watch_syms)))
        for i, sym in enumerate(watch_syms[:24]):
            with cols[i % len(cols)]:
                st.button(sym, key=f"qp_w_{i}_{sym}", on_click=_pick, args=(sym,))
    if st.session_state["scanner_recent"]:
        st.caption("Recent")
        recs = st.session_state["scanner_recent"]
        cols = st.columns(min(8, len(recs)))
        for i, sym in enumerate(recs):
            with cols[i % len(cols)]:
                st.button(sym, key=f"qp_r_{i}_{sym}", on_click=_pick, args=(sym,))
    st.caption("Examples")
    cols = st.columns(min(8, len(examples)))
    for i, sym in enumerate(examples):
        with cols[i % len(cols)]:
            st.button(sym, key=f"qp_e_{i}_{sym}", on_click=_pick, args=(sym,))

ticker = st.session_state["scanner_symbol"]


# ---------------- Main controls ----------------
left, right = st.columns([1,1])
with left:
    period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()), index=2, key="scanner_period")
    interval = st.selectbox("Resolution", INTERVAL_OPTIONS, index=2, key="scanner_interval")
    show_volume = st.toggle("Show volume", value=True, key="scanner_showvol")
with right:
    st.markdown("**Engines**")
    show_patterns = st.checkbox("Show candlestick/classical patterns", value=True, key="scanner_showpat")
    show_strats = st.checkbox("Show strategy signals", value=True, key="scanner_showstrat")


# ---------------- Patterns config ----------------
if "patterns_state" not in st.session_state:
    st.session_state["patterns_state"] = {}
pkey = f"{ticker}@{interval}"
pst = st.session_state["patterns_state"].setdefault(
    pkey,
    {"enabled": True, "min_conf": float(PAT_DEFAULTS.get("min_confidence", 0.6)),
     "enabled_names": [], "cfg": PAT_DEFAULTS.copy(), "last_alert_idx": {}}
)

if show_patterns:
    with st.expander("Pattern Controls", expanded=False):
        pst["enabled"] = st.toggle("Enable pattern engine", value=bool(pst.get("enabled", True)), key=f"pat_on_{pkey}")
        mc = float(pst.get("min_conf", 0.6))
        mc = max(0.0, min(1.0, mc))
        pst["min_conf"] = float(st.slider("Only alert if confidence ≥", 0.0, 1.0, mc, 0.05, key=f"pat_min_{pkey}"))
        if st.button("Restore default thresholds", key=f"pat_rst_{pkey}"):
            pst["cfg"] = PAT_DEFAULTS.copy()
            st.success("Pattern thresholds restored.")
        all_names = [
            "Hammer","Inverted Hammer","Bullish Engulfing","Bearish Engulfing","Doji",
            "Morning Star","Evening Star","Bullish Harami","Bearish Harami","Tweezer Top","Tweezer Bottom",
            "Head & Shoulders","Inverse Head & Shoulders","Piercing Line","Dark Cloud Cover",
            "Three White Soldiers","Three Black Crows","Three Inside Up","Three Inside Down",
            "Three Outside Up","Three Outside Down","Marubozu","Rising Window","Falling Window",
            "Tasuki Up","Tasuki Down","Kicker Bull","Kicker Bear","Rising Three Methods","Falling Three Methods","Mat Hold",
        ]
        chosen = set(pst.get("enabled_names", []))
        sel: List[str] = []
        cols = st.columns(3)
        for i, nm in enumerate(all_names):
            with cols[i % 3]:
                if st.checkbox(nm, value=(nm in chosen), key=f"pat_{pkey}_{i}_{nm}"):
                    sel.append(nm)
        pst["enabled_names"] = sel


# ---------------- Strategies selection ----------------
cat = get_catalog()
cat_names = sorted({v["name"] for v in cat.values()})
enabled_block = s.get("strategies", {}).get("enabled", {})
if isinstance(enabled_block, list):  # legacy shape
    enabled_block = {n: {"enabled": True, "params": {}, "approved": True} for n in enabled_block}
enabled_names = [n for n, row in enabled_block.items() if row.get("enabled", True)]
valid_defaults = sorted([n for n in enabled_names if n in cat_names])
ignored = sorted([n for n in enabled_names if n not in cat_names])

with st.expander("Strategy Signals (select to compute locally)", expanded=False):
    if ignored:
        st.caption(f"Note: dropped unknown/legacy entries: {', '.join(ignored)}")
    sel_strats = st.multiselect(
        "Strategies",
        options=cat_names,
        default=valid_defaults,
        key=f"strats_{ticker}_{interval}"
    )
    min_conf_strat = st.slider("Min confidence (strategy)", 0.0, 1.0, 0.6, 0.05, key=f"minconf_{ticker}_{interval}")
    max_per_strat = st.number_input("Max signals/strategy (recent window)", min_value=1, max_value=10, value=3, step=1, key=f"maxper_{ticker}_{interval}")

    # persist selection back to settings (keep existing params)
    new_enabled: Dict[str, Dict[str, Any]] = {}
    for nm in sel_strats:
        row = dict(enabled_block.get(nm, {"enabled": True, "params": {}, "approved": True}))
        row["enabled"] = True
        new_enabled[nm] = row
    for nm in enabled_names:
        if (nm in cat_names) and (nm not in sel_strats):
            row = dict(enabled_block.get(nm, {"enabled": False, "params": {}, "approved": True}))
            row["enabled"] = False
            new_enabled[nm] = row

    s.setdefault("strategies", {})
    s["strategies"]["enabled"] = new_enabled
    save_settings(s)

s_enabled = s.get("strategies", {}).get("enabled", {})


# ---------------- Data fetch ----------------
yf_period, clamp_msg = clamp_period_for_interval(period_label, interval)
if clamp_msg:
    st.info(clamp_msg)

df = fetch_ohlcv(ticker, yf_period, interval)
if df.empty:
    st.warning("No data returned for this selection.")
    st.stop()

N = min(len(df), 400)
df_slice = df.iloc[-N:].copy()


# ---------------- Detect patterns ----------------
hits = []
pat_markers: List[Dict[str, Any]] = []
if show_patterns and pst.get("enabled", True) and pst.get("enabled_names"):
    try:
        hits = detect_all(df_slice, pst["enabled_names"], pst["cfg"])
        hits = [h for h in hits if h.confidence >= float(pst.get("min_conf", 0.6))]
    except Exception as e:
        st.error(f"Pattern detection failed: {e}")

if hits:
    offset = len(df) - len(df_slice)
    for h in hits:
        h.index += offset
        h.bars = [b + offset for b in h.bars]
    pat_markers = hits_to_markers(hits, df)


# ---------------- Detect strategy signals ----------------
strat_hits: List[Dict[str,Any]] = []
if show_strats:
    for nm in st.session_state.get(f"strats_{ticker}_{interval}", []):
        params = s_enabled.get(nm, {}).get("params", {})
        try:
            strat = build_strategy(nm, params)
            sigs = strat.signals(
                pd.DataFrame({
                    "open": df_slice["Open"].values,
                    "high": df_slice["High"].values,
                    "low": df_slice["Low"].values,
                    "close": df_slice["Close"].values,
                    "volume": (df_slice["Volume"].values if "Volume" in df_slice else np.zeros(len(df_slice)))
                }, index=df_slice.index)
            )
            sigs = [s for s in sigs if s.confidence >= float(min_conf_strat)]
            sigs = sigs[-int(max_per_strat):]
            for s1 in sigs:
                strat_hits.append({
                    "name": s1.name, "side": s1.side, "idx": int(s1.index),
                    "ts": df_slice.index[int(s1.index)], "conf": float(s1.confidence),
                })
        except Exception as e:
            st.warning(f"{nm} failed: {e}")

if strat_hits:
    offset = len(df) - len(df_slice)
    for sh in strat_hits:
        sh["idx"] = sh["idx"] + offset
        sh["ts"] = df.index[sh["idx"]]

strat_markers = strat_signals_to_markers(strat_hits, df)


# ---------------- Optional backtrade preview ----------------
preview: List[Dict[str, Any]] = []
with st.expander("Backtrade preview (pick one row below)"):
    pick_src = st.radio("Source", ["Strategy", "Pattern"], horizontal=True, key=f"picksrc_{ticker}_{interval}")
    if pick_src == "Strategy" and strat_hits:
        labels = [f"{h['ts']} — {h['name']} ({h['side']})" for h in strat_hits]
        pick = st.selectbox("Pick strategy signal", labels, key=f"pickstr_{ticker}_{interval}")
        sel = strat_hits[labels.index(pick)]
        i = sel["idx"]; side = sel["side"]
        a = atr(df)
        entry = float(df["Close"].iat[i])
        risk_val = float(1.5 * a.iat[i])
        if side == "long":
            sl = entry - risk_val; tps = [entry + 2*risk_val]
        else:
            sl = entry + risk_val; tps = [entry - 2*risk_val]
        preview = preview_markers(df.index[i], entry, sl, tps)
        st.caption(f"Preview → Entry {entry:.2f}, SL {sl:.2f}, TP1 {tps[0]:.2f}")
    elif pick_src == "Pattern" and hits:
        labels = [f"{df.index[h.index]} — {h.name} ({h.direction})" for h in hits]
        pick = st.selectbox("Pick pattern hit", labels, key=f"pickpat_{ticker}_{interval}")
        hsel = hits[labels.index(pick)]
        i = int(hsel.index)
        side = "long" if hsel.direction.lower().startswith("bull") else "short"
        a = atr(df)
        entry = float(df["Close"].iat[i])
        risk_val = float(1.5 * a.iat[i])
        if side == "long":
            sl = entry - risk_val; tps = [entry + 2*risk_val]
        else:
            sl = entry + risk_val; tps = [entry - 2*risk_val]
        preview = preview_markers(df.index[i], entry, sl, tps)
        st.caption(f"Preview → Entry {entry:.2f}, SL {sl:.2f}, TP1 {tps[0]:.2f}")
    else:
        st.caption("No rows yet.")


# ---------------- Render chart ----------------
ohlcv_payload = [
    dict(time=int(ts.timestamp()), open=float(o), high=float(h), low=float(l),
         close=float(c), volume=float(v) if ("Volume" in df and show_volume) else None)
    for ts, o, h, l, c, v in zip(
        df.index, df["Open"], df["High"], df["Low"], df["Close"],
        (df["Volume"] if "Volume" in df else [0]*len(df))
    )
]
markers: List[Dict[str, Any]] = []
if show_patterns: markers += pat_markers
if show_strats:   markers += strat_markers
if preview:       markers += preview

_ = st_trading_draw(
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


# ---------------- Tables ----------------
cA, cB = st.columns(2)

with cA:
    st.markdown("### Strategy signals")
    if strat_hits:
        df_s = pd.DataFrame([{
            "time": h["ts"].isoformat(), "symbol": ticker, "tf": interval,
            "strategy": h["name"], "dir": h["side"], "conf": round(h["conf"], 3),
            "price": float(df["Close"].iat[h["idx"]]),
        } for h in strat_hits])
        st.dataframe(df_s, use_container_width=True, height=320)
        st.download_button("Export strategy signals CSV", df_s.to_csv(index=False),
                           file_name=f"scanner_strat_{ticker}_{interval}.csv", key=f"dl_strat_{ticker}_{interval}")
    else:
        st.caption("No strategy signals at current settings.")

with cB:
    st.markdown("### Pattern hits")
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
        st.dataframe(df_hits, use_container_width=True, height=320)
        st.download_button("Export pattern hits CSV", df_hits.to_csv(index=False),
                           file_name=f"scanner_patterns_{ticker}_{interval}.csv", key=f"dl_pats_{ticker}_{interval}")
    else:
        st.caption("No patterns at current settings.")
