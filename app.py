# app.py — Signal-Ringer 4-Tile Dashboard (fixed)
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4
from dataclasses import fields as _dataclass_fields
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ui.sidebar import sidebar
from alerts.messages import maybe_toast
from risk.manager import RiskOptions
from storage import init_db, fetch_alerts, export_csv, insert_journal, fetch_journal
from patterns.engine import (
    DEFAULT_CONFIG as PAT_DEFAULTS,
    detect_all,
    hits_to_markers,
)

# -------- Optional / graceful imports --------
try:
    import yfinance as yf
except Exception:
    yf = None  # Fallback stubs used below

try:
    from st_trading_draw import st_trading_draw as tv_chart
except Exception:
    tv_chart = None

try:
    from engine.singleton import get_engine  # noqa: F401
except Exception:
    get_engine = None

try:
    from strategies.catalog import get_catalog as get_strategy_catalog  # returns {sid:{name, class,...}}
except Exception:
    get_strategy_catalog = None

# ---------------------- Page config ----------------------
st.set_page_config(
    page_title="Signal-Ringer — Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = DATA_DIR / "settings.json"

# ---------------------- Defaults ----------------------
DEFAULTS: Dict[str, Any] = {
    "watchlist": ["AAPL", "MSFT", "BTC-USD", "ETH-USD", "XAUUSD=X", "XAGUSD=X"],
    "timeframes": ["1m", "5m", "15m", "1H", "1D"],
    "dashboard": {
        "persist": True,
        "layout": [
            {
                "type": "Chart",
                "symbol": "AAPL",
                "timeframe": "1D",
                "studies": {"ema": [20, 50, 200], "bbands": {"length": 20, "stdev": 2.0}, "vwap": True},
                "compare": [],
            },
            {
                "type": "Chart",
                "symbol": "AAPL",
                "timeframe": "1H",
                "studies": {"ema": [9, 21, 50]},
                "compare": [],
            },
            {"type": "Metrics", "symbol": "AAPL", "timeframe": "15m"},
            {
                "type": "Backtest",
                "symbol": "AAPL",
                "timeframe": "1D",
                "strategies": ["EMA50/200 Golden Cross", "RSI Mean Revert", "Breakout 20-High"],
            },
        ],
    },
    "strategies": {"enabled": ["EMA50/200 Golden Cross", "RSI Mean Revert", "Breakout 20-High"], "params": {}},
    "risk": {"risk_pct": 1.0, "commission_bps": 1.0, "slippage_bps": 2.0},  # risk_pct stored as percent in UI
}

# ---------------------- Settings IO ----------------------
def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce_symbols(seq) -> List[str]:
    out: List[str] = []
    for x in (seq or []):
        if isinstance(x, dict):
            sym = x.get("symbol") or x.get("ticker") or ""
        else:
            sym = x
        s = str(sym).strip().upper()
        if s:
            out.append(s)
    return out


def load_settings() -> Dict[str, Any]:
    if SETTINGS_PATH.exists():
        try:
            disk = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            merged = _deep_merge(DEFAULTS, disk)
            # sanitize watchlist (avoid dicts -> unhashable)
            merged["watchlist"] = _coerce_symbols(merged.get("watchlist"))
            return merged
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULTS))


def save_settings(s: Dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(s, indent=2), encoding="utf-8")


SETTINGS: Dict[str, Any] = load_settings()

# ---------- Engine / DB / Sidebar integration ----------
init_db()
eng = get_engine() if get_engine else None
st.session_state.setdefault("seen_alert_ids", set())
st.session_state.setdefault("autorefresh_sec", 8)
if "patterns_state" not in st.session_state:
    st.session_state["patterns_state"] = {}
if "drawings" not in st.session_state:
    st.session_state["drawings"] = {}

nav, wl, opts, risk, qc = sidebar()

# --- normalize RiskOptions input (UI uses percent; engine expects fraction) ---
if not isinstance(risk, dict):
    risk = {}

rp_raw = risk.get("risk_pct", SETTINGS.get("risk", {}).get("risk_pct", 1.0))
try:
    rp = float(rp_raw)
except Exception:
    rp = 1.0
# convert percent→fraction if user entered 1..10
if rp > 1:
    rp /= 100.0
# clamp to sane range
rp = max(0.0005, min(0.10, rp))

# keep only fields RiskOptions accepts
_allowed = {f.name for f in _dataclass_fields(RiskOptions)}
base = {k: v for k, v in risk.items() if k in _allowed}
base["risk_pct"] = rp
risk_opts = RiskOptions(**base)

if eng:
    eng.configure(
        trackers=wl,
        strategies_cfg=opts,
        risk_opts=risk_opts,
        interval_sec=float(st.session_state["autorefresh_sec"]),
    )

# Auto-refresh while engine is running
if eng and (eng.is_running() or qc.get("engine_on")):
    st_autorefresh(interval=int(st.session_state["autorefresh_sec"]) * 1000, key="live_refresh")

# ---------------------- Data / Indicators ----------------------
INTERVAL_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "1H": "60m", "1D": "1d"}
PERIOD_BY_TIMEFRAME = {"1m": "7d", "5m": "60d", "15m": "60d", "1H": "730d", "1D": "5y"}

@st.cache_data(show_spinner=False, ttl=60)
def fetch_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    interval = INTERVAL_MAP.get(timeframe, "1d")
    period = PERIOD_BY_TIMEFRAME.get(timeframe, "1y")
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    # Normalize
    cols = {c.lower(): c for c in df.columns}
    need = {}
    for k in ["open", "high", "low", "close", "adj close", "volume"]:
        if k in cols:
            need[k] = cols[k]
        else:
            for c in df.columns:
                if c.lower().startswith(k):
                    need[k] = c
                    break
    out = pd.DataFrame(index=df.index.copy())
    for pretty, raw in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")):
        if raw in need:
            out[pretty] = pd.to_numeric(df[need[raw]], errors="coerce")
    if "volume" in need:
        out["Volume"] = pd.to_numeric(df[need["volume"]], errors="coerce")
    out = out.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in out.columns])
    if getattr(out.index, "tz", None) is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def ema(s: pd.Series, n: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").ewm(span=n, adjust=False, min_periods=1).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    d = c.diff()
    g = d.clip(lower=0.0).rolling(n).mean()
    l = (-d.clip(upper=0.0)).rolling(n).mean()
    rs = g / l.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, f: int = 12, s: int = 26, sig: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(close, f)
    slow = ema(close, s)
    mac = fast - slow
    sigl = ema(mac, sig)
    hist = mac - sigl
    return mac, sigl, hist


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    c_prev = close.shift(1).bfill()
    x = pd.concat(
        [
            (high - low),
            (high - c_prev).abs(),
            (low - c_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return x


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = _true_range(df["High"], df["Low"], df["Close"])
    return tr.ewm(span=n, adjust=False, min_periods=1).mean()


def bbands(close: pd.Series, length: int = 20, stdev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    width = (upper - lower) / ma.replace(0.0, np.nan)
    return ma, upper, lower, width


def stoch_k(df: pd.DataFrame, n: int = 14, smooth: int = 3) -> pd.Series:
    ll = df["Low"].rolling(n).min()
    hh = df["High"].rolling(n).max()
    k = 100 * (df["Close"] - ll) / (hh - ll).replace(0.0, np.nan)
    return k.rolling(smooth).mean()


def ema_slope(close: pd.Series, n: int = 50) -> pd.Series:
    e = ema(close, n)
    return e.diff()

# ---------------------- UI Helpers ----------------------
TILE_TYPES = ["Chart", "Scanner", "Backtest", "Journal", "Metrics"]
TIMEFRAMES = ["1m", "5m", "15m", "1H", "1D"]

nav_cols = st.columns(5)
with nav_cols[0]:
    st.page_link("app.py", label="Dashboard")
with nav_cols[1]:
    st.page_link("pages/1_Scanner.py", label="Scanner")
with nav_cols[2]:
    st.page_link("pages/2_Backtesting.py", label="Backtesting")
with nav_cols[3]:
    # Use a safe filename (no &). Make sure this file exists in pages/
    st.page_link("pages/3_History_and_Journal.py", label="History & Journal")
with nav_cols[4]:
    st.page_link("pages/4_User_Settings.py", label="User Settings")


def _safe_symbol_list() -> List[str]:
    return _coerce_symbols(SETTINGS.get("watchlist", []))


def _ensure_layout() -> List[Dict[str, Any]]:
    dash = SETTINGS.setdefault("dashboard", {})
    layout = dash.setdefault("layout", [])
    if not layout or len(layout) != 4:
        SETTINGS["dashboard"]["layout"] = json.loads(json.dumps(DEFAULTS["dashboard"]["layout"]))
        layout = SETTINGS["dashboard"]["layout"]
    # Fill missing keys defensively
    wl_safe = _safe_symbol_list()
    fallback_sym = wl_safe[0] if wl_safe else "AAPL"
    for tile in layout:
        tile.setdefault("type", "Chart")
        tile.setdefault("symbol", fallback_sym)
        tile.setdefault("timeframe", "1D")
        if tile["type"] == "Chart":
            tile.setdefault("studies", {"ema": [20, 50, 200], "bbands": {"length": 20, "stdev": 2.0}, "vwap": True})
            tile.setdefault("compare", [])
        if tile["type"] == "Backtest":
            tile.setdefault("strategies", SETTINGS.get("strategies", {}).get("enabled", []))
        if tile["type"] == "Scanner":
            tile.setdefault("filters", {"min_conf": 0.65, "min_rr": 1.8})
            tile.setdefault("scope", "watchlist")  # "watchlist" or "symbol"
        if tile["type"] == "Journal":
            tile.setdefault("max_rows", 50)
    return layout


def _markers_from_alerts(symbol: str, tf: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        a = fetch_alerts()
        if isinstance(a, pd.DataFrame) and not a.empty:
            f = a[(a["symbol"].astype(str).str.upper() == symbol.upper()) & (a["timeframe"] == tf)].tail(400)
            for _, r in f.iterrows():
                ts = pd.to_datetime(r.get("ts_utc") or r.get("time") or r.get("ts"), utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                pos = df.index.get_indexer([ts], method="nearest")
                if len(pos) == 0:
                    continue
                i = max(0, min(int(pos[0]), len(df) - 1))
                is_long = str(r.get("side", "")).lower() in ("long", "buy", "bull", "bullish")
                out.append(
                    {
                        "time": int(df.index[i].timestamp()),
                        "position": "belowBar" if is_long else "aboveBar",
                        "shape": "arrowUp" if is_long else "arrowDown",
                        "color": "#10b981" if is_long else "#ef4444",
                        "text": str(r.get("strategy", "Signal")),
                    }
                )
    except Exception:
        pass
    return out

# ---------------------- Sidebar: Global & Per-tile Editors ----------------------
st.sidebar.markdown("### Dashboard Controls")

# Watchlist / Timeframes
options_watch = sorted(set(_coerce_symbols(DEFAULTS["watchlist"]) + _safe_symbol_list()))
wl = st.sidebar.multiselect(
    "Watchlist",
    options=options_watch,
    default=_safe_symbol_list() or _coerce_symbols(DEFAULTS["watchlist"]),
)

tfs = st.sidebar.multiselect(
    "Timeframes",
    TIMEFRAMES,
    default=[x for x in SETTINGS.get("timeframes", TIMEFRAMES) if x in TIMEFRAMES],
)

persist = st.sidebar.checkbox(
    "Persist layout to settings.json",
    value=bool(SETTINGS.get("dashboard", {}).get("persist", True)),
)

# Risk/Costs (displayed in PERCENT; stored as percent)
st.sidebar.markdown("**Risk & Costs**")
risk_pct = st.sidebar.number_input(
    "Risk % (display only)", min_value=0.1, max_value=10.0,
    value=float(SETTINGS.get("risk", {}).get("risk_pct", 1.0)), step=0.1,
)
comm_bps = st.sidebar.number_input(
    "Commission (bps)", min_value=0.0, max_value=50.0,
    value=float(SETTINGS.get("risk", {}).get("commission_bps", 1.0)), step=0.5,
)
slip_bps = st.sidebar.number_input(
    "Slippage (bps)", min_value=0.0, max_value=100.0,
    value=float(SETTINGS.get("risk", {}).get("slippage_bps", 2.0)), step=0.5,
)

# Persist sidebar values back to SETTINGS (strings only)
SETTINGS["watchlist"] = _coerce_symbols(wl)
SETTINGS["timeframes"] = [tf for tf in tfs if tf in TIMEFRAMES]
SETTINGS.setdefault("risk", {})
SETTINGS["risk"].update({
    "risk_pct": float(risk_pct),  # still percent here!
    "commission_bps": float(comm_bps),
    "slippage_bps": float(slip_bps),
})
SETTINGS.setdefault("dashboard", {})
SETTINGS["dashboard"]["persist"] = bool(persist)

# Per-tile editors
st.sidebar.markdown("---")
st.sidebar.markdown("### Tile Editors")

layout = _ensure_layout()


def _tile_editor(i: int, tile: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.markdown(f"**Tile {i+1}**")
    ttype = st.sidebar.selectbox(
        f"Type #{i+1}", TILE_TYPES, index=TILE_TYPES.index(tile.get("type", "Chart")), key=f"tile_type_{i}"
    )
    sym = st.sidebar.text_input(f"Symbol #{i+1}", value=str(tile.get("symbol", "AAPL")), key=f"tile_sym_{i}")
    tf = st.sidebar.selectbox(
        f"Timeframe #{i+1}", TIMEFRAMES, index=max(0, TIMEFRAMES.index(tile.get("timeframe", "1D"))), key=f"tile_tf_{i}"
    )

    new_tile = {"type": ttype, "symbol": sym.strip().upper(), "timeframe": tf}

    if ttype == "Chart":
        studies = tile.get("studies", {"ema": [20, 50, 200], "bbands": {"length": 20, "stdev": 2.0}, "vwap": True})
        ema_csv = st.sidebar.text_input(
            f"EMAs #{i+1} (csv)", value=",".join(map(str, studies.get("ema", [20, 50, 200]))), key=f"tile_ema_{i}"
        )
        bb_len = st.sidebar.number_input(
            f"BB length #{i+1}", 5, 200, int(studies.get("bbands", {}).get("length", 20)), key=f"tile_bb_len_{i}"
        )
        bb_sd = st.sidebar.number_input(
            f"BB stdev #{i+1}", 0.5, 6.0, float(studies.get("bbands", {}).get("stdev", 2.0)), step=0.1, key=f"tile_bb_sd_{i}"
        )
        vwap_on = st.sidebar.checkbox(f"VWAP #{i+1}", value=bool(studies.get("vwap", True)), key=f"tile_vwap_{i}")
        compare_csv = st.sidebar.text_input(
            f"Compare (0–3, csv) #{i+1}", value=",".join(tile.get("compare", []))[:200], key=f"tile_cmp_{i}"
        )
        new_tile.update(
            {
                "studies": {
                    "ema": [int(x) for x in ema_csv.split(",") if x.strip().isdigit()][:6],
                    "bbands": {"length": int(bb_len), "stdev": float(bb_sd)},
                    "vwap": bool(vwap_on),
                },
                "compare": [x.strip().upper() for x in compare_csv.split(",") if x.strip()][:3],
            }
        )

    elif ttype == "Backtest":
        if get_strategy_catalog:
            try:
                cat = get_strategy_catalog()
                strat_names = sorted({v["name"] for v in cat.values()})
            except Exception:
                strat_names = DEFAULTS["strategies"]["enabled"]
        else:
            strat_names = DEFAULTS["strategies"]["enabled"]
        default_sel = tile.get("strategies", strat_names)
        sel = st.sidebar.multiselect(
            f"Strategies #{i+1}", options=strat_names, default=[x for x in default_sel if x in strat_names], key=f"tile_bt_strats_{i}"
        )
        new_tile["strategies"] = sel

    elif ttype == "Scanner":
        scope = st.sidebar.selectbox(
            f"Scope #{i+1}", ["watchlist", "symbol"], index=0 if tile.get("scope", "watchlist") == "watchlist" else 1, key=f"tile_sc_scope_{i}"
        )
        minc = st.sidebar.slider(
            f"Min Confidence #{i+1}", 0.0, 1.0, float(tile.get("filters", {}).get("min_conf", 0.65)), 0.05, key=f"tile_sc_minc_{i}"
        )
        minrr = st.sidebar.number_input(
            f"Min RR #{i+1}", 0.5, 5.0, float(tile.get("filters", {}).get("min_rr", 1.8)), step=0.1, key=f"tile_sc_minrr_{i}"
        )
        new_tile.update({"scope": scope, "filters": {"min_conf": float(minc), "min_rr": float(minrr)}})

    elif ttype == "Journal":
        mr = st.sidebar.number_input(f"Max rows #{i+1}", 10, 500, int(tile.get("max_rows", 50)), key=f"tile_journal_rows_{i}")
        new_tile.update({"max_rows": int(mr)})

    return new_tile


new_layout = []
for i, tile in enumerate(layout):
    new_layout.append(_tile_editor(i, tile))

if st.sidebar.button("Apply & Refresh", use_container_width=True):
    SETTINGS["dashboard"]["layout"] = new_layout
    SETTINGS["dashboard"]["persist"] = persist
    save_settings(SETTINGS) if persist else None
    st.rerun()

# ---------------------- Tile Renderers ----------------------
def _candles_altair(df: pd.DataFrame, title: str = ""):
    import altair as alt

    data = df.reset_index().rename(columns={"index": "ts"})
    data["ts"] = pd.to_datetime(data["Datetime"] if "Datetime" in data.columns else data["ts"])
    base = alt.Chart(data).properties(height=300, title=title)
    rule = base.mark_rule().encode(x="ts:T", y="Low:Q", y2="High:Q")
    bars = base.mark_bar().encode(
        x="ts:T",
        y="Open:Q",
        y2="Close:Q",
        color=alt.condition("datum.Open <= datum.Close", alt.value("#10b981"), alt.value("#ef4444")),
    )
    return rule + bars


def _chart_tile(tile: Dict[str, Any]):
    sym = tile["symbol"]
    tf = tile["timeframe"]
    df = fetch_ohlcv(sym, tf)
    if df.empty:
        st.warning(f"No data for {sym} {tf}.")
        return

    # --- Patterns (per symbol@tf) ---
    pkey = f"{sym}@{tf}"
    pst = st.session_state["patterns_state"].setdefault(
        pkey,
        {
            "enabled": True,
            "min_conf": PAT_DEFAULTS["min_confidence"],
            "enabled_names": [],
            "cfg": PAT_DEFAULTS.copy(),
            "last_alert_idx": {},
        },
    )
    with st.expander(f"Patterns — {sym} {tf}", expanded=False):
        colsP = st.columns(2)
        pst["enabled"] = colsP[0].toggle("Enable", value=bool(pst["enabled"]), key=f"pat_on_{pkey}")
        pst["min_conf"] = float(
            colsP[1].slider("Min confidence", 0.0, 1.0, float(pst["min_conf"]), 0.05, key=f"pat_min_{pkey}")
        )
        all_names = [
            "Hammer","Inverted Hammer","Bullish Engulfing","Bearish Engulfing","Doji",
            "Morning Star","Evening Star","Bullish Harami","Bearish Harami","Tweezer Top","Tweezer Bottom",
            "Head & Shoulders","Inverse Head & Shoulders","Piercing Line","Dark Cloud Cover",
            "Three White Soldiers","Three Black Crows","Three Inside Up","Three Inside Down",
            "Three Outside Up","Three Outside Down","Marubozu","Rising Window","Falling Window",
            "Tasuki Up","Tasuki Down","Kicker Bull","Kicker Bear","Rising Three Methods","Falling Three Methods","Mat Hold",
        ]
        sel: List[str] = []
        colsN = st.columns(3)
        for i, nm in enumerate(all_names):
            if colsN[i % 3].checkbox(nm, value=(nm in pst["enabled_names"]), key=f"pat_{pkey}_{nm}"):
                sel.append(nm)
        pst["enabled_names"] = sel
        if st.button("Restore default thresholds", key=f"pat_rst_{pkey}"):
            pst["cfg"] = PAT_DEFAULTS.copy()
            st.success("Restored.")

    # Build overlays (EMAs/BB/VWAP)
    studies = tile.get("studies", {})
    ema_list = studies.get("ema", [])
    bb_cfg = studies.get("bbands", {"length": 20, "stdev": 2.0})
    vwap_on = bool(studies.get("vwap", True))

    # --- Pattern markers (recent window) ---
    pat_markers: List[Dict[str, Any]] = []
    if pst.get("enabled") and pst["enabled_names"]:
        df_slice = df.iloc[-min(len(df), 400) :].copy()
        try:
            hits = detect_all(df_slice, pst["enabled_names"], pst["cfg"])
            hits = [h for h in hits if h.confidence >= pst["min_conf"]]
            # align to full df
            off = len(df) - len(df_slice)
            for h in hits:
                h.index += off
                h.bars = [b + off for b in h.bars]
            pat_markers = hits_to_markers(hits, df)
        except Exception:
            pat_markers = []

    # --- Strategy markers from storage ---
    strat_markers = _markers_from_alerts(sym, tf, df)
    all_markers = pat_markers + strat_markers

    # --- Drawing persistence ---
    initial_drawings = st.session_state["drawings"].get(pkey, {})

    if tv_chart:
        ohlcv = [
            dict(
                time=int(ts.timestamp()),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v) if "Volume" in df else None,
            )
            for ts, o, h, l, c, v in zip(
                df.index, df["Open"], df["High"], df["Low"], df["Close"], (df["Volume"] if "Volume" in df else [0] * len(df))
            )
        ]
        overlays = []
        for n in ema_list[:6]:
            overlays.append({"type": "ema", "params": {"length": int(n)}})
        if bb_cfg:
            overlays.append(
                {"type": "bbands", "params": {"length": int(bb_cfg.get("length", 20)), "stdev": float(bb_cfg.get("stdev", 2.0))}}
            )
        if vwap_on:
            overlays.append({"type": "vwap", "params": {}})

        panes = [{"type": "rsi", "params": {"length": 14}}, {"type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}]

        state = tv_chart(
            ohlcv=ohlcv,
            symbol=sym,
            timeframe=tf,
            initial_drawings=initial_drawings,
            magnet=True,
            toolbar_default="docked-right",
            overlay_indicators=overlays,
            pane_indicators=panes,
            markers=all_markers,
            key=f"chart_{sym}_{tf}",
        )
        if isinstance(state, dict) and "drawings" in state:
            st.session_state["drawings"][pkey] = state["drawings"]

        with st.expander("Drawings: Export / Import", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Export drawings (JSON)",
                    data=json.dumps(st.session_state["drawings"].get(pkey, {}), indent=2),
                    file_name=f"drawings_{pkey}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with c2:
                up = st.file_uploader("Import drawings (JSON)", type=["json"], key=f"import_{pkey}")
                if up:
                    try:
                        payload = json.loads(up.read().decode("utf-8"))
                        st.session_state["drawings"][pkey] = payload
                        st.success("Imported.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {e}")
    else:
        # Altair fallback (no markers/drawings)
        ch = _candles_altair(df, title=f"{sym} — {tf}")
        for n in ema_list[:4]:
            df[f"EMA{n}"] = ema(df["Close"], int(n))
        import altair as alt

        chart = ch
        for n in ema_list[:4]:
            line = (
                alt.Chart(df.reset_index())
                .mark_line()
                .encode(x="index:T", y=alt.Y(f"EMA{n}:Q", axis=alt.Axis(title=None)))
            )
            chart = chart + line
        st.altair_chart(chart, use_container_width=True)


def _dedupe_alerts(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["symbol", "tf", "strategy", "side"]:
        if col not in df.columns:
            df[col] = ""
    df["key"] = (
        df["symbol"].astype(str)
        + "|"
        + df["tf"].astype(str)
        + "|"
        + df["strategy"].astype(str)
        + "|"
        + df["side"].astype(str)
    )
    df = df.sort_values(by=["confidence", "msg"], ascending=[False, True]).drop_duplicates(subset=["key"], keep="first")
    df = df.sort_values(by=["confidence", "price"], ascending=[False, True])
    return df


def _scanner_tile(tile: Dict[str, Any]):
    # Fetch alerts, normalize to list-of-dicts
    rows: List[Dict[str, Any]] = []
    try:
        dfA = fetch_alerts(limit=500)
        if isinstance(dfA, pd.DataFrame):
            rows = dfA.to_dict("records")
        elif isinstance(dfA, list):
            rows = dfA
        else:
            rows = []
    except Exception:
        rows = []
    df = _dedupe_alerts(rows)
    if df.empty:
        st.info("No live signals yet.")
        return

    # Apply scope / filters
    scope = tile.get("scope", "watchlist")
    filters = tile.get("filters", {"min_conf": 0.65, "min_rr": 1.8})
    minc = float(filters.get("min_conf", 0.65))
    minrr = float(filters.get("min_rr", 1.8))

    wl_upper = set(_safe_symbol_list())
    if scope == "symbol":
        df = df[df["symbol"].astype(str).str.upper() == tile["symbol"].upper()]
    else:
        df = df[df["symbol"].astype(str).str.upper().isin(wl_upper)]
    if "confidence" in df.columns:
        df = df[df["confidence"].fillna(0.0) >= minc]
    if "rr" in df.columns:
        df = df[(df["rr"].fillna(0.0)) >= minrr]

    keep_cols = [c for c in ["symbol", "tf", "strategy", "side", "price", "confidence", "rr", "msg", "time"] if c in df.columns]
    if keep_cols:
        st.dataframe(df[keep_cols], use_container_width=True, height=320)
        csv = df[keep_cols].to_csv(index=False)
        st.download_button("Export scanner CSV", data=csv, file_name="scanner_signals.csv")
    else:
        st.dataframe(df, use_container_width=True, height=320)


def _backtest_builtin(
    df: pd.DataFrame,
    tf: str,
    strategies: List[str],
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:  # equity DF, KPIs
    if df.empty or len(df) < 50:
        return pd.DataFrame(), {}

    close = df["Close"].copy()
    ret = close.pct_change().fillna(0.0)

    # Simple signal blend: union of three common patterns if names match
    sig = pd.Series(0, index=df.index, dtype=float)

    if any("golden" in s.lower() or "ema50/200" in s.lower() or "crossover" in s.lower() for s in strategies):
        e50 = ema(close, 50)
        e200 = ema(close, 200)
        cross_up = (e50 > e200) & (e50.shift(1) <= e200.shift(1))
        cross_dn = (e50 < e200) & (e50.shift(1) >= e200.shift(1))
        pos = pd.Series(0, index=df.index, dtype=int)
        pos = np.where(cross_up, 1, np.where(cross_dn, 0, np.nan))
        pos = pd.Series(pos, index=df.index).ffill().fillna(0)
        sig = np.maximum(sig, pos)

    if any("rsi mean" in s.lower() for s in strategies):
        rs = rsi(close, 14)
        mr_pos = ((rs < 30) & (close > ema(close, 200))).astype(int)
        sig = np.maximum(sig, mr_pos.rolling(2).max().fillna(0))

    if any("breakout 20" in s.lower() or "20-high" in s.lower() for s in strategies):
        high20 = df["High"].rolling(20).max()
        bo = (close > high20.shift(1)).astype(int)
        sig = np.maximum(sig, bo)

    sig = pd.Series(sig, index=df.index).astype(float).fillna(0.0)
    # Transaction costs (bps applied on change in position)
    pos = sig.round().clip(0, 1)
    pos_change = pos.diff().abs().fillna(0.0)
    gross = pos.shift(1).fillna(0) * ret
    tcost = pos_change * ((commission_bps + slippage_bps) / 10000.0)
    strat_ret = gross - tcost

    eq = (1.0 + strat_ret).cumprod()
    trades = int(pos_change.sum())
    exp = pos.mean()
    dd = (eq / eq.cummax() - 1.0).min()
    ann = 252 if tf in ("1D",) else (78 * 252 if tf == "5m" else 252)
    cagr = eq.iloc[-1] ** (ann / max(1, len(eq))) - 1.0
    sharpe = np.sqrt(ann) * (strat_ret.mean() / (strat_ret.std() + 1e-9))
    neg = strat_ret[strat_ret < 0]
    sortino = np.sqrt(ann) * (strat_ret.mean() / (neg.std() + 1e-9))
    wins = (strat_ret > 0).sum()
    winrate = wins / max(1, trades)
    avg_trade = strat_ret[strat_ret != 0].mean() if trades else 0.0
    turnover = pos_change.sum() / max(1, len(pos))
    kpis = {
        "CAGR": float(cagr),
        "MaxDD": float(dd),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "WinRate": float(winrate),
        "AvgTrade": float(avg_trade),
        "Exposure": float(exp),
        "Turnover": float(turnover),
    }
    out = pd.DataFrame({"equity": eq, "returns": strat_ret, "position": pos})
    return out, kpis


def _backtest_tile(tile: Dict[str, Any]):
    sym = tile["symbol"]
    tf = tile["timeframe"]
    strats = tile.get("strategies", []) or DEFAULTS["strategies"]["enabled"]
    df = fetch_ohlcv(sym, tf)
    if df.empty:
        st.warning(f"No data for {sym} {tf}.")
        return
    eq, kpi = _backtest_builtin(df, tf, strats, SETTINGS["risk"]["commission_bps"], SETTINGS["risk"]["slippage_bps"])
    if eq.empty:
        st.info("Not enough data to backtest.")
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        st.line_chart(eq["equity"], height=260)
    with c2:
        show = {
            k: (round(v * 100, 2) if k in ("CAGR", "MaxDD", "WinRate", "Exposure", "Turnover") else round(v, 3))
            for k, v in kpi.items()
        }
        st.dataframe(pd.DataFrame([show]).T.rename(columns={0: "value"}), use_container_width=True, height=260)


def _journal_tile(tile: Dict[str, Any]):
    sym = tile["symbol"]
    tf = tile["timeframe"]
    key = f"{sym}|{tf}"

    st.markdown(f"**Journal for {sym} — {tf}**")
    note = st.text_area("Add note", value="", height=100, key=f"jrnl_text_{key}")
    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("Save Note", key=f"jrnl_save_{key}"):
        try:
            # dict-shaped API
            insert_journal(
                {
                    "id": f"note_{uuid4().hex[:10]}",
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "type": "note",
                    "text": note,
                    "tags": f"symbol:{sym},tf:{tf}",
                    "link_id": "",
                    "meta": {},
                }
            )
        except TypeError:
            # positional fallback
            insert_journal(sym, tf, datetime.now(timezone.utc).isoformat(), note)
        st.success("Saved.")

    max_rows = int(tile.get("max_rows", 50))
    # Fetch journal rows (supports both list-of-dicts and DataFrame)
    try:
        rj = fetch_journal(sym=sym, tf=tf, limit=max_rows)
    except Exception:
        rj = []
    if isinstance(rj, pd.DataFrame):
        df = rj
    else:
        df = pd.DataFrame(rj)
    if not df.empty:
        st.dataframe(df, use_container_width=True, height=240)
        st.download_button("Export Journal CSV", df.to_csv(index=False), file_name=f"journal_{sym}_{tf}.csv")
    else:
        st.caption("No entries yet.")


def _metrics_tile(tile: Dict[str, Any]):
    sym = tile["symbol"]
    tf = tile["timeframe"]
    df = fetch_ohlcv(sym, tf)
    if df.empty:
        st.warning(f"No data for {sym} {tf}.")
        return
    vol_surge = (df["Volume"] / df["Volume"].rolling(20).mean()).iloc[-1] if "Volume" in df else np.nan
    rsi14 = rsi(df["Close"], 14).iloc[-1]
    _, _, macd_hist = macd(df["Close"], 12, 26, 9)
    macd_h = macd_hist.iloc[-1]
    stoch = stoch_k(df, 14, 3).iloc[-1]
    a = atr(df, 14).iloc[-1]
    _, _, _, bb_w = bbands(df["Close"], 20, 2.0)
    bbw = bb_w.iloc[-1]
    trend = ema_slope(df["Close"], 50).iloc[-1]
    liq = (df["Close"] * (df["Volume"] if "Volume" in df else 0)).rolling(20).mean().iloc[-1]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Volume Surge (x)", value=f"{vol_surge:.2f}")
        st.metric("RSI(14)", value=f"{rsi14:.1f}")
    with c2:
        st.metric("MACD Hist", value=f"{macd_h:.4f}")
        st.metric("Stoch %K", value=f"{stoch:.1f}")
    with c3:
        st.metric("ATR(14)", value=f"{a:.2f}")
        st.metric("BB Width", value=f"{bbw:.3f}")

    # Sparklines
    s1 = (df["Volume"] / df["Volume"].rolling(20).mean()).tail(120).bfill() if "Volume" in df else pd.Series([np.nan])
    s2 = rsi(df["Close"], 14).tail(120)
    s3 = macd_hist.tail(120)
    st.caption("Sparklines")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.line_chart(s1, height=120)
    with sc2:
        st.line_chart(s2, height=120)
    with sc3:
        st.line_chart(s3, height=120)

# ---------------------- Layout: 2×2 grid ----------------------
st.markdown("## Dashboard")
layout = _ensure_layout()


def _render_tile(idx: int, tile: Dict[str, Any]):
    st.markdown(f"#### Tile {idx+1}: {tile['type']} — {tile['symbol']} ({tile['timeframe']})")
    if tile["type"] == "Chart":
        _chart_tile(tile)
    elif tile["type"] == "Scanner":
        _scanner_tile(tile)
    elif tile["type"] == "Backtest":
        _backtest_tile(tile)
    elif tile["type"] == "Journal":
        _journal_tile(tile)
    elif tile["type"] == "Metrics":
        _metrics_tile(tile)
    else:
        st.info("Unknown tile type.")


row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    _render_tile(0, layout[0])
with row1_col2:
    _render_tile(1, layout[1])

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    _render_tile(2, layout[2])
with row2_col2:
    _render_tile(3, layout[3])

# ---------------------- Persist on first load if requested ----------------------
if SETTINGS.get("dashboard", {}).get("persist", True) and not st.session_state.get("persisted_once", False):
    save_settings(SETTINGS)
    st.session_state["persisted_once"] = True

# ---------------------- Live Alerts Board ----------------------
with st.expander("🔔 Live — New Alerts (new only)", expanded=True):
    pending: List[Dict[str, Any]] = []
    if eng:
        try:
            while True:
                pending.append(eng.q.get_nowait())
        except Exception:
            pass
    new_hits = [h for h in pending if h["id"] not in st.session_state["seen_alert_ids"]]
    for h in new_hits:
        st.session_state["seen_alert_ids"].add(h["id"])
        maybe_toast(h.get("msg", "Signal"))
    if new_hits:
        cols = ["time", "symbol", "tf", "strategy", "side", "price", "confidence", "msg", "id"]
        st.dataframe(
            pd.DataFrame(new_hits)[[c for c in cols if c in new_hits[0]]],
            use_container_width=True,
            height=260,
            hide_index=True,
        )
    c1, c2 = st.columns(2)
    if c1.button("Mark all as read", use_container_width=True):
        for h in pending:
            st.session_state["seen_alert_ids"].add(h["id"])
        st.success("Marked.")
    if c2.button("Export alerts CSV", use_container_width=True):
        path = export_csv("alerts", "data/alerts_export.csv")
        st.success(f"Saved → {path}")
