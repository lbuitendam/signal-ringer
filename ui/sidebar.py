from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

from data.provider import normalize_watchlist
from engine.singleton import get_engine
from risk.manager import RiskOptions

SETTINGS = Path("data/settings.json")
SETTINGS.parent.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "watchlist": [
        {"symbol": "AAPL",    "asset": "stock",  "timeframe": "5m", "adapter": "yfinance", "enabled": True},
        {"symbol": "MSFT",    "asset": "stock",  "timeframe": "5m", "adapter": "yfinance", "enabled": True},
        {"symbol": "BTC-USD", "asset": "crypto", "timeframe": "5m", "adapter": "yfinance", "enabled": False},
    ],
"strategies": {
    "enabled": {
        "EMA20/50 Pullback": {"enabled": True, "params": {}, "approved": True},
        "MACD Trend":        {"enabled": True, "params": {}, "approved": True},
        "Range Breakout":    {"enabled": True, "params": {"lookback": 20, "retest": 5}, "approved": True},
    },

    },
    "risk": {
        "equity": 10000.0, "risk_pct": 0.01, "atr_mult_sl": 1.5, "rr": 2.0,
        "tp_count": 2, "cooldown_min": 15, "max_positions": 6
    }
}

def load_settings() -> Dict[str, Any]:
    if SETTINGS.exists():
        try:
            return json.loads(SETTINGS.read_text())
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULTS))

def save_settings(obj: Dict[str, Any]) -> None:
    SETTINGS.write_text(json.dumps(obj, indent=2))

def _chip(text: str, color="#6b7280"):
    st.markdown(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:{color}1A;color:{color};border:1px solid {color}55;font-size:12px'>{text}</span>",
        unsafe_allow_html=True,
    )

def sidebar():
    s = load_settings()
    wl: List[Dict[str, Any]] = s.get("watchlist", [])
    opts: Dict[str, Any] = s.get("strategies", {})
    risk: Dict[str, Any] = s.get("risk", {})

    st.sidebar.markdown("### Navigation")
    st.sidebar.page_link("app.py", label="📊 Dashboard")
    st.sidebar.page_link("pages/1_Scanner.py", label="🧠 Scanner (Patterns & Strategies)")
    st.sidebar.page_link("pages/2_Backtesting.py", label="🔁 Backtesting")
    st.sidebar.page_link("pages/3_History_&_Journal.py", label="🗂 History & Journal")
    st.sidebar.page_link("pages/4_User_Settings.py", label="👤 User Settings")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")

    eng = get_engine()
    est = getattr(eng, "status", lambda: {"running": eng.is_running(), "active_symbols": 0, "last_tick_utc": None, "error": ""})()

    with st.sidebar.expander("Quick Controls", expanded=True):
        c1, c2 = st.columns(2)
        engine_on = c1.toggle("Engine", value=bool(est.get("running", False)))
        opts.setdefault("big_boss", {})
        big_boss = c2.toggle("Big Boss", value=bool(opts["big_boss"].get("enabled", True)))

        conf_min = st.slider("Confidence ≥", 0.0, 1.0, float(opts.get("min_conf", 0.0)), 0.05)
        cooldown = st.number_input("Cooldown (min)", 1, 120, int(risk.get("cooldown_min", 15)))

        c3, c4, c5 = st.columns(3)
        rr = c3.number_input("RR target", value=float(risk.get("rr", 2.0)))

        saved_rp = float(risk.get("risk_pct", 0.01))
        if saved_rp > 1.0:
            saved_rp = saved_rp / 100.0
        risk_pct = c4.number_input("Risk %/trade", min_value=0.001, max_value=0.10, step=0.001, format="%.3f", value=saved_rp)

        max_pos = c5.number_input("Max positions", min_value=1, max_value=50, value=int(risk.get("max_positions", 6)))

        refresh = st.slider("Autorefresh (s)", 2, 30, int(st.session_state.get("autorefresh_sec", 8)))
        st.session_state["autorefresh_sec"] = refresh

        st.markdown("<div style='display:flex;gap:6px;flex-wrap:wrap'>", unsafe_allow_html=True)
        _chip("Running" if est.get("running") else "Stopped", "#10b981" if est.get("running") else "#6b7280")
        _chip(f"Active {int(est.get('active_symbols', 0))}", "#10b981" if int(est.get("active_symbols", 0)) > 0 else "#6b7280")
        _chip("Last tick ✓" if est.get("last_tick_utc") else "No tick", "#10b981" if est.get("last_tick_utc") else "#6b7280")
        if est.get("error"): _chip("Error", "#ef4444")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Watchlist editor
    with st.sidebar.expander("Watchlist", expanded=False):
        df = pd.DataFrame(wl)
        edited = st.data_editor(
            df, num_rows="dynamic", use_container_width=True, height=240,
            column_config={"enabled": st.column_config.CheckboxColumn(default=True)}
        )
        if st.button("Save watchlist"):
            s2 = load_settings()
            new_wl = edited.fillna("").to_dict("records")
            new_wl = normalize_watchlist(new_wl)
            s2["watchlist"] = new_wl
            save_settings(s2)
            st.success("Saved. Engine will pick this up on next rerun.")
            st.rerun()

    # Persist and push to engine every run
    opts["big_boss"]["enabled"] = bool(big_boss)
    opts["min_conf"] = float(conf_min)
    risk.update({
        "cooldown_min": int(cooldown),
        "rr": float(rr),
        "risk_pct": float(risk_pct),
        "max_positions": int(max_pos)
    })
    s.update({"watchlist": wl, "strategies": opts, "risk": risk})
    save_settings(s)

    try:
        eng.configure(
            trackers=wl,
            strategies_cfg=opts,
            risk_opts=RiskOptions(**risk),
            interval_sec=float(st.session_state.get("autorefresh_sec", 8))
        )
    except Exception as e:
        st.sidebar.error(f"Engine config error: {e}")

    if engine_on and not eng.is_running():
        eng.start()
    if (not engine_on) and eng.is_running():
        eng.stop()

    nav = {"page": "dashboard"}
    qc = {"engine_on": engine_on, "refresh_sec": int(st.session_state.get("autorefresh_sec", 8))}
    return nav, wl, opts, risk, qc
