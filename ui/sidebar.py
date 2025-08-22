# ui/sidebar.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import streamlit as st

from engine.singleton import get_engine
from risk.manager import RiskOptions

SETTINGS = Path("data/settings.json")
SETTINGS.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_WATCHLIST = [
    {"symbol": "AAPL",    "asset": "stock",  "timeframe": "5m", "adapter": "yfinance", "enabled": True},
    {"symbol": "MSFT",    "asset": "stock",  "timeframe": "5m", "adapter": "yfinance", "enabled": True},
    {"symbol": "BTC-USD", "asset": "crypto", "timeframe": "5m", "adapter": "yfinance", "enabled": False},
]

DEFAULT_STRATEGIES = {
    "enabled": {
        "EMA20/50 Pullback": {"enabled": True,  "params": {}, "approved": True},
        "MACD Trend":        {"enabled": True,  "params": {}, "approved": True},
        "Range Breakout":    {"enabled": True,  "params": {"lookback": 20, "retest": 5}, "approved": True},
        "Bullish Engulfing": {"enabled": True,  "params": {}, "approved": True},
        "Bearish Engulfing": {"enabled": True,  "params": {}, "approved": True},
    },
    "big_boss": {"enabled": True, "k_bars": 3, "tol": 0.003},
    "min_conf": 0.0,
}

DEFAULT_RISK = {
    "equity": 10000.0, "risk_pct": 0.01, "atr_mult_sl": 1.5, "rr": 2.0,
    "tp_count": 2, "cooldown_min": 15, "max_positions": 6
}

def load_settings() -> Dict[str, Any]:
    if SETTINGS.exists():
        try:
            return json.loads(SETTINGS.read_text())
        except Exception:
            pass
    return {"watchlist": DEFAULT_WATCHLIST, "strategies": DEFAULT_STRATEGIES, "risk": DEFAULT_RISK}

def save_settings(obj: Dict[str, Any]) -> None:
    SETTINGS.write_text(json.dumps(obj, indent=2))

def _chip(label: str, kind: str = "ok"):
    col = {"ok": "#10b981", "warn": "#f59e0b", "bad": "#ef4444", "idle": "#6b7280"}.get(kind, "#6b7280")
    st.markdown(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:{col}20;color:{col};font-size:12px;border:1px solid {col}40'>{label}</span>",
        unsafe_allow_html=True,
    )

def sidebar() -> Tuple[str, List[dict], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Returns: (nav, watchlist, strategies_cfg, risk, engine_controls)
    """
    s = load_settings()
    s.setdefault("watchlist", DEFAULT_WATCHLIST)
    s.setdefault("strategies", DEFAULT_STRATEGIES)
    s.setdefault("risk", DEFAULT_RISK)

    st.sidebar.markdown("### Signal Ringer")

    # ---- NAV ----
    nav = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Watchlist", "Options", "History & Journal"],
        index=0,
        label_visibility="collapsed",
    )

    eng = get_engine()
    est = eng.status() if hasattr(eng, "status") else {"running": False, "active_symbols": 0, "last_tick_utc": None, "error": ""}

    # ---- Quick Controls ----
    with st.sidebar.expander("Quick Controls", expanded=True):
        c1, c2 = st.columns(2)
        engine_on = c1.toggle("Engine", value=bool(est.get("running", False)))
        big_boss_on = c2.toggle("Big Boss", value=bool(s["strategies"].get("big_boss", {}).get("enabled", True)))

        conf_min = st.slider("Confidence ≥", 0.0, 1.0, float(s["strategies"].get("min_conf", 0.0)), 0.05)
        cooldown = st.number_input("Cooldown (min)", min_value=1, max_value=120,
                                   value=int(s["risk"].get("cooldown_min", 15)))

        c3, c4, c5 = st.columns(3)
        rr = c3.number_input("RR target", value=float(s["risk"].get("rr", 2.0)))

        # -------- normalize the saved risk% BEFORE using it as widget value --------
        saved_rp = float(s["risk"].get("risk_pct", 0.01))
        if saved_rp > 1:   # e.g., 10 -> 0.10, 2 -> 0.02
            saved_rp = saved_rp / 100.0
        saved_rp = max(0.001, min(saved_rp, 0.10))  # clamp to widget range

        risk_pct = c4.number_input(
            "Risk %/trade", min_value=0.001, max_value=0.1,
            step=0.001, format="%.3f", value=saved_rp
        )
        max_pos = c5.number_input("Max positions", min_value=1, max_value=50,
                                  value=int(s["risk"].get("max_positions", 6)))

        refresh = st.slider("Autorefresh (s)", 2, 30, int(st.session_state.get("autorefresh_sec", 8)))
        st.session_state["autorefresh_sec"] = refresh

        # status chips
        st.markdown("<div style='display:flex;gap:6px;flex-wrap:wrap'>", unsafe_allow_html=True)
        _chip("Running" if est.get("running") else "Stopped", "ok" if est.get("running") else "idle")
        _chip(f"Active {int(est.get('active_symbols', 0))}", "ok" if int(est.get("active_symbols", 0)) > 0 else "idle")
        if est.get("last_tick_utc"):
            _chip("Last tick ✓", "ok")
        else:
            _chip("No tick", "idle")
        if est.get("error"):
            _chip("Error", "bad")
            st.caption(f"⚠️ {est['error']}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Watchlist & Options editors (for persistence) ----
    wl = s.get("watchlist", DEFAULT_WATCHLIST)
    opts = s.get("strategies", DEFAULT_STRATEGIES)
    risk = s.get("risk", DEFAULT_RISK)

    # reflect quick controls into in-memory config (not all are persisted)
    opts.setdefault("big_boss", {}).update({"enabled": bool(big_boss_on)})
    opts["min_conf"] = conf_min
    risk["cooldown_min"] = int(cooldown)
    risk["rr"] = float(rr)
    risk["risk_pct"] = float(risk_pct)         # already normalized by the widget
    risk["max_positions"] = int(max_pos)

    # configure live engine (doesn't start it yet)
    try:
        risk_opts = RiskOptions(**risk)
        eng.configure(trackers=wl, strategies_cfg=opts, risk_opts=risk_opts,
                      interval_sec=float(st.session_state.get("autorefresh_sec", 8)))
    except Exception as e:
        st.sidebar.error(f"Engine config error: {e}")

    # allow start/stop now
    if engine_on and not getattr(eng, "is_running", lambda: False)():
        eng.start()
    if (not engine_on) and getattr(eng, "is_running", lambda: False)():
        eng.stop()

    return nav, wl, opts, risk, {
        "engine_on": engine_on,
        "refresh_sec": refresh,
        "big_boss": big_boss_on,
        "confidence_min": conf_min,
        "cooldown": cooldown,
        "rr": rr,
        "risk_pct": risk_pct,
        "max_positions": max_pos,
    }
