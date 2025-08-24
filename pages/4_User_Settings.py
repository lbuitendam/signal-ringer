# pages/4_User_Settings.py
from __future__ import annotations
import json
import pandas as pd
import streamlit as st

from ui.sidebar import load_settings, save_settings
from engine.singleton import get_engine
from risk.manager import RiskOptions

st.set_page_config(layout="wide", page_title="User Settings — Signal Ringer")
st.title("👤 User Settings")

s = load_settings()
wl = s.get("watchlist", [])
opts = s.get("strategies", {})
risk = s.get("risk", {})

st.subheader("Watchlist")
df = pd.DataFrame(wl)
edited = st.data_editor(
    df, num_rows="dynamic", use_container_width=True, height=300,
    column_config={"enabled": st.column_config.CheckboxColumn(default=True)}
)
if st.button("Save Watchlist"):
    s["watchlist"] = edited.fillna("").to_dict("records")
    save_settings(s)
    st.success("Saved.")

st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Strategies")
    en = opts.setdefault("enabled", {})
    for name in sorted(en.keys()):
        row = en[name]
        row["enabled"] = st.checkbox(f"{name}", value=bool(row.get("enabled", True)))
    st.markdown("**Big Boss filter**")
    bb = opts.setdefault("big_boss", {"enabled": True, "k_bars": 3, "tol": 0.003})
    bb["enabled"] = st.checkbox("Enable Big Boss", value=bool(bb.get("enabled", True)))
    bb["k_bars"] = int(st.number_input("k_bars", min_value=1, max_value=20, value=int(bb.get("k_bars", 3))))
    bb["tol"] = float(st.number_input("tol", min_value=0.0, max_value=0.05, value=float(bb.get("tol", 0.003)), step=0.0005))
    opts["min_conf"] = float(st.slider("Global min confidence", 0.0, 1.0, float(opts.get("min_conf", 0.0)), 0.05))

with c2:
    st.subheader("Risk")
    risk["equity"] = float(st.number_input("Equity", min_value=0.0, value=float(risk.get("equity", 10000.0))))
    rp = float(risk.get("risk_pct", 0.01))
    if rp > 1: rp = rp / 100.0
    risk["risk_pct"] = float(st.number_input("Risk % (fraction)", min_value=0.0005, max_value=0.1, value=rp, step=0.0005, format="%.4f"))
    risk["atr_mult_sl"] = float(st.number_input("ATR SL multiple", min_value=0.1, max_value=10.0, value=float(risk.get("atr_mult_sl", 1.5))))
    risk["rr"] = float(st.number_input("RR target", min_value=0.1, max_value=10.0, value=float(risk.get("rr", 2.0))))
    risk["tp_count"] = int(st.number_input("TP count", min_value=1, max_value=5, value=int(risk.get("tp_count", 2))))
    risk["cooldown_min"] = int(st.number_input("Cooldown (min)", min_value=1, max_value=240, value=int(risk.get("cooldown_min", 15))))
    risk["max_positions"] = int(st.number_input("Max positions", min_value=1, max_value=50, value=int(risk.get("max_positions", 6))))

if st.button("Save All Settings", type="primary"):
    save_settings(s)
    st.success("Settings saved.")

st.markdown("---")
st.subheader("Apply to Engine")
if st.button("Push current settings to engine"):
    eng = get_engine()
    try:
        eng.configure(trackers=s["watchlist"], strategies_cfg=s["strategies"], risk_opts=RiskOptions(**s["risk"]))
        st.success("Pushed to engine.")
    except Exception as e:
        st.error(f"Engine config error: {e}")
