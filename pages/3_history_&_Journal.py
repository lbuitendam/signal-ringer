# pages/3_History_&_Journal.py
from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import streamlit as st

from storage import fetch_alerts, insert_journal, fetch_journal

st.set_page_config(layout="wide", page_title="History & Journal — Signal Ringer")
st.title("🗂 History & Journal")

tabs = st.tabs(["History","Data Collection","Setup & Strategy","Performance","Emotion Log","Trading Plan"])

# History
with tabs[0]:
    st.subheader("Alert Archive")
    dfA = fetch_alerts()
    if dfA.empty:
        st.caption("No alerts yet.")
    else:
        c1, c2, c3 = st.columns(3)
        syms = sorted(dfA["symbol"].unique().tolist())
        sy_f = c1.multiselect("Symbols", syms, default=syms[: min(5, len(syms))])
        strats = sorted(dfA["strategy"].unique().tolist())
        st_f = c2.multiselect("Strategies", strats, default=strats[: min(5, len(strats))])
        fdf = dfA
        if sy_f: fdf = fdf[fdf["symbol"].isin(sy_f)]
        if st_f: fdf = fdf[fdf["strategy"].isin(st_f)]
        st.dataframe(fdf, use_container_width=True, height=460, hide_index=True)

# Data Collection
with tabs[1]:
    st.subheader("Raw Signal Data (editable notes)")
    dfA = fetch_alerts()
    if dfA.empty:
        st.caption("No data yet.")
    else:
        dfE = dfA[["id","ts_utc","symbol","timeframe","strategy","side","price","confidence","msg"]].copy()
        dfE["note"] = ""
        out = st.data_editor(dfE, use_container_width=True, height=420, key="data_collect_history")
        if st.button("Save notes to Journal"):
            now = datetime.now(timezone.utc).isoformat()
            n_saved = 0
            for _, row in out.iterrows():
                note = (row.get("note") or "").strip()
                if not note:
                    continue
                insert_journal({
                    "id": f"note_{row['id']}",
                    "ts_utc": now,
                    "type": "note",
                    "text": note,
                    "tags": f"symbol:{row['symbol']},strategy:{row['strategy']}",
                    "link_id": row["id"],
                    "meta": {},
                })
                n_saved += 1
            st.success(f"Saved {n_saved} notes.")

# Setup & Strategy
with tabs[2]:
    st.subheader("Pre-Trade Checklist")
    with st.form("setup_form_hist"):
        c1, c2, c3 = st.columns(3)
        sym = c1.text_input("Symbol", value="AAPL")
        strat = c2.text_input("Strategy", value="EMA Pullback")
        plan_ok = c3.checkbox("Plan validated (rules met?)", value=True)
        notes = st.text_area("Notes / Plan", height=120, value="Entry: ...\nRisk: ...\nInvalidation: ...")
        submitted = st.form_submit_button("Save setup")
    if submitted:
        insert_journal({
            "id": f"setup_{uuid4().hex[:10]}",
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "type": "setup",
            "text": notes,
            "tags": f"symbol:{sym},strategy:{strat},ok:{plan_ok}",
            "link_id": "",
            "meta": {"symbol": sym, "strategy": strat, "ok": bool(plan_ok)},
        })
        st.success("Setup saved.")

    st.markdown("#### Recent setups")
    j = fetch_journal("setup")
    if not j.empty:
        st.dataframe(j[["ts_utc","text","tags"]], use_container_width=True, hide_index=True, height=240)
    else:
        st.caption("No setups yet.")

# Performance (placeholder KPIs)
with tabs[3]:
    st.subheader("Performance Dashboard")
    st.caption("Hook this to your 'trades' table when fills/exits are logged.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trades", 0)
    c2.metric("Win rate", "—")
    c3.metric("Profit Factor", "—")
    c4, c5, c6 = st.columns(3)
    c4.metric("Avg R", "—")
    c5.metric("Expectancy", "—")
    c6.metric("Max DD", "—")

# Emotion Log
with tabs[4]:
    st.subheader("Emotion Log")
    calm = st.slider("Calm ↔ Anxious", 0, 100, 60)
    disciplined = st.slider("Disciplined ↔ Impulsive", 0, 100, 70)
    confident = st.slider("Confident ↔ Afraid", 0, 100, 65)
    tags = st.multiselect("Tags", ["FOMO","Revenge","Overtrading","Hesitation","Chasing","Boredom"], [])
    note = st.text_area("Note", height=100)
    if st.button("Save emotion entry"):
        insert_journal({
            "id": f"emo_{uuid4().hex[:10]}",
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "type": "emotion",
            "text": note,
            "tags": ",".join(tags),
            "link_id": "",
            "meta": {"calm": int(calm), "disciplined": int(disciplined), "confident": int(confident)},
        })
        st.success("Saved.")
    st.markdown("#### Recent emotions")
    j = fetch_journal("emotion")
    if not j.empty:
        st.dataframe(j[["ts_utc","tags","text"]], use_container_width=True, hide_index=True, height=240)
    else:
        st.caption("No entries yet.")

# Trading Plan (versioned snapshots)
with tabs[5]:
    st.subheader("Trading Plan")
    plan_txt = st.text_area("Plan (edit and save a new version)", height=200)
    c1, c2 = st.columns(2)
    if c1.button("Save new version"):
        insert_journal({
            "id": f"plan_{uuid4().hex[:10]}",
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "type": "plan",
            "text": plan_txt,
            "tags": "plan",
            "link_id": "",
            "meta": {},
        })
        st.success("Saved.")
    st.markdown("#### Versions")
    j = fetch_journal("plan")
    if not j.empty:
        st.dataframe(j[["ts_utc","text"]], use_container_width=True, hide_index=True, height=300)
    else:
        st.caption("No versions yet.")
