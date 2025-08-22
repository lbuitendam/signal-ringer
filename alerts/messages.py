# alerts/messages.py
from __future__ import annotations
from typing import List
import pandas as pd
from engine.utils import OrderSuggestion

def format_entry(o: OrderSuggestion) -> str:
    tps = ", ".join(f"{x:.2f}" for x in o.tp)
    side = "BUY" if o.side == "buy" else "SELL"
    return f"🔔 ENTRY → [{o.symbol} {o.timeframe}] {side} {o.qty:.0f} @ {o.entry:.2f} | SL {o.sl:.2f} | TP [{tps}] — {o.strategy} (conf {o.confidence:.2f})"

def format_exit(o: OrderSuggestion, reason: str) -> str:
    side = "SELL" if o.side == "buy" else "BUY"
    return f"🔔 EXIT  → [{o.symbol} {o.timeframe}] {side} {o.qty:.0f} @ {o.entry:.2f} — {reason} ({o.strategy})"

def maybe_toast(msg: str):
    try:
        import streamlit as st
        st.toast(msg)
    except Exception:
        pass

def csv_log(path: str, rows: List[dict]):
    import csv, os
    header = list(rows[0].keys())
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)
