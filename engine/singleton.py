# engine/singleton.py
from __future__ import annotations
import threading, queue, time, traceback
from datetime import datetime, timezone
from typing import Any, Dict, List
import streamlit as st

from engine.runner import scan_once
from engine.utils import OrderSuggestion
from storage import upsert_alerts

import hashlib

def _alert_id(symbol: str, timeframe: str, t_iso: str, name: str, side: str) -> str:
    key = f"{symbol}|{timeframe}|{t_iso}|{name}|{side}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]

class LiveEngine:
    """
    Threaded engine that calls engine.runner.scan_once() repeatedly and pushes
    *only* NEW hits into a Queue. Use get_engine() to obtain the singleton.
    """
    def __init__(self):
        self.q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cfg: Dict[str, Any] = {
            "trackers": [],
            "strategies_cfg": {},
            "risk_opts": None,
            "interval_sec": 8.0,
        }

    # ---------- lifecycle ----------
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        t = threading.Thread(target=self._run, daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None

    # ---------- configuration ----------
    def configure(self, *, trackers, strategies_cfg, risk_opts, interval_sec: float = 8.0) -> None:
        with self._lock:
            self._cfg.update(
                trackers=trackers or [],
                strategies_cfg=strategies_cfg or {},
                risk_opts=risk_opts,
                interval_sec=float(interval_sec or 8.0),
            )

    # ---------- engine thread ----------
    def _emit(self, sug: OrderSuggestion) -> None:
        ts_utc = datetime.now(timezone.utc).isoformat()
        row = {
            "id": _alert_id(sug.symbol, sug.timeframe, ts_utc, sug.strategy, sug.side),
            "time": ts_utc,
            "symbol": sug.symbol,
            "tf": sug.timeframe,
            "strategy": sug.strategy,
            "side": sug.side,
            "price": float(sug.entry),
            "confidence": float(getattr(sug, "confidence", 0.0)),
            "msg": f"{sug.symbol} {sug.timeframe} {sug.side.upper()} {sug.strategy} @ {float(sug.entry):.4f}",
            "meta": {
                "qty": getattr(sug, "qty", 0),
                "sl": getattr(sug, "sl", None),
                "tp": getattr(sug, "tp", []),
                "reason": getattr(sug, "reason", ""),
            },
        }
        # persist to SQLite (best-effort)
        try:
            upsert_alerts([{
                "id": row["id"],
                "time": row["time"],
                "symbol": row["symbol"],
                "tf": row["tf"],
                "strategy": row["strategy"],
                "side": row["side"],
                "price": row["price"],
                "confidence": row["confidence"],
                "rr": None,
                "msg": row["msg"],
                "meta": row["meta"],
            }])
        except Exception:
            pass

        # push to live queue
        self.q.put(row)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                trackers = self._cfg["trackers"]
                strategies_cfg = self._cfg["strategies_cfg"]
                risk_opts = self._cfg["risk_opts"]
                sleep_s = float(self._cfg["interval_sec"])

            try:
                # one pass; engine.runner does the heavy lifting
                scan_once(
                    trackers=trackers,
                    strategies_cfg=strategies_cfg,
                    risk_opts=risk_opts,
                    on_suggestion=self._emit,
                )
            except Exception:
                traceback.print_exc()

            time.sleep(max(1.5, sleep_s))

@st.cache_resource
def get_engine() -> LiveEngine:
    return LiveEngine()
