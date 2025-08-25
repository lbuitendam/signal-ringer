# engine/singleton.py
from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

import streamlit as st

from engine.runner import scan_once          # async or sync; we handle both
from engine.utils import OrderSuggestion
from storage import upsert_alerts


def _alert_id(symbol: str, timeframe: str, t_iso: str, name: str, side: str) -> str:
    key = f"{symbol}|{timeframe}|{t_iso}|{name}|{side}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


class LiveEngine:
    """
    Background scanner engine:
      - Runs scan_once(...) in a dedicated thread.
      - If scan_once is async, uses asyncio.run() per cycle.
      - Emits OrderSuggestion via _emit() into a thread-safe Queue for the UI.
    """

    def __init__(self) -> None:
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

        self._last_tick_utc: str | None = None
        self._error_msg: str = ""

    # ---------- helpers ----------
    @staticmethod
    def _norm_tracker(t: Any) -> Dict[str, Any]:
        if isinstance(t, dict):
            out = dict(t)
            out["symbol"] = str(out.get("symbol", out.get("ticker", ""))).upper()
            out["enabled"] = bool(out.get("enabled", True))
            return out
        return {"symbol": str(t).upper(), "enabled": True}

    # ---------- lifecycle ----------
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="LiveEngineThread")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    # ---------- configuration ----------
    def configure(
        self,
        *,
        trackers,
        strategies_cfg,
        risk_opts,
        interval_sec: float = 8.0,
    ) -> None:
        """Safe to call on every Streamlit rerun. Does NOT auto-start the engine."""
        with self._lock:
            self._cfg.update(
                trackers=[self._norm_tracker(t) for t in (trackers or [])],
                strategies_cfg=dict(strategies_cfg or {}),
                risk_opts=risk_opts,
                interval_sec=float(interval_sec or 8.0),
            )

    # ---------- status for UI ----------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            trackers = [self._norm_tracker(t) for t in self._cfg.get("trackers", [])]
            return {
                "running": self.is_running(),
                "active_symbols": sum(1 for t in trackers if t.get("enabled", True)),
                "last_tick_utc": self._last_tick_utc,
                "error": self._error_msg,
            }

    # ---------- suggestion sink ----------
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
        try:
            upsert_alerts([{
                "id": row["id"], "time": row["time"], "symbol": row["symbol"], "tf": row["tf"],
                "strategy": row["strategy"], "side": row["side"], "price": row["price"],
                "confidence": row["confidence"], "rr": None, "msg": row["msg"], "meta": row["meta"],
            }])
        except Exception:
            pass
        self.q.put(row)

    # ---------- worker thread ----------
    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                trackers = list(self._cfg["trackers"])
                strategies_cfg = dict(self._cfg["strategies_cfg"])
                risk_opts = self._cfg["risk_opts"]
                sleep_s = float(self._cfg["interval_sec"])

            try:
                if asyncio.iscoroutinefunction(scan_once):
                    asyncio.run(
                        scan_once(
                            trackers=trackers,
                            strategies_cfg=strategies_cfg,
                            risk_opts=risk_opts,
                            on_suggestion=self._emit,
                        )
                    )
                else:
                    scan_once(
                        trackers=trackers,
                        strategies_cfg=strategies_cfg,
                        risk_opts=risk_opts,
                        on_suggestion=self._emit,
                    )
                self._last_tick_utc = datetime.now(timezone.utc).isoformat()
                self._error_msg = ""
            except Exception:
                self._error_msg = traceback.format_exc()

            if self._stop.wait(timeout=max(1.5, sleep_s)):
                break


@st.cache_resource
def get_engine() -> LiveEngine:
    """Streamlit-cached singleton."""
    return LiveEngine()
