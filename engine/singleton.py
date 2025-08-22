# engine/singleton.py
from __future__ import annotations

import threading
import queue
import time
import asyncio
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple

import streamlit as st

from engine.runner import scan_once  # async
from risk.manager import RiskOptions
from storage import upsert_alerts, alert_id as make_alert_id


class LiveEngine:
    """
    Threaded live engine that:
      - runs scan_once(...) in a loop (async -> wrapped via asyncio.run)
      - pushes only NEW hits into a thread-safe queue
      - writes alerts into SQLite (alerts table)
    """
    def __init__(self):
        self.q: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # dynamic config from sidebar
        self.trackers: List[Dict[str, Any]] = []
        self.strategies_cfg: Dict[str, Any] = {}
        self.risk_opts = RiskOptions(equity=10000.0, risk_pct=0.01, atr_mult_sl=1.5, rr=2.0,
                                     tp_count=2, cooldown_min=15, max_positions=6)
        self.interval_sec: float = 10.0

        # telemetry
        self.last_tick_utc: Optional[str] = None
        self.last_error: Optional[str] = None

        # de-dupe (engine-scope)
        self._sent_ids: Set[str] = set()
        # throttle repeated same suggestion (symbol|tf|strategy|side) within short window
        self._last_emit_by_key: Dict[str, Tuple[float, float]] = {}  # key -> (last_price, unixtime)

    # ---- public API ----
    def configure(self,
                  trackers: List[Dict[str, Any]],
                  strategies_cfg: Dict[str, Any],
                  risk_opts: RiskOptions,
                  interval_sec: float) -> None:
        self.trackers = list(trackers or [])
        self.strategies_cfg = dict(strategies_cfg or {})
        self.risk_opts = risk_opts
        self.interval_sec = max(2.0, float(interval_sec or 10.0))

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="LiveEngine")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ---- internals ----
    def _run(self):
        while not self._stop.is_set():
            try:
                # callback from runner → convert to alert rows + queue
                async def _sink_async(sug):
                    # stable key w/out timestamp to fight spam
                    key = f"{sug.symbol}|{sug.timeframe}|{sug.strategy}|{sug.side}"
                    price = float(sug.entry)
                    now = time.time()
                    last = self._last_emit_by_key.get(key)
                    if last and (now - last[1] < 120) and abs(last[0] - price) < 1e-6:
                        # same suggestion within 2 min at same price → drop
                        return
                    self._last_emit_by_key[key] = (price, now)

                    t_iso = datetime.now(timezone.utc).isoformat()
                    aid = make_alert_id(sug.symbol, sug.timeframe, t_iso, sug.strategy, sug.side)
                    if aid in self._sent_ids:
                        return

                    msg = f"[{sug.symbol} {sug.timeframe}] {sug.side.upper()} {sug.qty:g} @ {sug.entry:.4f} | " \
                          f"SL {sug.sl:.4f} | TP {', '.join(f'{x:.4f}' for x in (sug.tp or []))} — " \
                          f"{sug.strategy} (conf {sug.confidence:.2f})"

                    row = {
                        "id": aid,
                        "time": t_iso,
                        "symbol": sug.symbol,
                        "tf": sug.timeframe,
                        "strategy": sug.strategy,
                        "side": sug.side,
                        "price": float(sug.entry),
                        "confidence": float(sug.confidence),
                        "rr": None,
                        "msg": msg,
                        "meta": {
                            "sl": float(sug.sl),
                            "tp": [float(x) for x in (sug.tp or [])],
                            "type": sug.type,
                            "qty": float(sug.qty),
                            "reason": sug.reason,
                            "paper": bool(getattr(sug, "paper", True)),
                        },
                    }

                    # queue + db
                    self.q.put(row)
                    upsert_alerts([row])
                    self._sent_ids.add(aid)

                # run one scan pass
                if self.trackers and any(t.get("enabled", True) for t in self.trackers):
                    asyncio.run(scan_once(
                        trackers=self.trackers,
                        strategies_cfg=self.strategies_cfg,
                        risk_opts=self.risk_opts,
                        on_suggestion=_sink_async,
                        lookback=600,
                    ))
                    self.last_tick_utc = datetime.now(timezone.utc).isoformat()
                    self.last_error = None
                else:
                    # idle if nothing to do
                    time.sleep(0.25)

            except Exception as e:
                self.last_error = str(e)

            # wait between passes
            time.sleep(self.interval_sec)

    # convenience for sidebar chips
    def status(self) -> Dict[str, Any]:
        return {
            "running": self.is_running(),
            "active_symbols": sum(1 for t in self.trackers if t.get("enabled", True)),
            "last_tick_utc": self.last_tick_utc,
            "error": self.last_error,
        }


@st.cache_resource
def get_engine() -> LiveEngine:
    return LiveEngine()
