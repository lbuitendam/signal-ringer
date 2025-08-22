from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# make sure package root is on path when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.data import get_adapter  # noqa
from engine.runner import scan_once  # noqa
from engine.utils import Signal      # noqa
from strategies.library.macd_trend import MacdTrend  # noqa
from strategies.library.range_breakout import RangeBreakout  # noqa
from patterns.engine import detect_all, DEFAULT_CONFIG  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="Run deep checks incl. scan_once")
    args = ap.parse_args()

    report = {
        "env": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "meta": {"project_root": str(ROOT)},
        "checks": [],
    }

    def ok(name, detail=""):
        report["checks"].append({"name": name, "status": "OK", "detail": detail})

    def fail(name, detail):
        report["checks"].append({"name": name, "status": "FAIL", "detail": str(detail)})

    # imports
    try:
        import engine.runner  # noqa
        ok("import engine.runner")
    except Exception as e:
        fail("import engine.runner", e)

    try:
        import engine.data  # noqa
        ok("import engine.data")
    except Exception as e:
        fail("import engine.data", e)

    try:
        import patterns.engine  # noqa
        ok("import patterns.engine")
    except Exception as e:
        fail("import patterns.engine", e)

    # quick strategy sanity
    try:
        yf = get_adapter("yfinance")
        df = yf.fetch_history("AAPL", "5m", 600)
        m = MacdTrend({})
        out = m.signals(df)
        ok("quick: MacdTrend signals", f"count={len(out)}")
    except Exception as e:
        fail("quick: MacdTrend signals", e)

    try:
        df = get_adapter("yfinance").fetch_history("AAPL", "5m", 600)
        r = RangeBreakout(lookback=20, retest=5, params={})
        out = r.signals(df)
        ok("quick: RangeBreakout signals", f"count={len(out)}")
    except Exception as e:
        fail("quick: RangeBreakout signals", e)

    try:
        df = get_adapter("yfinance").fetch_history("AAPL", "5m", 600)
        hits = detect_all(df.rename(columns=str.capitalize), list(DEFAULT_CONFIG.keys()), DEFAULT_CONFIG)
        ok("quick: patterns.engine", f"hits={len(hits)}")
    except Exception as e:
        fail("quick: patterns.engine", e)

    if args.deep:
        try:
            # synthetic adapter to avoid API rate/availability issues
            df = get_adapter("synthetic").fetch_history("TEST", "5m", 600)

            async def _once():
                wl = [{"symbol": "TEST", "timeframe": "5m", "enabled": True, "adapter": "synthetic"}]
                opts = {
                    "enabled": {
                        "MACD Trend": {"enabled": True, "params": {}, "approved": True},
                        "Range Breakout": {"enabled": True, "params": {"lookback": 20, "retest": 5}, "approved": True},
                    },
                    "big_boss": {"enabled": True, "k_bars": 3, "tol": 0.003},
                }
                class DummyRisk: pass  # not used
                async def _sink(sug):  # noqa
                    return
                from risk.manager import RiskOptions
                ro = RiskOptions(equity=10000.0, risk_pct=0.01, atr_mult_sl=1.5, rr=2.0, tp_count=2, cooldown_min=15, max_positions=6)
                await scan_once(wl, opts, ro, _sink, lookback=300)

            import asyncio
            asyncio.run(_once())
            ok("deep: scan_once")
        except Exception as e:
            fail("deep: scan_once", e)

    print("\n==== Signal Ringer Diagnosis ====")
    print(json.dumps(report, indent=2))
    print("=================================\n")


if __name__ == "__main__":
    main()
