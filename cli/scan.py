# cli/scan.py
import argparse, asyncio, json
import pandas as pd
from engine.runner import run_engine_loop
from risk.manager import RiskOptions
from alerts.messages import format_entry, csv_log
from engine.utils import OrderSuggestion

def on_suggestion(sug: OrderSuggestion):
    print(format_entry(sug))
    csv_log("data/logs.csv", [{
        "time": pd.Timestamp.utcnow().isoformat(), "symbol": sug.symbol, "tf": sug.timeframe,
        "side": sug.side, "qty": sug.qty, "entry": sug.entry, "sl": sug.sl,
        "tp": "|".join(map(str,sug.tp)), "strategy": sug.strategy, "conf": sug.confidence, "reason": sug.reason
    }])

def parse_symbols(s: str):
    out = []
    for x in s.split(","):
        x = x.strip()
        out.append({"symbol": x, "timeframe": args.tf, "adapter":"yfinance", "enabled": True})
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True, help="Comma separated symbols")
    p.add_argument("--tf", default="5m")
    p.add_argument("--big-boss", type=int, default=1)
    args = p.parse_args()
    wl = parse_symbols(args.symbols)
    opts = {
        "enabled": {
            "EMA20/50 Pullback": {"enabled": True, "params": {}, "approved": True},
            "MACD Trend":        {"enabled": True, "params": {}, "approved": True},
            "Range Breakout":    {"enabled": True, "params": {"lookback":20,"retest":5}, "approved": True},
            "Bullish Engulfing": {"enabled": True, "params": {}, "approved": True},
            "Bearish Engulfing": {"enabled": True, "params": {}, "approved": True},
        },
        "big_boss": {"enabled": bool(args.big_boss), "k_bars": 3, "tol": 0.003}
    }
    risk = RiskOptions()
    asyncio.run(run_engine_loop(wl, opts, risk, on_suggestion, 30))
