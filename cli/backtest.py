# cli/backtest.py
import argparse, json
import pandas as pd
from engine.data import get_adapter
from backtest.backtrader_bridge import cached_backtest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True)
    p.add_argument("--tf", default="5m")
    p.add_argument("--strategies", default="EMA Pullback,MACD Trend")
    p.add_argument("--years", type=int, default=2)
    args = p.parse_args()

    syms = [s.strip() for s in args.symbols.split(",")]
    strs = [s.strip() for s in args.strategies.split(",")]

    adapter = get_adapter("yfinance")
    for sym in syms:
        df = adapter.fetch_history(sym, args.tf, lookback=6000)
        df = df.dropna()
        for s in strs:
            res = cached_backtest(df, s, params={})
            print(sym, args.tf, s, res)
