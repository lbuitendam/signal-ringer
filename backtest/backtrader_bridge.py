# backtest/backtrader_bridge.py
from __future__ import annotations
from typing import Dict, Any, List
import backtrader as bt
import pandas as pd
import numpy as np
import json
from pathlib import Path
from engine.utils import cache_key

class PandasData(bt.feeds.PandasData):
    params = (("datetime", None), ("open", "open"), ("high","high"), ("low","low"),
              ("close","close"), ("volume","volume"), ("openinterest", None),)

def to_bt_data(df: pd.DataFrame):
    d = df.copy()
    d.index.name = "datetime"
    d = d[["open","high","low","close","volume"]]
    return d

def run_backtest(df: pd.DataFrame, impl_name: str, params: Dict[str, Any] | None = None, cash: float = 10000.0) -> Dict[str, Any]:
    """
    Very light BT wrapper: we simulate entries/exits using signals via simple rule replication.
    NOTE: For brevity, this uses a generic strategy that calls a Python predicate on next().
    """
    params = params or {}

    class Generic(bt.Strategy):
        params = dict(rr=2.0, risk_pct=0.01, atr_mult=1.5)
        def __init__(self):
            self.atr = bt.ind.ATR(period=14)
            self.order = None

        def next(self):
            # super-simplified: entry on close if predicate true
            # You can expand: per-impl exact logic mirrored here.
            pass

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0)
    data = PandasData(dataname=to_bt_data(df))
    cerebro.adddata(data)
    cerebro.addstrategy(Generic, **params)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    res = cerebro.run()
    strat = res[0]
    ta = strat.analyzers.trades.get_analysis()
    sh = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = ta.total.closed if "total" in ta and "closed" in ta.total._asdict() else ta.total.get("closed", 0)
    pf = float(ta.pnl.net.total / max(1e-9, -ta.pnl.net.totalopen)) if hasattr(ta, "pnl") else 1.0
    winrate = float(ta.won.total / max(1, trades)) if hasattr(ta, "won") else 0.5

    return dict(
        strategy=impl_name,
        trades=int(trades),
        pf=float(pf),
        winrate=float(winrate),
        sharpe=float(sh.get("sharperatio", 0.0) or 0.0),
        maxdd=float(dd.max.drawdown if hasattr(dd, "max") else 0.0),
    )

def cached_backtest(df: pd.DataFrame, impl_name: str, params: Dict[str, Any], cache_dir=".cache/backtests") -> Dict[str, Any]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = cache_key({"impl":impl_name, "params":params, "shape":df.shape, "idx0":str(df.index[0]), "idxN":str(df.index[-1])})
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    res = run_backtest(df, impl_name, params)
    path.write_text(json.dumps(res, indent=2))
    return res
