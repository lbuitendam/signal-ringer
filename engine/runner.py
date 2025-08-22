from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Callable, Awaitable, Union

import pandas as pd

from engine.data import get_adapter, BaseAdapter
from engine.utils import Signal, OrderSuggestion
from strategies.base import BaseStrategy
from strategies.library.ema_pullback import EmaPullback
from strategies.library.macd_trend import MacdTrend
from strategies.library.candlesticks import CandlesStrategy
from strategies.library.range_breakout import RangeBreakout
from strategies.library.ensemble_bag import EnsembleBAG
from risk.manager import RiskManager, RiskOptions
from sizing.position import position_size


StrategySpec = Dict[str, Any]  # {"name":..., "params":{...}, "enabled":True}
TrackerSpec = Dict[str, Any]   # {"symbol":..., "timeframe":..., "adapter":"yfinance"|"csv"|..., "enabled":True}
SuggestionCb = Callable[[OrderSuggestion], Union[None, Awaitable[None]]]


# --------------------- helpers ---------------------
async def _maybe_await(callable_or_coro, *args, **kwargs):
    """
    Call a callable; if it returns an awaitable, await it.
    Works for both sync and async callbacks.
    """
    res = callable_or_coro(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


def _lower_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has lowercase 'open','high','low','close','volume' if possible.
    Strategies expect lowercase.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    lower_map = {c.lower(): c for c in df.columns}
    for k in ("open", "high", "low", "close", "volume"):
        if k in lower_map:
            out[k] = pd.to_numeric(df[lower_map[k]], errors="coerce")
        elif k in df.columns:
            out[k] = pd.to_numeric(df[k], errors="coerce")
    # drop if essential OHLC missing
    if {"open", "high", "low", "close"}.difference(out.columns):
        return pd.DataFrame()
    return out.dropna(subset=["open", "high", "low", "close"])


# --------------------- strategy factory ---------------------
def build_strategy(name: str, params: Dict[str, Any] | None = None) -> BaseStrategy:
    params = params or {}
    n = name.lower()
    if n in {"ema pullback", "ema20/50 pullback", "ema"}:
        return EmaPullback(params)
    if n in {"macd trend", "macd"}:
        return MacdTrend(params)
    if n in {"range breakout", "range breakout + retest", "breakout"}:
        return RangeBreakout(params.get("lookback", 20), params.get("retest", 5), params)
    # candlesticks
    if n in {"hammer", "inverted hammer", "inv hammer", "inv_hammer"}:
        which = "hammer" if "invert" not in n else "inv_hammer"
        return CandlesStrategy(which, params)
    if n in {"hanging man", "hanging_man"}: return CandlesStrategy("hanging_man", params)
    if n in {"shooting star", "shooting_star"}: return CandlesStrategy("shooting_star", params)
    if n in {"bullish engulfing", "engulf bull", "engulf_bull"}: return CandlesStrategy("engulf_bull", params)
    if n in {"bearish engulfing", "engulf bear", "engulf_bear"}: return CandlesStrategy("engulf_bear", params)
    if n in {"tweezer top", "tweezer_top"}: return CandlesStrategy("tweezer_top", params)
    if n in {"tweezer bottom", "tweezer_bottom"}: return CandlesStrategy("tweezer_bottom", params)
    raise ValueError(f"Unknown strategy: {name}")


# --------------------- IO ---------------------
async def fetch_df(executor: ThreadPoolExecutor, adapter: BaseAdapter, symbol: str, tf: str, lookback: int) -> pd.DataFrame:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, adapter.fetch_history, symbol, tf, lookback)


def propose_orders_for_signal(
    symbol: str, timeframe: str, sig: Signal, risk: RiskManager, equity: float, rr: float, df: pd.DataFrame
) -> OrderSuggestion:
    side = "buy" if sig.side == "long" else "sell"
    # scalar-safe access
    entry = float(pd.to_numeric(df["close"].iat[int(sig.index)], errors="coerce"))
    sl, tps = risk.build_sl_tp(df, int(sig.index), sig.side, entry)
    qty = position_size(equity, risk.opts.risk_pct, entry, sl)
    return OrderSuggestion(
        symbol=symbol, timeframe=timeframe, side=side, type="market",
        qty=float(qty), entry=float(entry), sl=float(sl), tp=[float(x) for x in tps], time_in_force="GTC",
        strategy=sig.name, confidence=float(sig.confidence), reason="; ".join(sig.reasons), paper=True
    )


# --------------------- scan ---------------------
async def scan_once(
    trackers: List[TrackerSpec],
    strategies_cfg: Dict[str, Dict[str, Any]],
    risk_opts: RiskOptions,
    on_suggestion: SuggestionCb,
    lookback: int = 600,
) -> None:
    # Prepare adapters per tracker
    adapters: Dict[str, BaseAdapter] = {}
    for t in trackers:
        if t.get("enabled", True) is False:
            continue
        akey = f'{t.get("adapter","yfinance")}::{t.get("adapter_path","")}'
        if akey not in adapters:
            if t.get("adapter") == "csv":
                adapters[akey] = get_adapter("csv", path=t.get("adapter_path", "data/{symbol}.csv"))
            else:
                adapters[akey] = get_adapter(t.get("adapter", "yfinance"))

    risk = RiskManager(risk_opts)
    bag = EnsembleBAG(
        k_bars=strategies_cfg.get("big_boss", {}).get("k_bars", 3),
        tol=strategies_cfg.get("big_boss", {}).get("tol", 0.003),
    )

    # Fetch concurrently
    executor = ThreadPoolExecutor(max_workers=min(16, max(1, len(trackers) + 4)))
    tasks = []
    ctxs: List[Dict[str, Any]] = []
    for t in trackers:
        if not t.get("enabled", True):
            continue
        adapter = adapters[f'{t.get("adapter","yfinance")}::{t.get("adapter_path","")}']
        tasks.append(fetch_df(executor, adapter, t["symbol"], t["timeframe"], lookback))
        ctxs.append(t)
    dfs = await asyncio.gather(*tasks, return_exceptions=True)

    for t, dfor in zip(ctxs, dfs):
        if isinstance(dfor, Exception):
            # compact warning
            print(f"[WARN] fetch failed {t['symbol']} {t['timeframe']}: {dfor}")
            continue

        raw: pd.DataFrame = dfor
        df = _lower_ohlcv(raw)
        if df is None or df.empty:
            continue

        # Build & run strategies (only enabled & approved)
        all_signals: List[Signal] = []
        for name, cfg in strategies_cfg.get("enabled", {}).items():
            if not cfg.get("enabled", True):
                continue
            if not cfg.get("approved", True):  # gate from backtests
                continue
            strat = build_strategy(name, cfg.get("params", {}))
            try:
                sigs = strat.signals(df.tail(400))  # evaluate recent window
                # keep only the most-recent few per strategy to avoid spamming
                all_signals.extend(sigs[-3:])
            except Exception as e:
                print(f"[WARN] strategy {name} failed: {e}")

        # Big Boss combination (if enabled)
        if strategies_cfg.get("big_boss", {}).get("enabled", False) and all_signals:
            try:
                all_signals.extend(bag.combine(df, all_signals))
            except Exception as e:
                print(f"[WARN] ensemble combine failed: {e}")

        # Risking + cooldown → suggestions
        if len(df.index) == 0:
            continue
        now = df.index[-1]
        for s in all_signals:
            key = f"{t['symbol']}:{s.name}:{s.side}"
            if risk.in_cooldown(key, now):
                continue
            sug = propose_orders_for_signal(t["symbol"], t["timeframe"], s, risk, risk_opts.equity, risk_opts.rr, df)
            if sug.qty <= 0:
                continue
            risk.arm_cooldown(key, now)
            # <-- handle sync or async sink
            await _maybe_await(on_suggestion, sug)


# --------------------- loop ---------------------
async def run_engine_loop(
    trackers: List[TrackerSpec],
    strategies_cfg: Dict[str, Dict[str, Any]],
    risk_opts: RiskOptions,
    on_suggestion: SuggestionCb,
    interval_sec: int = 30,
):
    while True:
        try:
            await scan_once(trackers, strategies_cfg, risk_opts, on_suggestion)
        except Exception as e:
            print("[ENGINE] scan error:", e)
        await asyncio.sleep(int(max(5, interval_sec)))
