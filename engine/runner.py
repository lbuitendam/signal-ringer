from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Callable, Awaitable, Union

import pandas as pd

from data.provider import get_provider
from engine.utils import Signal, OrderSuggestion
from strategies.base import BaseStrategy
from strategies.library.ema_pullback import EmaPullback
from strategies.library.macd_trend import MacdTrend
from strategies.library.candlesticks import CandlesStrategy
from strategies.library.range_breakout import RangeBreakout
from strategies.library.ensemble_bag import EnsembleBAG
from risk.manager import RiskManager, RiskOptions
from sizing.position import position_size

StrategySpec = Dict[str, Any]
TrackerSpec = Dict[str, Any]
SuggestionCb = Callable[[OrderSuggestion], Union[None, Awaitable[None]]]

# --------------------- helpers ---------------------
async def _maybe_await(fn, *args, **kwargs):
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res

def _lower_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    lower_map = {c.lower(): c for c in df.columns}
    for k in ("open", "high", "low", "close", "volume"):
        if k in lower_map:
            out[k] = pd.to_numeric(df[lower_map[k]], errors="coerce")
        elif k in df.columns:
            out[k] = pd.to_numeric(df[k], errors="coerce")
        else:
            if k != "volume":
                out[k] = pd.NA
            else:
                out[k] = 0.0
    if {"open", "high", "low", "close"}.difference(out.columns):
        return pd.DataFrame()
    return out.dropna(subset=["open", "high", "low", "close"])

# --------------------- strategy factory ---------------------
def build_strategy(name: str, params: Dict[str, Any] | None = None) -> BaseStrategy:
    params = params or {}
    n = (name or "").strip().lower()

    if n in {"ema pullback", "ema20/50 pullback", "ema"}:
        return EmaPullback(params)
    if n in {"macd trend", "macd"}:
        return MacdTrend(params)
    if n in {"range breakout", "range breakout + retest", "breakout"}:
        return RangeBreakout(params.get("lookback", 20), params.get("retest", 5), params)

    # candlesticks
    if n in {"hammer"}: return CandlesStrategy("hammer", params)
    if n in {"inverted hammer", "inv hammer", "inv_hammer"}: return CandlesStrategy("inv_hammer", params)
    if n in {"hanging man", "hanging_man"}: return CandlesStrategy("hanging_man", params)
    if n in {"shooting star", "shooting_star"}: return CandlesStrategy("shooting_star", params)
    if n in {"bullish engulfing", "engulf bull", "engulf_bull"}: return CandlesStrategy("engulf_bull", params)
    if n in {"bearish engulfing", "engulf bear", "engulf_bear"}: return CandlesStrategy("engulf_bear", params)
    if n in {"tweezer top", "tweezer_top"}: return CandlesStrategy("tweezer_top", params)
    if n in {"tweezer bottom", "tweezer_bottom"}: return CandlesStrategy("tweezer_bottom", params)

    raise ValueError(f"Unknown strategy: {name}")

# --------------------- data fetch ---------------------
async def fetch_df(executor: ThreadPoolExecutor, symbol: str, tf: str, lookback: int) -> pd.DataFrame:
    prov = get_provider()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, prov.get_ohlcv, symbol, tf, lookback)

# --------------------- order proposal ---------------------
def propose_orders_for_signal(
    symbol: str, timeframe: str, sig: Signal, risk: RiskManager, equity: float, rr: float, df: pd.DataFrame
) -> OrderSuggestion:
    side = "buy" if sig.side == "long" else "sell"
    i = int(sig.index)
    entry = float(pd.to_numeric(df["close"].iat[i], errors="coerce"))
    sl, tps = risk.build_sl_tp(df, i, sig.side, entry)
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
    # Risk + optional ensemble
    risk = RiskManager(risk_opts)
    bag = EnsembleBAG(
        k_bars=strategies_cfg.get("big_boss", {}).get("k_bars", 3),
        tol=strategies_cfg.get("big_boss", {}).get("tol", 0.003),
    )

    # Concurrent fetch
    active = [t for t in (trackers or []) if t.get("enabled", True)]
    if not active:
        return

    executor = ThreadPoolExecutor(max_workers=min(16, max(1, len(active) + 2)))
    tasks = [fetch_df(executor, t["symbol"], t.get("timeframe", "5m"), lookback) for t in active]
    dfs = await asyncio.gather(*tasks, return_exceptions=True)

    for t, dfor in zip(active, dfs):
        if isinstance(dfor, Exception):
            print(f"[WARN] fetch failed {t['symbol']} {t.get('timeframe')}: {dfor}")
            continue

        df = _lower_ohlcv(dfor)
        if df.empty:
            continue

        # Run strategies (enabled + approved)
        all_signals: List[Signal] = []
        for name, cfg in (strategies_cfg.get("enabled", {}) or {}).items():
            if not cfg.get("enabled", True) or not cfg.get("approved", True):
                continue
            try:
                strat = build_strategy(name, cfg.get("params", {}))
                sigs = strat.signals(df.tail(400))
                all_signals.extend(sigs[-3:])
            except Exception as e:
                print(f"[WARN] strategy {name} failed: {e}")

        # Ensemble
        if strategies_cfg.get("big_boss", {}).get("enabled", False) and all_signals:
            try:
                all_signals.extend(bag.combine(df, all_signals))
            except Exception as e:
                print(f"[WARN] ensemble combine failed: {e}")

        if df.empty:
            continue
        now = df.index[-1]
        for s in all_signals:
            key = f"{t['symbol']}:{s.name}:{s.side}"
            if risk.in_cooldown(key, now):
                continue
            sug = propose_orders_for_signal(t["symbol"], t.get("timeframe", "5m"), s, risk, risk_opts.equity, risk_opts.rr, df)
            if sug.qty <= 0:
                continue
            risk.arm_cooldown(key, now)
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
