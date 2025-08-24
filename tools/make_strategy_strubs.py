from __future__ import annotations
import os, textwrap, re
from pathlib import Path

# === 60+ strategies master list (name, category, schema dict) ===
STRATS = [
    # Trend/MA
    ("EMA20/50 Pullback", "Trend", {"ema_fast":20,"ema_slow":50,"src":"close"}),
    ("EMA 9/21 Crossover", "Trend", {"ema_fast":9,"ema_slow":21,"pullback":True}),
    ("SMA 50/200 Golden Cross", "Trend", {"sma_short":50,"sma_long":200,"src":"close"}),
    ("Hull MA Trend", "Trend", {"period":55}),
    ("TEMA Pullback", "Trend", {"period":34}),
    ("ZLSMA Pullback", "Trend", {"period":50}),
    ("EMA Ribbon Trend", "Trend", {"periods_csv":"8,13,21,34,55"}),
    ("S/R + Trend Alignment", "Trend", {"ema_fast":20,"ema_slow":50,"level_lookback":200}),

    # Breakout
    ("Range Breakout", "Breakout", {"lookback":20,"retest":5}),
    ("Donchian 20 Breakout", "Breakout", {"lookback":20,"retest":True}),
    ("Donchian 55 Trend Follow", "Breakout", {"lookback":55}),
    ("Opening Range Breakout 15m", "Breakout", {"minutes":15}),
    ("Opening Range Breakout 30m", "Breakout", {"minutes":30}),
    ("Gap and Go", "Breakout", {"min_gap_pct":3.0}),
    ("Triangle Breakout", "Breakout", {"lookback":120}),
    ("Flag/Pennant Continuation", "Breakout", {"max_flag_bars":30}),
    ("Cup and Handle", "Breakout", {"min_base_bars":90}),
    ("Channel Breakout", "Breakout", {"lookback":100}),
    ("ATR Channel Break", "Breakout", {"sma":20,"atr_mult":2.0}),
    ("Pivot Breakout", "Breakout", {"pivot_type":"classic"}),
    ("Range Break + Momentum Filter", "Breakout", {"lookback":20,"adx_thr":25,"rsi_period":14}),

    # Mean Reversion / Volatility
    ("Bollinger Mean Reversion", "Mean Reversion", {"period":20,"stddev":2.0,"src":"close"}),
    ("BB Squeeze → Keltner", "Volatility", {"period":20,"stddev":2.0,"keltner_mult":1.5}),
    ("Keltner Channel Reversion", "Mean Reversion", {"ema":20,"atr_mult":2.0}),
    ("Stochastic Oversold Pop", "Mean Reversion", {"k":14,"d":3,"thr":20}),
    ("Williams %R Reversion", "Mean Reversion", {"period":14,"oversold":-80,"overbought":-20}),
    ("CCI Trend Pullback", "Mean Reversion", {"period":20,"reset":0}),
    ("Mean Revert to SMA", "Mean Reversion", {"sma":50,"dev_atr":1.5}),
    ("52-Week Low Mean Reversion", "Mean Reversion", {"min_bounce_pct":2.0}),

    # Momentum
    ("MACD Trend", "Momentum", {"fast":12,"slow":26,"signal":9,"trend_len":50}),
    ("MACD Trend + Pullback", "Momentum", {"fast":12,"slow":26,"signal":9,"pull_ema":20}),
    ("MACD Zero-line Reclaim", "Momentum", {"fast":12,"slow":26,"signal":9}),
    ("ADX Trend Breakout", "Momentum", {"adx_thr":25,"lookback":20}),
    ("RVI Divergence", "Momentum", {"period":14}),
    ("DMI + ADX Alignment", "Momentum", {"adx_thr":20}),
    ("Relative Strength Breakout", "Momentum", {"lookback":20,"benchmark":"SPY"}),
    ("52-Week High Momentum", "Momentum", {"min_volume":0.0}),
    ("VWAP Trend Follow", "Momentum", {"thr_atr":0.0}),

    # Reversals / Levels
    ("RSI Divergence at Level", "Reversal", {"period":14,"level_lookback":100}),
    ("Supply/Demand Flip", "Reversal", {"lookback":150,"confirm":True}),
    ("Swing Failure Pattern (SFP)", "Reversal", {"lookback":60}),
    ("Pivot Reversal", "Reversal", {"pivot_type":"classic"}),
    ("Double Bottom", "Reversal", {"tol_pct":0.5}),
    ("Double Top", "Reversal", {"tol_pct":0.5}),
    ("Head & Shoulders", "Reversal", {"min_span":120}),
    ("Inverse Head & Shoulders", "Reversal", {"min_span":120}),

    # VWAP
    ("VWAP Mean Reversion (intraday)", "VWAP", {"thr_atr":1.5}),
    ("Anchored VWAP Pullback", "VWAP", {"anchor_dt":"","confirm":True}),

    # Systems
    ("Ichimoku Kumo Breakout", "Systems", {"tenkan":9,"kijun":26,"senkou":52}),
    ("Ichimoku TK Cross", "Systems", {"filter_cloud":True}),
    ("Parabolic SAR Trend", "Systems", {"step":0.02,"max":0.2}),
    ("Heikin-Ashi Trend Ride", "Systems", {"min_bars":3}),
    ("Supertrend Trend Following", "Systems", {"atr_period":10,"mult":3.0}),

    # Range
    ("Range Fade", "Range", {"lookback":120,"mid_take":True}),
    ("S/R Break + Retest", "Range", {"level_lookback":200,"confirm":True}),

    # Patterns (non-candlestick-engine)
    ("Rising Three Methods", "Pattern", {"lookback":60}),
    ("Falling Three Methods", "Pattern", {"lookback":60}),
    ("Tasuki Up", "Pattern", {"lookback":40}),
    ("Tasuki Down", "Pattern", {"lookback":40}),
    ("Marubozu Break", "Pattern", {"min_body_pct":1.0}),
    ("Kicker (Bull/Bear)", "Pattern", {"min_body":0.5}),

    # Candlestick set (you already have a candlestick engine; this keeps names visible)
    ("Bullish Engulfing", "Candlestick", {"dummy":0}),
    ("Bearish Engulfing", "Candlestick", {"dummy":0}),
    ("Hammer", "Candlestick", {"dummy":0}),
    ("Inverted Hammer", "Candlestick", {"dummy":0}),
    ("Hanging Man", "Candlestick", {"dummy":0}),
    ("Shooting Star", "Candlestick", {"dummy":0}),
    ("Tweezer Top", "Candlestick", {"dummy":0}),
    ("Tweezer Bottom", "Candlestick", {"dummy":0}),
]

# --- helpers ---
def to_module(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return base + ".py"

def class_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", name.title())

TEMPLATE = '''\
from __future__ import annotations
import pandas as pd
from strategies.base import BaseStrategy
from engine.utils import Signal

class {cls}(BaseStrategy):
    name = "{name}"
    CATEGORY = "{cat}"
    PARAMS_SCHEMA = {schema}

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        # TODO: implement logic
        # You receive df with columns: open, high, low, close (lowercase recommended)
        # Return a list[Signal(name, side, index, timestamp, confidence, reasons, price)]
        return []
'''

def main():
    lib = Path("strategies/library")
    lib.mkdir(parents=True, exist_ok=True)

    for name, cat, schema in STRATS:
        p = lib / to_module(name)
        if p.exists():
            continue
        cls = class_name(name)
        code = TEMPLATE.format(cls=cls, name=name, cat=cat, schema=repr(schema))
        p.write_text(code, encoding="utf-8")
        print("Wrote", p)

    init = lib / "__init__.py"
    if not init.exists():
        init.write_text("# generated\n", encoding="utf-8")

if __name__ == "__main__":
    main()
