# strategies/library/candlesticks.py
from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from engine.utils import Signal

def _body(o, c): return (c - o).abs()
def _is_bull(o, c): return c > o
def _is_bear(o, c): return c < o
def _upper(o, h, c): return h - np.maximum(o, c)
def _lower(o, l, c): return np.minimum(o, c) - l

def detect_engulfing(df: pd.DataFrame) -> List[int]:
    o,c,h,l = df["open"], df["close"], df["high"], df["low"]
    bull = (_is_bull(o,c) & _is_bear(o.shift(1), c.shift(1)) &
           (c >= o.shift(1)) & (o <= c.shift(1)))
    bear = (_is_bear(o,c) & _is_bull(o.shift(1), c.shift(1)) &
           (c <= o.shift(1)) & (o >= c.shift(1)))
    idx = list(df.index[bull | bear])
    return idx

def detect_hammer_like(df: pd.DataFrame, inverted=False) -> List[int]:
    o,c,h,l = df["open"], df["close"], df["high"], df["low"]
    body = _body(o,c)
    U = _upper(o,h,c)
    L = _lower(o,l,c)
    if inverted:
        cond = (U >= 2*body) & (L <= 0.5*body)
    else:
        cond = (L >= 2*body) & (U <= 0.5*body)
    return list(df.index[cond])

def detect_tweezer(df: pd.DataFrame, top=False, tol=0.0005) -> List[int]:
    # successive highs (top) or lows (bottom) within tolerance
    h,l = df["high"], df["low"]
    if top:
        cond = (h.shift(1).notna()) & ((h - h.shift(1)).abs() <= tol * h)
    else:
        cond = (l.shift(1).notna()) & ((l - l.shift(1)).abs() <= tol * l)
    return list(df.index[cond])

class CandlesStrategy(BaseStrategy):
    name = "Candlestick Patterns"
    def __init__(self, which: str, params=None):
        super().__init__(params)
        self.which = which  # e.g. "hammer","inv_hammer","engulf_bull","engulf_bear","tweezer_top","tweezer_bottom"

    def signals(self, df: pd.DataFrame):
        df = df.copy()
        o,c = df["open"], df["close"]
        out: List[Signal] = []
        if self.which == "hammer":
            for ts in detect_hammer_like(df, inverted=False):
                side = "long"
                conf = 0.6
                price = float(df.loc[ts, "close"])
                out.append(Signal("Hammer", side, df.index.get_loc(ts), ts, conf, ["hammer"], price))
        elif self.which == "inv_hammer":
            for ts in detect_hammer_like(df, inverted=True):
                side = "long"
                conf = 0.55
                price = float(df.loc[ts, "close"])
                out.append(Signal("Inverted Hammer", side, df.index.get_loc(ts), ts, conf, ["inv_hammer"], price))
        elif self.which == "hanging_man":
            for ts in detect_hammer_like(df, inverted=False):
                side = "short"
                conf = 0.55
                price = float(df.loc[ts, "close"])
                out.append(Signal("Hanging Man", side, df.index.get_loc(ts), ts, conf, ["hanging_man"], price))
        elif self.which == "shooting_star":
            for ts in detect_hammer_like(df, inverted=True):
                side = "short"
                conf = 0.6
                price = float(df.loc[ts, "close"])
                out.append(Signal("Shooting Star", side, df.index.get_loc(ts), ts, conf, ["shooting_star"], price))
        elif self.which == "engulf_bull" or self.which == "engulf_bear":
            idxs = detect_engulfing(df)
            for ts in idxs:
                bull = df.loc[ts, "close"] > df.loc[ts, "open"]
                if self.which == "engulf_bull" and bull:
                    out.append(Signal("Bullish Engulfing", "long", df.index.get_loc(ts), ts, 0.65, ["engulf"], float(df.loc[ts,"close"])))
                if self.which == "engulf_bear" and not bull:
                    out.append(Signal("Bearish Engulfing", "short", df.index.get_loc(ts), ts, 0.65, ["engulf"], float(df.loc[ts,"close"])))
        elif self.which == "tweezer_top":
            for ts in detect_tweezer(df, top=True):
                out.append(Signal("Tweezer Top", "short", df.index.get_loc(ts), ts, 0.55, ["tweezer_top"], float(df.loc[ts,"close"])))
        elif self.which == "tweezer_bottom":
            for ts in detect_tweezer(df, top=False):
                out.append(Signal("Tweezer Bottom", "long", df.index.get_loc(ts), ts, 0.55, ["tweezer_bottom"], float(df.loc[ts,"close"])))
        return out
