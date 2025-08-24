from __future__ import annotations
import pandas as pd, numpy as np
from strategies.base import BaseStrategy
from engine.utils import Signal

def intraday_vwap(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns or df["volume"].fillna(0).sum() == 0:
        return pd.Series(np.nan, index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    byday = pd.Index(df.index.tz_convert("UTC").date, name="day") if df.index.tz is not None else pd.Index(df.index.date, name="day")
    pv = (tp * df["volume"]).groupby(byday).cumsum()
    vv = df["volume"].groupby(byday).cumsum()
    return pv / vv

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    tr = (h - l).combine_abs(h - c.shift(1)).combine(lambda a,b: max(a,b), l - c.shift(1)).abs()
    return tr.rolling(int(max(1,n)), min_periods=1).mean()

class VwapReversion(BaseStrategy):
    name = "VWAP Mean Reversion (intraday)"
    CATEGORY = "VWAP"
    PARAMS_SCHEMA = {"thr_atr":{"type":"float","min":0.5,"max":5.0,"step":0.1,"default":1.5,"label":"Distance (ATR)"}}

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        p = df.copy()
        if p.index.tz is None: p.index = p.index.tz_localize("UTC")
        v = intraday_vwap(p)
        if v.isna().all():
            return []
        a = atr(p, 14)
        c = p["close"]
        dist = (c - v).abs() / a.replace(0, np.nan)
        thr = float(self.params.get("thr_atr", 1.5))

        # Enter when distance was > thr and then price closes back toward vwap
        out: list[Signal] = []
        back_long = (c.shift(1) < v.shift(1)) & (dist.shift(1) > thr) & (c >= v)
        back_short = (c.shift(1) > v.shift(1)) & (dist.shift(1) > thr) & (c <= v)
        for ts in p.index[back_long]:
            i = p.index.get_loc(ts); out.append(Signal(self.name, "long", i, ts, 0.57, ["vwap revert"], float(c.loc[ts])))
        for ts in p.index[back_short]:
            i = p.index.get_loc(ts); out.append(Signal(self.name, "short", i, ts, 0.57, ["vwap revert"], float(c.loc[ts])))
        return out
