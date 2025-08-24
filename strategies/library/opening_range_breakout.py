from __future__ import annotations
import pandas as pd
from strategies.base import BaseStrategy
from engine.utils import Signal

class OpeningRangeBreakout(BaseStrategy):
    name = "Opening Range Breakout 15m"
    CATEGORY = "Breakout"
    PARAMS_SCHEMA = {"minutes":{"type":"int","min":5,"max":60,"step":5,"default":15,"label":"OR minutes"}}

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        # Assumes intraday data with timezone-aware index
        p = df.copy()
        p.index = p.index.tz_convert("UTC") if p.index.tz is not None else p.index.tz_localize("UTC")
        mins = int(self.params.get("minutes", 15))

        out: list[Signal] = []
        # group by date, compute first N-minute high/low, mark later breakouts
        for day, g in p.groupby(p.index.date):
            g = g.sort_index()
            start = g.index[0]
            or_end = start + pd.Timedelta(minutes=mins)
            gr = g[(g.index >= start) & (g.index < or_end)]
            if gr.empty: 
                continue
            hi, lo = gr["high"].max(), gr["low"].min()
            later = g[g.index >= or_end]
            if later.empty:
                continue
            cross_up = (later["close"] > hi) & (later["close"].shift(1) <= hi)
            cross_dn = (later["close"] < lo) & (later["close"].shift(1) >= lo)
            for ts in later.index[cross_up]:
                i = p.index.get_loc(ts)
                out.append(Signal(self.name, "long", i, ts, 0.6, ["OR breakout up"], float(p.loc[ts,"close"])))
            for ts in later.index[cross_dn]:
                i = p.index.get_loc(ts)
                out.append(Signal(self.name, "short", i, ts, 0.6, ["OR breakout down"], float(p.loc[ts,"close"])))
        return out
