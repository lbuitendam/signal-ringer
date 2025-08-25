# strategies/library/ema_crossover_with_filter.py
from __future__ import annotations
import pandas as pd
from strategies.base import BaseStrategy
from engine.utils import Signal

class EmaCrossoverWithFilter(BaseStrategy):
    name = "EMA Crossover + RSI/ADX Filter"
    CATEGORY = "Trend"
    DESC = "50/200 cross gated by RSI<70 & ADX>20"
    PARAMS_SCHEMA = {
        "fast":  {"type":"int","min":5,"max":200,"step":1,"default":50,"label":"EMA fast"},
        "slow":  {"type":"int","min":10,"max":400,"step":1,"default":200,"label":"EMA slow"},
        "rsi":   {"type":"int","min":5,"max":50,"step":1,"default":14,"label":"RSI len"},
        "adx":   {"type":"int","min":5,"max":50,"step":1,"default":14,"label":"ADX len"},
        "adx_thr":{"type":"float","min":5,"max":60,"step":0.5,"default":20,"label":"ADX threshold"},
    }

    def _ema(self, s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
    def _rsi(self, s: pd.Series, n: int):
        d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        rs = up / dn.replace(0, pd.NA); return 100 - 100/(1+rs)
    def _adx(self, df: pd.DataFrame, n: int):
        h,l,c = df["high"],df["low"],df["close"]
        plus_dm  = (h.diff().clip(lower=0) > (-l.diff().clip(upper=0))).astype(float) * h.diff().clip(lower=0)
        minus_dm = ((-l.diff().clip(upper=0)) > h.diff().clip(lower=0)).astype(float) * (-l.diff().clip(upper=0))
        tr = (pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1)).max(axis=1)
        atr = tr.rolling(n).mean()
        pdi = 100*(plus_dm.rolling(n).mean()/atr); mdi = 100*(minus_dm.rolling(n).mean()/atr)
        dx = ( (pdi - mdi).abs() / (pdi + mdi).replace(0, pd.NA) ) * 100
        return dx.rolling(n).mean()

    def signals(self, df: pd.DataFrame):
        p = df.copy()
        p.columns = [c.lower() for c in p.columns]
        close = p["close"].astype(float)
        fast = int(self.params.get("fast", 50))
        slow = int(self.params.get("slow", 200))
        rlen = int(self.params.get("rsi", 14))
        alen = int(self.params.get("adx", 14))
        adx_thr = float(self.params.get("adx_thr", 20))

        ema_f, ema_s = self._ema(close, fast), self._ema(close, slow)
        rsi = self._rsi(close, rlen).fillna(50)
        adx = self._adx(p, alen).fillna(15)

        cross_up   = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1)) & (rsi < 70) & (adx > adx_thr)
        cross_down = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1)) & (rsi > 30) & (adx > adx_thr)

        out = []
        for ts in p.index[cross_up]:
            i = p.index.get_loc(ts); out.append(Signal(self.name,"long",i,ts,0.68,["ema cross + filters"],float(close.loc[ts])))
        for ts in p.index[cross_down]:
            i = p.index.get_loc(ts); out.append(Signal(self.name,"short",i,ts,0.68,["ema cross + filters"],float(close.loc[ts])))
        return out