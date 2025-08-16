# patterns/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple, Optional
import numpy as np
import pandas as pd

# ----------------- Config -----------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # generic context
    "trend_lookback": 5,
    "atr_lookback": 14,
    "gap_min_atr_mult": 0.15,

    # doji / hammer family
    "doji_body_max_k": 0.1,  # body <= k * median(range)
    "hammer_lower_wick_min_mult": 2.0,
    "hammer_upper_wick_max_mult": 0.25,
    "hammer_body_pos_min": 0.66,
    "inv_hammer_upper_wick_min_mult": 2.0,
    "inv_hammer_lower_wick_max_mult": 0.25,
    "inv_hammer_body_pos_max": 0.34,

    # engulfing / misc
    "engulfing_min_body_frac": 0.4,  # vs median range
    "harami_body_inside_tol": 0.0,   # strict inside
    "tweezer_tol_atr_mult": 0.15,
    "star_middle_body_frac": 0.2,    # 2nd bar "small"

    # alerts
    "min_bars_between_alerts": 3,
    "min_confidence": 0.6,

    # --- Head & Shoulders specific ---
    "hs_swing": 3,                     # pivot radius (bars on each side)
    "hs_min_separation": 6,            # min bars between key pivots
    "hs_shoulder_tol": 0.03,           # shoulders equal within 3%
    "hs_head_min_diff": 0.03,          # head ≥3% above/ below shoulders
    "hs_confirm_lookahead": 25,        # bars after R-shoulder to confirm
    "hs_neckline_buffer_atr_mult": 0.05, # ATR buffer through neckline
}

# ----------------- Data model -----------------
@dataclass
class PatternHit:
    name: str
    index: int
    bars: List[int]
    direction: str  # "bull" | "bear" | "neutral"
    confidence: float
    explanation: str

# ----------------- Helpers -----------------
def _features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    up = h - np.maximum(o, c)
    dn = np.minimum(o, c) - l
    body_pos = (np.maximum(o, c) - l) / rng  # 0 bottom .. 1 top
    color = np.where(c >= o, 1, -1)  # green/red
    out = pd.DataFrame({
        "o": o, "h": h, "l": l, "c": c, "body": body, "range": rng,
        "up": up, "dn": dn, "body_pos": body_pos, "color": color
    }, index=df.index)
    return out

def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=1).mean()

def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _median_range(feat: pd.DataFrame, n: int = 100) -> float:
    return float(pd.Series(feat["range"]).tail(n).median())

def _is_downtrend(close: pd.Series, lookback: int) -> pd.Series:
    ma = _sma(close, lookback)
    slope = ma - ma.shift(3)
    return (close < ma) & (slope < 0)

def _is_uptrend(close: pd.Series, lookback: int) -> pd.Series:
    ma = _sma(close, lookback)
    slope = ma - ma.shift(3)
    return (close > ma) & (slope > 0)

def _gap_up(df: pd.DataFrame, atr: pd.Series, i: int, cfg: Dict[str, Any]) -> bool:
    if i <= 0: return False
    return df["Low"].iat[i] > df["High"].iat[i-1] + cfg["gap_min_atr_mult"] * atr.iat[i]

def _gap_down(df: pd.DataFrame, atr: pd.Series, i: int, cfg: Dict[str, Any]) -> bool:
    if i <= 0: return False
    return df["High"].iat[i] < df["Low"].iat[i-1] - cfg["gap_min_atr_mult"] * atr.iat[i]

# ----------------- Candlestick Detectors -----------------
def detect_hammer(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    f = _features(df); close = df["Close"]
    down = _is_downtrend(close, cfg["trend_lookback"])
    hits: List[PatternHit] = []
    for i in range(len(df)):
        if not bool(down.iat[i]): continue
        if f["range"].iat[i] == 0: continue
        cond_body_top = f["body_pos"].iat[i] >= cfg["hammer_body_pos_min"]
        cond_long_lower = f["dn"].iat[i] >= cfg["hammer_lower_wick_min_mult"] * max(f["body"].iat[i], 1e-12)
        cond_small_upper = f["up"].iat[i] <= cfg["hammer_upper_wick_max_mult"] * max(f["body"].iat[i], 1e-12)
        if cond_body_top and cond_long_lower and cond_small_upper:
            conf = min(1.0, (f["dn"].iat[i] / max(f["body"].iat[i], 1e-12)) / cfg["hammer_lower_wick_min_mult"])
            hits.append(PatternHit("Hammer", i, [i], "bull", conf, "Small body near top, long lower wick after downtrend"))
    return hits

def detect_inverted_hammer(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    f = _features(df); close = df["Close"]
    down = _is_downtrend(close, cfg["trend_lookback"])
    hits: List[PatternHit] = []
    for i in range(len(df)):
        if not bool(down.iat[i]): continue
        if f["range"].iat[i] == 0: continue
        cond_body_bottom = f["body_pos"].iat[i] <= cfg["inv_hammer_body_pos_max"]
        cond_long_upper = f["up"].iat[i] >= cfg["inv_hammer_upper_wick_min_mult"] * max(f["body"].iat[i], 1e-12)
        cond_small_lower = f["dn"].iat[i] <= cfg["inv_hammer_lower_wick_max_mult"] * max(f["body"].iat[i], 1e-12)
        if cond_body_bottom and cond_long_upper and cond_small_lower:
            conf = min(1.0, (f["up"].iat[i] / max(f["body"].iat[i], 1e-12)) / cfg["inv_hammer_upper_wick_min_mult"])
            hits.append(PatternHit("Inverted Hammer", i, [i], "bull", conf, "Small body near bottom, long upper wick after downtrend"))
    return hits

def detect_doji(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    f = _features(df)
    med_rng = _median_range(f)
    k = cfg["doji_body_max_k"]
    hits: List[PatternHit] = []
    for i in range(len(df)):
        if f["body"].iat[i] <= k * med_rng:
            # subtype
            sub = "Doji"
            if f["dn"].iat[i] >= 2.5 * f["body"].iat[i] and f["up"].iat[i] <= 0.5 * f["body"].iat[i]:
                sub = "Dragonfly Doji"
            elif f["up"].iat[i] >= 2.5 * f["body"].iat[i] and f["dn"].iat[i] <= 0.5 * f["body"].iat[i]:
                sub = "Gravestone Doji"
            elif f["up"].iat[i] >= 1.5 * f["body"].iat[i] and f["dn"].iat[i] >= 1.5 * f["body"].iat[i]:
                sub = "Long-Legged Doji"
            conf = min(1.0, 1.0 - (f["body"].iat[i] / max(k * med_rng, 1e-12)))
            hits.append(PatternHit(sub, i, [i], "neutral", conf, "Small real body"))
    return hits

def detect_engulfing(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    f = _features(df)
    med_rng = _median_range(f)
    hits: List[PatternHit] = []
    for i in range(1, len(df)):
        o1, c1 = f["o"].iat[i-1], f["c"].iat[i-1]
        o2, c2 = f["o"].iat[i], f["c"].iat[i]
        body1 = abs(c1 - o1); body2 = abs(c2 - o2)
        if body2 < cfg["engulfing_min_body_frac"] * med_rng: continue
        # opposite colors + body engulf
        if (c2 > o2 and c1 < o1) and (o2 <= c1 and c2 >= o1):
            ratio = body2 / max(body1, 1e-12)
            conf = min(1.0, ratio / 1.2)
            hits.append(PatternHit("Bullish Engulfing", i, [i-1, i], "bull", conf, "Up real body engulfs prior down body"))
        elif (c2 < o2 and c1 > o1) and (o2 >= c1 and c2 <= o1):
            ratio = body2 / max(body1, 1e-12)
            conf = min(1.0, ratio / 1.2)
            hits.append(PatternHit("Bearish Engulfing", i, [i-1, i], "bear", conf, "Down real body engulfs prior up body"))
    return hits

def _small_body(f: pd.DataFrame, i: int, med_rng: float, frac: float) -> bool:
    return f["body"].iat[i] <= frac * med_rng

def detect_morning_evening_star(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    f = _features(df); close = df["Close"]
    up = _is_uptrend(close, cfg["trend_lookback"])
    down = _is_downtrend(close, cfg["trend_lookback"])
    med_rng = _median_range(f)
    hits: List[PatternHit] = []
    for i in range(2, len(df)):
        # Morning Star (bullish)
        if bool(down.iat[i-2]) and f["c"].iat[i-2] < f["o"].iat[i-2]:
            if _small_body(f, i-1, med_rng, cfg["star_middle_body_frac"]):
                mid1 = (f["o"].iat[i-2] + f["c"].iat[i-2]) / 2.0
                if f["c"].iat[i] > f["o"].iat[i] and f["c"].iat[i] >= mid1:
                    conf = min(1.0, (f["c"].iat[i] - mid1) / max(f["range"].iat[i-2], 1e-12) + 0.5)
                    hits.append(PatternHit("Morning Star", i, [i-2, i-1, i], "bull", float(conf), "Three-bar bullish reversal"))
        # Evening Star (bearish)
        if bool(up.iat[i-2]) and f["c"].iat[i-2] > f["o"].iat[i-2]:
            if _small_body(f, i-1, med_rng, cfg["star_middle_body_frac"]):
                mid1 = (f["o"].iat[i-2] + f["c"].iat[i-2]) / 2.0
                if f["c"].iat[i] < f["o"].iat[i] and f["c"].iat[i] <= mid1:
                    conf = min(1.0, (mid1 - f["c"].iat[i]) / max(f["range"].iat[i-2], 1e-12) + 0.5)
                    hits.append(PatternHit("Evening Star", i, [i-2, i-1, i], "bear", float(conf), "Three-bar bearish reversal"))
    return hits

def detect_harami(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    f = _features(df)
    hits: List[PatternHit] = []
    for i in range(1, len(df)):
        o1, c1 = f["o"].iat[i-1], f["c"].iat[i-1]
        o2, c2 = f["o"].iat[i], f["c"].iat[i]
        b1_lo, b1_hi = min(o1, c1), max(o1, c1)
        b2_lo, b2_hi = min(o2, c2), max(o2, c2)
        inside = (b2_lo >= b1_lo) and (b2_hi <= b1_hi)
        if not inside: continue
        if (c1 < o1) and (c2 > o2):
            hits.append(PatternHit("Bullish Harami", i, [i-1, i], "bull", 0.7, "Small up body inside prior down body"))
        elif (c1 > o1) and (c2 < o2):
            hits.append(PatternHit("Bearish Harami", i, [i-1, i], "bear", 0.7, "Small down body inside prior up body"))
    return hits

def detect_tweezer(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    atr = _atr(df, cfg["atr_lookback"])
    hits: List[PatternHit] = []
    for i in range(1, len(df)):
        tol = cfg["tweezer_tol_atr_mult"] * atr.iat[i]
        if abs(df["High"].iat[i] - df["High"].iat[i-1]) <= tol:
            hits.append(PatternHit("Tweezer Top", i, [i-1, i], "bear", 0.65, "Similar highs across two bars"))
        if abs(df["Low"].iat[i] - df["Low"].iat[i-1]) <= tol:
            hits.append(PatternHit("Tweezer Bottom", i, [i-1, i], "bull", 0.65, "Similar lows across two bars"))
    return hits

# ---- Stubs (placeholders for v2; return [] for now) ----
def detect_piercing_line(df, cfg): return []
def detect_dark_cloud(df, cfg): return []
def detect_three_soldiers(df, cfg): return []
def detect_three_crows(df, cfg): return []
def detect_three_inside_up(df, cfg): return []
def detect_three_inside_down(df, cfg): return []
def detect_three_outside_up(df, cfg): return []
def detect_three_outside_down(df, cfg): return []
def detect_marubozu(df, cfg): return []
def detect_windows(df, cfg): return []
def detect_tasuki_up(df, cfg): return []
def detect_tasuki_down(df, cfg): return []
def detect_kicker_bull(df, cfg): return []
def detect_kicker_bear(df, cfg): return []
def detect_rising_three_methods(df, cfg): return []
def detect_falling_three_methods(df, cfg): return []
def detect_mat_hold(df, cfg): return []

# ----------------- Head & Shoulders helpers -----------------
def _pivot_highs_lows(df: pd.DataFrame, swing: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    H = df["High"].values
    L = df["Low"].values
    n = len(df)
    ph: List[Tuple[int, float]] = []
    pl: List[Tuple[int, float]] = []
    left = swing
    right = swing
    for i in range(left, n - right):
        segH = H[i - left : i + right + 1]
        segL = L[i - left : i + right + 1]
        if H[i] == segH.max() and np.isfinite(H[i]):
            ph.append((i, H[i]))
        if L[i] == segL.min() and np.isfinite(L[i]):
            pl.append((i, L[i]))
    return ph, pl

def _first_low_between(pl: List[Tuple[int, float]], i1: int, i2: int) -> Optional[Tuple[int, float]]:
    cands = [p for p in pl if i1 < p[0] < i2]
    if not cands:
        return None
    return min(cands, key=lambda x: x[1])

def _first_high_between(ph: List[Tuple[int, float]], i1: int, i2: int) -> Optional[Tuple[int, float]]:
    cands = [p for p in ph if i1 < p[0] < i2]
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])

def _neckline_y(i_left: int, y_left: float, i_right: int, y_right: float, i: int) -> float:
    if i_right == i_left:
        return y_left
    t = (i - i_left) / float(i_right - i_left)
    return y_left + t * (y_right - y_left)

def _conf_hs(h1: float, h2: float, h3: float, shoulder_tol: float, head_min_diff: float, break_amt_atr: float) -> float:
    sh_dev = abs(h1 - h3) / max(h1, h3)
    sh_score = max(0.0, 1.0 - (sh_dev / max(shoulder_tol, 1e-9)))
    head_ratio = (h2 / max(h1, h3)) - 1.0
    head_score = max(0.0, min(1.0, head_ratio / max(head_min_diff, 1e-9)))
    brk_score = max(0.0, min(1.0, break_amt_atr))
    return float(np.clip(0.2 * sh_score + 0.6 * head_score + 0.2 * brk_score, 0, 1))

def _conf_inverse_hs(l1: float, l2: float, l3: float, shoulder_tol: float, head_min_diff: float, break_amt_atr: float) -> float:
    sh_dev = abs(l1 - l3) / max(l1, l3)
    sh_score = max(0.0, 1.0 - (sh_dev / max(shoulder_tol, 1e-9)))
    head_ratio = (min(l1, l3) / max(l2, 1e-9)) - 1.0  # positive when l2 is lower
    head_score = max(0.0, min(1.0, head_ratio / max(head_min_diff, 1e-9)))
    brk_score = max(0.0, min(1.0, break_amt_atr))
    return float(np.clip(0.2 * sh_score + 0.6 * head_score + 0.2 * brk_score, 0, 1))

# ----------------- Head & Shoulders Detectors -----------------
def detect_head_shoulders(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    swing = int(cfg.get("hs_swing", 3))
    min_sep = int(cfg.get("hs_min_separation", 6))
    shoulder_tol = float(cfg.get("hs_shoulder_tol", 0.03))
    head_min_diff = float(cfg.get("hs_head_min_diff", 0.03))
    look = int(cfg.get("hs_confirm_lookahead", 25))
    buf_mult = float(cfg.get("hs_neckline_buffer_atr_mult", 0.05))

    ph, pl = _pivot_highs_lows(df, swing)
    atr = _atr(df, cfg["atr_lookback"]).values
    close = df["Close"].values

    hits: List[PatternHit] = []
    for a in range(len(ph) - 2):
        i1, h1 = ph[a]
        i2, h2 = ph[a + 1]
        i3, h3 = ph[a + 2]
        if not (i1 < i2 < i3): continue
        if (i2 - i1) < min_sep or (i3 - i2) < min_sep: continue

        low1 = _first_low_between(pl, i1, i2)
        low2 = _first_low_between(pl, i2, i3)
        if not low1 or not low2: continue
        j1, nl1 = low1
        j2, nl2 = low2

        if h2 < max(h1, h3) * (1.0 + head_min_diff): continue
        if abs(h1 - h3) / max(h1, h3) > shoulder_tol: continue

        break_index: Optional[int] = None
        for j in range(i3 + 1, min(i3 + 1 + look, len(df))):
            nl = _neckline_y(j1, nl1, j2, nl2, j)
            margin = atr[j] * buf_mult
            if close[j] < (nl - margin):
                break_index = j
                break
        if break_index is None: continue

        nl = _neckline_y(j1, nl1, j2, nl2, break_index)
        break_amt_atr = float((nl - close[break_index]) / max(atr[break_index], 1e-9))
        conf = _conf_hs(h1, h2, h3, shoulder_tol, head_min_diff, break_amt_atr)

        hits.append(PatternHit(
            "Head & Shoulders",
            int(break_index),
            [int(i1), int(j1), int(i2), int(j2), int(i3), int(break_index)],
            "bear",
            float(np.clip(conf, 0, 1)),
            "Three-peak pattern with higher middle head; neckline break down."
        ))
    return hits

def detect_inverse_head_shoulders(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    swing = int(cfg.get("hs_swing", 3))
    min_sep = int(cfg.get("hs_min_separation", 6))
    shoulder_tol = float(cfg.get("hs_shoulder_tol", 0.03))
    head_min_diff = float(cfg.get("hs_head_min_diff", 0.03))
    look = int(cfg.get("hs_confirm_lookahead", 25))
    buf_mult = float(cfg.get("hs_neckline_buffer_atr_mult", 0.05))

    ph, pl = _pivot_highs_lows(df, swing)
    atr = _atr(df, cfg["atr_lookback"]).values
    close = df["Close"].values

    hits: List[PatternHit] = []
    for a in range(len(pl) - 2):
        i1, l1 = pl[a]
        i2, l2 = pl[a + 1]
        i3, l3 = pl[a + 2]
        if not (i1 < i2 < i3): continue
        if (i2 - i1) < min_sep or (i3 - i2) < min_sep: continue

        hi1 = _first_high_between(ph, i1, i2)
        hi2 = _first_high_between(ph, i2, i3)
        if not hi1 or not hi2: continue
        j1, nh1 = hi1
        j2, nh2 = hi2

        if l2 > min(l1, l3) * (1.0 - head_min_diff): continue
        if abs(l1 - l3) / max(l1, l3) > shoulder_tol: continue

        break_index: Optional[int] = None
        for j in range(i3 + 1, min(i3 + 1 + look, len(df))):
            nl = _neckline_y(j1, nh1, j2, nh2, j)
            margin = atr[j] * buf_mult
            if close[j] > (nl + margin):
                break_index = j
                break
        if break_index is None: continue

        nl = _neckline_y(j1, nh1, j2, nh2, break_index)
        break_amt_atr = float((close[break_index] - nl) / max(atr[break_index], 1e-9))
        conf = _conf_inverse_hs(l1, l2, l3, shoulder_tol, head_min_diff, break_amt_atr)

        hits.append(PatternHit(
            "Inverse Head & Shoulders",
            int(break_index),
            [int(i1), int(j1), int(i2), int(j2), int(i3), int(break_index)],
            "bull",
            float(np.clip(conf, 0, 1)),
            "Three-trough pattern with lower middle head; neckline break up."
        ))
    return hits

# ----------------- Registry -----------------
RULES: Dict[str, Tuple[str, callable]] = {
    # name -> (direction_hint, func)
    "Hammer": ("bull", detect_hammer),
    "Inverted Hammer": ("bull", detect_inverted_hammer),
    "Bullish Engulfing": ("bull", detect_engulfing),
    "Bearish Engulfing": ("bear", detect_engulfing),
    "Doji": ("neutral", detect_doji),
    "Morning Star": ("bull", detect_morning_evening_star),
    "Evening Star": ("bear", detect_morning_evening_star),
    "Bullish Harami": ("bull", detect_harami),
    "Bearish Harami": ("bear", detect_harami),
    "Tweezer Top": ("bear", detect_tweezer),
    "Tweezer Bottom": ("bull", detect_tweezer),

    # Head & Shoulders (new)
    "Head & Shoulders": ("bear", detect_head_shoulders),
    "Inverse Head & Shoulders": ("bull", detect_inverse_head_shoulders),

    # stubs below (return [])
    "Piercing Line": ("bull", detect_piercing_line),
    "Dark Cloud Cover": ("bear", detect_dark_cloud),
    "Three White Soldiers": ("bull", detect_three_soldiers),
    "Three Black Crows": ("bear", detect_three_crows),
    "Three Inside Up": ("bull", detect_three_inside_up),
    "Three Inside Down": ("bear", detect_three_inside_down),
    "Three Outside Up": ("bull", detect_three_outside_up),
    "Three Outside Down": ("bear", detect_three_outside_down),
    "Marubozu": ("neutral", detect_marubozu),
    "Rising Window": ("bull", detect_windows),
    "Falling Window": ("bear", detect_windows),
    "Tasuki Up": ("bull", detect_tasuki_up),
    "Tasuki Down": ("bear", detect_tasuki_down),
    "Kicker Bull": ("bull", detect_kicker_bull),
    "Kicker Bear": ("bear", detect_kicker_bear),
    "Rising Three Methods": ("bull", detect_rising_three_methods),
    "Falling Three Methods": ("bear", detect_falling_three_methods),
    "Mat Hold": ("bull", detect_mat_hold),
}

# ----------------- Orchestrator -----------------
def detect_all(df: pd.DataFrame, enabled: Iterable[str], cfg: Dict[str, Any]) -> List[PatternHit]:
    enabled = list(enabled)
    out: List[PatternHit] = []
    for name, (_, fn) in RULES.items():
        if name not in enabled:
            continue
        hits = fn(df, cfg) or []
        for h in hits:
            if h.name == name:
                out.append(h)
    return out

# ----------------- Markers conversion (MATCHES YOUR Chart.tsx) -----------------
def hits_to_markers(hits: List[PatternHit], df: pd.DataFrame):
    markers = []
    for h in hits:
        i = h.index
        ts = int(df.index[i].timestamp())
        # pick a reference price for the marker
        if h.direction == "bear":
            price = float(df["High"].iat[i])
            side = "above"
        elif h.direction == "bull":
            price = float(df["Low"].iat[i])
            side = "below"
        else:
            price = float(df["Close"].iat[i])
            side = "inBar"
        color = "#ef5350" if h.direction == "bear" else ("#26a69a" if h.direction == "bull" else "#60a5fa")
        label = f"{h.name} ({h.confidence:.2f})"
        markers.append({
            "time": ts,
            "price": price,
            "side": side,
            "color": color,
            "label": label,
        })
    return markers
