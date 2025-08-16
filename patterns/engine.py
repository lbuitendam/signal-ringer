# patterns/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple, Optional
import numpy as np
import pandas as pd

# ----------------- Config -----------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # generic
    "trend_lookback": 5,
    "atr_lookback": 14,
    "gap_min_atr_mult": 0.15,
    "doji_body_max_k": 0.1,  # body <= k * median(range)
    "hammer_lower_wick_min_mult": 2.0,
    "hammer_upper_wick_max_mult": 0.25,
    "hammer_body_pos_min": 0.66,
    "inv_hammer_upper_wick_min_mult": 2.0,
    "inv_hammer_lower_wick_max_mult": 0.25,
    "inv_hammer_body_pos_max": 0.34,
    "engulfing_min_body_frac": 0.4,  # vs median range
    "harami_body_inside_tol": 0.0,   # strict inside
    "tweezer_tol_atr_mult": 0.15,
    "star_middle_body_frac": 0.2,    # 2nd bar "small"
    "min_bars_between_alerts": 3,
    "min_confidence": 0.6,

    # Head & Shoulders (and inverse) specific
    "hs_swing_lookback": 3,          # pivot window on each side
    "hs_shoulders_tol_atr": 0.6,     # shoulders (L vs R) within X * ATR
    "hs_head_min_prom_atr": 0.8,     # head must exceed shoulders by >= X * ATR
    "hs_confirm_max_bars": 5,        # confirmation (neckline break) window after R
}

# ----------------- Data model -----------------
@dataclass
class PatternHit:
    name: str
    index: int             # the bar where pattern completes (R or confirmation break)
    bars: List[int]        # involved bars (e.g., [L,H,R] or [i-1,i] etc.)
    direction: str         # "bull" | "bear" | "neutral"
    confidence: float      # 0..1
    explanation: str

# ----------------- Helpers -----------------
def _features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    up = h - np.maximum(o, c)
    dn = np.minimum(o, c) - l
    body_pos = (np.maximum(o, c) - l) / rng  # 0 bottom .. 1 top
    color = np.where(c >= o, 1, -1)
    return pd.DataFrame(
        {"o": o, "h": h, "l": l, "c": c, "body": body, "range": rng,
         "up": up, "dn": dn, "body_pos": body_pos, "color": color},
        index=df.index,
    )

def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=1).mean()

def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _median_range(feat: pd.DataFrame, n: int = 100) -> float:
    return float(pd.Series(feat["range"]).tail(n).median())

def _is_downtrend(close: pd.Series, lookback: int) -> pd.Series:
    ma = _sma(close, lookback); slope = ma - ma.shift(3)
    return (close < ma) & (slope < 0)

def _is_uptrend(close: pd.Series, lookback: int) -> pd.Series:
    ma = _sma(close, lookback); slope = ma - ma.shift(3)
    return (close > ma) & (slope > 0)

def _gap_up(df: pd.DataFrame, atr: pd.Series, i: int, cfg: Dict[str, Any]) -> bool:
    if i <= 0: return False
    return df["Low"].iat[i] > df["High"].iat[i-1] + cfg["gap_min_atr_mult"] * atr.iat[i]

def _gap_down(df: pd.DataFrame, atr: pd.Series, i: int, cfg: Dict[str, Any]) -> bool:
    if i <= 0: return False
    return df["High"].iat[i] < df["Low"].iat[i-1] - cfg["gap_min_atr_mult"] * atr.iat[i]

# ---------- Pivot utilities (for Head & Shoulders) ----------
def _pivot_highs(high: np.ndarray, L: int) -> List[int]:
    n = len(high); idx: List[int] = []
    for i in range(L, n - L):
        if high[i] == np.max(high[i - L:i + L + 1]):
            # ensure strict local max (avoid flat runs picking many)
            if high[i] > high[i-1] or high[i] > high[i+1]:
                idx.append(i)
    return idx

def _pivot_lows(low: np.ndarray, L: int) -> List[int]:
    n = len(low); idx: List[int] = []
    for i in range(L, n - L):
        if low[i] == np.min(low[i - L:i + L + 1]):
            if low[i] < low[i-1] or low[i] < low[i+1]:
                idx.append(i)
    return idx

def _liny(x1: int, y1: float, x2: int, y2: float, x: int) -> float:
    if x2 == x1: return y1
    t = (x - x1) / float(x2 - x1)
    return y1 + t * (y2 - y1)

# ----------------- Basic candlestick detectors -----------------
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
    f = _features(df); med_rng = _median_range(f)
    hits: List[PatternHit] = []
    for i in range(1, len(df)):
        o1, c1 = f["o"].iat[i-1], f["c"].iat[i-1]
        o2, c2 = f["o"].iat[i], f["c"].iat[i]
        body1 = abs(c1 - o1); body2 = abs(c2 - o2)
        if body2 < cfg["engulfing_min_body_frac"] * med_rng: continue
        if (c2 > o2 and c1 < o1) and (o2 <= c1 and c2 >= o1):
            ratio = body2 / max(body1, 1e-12)
            conf = min(1.0, ratio / 1.2)
            hits.append(PatternHit("Bullish Engulfing", i, [i-1, i], "bull", conf, "Up body engulfs prior down body"))
        elif (c2 < o2 and c1 > o1) and (o2 >= c1 and c2 <= o1):
            ratio = body2 / max(body1, 1e-12)
            conf = min(1.0, ratio / 1.2)
            hits.append(PatternHit("Bearish Engulfing", i, [i-1, i], "bear", conf, "Down body engulfs prior up body"))
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
        # Morning Star
        if bool(down.iat[i-2]) and f["c"].iat[i-2] < f["o"].iat[i-2]:
            if _small_body(f, i-1, med_rng, cfg["star_middle_body_frac"]):
                mid1 = (f["o"].iat[i-2] + f["c"].iat[i-2]) / 2.0
                if f["c"].iat[i] > f["o"].iat[i] and f["c"].iat[i] >= mid1:
                    conf = min(1.0, (f["c"].iat[i] - mid1) / max(f["range"].iat[i-2], 1e-12) + 0.5)
                    hits.append(PatternHit("Morning Star", i, [i-2, i-1, i], "bull", float(conf), "Three-bar bullish reversal"))
        # Evening Star
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

# ----------------- Head & Shoulders -----------------
def _neckline_points(low: np.ndarray, iL: int, iH: int, iR: int) -> Optional[Tuple[int, float, int, float]]:
    """Pick troughs between L-H and H-R for neckline."""
    if iL >= iH or iH >= iR: return None
    iNL = int(np.argmin(low[iL:iH+1])) + iL
    iNR = int(np.argmin(low[iH:iR+1])) + iH
    return (iNL, low[iNL], iNR, low[iNR])

def _neckline_points_inv(high: np.ndarray, iL: int, iH: int, iR: int) -> Optional[Tuple[int, float, int, float]]:
    """For inverse H&S, pick peaks between L-H and H-R (neckline above)."""
    if iL >= iH or iH >= iR: return None
    iNL = int(np.argmax(high[iL:iH+1])) + iL
    iNR = int(np.argmax(high[iH:iR+1])) + iH
    return (iNL, high[iNL], iNR, high[iNR])

def detect_head_shoulders(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    """Bearish H&S: three pivot highs L<H>R, shoulders ~equal, head prominent; confirm on break below neckline."""
    h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    atr = _atr(df, cfg["atr_lookback"]).values
    L = int(cfg["hs_swing_lookback"])
    piv = _pivot_highs(h, L)
    hits: List[PatternHit] = []

    for a, b, d in zip(piv, piv[1:], piv[2:]):  # consecutive highs as (L,H,R)
        iL, iH, iR = a, b, d
        if not (h[iH] > h[iL] and h[iH] > h[iR]):  # head higher
            continue

        # shoulder symmetry (within tol * ATR)
        shoulder_tol = cfg["hs_shoulders_tol_atr"] * atr[iH]
        if abs(h[iL] - h[iR]) > max(shoulder_tol, 1e-9):
            continue

        # head prominence
        head_prom = h[iH] - max(h[iL], h[iR])
        if head_prom < cfg["hs_head_min_prom_atr"] * atr[iH]:
            continue

        nl = _neckline_points(l, iL, iH, iR)
        if nl is None: continue
        n1x, n1y, n2x, n2y = nl

        # confirm break below neckline within window; otherwise mark at R
        break_idx = iR
        conf_bonus = 0.0
        for j in range(iR + 1, min(len(df), iR + 1 + int(cfg["hs_confirm_max_bars"]))):
            neck_y = _liny(n1x, n1y, n2x, n2y, j)
            if c[j] < neck_y:
                break_idx = j
                conf_bonus = 0.15
                break

        # confidence from symmetry + prominence (+ bonus if broke neckline)
        sym = 1.0 - min(1.0, abs(h[iL] - h[iR]) / max(shoulder_tol, 1e-9))
        prom = min(1.0, head_prom / max(cfg["hs_head_min_prom_atr"] * atr[iH], 1e-9))
        confidence = max(0.0, min(1.0, 0.5 * sym + 0.5 * prom + conf_bonus))

        hits.append(PatternHit(
            name="Head and Shoulders",
            index=break_idx,
            bars=[iL, iH, iR],
            direction="bear",
            confidence=float(confidence),
            explanation="Left/Head/Right highs with similar shoulders; neckline break confirms.",
        ))
    return hits

def detect_inverse_head_shoulders(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[PatternHit]:
    """Bullish inverse H&S using pivot lows and break above neckline."""
    h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    atr = _atr(df, cfg["atr_lookback"]).values
    L = int(cfg["hs_swing_lookback"])
    piv = _pivot_lows(l, L)
    hits: List[PatternHit] = []

    for a, b, d in zip(piv, piv[1:], piv[2:]):  # (L,H,R) but for lows, H is the deepest head
        iL, iH, iR = a, b, d
        if not (l[iH] < l[iL] and l[iH] < l[iR]):  # head lower (deeper)
            continue

        shoulder_tol = cfg["hs_shoulders_tol_atr"] * atr[iH]
        if abs(l[iL] - l[iR]) > max(shoulder_tol, 1e-9):
            continue

        head_prom = min(l[iL], l[iR]) - l[iH]  # depth vs shoulders
        if head_prom < cfg["hs_head_min_prom_atr"] * atr[iH]:
            continue

        nl = _neckline_points_inv(h, iL, iH, iR)
        if nl is None: continue
        n1x, n1y, n2x, n2y = nl

        break_idx = iR
        conf_bonus = 0.0
        for j in range(iR + 1, min(len(df), iR + 1 + int(cfg["hs_confirm_max_bars"]))):
            neck_y = _liny(n1x, n1y, n2x, n2y, j)
            if c[j] > neck_y:
                break_idx = j
                conf_bonus = 0.15
                break

        sym = 1.0 - min(1.0, abs(l[iL] - l[iR]) / max(shoulder_tol, 1e-9))
        prom = min(1.0, head_prom / max(cfg["hs_head_min_prom_atr"] * atr[iH], 1e-9))
        confidence = max(0.0, min(1.0, 0.5 * sym + 0.5 * prom + conf_bonus))

        hits.append(PatternHit(
            name="Inverse Head and Shoulders",
            index=break_idx,
            bars=[iL, iH, iR],
            direction="bull",
            confidence=float(confidence),
            explanation="Left/Head/Right lows with similar shoulders; neckline break confirms.",
        ))
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

# ----------------- Registry -----------------
RULES: Dict[str, Tuple[str, callable]] = {
    # simple
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

    # new
    "Head and Shoulders": ("bear", detect_head_shoulders),
    "Inverse Head and Shoulders": ("bull", detect_inverse_head_shoulders),

    # stubs
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

# ----------------- Markers conversion (to your frontend shape) -----------------
def hits_to_markers(hits: List[PatternHit], df: pd.DataFrame):
    markers = []
    for h in hits:
        i = h.index
        t = int(df.index[i].timestamp())
        if h.direction == "bear":
            side = "above"; price = float(df["High"].iat[i]); color = "#ef5350"
        elif h.direction == "bull":
            side = "below"; price = float(df["Low"].iat[i]); color = "#26a69a"
        else:
            side = "above"; price = float(df["Close"].iat[i]); color = "#60a5fa"
        markers.append({
            "time": t,
            "price": price,
            "side": side,                 # "above" | "below"
            "color": color,
            "label": f"{h.name} ({h.confidence:.2f})",
        })
    return markers
