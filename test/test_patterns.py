# tests/test_patterns.py
import pandas as pd
from patterns.engine import detect_all, DEFAULT_CONFIG

def _df(rows):
    # rows: list of (t,o,h,l,c)
    return pd.DataFrame([
        {"time": 1+i, "open":o, "high":h, "low":l, "close":c, "volume":1000}
        for i,(t,o,h,l,c) in enumerate(rows)
    ])

def test_bullish_engulfing():
    rows = [
        (0, 10, 10.5, 9.5, 9.8),   # down bar
        (0, 9.7, 10.8, 9.6, 10.6), # up, engulfs body
    ]
    df = _df(rows*10)  # repeat to build medians
    cfg = {**DEFAULT_CONFIG, "enabled": {"Bullish Engulfing": True}}
    hits = detect_all(df, cfg)
    assert any(h.name=="Bullish Engulfing" for h in hits)

def test_hammer():
    # simple downtrend then hammer
    rows=[]
    price=100
    for _ in range(5):
        rows.append((0, price, price+1, price-1, price-0.5)); price-=1
    # hammer: long lower wick, small body near top
    rows.append((0, 95, 96, 92.5, 95.8))
    df = _df(rows)
    cfg = {**DEFAULT_CONFIG, "enabled": {"Hammer": True}}
    hits = detect_all(df, cfg)
    assert any(h.name=="Hammer" for h in hits)

def test_no_false_on_flat():
    rows = [(0, 10,10.1,9.9,10.0) for _ in range(30)]
    df = _df(rows)
    cfg = {**DEFAULT_CONFIG, "enabled": {k:True for k in DEFAULT_CONFIG["enabled"].keys()}}
    hits = detect_all(df, cfg)
    # should be very few or none
    assert len(hits) <= 2
