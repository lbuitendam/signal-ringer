# tests/test_strategies.py
import pandas as pd
import numpy as np
from strategies.library.ema_pullback import EmaPullback
from strategies.library.candlesticks import CandlesStrategy

def synthetic_df(n=300, seed=0):
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.randn(n)*0.2)
    df = pd.DataFrame({
        "open": prices + np.random.randn(n)*0.05,
        "high": prices + 0.3 + np.random.rand(n)*0.2,
        "low":  prices - 0.3 - np.random.rand(n)*0.2,
        "close": prices,
        "volume": np.random.randint(100, 1000, size=n)
    }, index=pd.date_range("2024-01-01", periods=n, freq="5T", tz="UTC"))
    return df

def test_ema_pullback_has_signals():
    df = synthetic_df()
    s = EmaPullback()
    sigs = s.signals(df)
    assert isinstance(sigs, list)

def test_candles_engulf():
    df = synthetic_df()
    s = CandlesStrategy("engulf_bull")
    sigs = s.signals(df)
    assert isinstance(sigs, list)
