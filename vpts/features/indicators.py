"""Classic swing-trading indicators as a vpts feature family — to be put on trial.

The newsletter-staple toolkit — **RSI, MACD, a moving-average crossover, momentum, and a
Fibonacci-retracement position** — built as a no-look-ahead :class:`~vpts.ml.models.FactorDataset`
so it clears (or fails) exactly the same bars as everything else in the study: purged CPCV
out-of-sample IC, a block-permutation null, the Deflated Sharpe / PBO, and the
survivorship-injection sweep. The repo's standing finding (experiments 2–6) is that this
*class* of input does not produce a survivorship-robust edge; this lets anyone re-run it.

All features at bar *t* use only data ≤ *t* (trailing EWMs / rolling windows); the label is the
strictly-future ``horizon``-bar forward return. Enforced by a no-look-ahead unit test.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from vpts.ml.models import FactorDataset

_FEATURES = ("rsi", "macd", "ma_cross", "mom_12", "fib_pos")


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (trailing EWM of gains/losses)."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd - macd.ewm(span=signal, adjust=False).mean()


def build_indicator_dataset(
    frame: pd.DataFrame,
    *,
    lookback: int = 120,
    horizon: int = 20,
    stride: int = 8,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
) -> FactorDataset:
    """Build a ``(classic indicators → forward return)`` FactorDataset, no look-ahead.

    Features (all standardized-ish and trailing): RSI(14) centred, MACD(12,26,9) histogram
    over price, (SMA20−SMA50)/price crossover, 12-bar momentum, and the close's Fibonacci
    *position* within the trailing ``lookback`` high–low range (−0.5…+0.5). Label: the
    ``horizon``-bar forward return. Sampled every ``stride`` bars after warm-up.
    """
    close, high, low = frame["Close"], frame["High"], frame["Low"]
    feats = pd.DataFrame({
        "rsi": (_rsi(close, 14) - 50.0) / 50.0,
        "macd": _macd_hist(close) / close,
        "ma_cross": (close.rolling(20).mean() - close.rolling(50).mean()) / close,
        "mom_12": close.pct_change(12),
        "fib_pos": (close - low.rolling(lookback).min())
                   / (high.rolling(lookback).max() - low.rolling(lookback).min()) - 0.5,
    }, index=frame.index)
    fwd = close.shift(-horizon) / close - 1.0                 # strictly FUTURE label

    pos = np.arange(len(frame))
    keep = feats.notna().all(axis=1).to_numpy() & fwd.notna().to_numpy() & (pos >= lookback)
    idx = np.where(keep)[0][::stride]
    if idx.size < 30:
        raise ValueError(f"{symbol or '?'}: too few indicator samples ({idx.size}).")

    X = feats.to_numpy()[idx].astype(float)
    y = fwd.to_numpy()[idx].astype(float)
    baseline = feats["mom_12"].to_numpy()[idx].astype(float)   # momentum as the comparison factor
    return FactorDataset(X=X, y=y, baseline=baseline, feature_names=_FEATURES,
                         horizon=horizon, stride=stride,
                         timestamps=frame.index[idx], symbol=symbol)
