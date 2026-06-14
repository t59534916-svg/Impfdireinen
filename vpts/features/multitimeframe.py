"""Causal multi-timeframe utilities — true calendar resampling done safely.

The core behavioral feature set uses multi-horizon *rolling windows* (see
:mod:`vpts.features.models`), which are causal by construction. For users who want
genuine higher-timeframe bars (weekly/monthly), this module resamples OHLCV and —
crucially — aligns the coarse series back to the base index **without look-ahead**,
by exposing a completed coarse bar only on the *next* base bar after it closes.
"""
from __future__ import annotations

import pandas as pd

#: OHLCV aggregation for resampling.
_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, rule: str = "W") -> pd.DataFrame:
    """Resample an OHLCV frame to a coarser *rule* (e.g. ``"W"``, ``"ME"``).

    Requires a :class:`~pandas.DatetimeIndex`. Bars are right-labelled at the
    period end (the bar's *close* timestamp), which is what
    :func:`align_to_base` relies on for causal alignment.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_ohlcv requires a DatetimeIndex.")
    out = df.resample(rule, label="right", closed="right").agg(_AGG)
    return out.dropna(subset=["Close"])


def align_to_base(coarse: pd.Series, base_index: pd.DatetimeIndex) -> pd.Series:
    """Align a coarse-timeframe series onto *base_index* with **no look-ahead**.

    A coarse bar labelled at its period-end timestamp ``T`` summarises data up to
    and including ``T``; it must therefore only become visible on base bars *after*
    ``T``. We shift the coarse series one coarse-bar forward (so each value is
    available strictly after its bar closed) and forward-fill onto the base index.
    Base bars before the first completed coarse bar are NaN.
    """
    if not isinstance(base_index, pd.DatetimeIndex):
        raise TypeError("base_index must be a DatetimeIndex.")
    shifted = coarse.shift(1)                       # expose only after the bar closes
    return shifted.reindex(base_index, method="ffill")


def multi_timeframe_feature(
    df: pd.DataFrame,
    feature_fn,
    *,
    rule: str = "W",
    **feature_kwargs,
) -> pd.Series:
    """Compute *feature_fn* on a resampled timeframe, aligned causally to ``df.index``.

    Example: ``multi_timeframe_feature(df, clv_pressure, rule="W", window=8)`` is the
    8-week accumulation pressure, visible on daily bars only after each week closes.
    """
    coarse = resample_ohlcv(df, rule)
    coarse_feat = feature_fn(coarse, **feature_kwargs)
    return align_to_base(coarse_feat, df.index).fillna(0.0)
