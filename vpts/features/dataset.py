"""Assemble behavioral features → ``FactorDataset`` for the CPCV harness.

:func:`behavioral_feature_frame` builds the full causal feature matrix (all
features are rolling/shift series, so a row at bar *t* uses only data ``<= t``).
:func:`build_behavioral_dataset` samples it at decision bars and attaches the
strictly-future forward return, producing a :class:`~vpts.ml.models.FactorDataset`
that drops straight into ``cpcv_factor_eval`` / ``permutation_test_factor`` and the
``vpts.data.SurvivorshipInjector`` stress harness.

No edge is assumed. This is the machinery to *test* the behavioral hypotheses
honestly; the verdict is whatever the harness returns.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from vpts.features.models import BehavioralConfig, feature_spec
from vpts.ml.models import FactorDataset
from vpts.regime.indicators import ensure_ohlcv


def behavioral_feature_frame(
    df: pd.DataFrame, config: BehavioralConfig = BehavioralConfig()
) -> pd.DataFrame:
    """Return the causal behavioral feature matrix (one column per feature).

    **Warm-up caveat:** most primitives end in ``.fillna(0.0)``, so rows before a
    feature's window has filled hold a *neutral 0.0*, not NaN (only the RVOL columns
    stay NaN during their first window). Do **not** treat early rows as genuine —
    :func:`build_behavioral_dataset` skips the first ``warmup`` bars
    (``max(long, med, grab_lookback)``) for exactly this reason; direct callers
    should apply the same skip. Column order matches
    :func:`~vpts.features.models.feature_spec`.
    """
    ensure_ohlcv(df, min_bars=2)
    cols = {name: fn(df, **kw) for name, fn, kw in feature_spec(config)}
    return pd.DataFrame(cols, index=df.index)


def build_behavioral_dataset(
    df: pd.DataFrame,
    *,
    config: BehavioralConfig = BehavioralConfig(),
    horizon: int = 20,
    stride: int = 3,
    warmup: Optional[int] = None,
    symbol: Optional[str] = None,
) -> FactorDataset:
    """Behavioral features → forward ``horizon``-bar return (no look-ahead).

    Parameters
    ----------
    horizon:
        Forward return horizon in bars (the label, strictly future).
    stride:
        Sample every *stride* bars to reduce label overlap.
    warmup:
        Bars to skip at the start for rolling warm-up. Defaults to the largest
        configured window, which is where every feature first becomes finite.

    The single-feature ``baseline`` is the medium-horizon accumulation slope (a
    plain order-flow read) for an apples-to-apples comparison against the combo.
    """
    spec = feature_spec(config)
    names = tuple(name for name, _, _ in spec)
    frame = behavioral_feature_frame(df, config)
    close = df["Close"].to_numpy(float)
    n = len(df)
    warm = int(warmup) if warmup is not None else max(config.long, config.med, config.grab_lookback)

    feats: list[np.ndarray] = []
    ys: list[float] = []
    base: list[float] = []
    ts: list = []
    is_dt = isinstance(df.index, pd.DatetimeIndex)
    fvals = frame.to_numpy(float)
    base_col = names.index(f"accum_slope_{config.med}")

    for t in range(warm, n - horizon, max(1, stride)):
        row = fvals[t]
        if not np.all(np.isfinite(row)):
            continue
        if close[t] <= 0:
            continue
        feats.append(row)
        ys.append(float(close[t + horizon] / close[t] - 1.0))
        base.append(float(row[base_col]))
        ts.append(df.index[t] if is_dt else t)

    X = np.array(feats, dtype=float).reshape(-1, len(names))
    return FactorDataset(
        X=X,
        y=np.array(ys, dtype=float),
        baseline=np.array(base, dtype=float),
        feature_names=names,
        horizon=horizon,
        stride=max(1, stride),
        timestamps=pd.DatetimeIndex(ts) if (is_dt and ts) else None,
        symbol=symbol,
    )
