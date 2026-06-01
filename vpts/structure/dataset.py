"""Walk bars and assemble the structural feature matrix → a ``FactorDataset``.

Each decision bar gets the per-window structural features from
:mod:`vpts.structure.analytics` plus the two rolling/time-series signals that
need history — the **value-area compression z-score** and the **POC-migration
slope** — computed from a trailing window of *previously sampled* bars, so there
is no look-ahead. The output is a :class:`~vpts.ml.models.FactorDataset`, which
plugs straight into the validated ``cpcv_factor_eval`` / ``permutation_test_factor``
harness.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from vpts.profile.calculator import VolumeProfileCalculator
from vpts.regime.indicators import atr, ensure_ohlcv
from vpts.structure.analytics import (
    SHAPE_B,
    SHAPE_P,
    SHAPE_b,
    classify_shape,
    decayed_poc,
    detect_ledges,
    poor_high,
    synthetic_delta_stats,
    value_area_compression_ratio,
    weighted_moments,
)
from vpts.structure.models import STRUCTURAL_FEATURES, StructuralFeatures
from vpts.ml.models import FactorDataset

_EPS = 1e-12


def build_structural_dataset(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    horizon: int = 20,
    stride: int = 3,
    vacr_window: int = 20,
    poc_window: int = 5,
    halflife: float = 21.0,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    profile_calculator: Optional[VolumeProfileCalculator] = None,
) -> FactorDataset:
    """Build a (structural features → forward return) dataset with no look-ahead.

    Features at bar ``t`` use only the trailing ``lookback`` window and the
    rolling history of previously sampled bars; the label is the strictly-future
    ``horizon``-bar return. The single-feature ``baseline`` is the synthetic delta
    at the POC. Bars before the rolling warm-up (``max(vacr_window, poc_window)``
    sampled bars) are skipped.
    """
    ensure_ohlcv(df, min_bars=lookback + horizon + 2)
    pc = profile_calculator or VolumeProfileCalculator(bin_mode="auto")
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    volume = df["Volume"].to_numpy(float)
    atr_a = atr(df["High"], df["Low"], df["Close"], 14).to_numpy(float)
    n = len(df)

    hist_vacr: list[float] = []
    hist_poc: list[float] = []
    rows: list[StructuralFeatures] = []
    feats: list[np.ndarray] = []
    ys: list[float] = []
    base: list[float] = []
    ts: list = []

    for t in range(lookback - 1, n - horizon, max(1, stride)):
        window = df.iloc[t - lookback + 1 : t + 1]
        try:
            profile = pc.calculate(window, symbol, interval)
        except (ValueError, ZeroDivisionError):
            continue
        sl = slice(t - lookback + 1, t + 1)
        wh, wl, wc, wv = high[sl], low[sl], close[sl], volume[sl]

        net, poc_d = synthetic_delta_stats(wh, wl, wc, wv, profile)
        _, _, skew, kurt = weighted_moments(profile.bin_centers, profile.volume_distribution)
        shape = classify_shape(profile, skew)
        loc_rng = profile.price_high - profile.price_low
        poc_loc = float(np.clip((profile.poc - profile.price_low) / loc_rng, 0, 1)) \
            if loc_rng > _EPS else 0.5
        n_ledges = float(len(detect_ledges(profile)))
        ph = 1.0 if poor_high(profile) else 0.0
        atr_t = atr_a[t] if (np.isfinite(atr_a[t]) and atr_a[t] > 0) else max(profile.bin_size, _EPS)
        cbm = float((decayed_poc(wh, wl, wc, wv, profile, halflife) - profile.poc) / atr_t)
        vacr_raw = value_area_compression_ratio(profile, close[t])

        # --- rolling features from PRIOR sampled history (no look-ahead) ---
        warm = len(hist_vacr) >= vacr_window and len(hist_poc) >= poc_window - 1
        vacr_z = 0.0
        poc_slope = 0.0
        if len(hist_vacr) >= vacr_window:
            b = np.asarray(hist_vacr[-vacr_window:], float)
            sd = float(b.std())
            vacr_z = float((vacr_raw - b.mean()) / sd) if sd > _EPS else 0.0
        if len(hist_poc) >= poc_window - 1:
            pocs = np.asarray(hist_poc[-(poc_window - 1):] + [profile.poc], float)
            xs = np.arange(pocs.size, dtype=float)
            slope = float(linregress(xs, pocs).slope) if pocs.size >= 2 else 0.0
            poc_slope = slope / close[t] if close[t] > _EPS else 0.0

        hist_vacr.append(float(vacr_raw))
        hist_poc.append(float(profile.poc))
        if not warm:
            continue

        sf = StructuralFeatures(
            delta_net=net, delta_poc=poc_d, skew=float(skew), kurtosis=float(kurt),
            poc_loc=poc_loc, vacr_z=vacr_z, poc_slope=poc_slope,
            cost_basis_migration=cbm, n_ledges=n_ledges, poor_high=ph,
            is_P=1.0 if shape == SHAPE_P else 0.0,
            is_b=1.0 if shape == SHAPE_b else 0.0,
            is_B=1.0 if shape == SHAPE_B else 0.0,
            shape_class=int(shape), poc=float(profile.poc), vacr=float(vacr_raw),
            timestamp=df.index[t] if isinstance(df.index, pd.DatetimeIndex) else None,
        )
        vec = sf.to_vector()
        if not np.all(np.isfinite(vec)):
            continue
        rows.append(sf)
        feats.append(vec)
        ys.append(float(close[t + horizon] / close[t] - 1.0))
        base.append(poc_d)                          # single-feature baseline = delta@POC
        ts.append(df.index[t])

    is_dt = isinstance(df.index, pd.DatetimeIndex) and bool(ts)
    return FactorDataset(
        X=np.array(feats, dtype=float).reshape(-1, len(STRUCTURAL_FEATURES)),
        y=np.array(ys, dtype=float),
        baseline=np.array(base, dtype=float),
        feature_names=STRUCTURAL_FEATURES,
        horizon=horizon,
        stride=max(1, stride),
        timestamps=pd.DatetimeIndex(ts) if is_dt else None,
        symbol=symbol,
    )
