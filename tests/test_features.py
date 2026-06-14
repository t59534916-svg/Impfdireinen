"""Tests for vpts.features — behavioral features + multi-timeframe alignment.

The load-bearing test is **truncation invariance**: a feature at bar t must be
identical whether or not bars after t exist. That is the no-look-ahead guarantee,
proven empirically for every feature at once.

    python tests/test_features.py
    pytest tests/test_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data import synthetic_survivor_ohlcv  # noqa: E402
from vpts.features import (  # noqa: E402
    BEHAVIORAL_FEATURES,
    BehavioralConfig,
    align_to_base,
    behavioral_feature_frame,
    build_behavioral_dataset,
    resample_ohlcv,
)
from vpts.features import behavioral as bh  # noqa: E402
from vpts.features.multitimeframe import multi_timeframe_feature  # noqa: E402


# --------------------------------------------------------------------------- #
# No look-ahead — the critical invariant
# --------------------------------------------------------------------------- #
def test_truncation_invariance_no_lookahead() -> None:
    df = synthetic_survivor_ohlcv(420, seed=1)
    full = behavioral_feature_frame(df)
    for t in (150, 250, 380):
        partial = behavioral_feature_frame(df.iloc[: t + 1])
        a = full.iloc[t].to_numpy(float)
        b = partial.iloc[-1].to_numpy(float)
        assert np.allclose(a, b, atol=1e-9, equal_nan=True), f"look-ahead leak at t={t}"


def test_feature_frame_shape_and_names() -> None:
    df = synthetic_survivor_ohlcv(300, seed=2)
    frame = behavioral_feature_frame(df)
    assert list(frame.columns) == list(BEHAVIORAL_FEATURES)
    assert len(frame) == len(df)
    # Warmed rows are finite.
    assert np.all(np.isfinite(frame.iloc[200].to_numpy(float)))


# --------------------------------------------------------------------------- #
# Behavioral correctness sanity
# --------------------------------------------------------------------------- #
def _frame(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=idx)


def test_clv_pressure_sign() -> None:
    # Closes pinned near the high → positive accumulation pressure.
    up = _frame([{"Open": 10, "High": 11, "Low": 9.5, "Close": 10.95, "Volume": 1e6}] * 30)
    down = _frame([{"Open": 10, "High": 10.5, "Low": 9, "Close": 9.05, "Volume": 1e6}] * 30)
    assert bh.clv_pressure(up, 10).iloc[-1] > 0.7
    assert bh.clv_pressure(down, 10).iloc[-1] < -0.7


def test_relative_volume_spike() -> None:
    rows = [{"Open": 10, "High": 10.5, "Low": 9.5, "Close": 10, "Volume": 1e6}] * 25
    rows.append({"Open": 10, "High": 10.5, "Low": 9.5, "Close": 10, "Volume": 3e6})
    rvol = bh.relative_volume(_frame(rows), 20)
    assert rvol.iloc[-1] > 1.5                          # ~3x average ⇒ RVOL-1 ≈ 2


def test_liquidity_grab_bullish_sweep() -> None:
    # Flat range, then a bar that wicks below the prior low and closes back above it.
    rows = [{"Open": 10, "High": 10.4, "Low": 9.6, "Close": 10, "Volume": 1e6}] * 25
    rows.append({"Open": 9.8, "High": 10.1, "Low": 9.0, "Close": 10.0, "Volume": 2e6})
    grab = bh.liquidity_grab(_frame(rows), 20)
    assert grab.iloc[-1] > 0                            # bullish liquidity grab
    # Mirror: sweep above prior high, close back below ⇒ bearish (negative).
    rows2 = [{"Open": 10, "High": 10.4, "Low": 9.6, "Close": 10, "Volume": 1e6}] * 25
    rows2.append({"Open": 10.2, "High": 11.0, "Low": 9.9, "Close": 10.0, "Volume": 2e6})
    assert bh.liquidity_grab(_frame(rows2), 20).iloc[-1] < 0


# --------------------------------------------------------------------------- #
# Multi-timeframe causal alignment
# --------------------------------------------------------------------------- #
def test_resample_and_causal_alignment() -> None:
    df = synthetic_survivor_ohlcv(260, seed=3)
    weekly = resample_ohlcv(df, "W")
    assert {"Open", "High", "Low", "Close", "Volume"} <= set(weekly.columns)
    assert len(weekly) < len(df)
    feat = multi_timeframe_feature(df, bh.clv_pressure, rule="W", window=4)
    assert len(feat) == len(df) and np.all(np.isfinite(feat.to_numpy(float)))


def test_align_to_base_is_not_lookahead() -> None:
    df = synthetic_survivor_ohlcv(200, seed=4)
    weekly = resample_ohlcv(df, "W")
    # A monotone coarse series equal to its own position; aligned values on the base
    # index must never exceed the count of weeks that have already *closed*.
    coarse = pd.Series(np.arange(len(weekly), dtype=float), index=weekly.index)
    aligned = align_to_base(coarse, df.index)
    for ts, val in aligned.dropna().items():
        closed = int((weekly.index < ts).sum())        # weeks strictly before this bar
        assert val <= closed - 1 + 1e-9                 # only exposes already-closed weeks


# --------------------------------------------------------------------------- #
# Dataset build + harness integration
# --------------------------------------------------------------------------- #
def test_build_behavioral_dataset_shapes_and_labels() -> None:
    df = synthetic_survivor_ohlcv(500, seed=5)
    ds = build_behavioral_dataset(df, horizon=20, stride=3, symbol="SYN")
    assert ds.X.shape[1] == len(BEHAVIORAL_FEATURES) == len(ds.feature_names)
    assert ds.X.shape[0] == ds.y.shape[0] == ds.baseline.shape[0] > 0
    assert np.all(np.isfinite(ds.X)) and np.all(np.isfinite(ds.y))
    assert ds.horizon == 20 and ds.purge_samples >= 1


def test_behavioral_dataset_runs_through_cpcv_harness() -> None:
    from vpts.ml.factor_model import cpcv_factor_eval
    from vpts.validation.cpcv import CombinatorialPurgedCV

    df = synthetic_survivor_ohlcv(700, seed=6)
    ds = build_behavioral_dataset(df, horizon=20, stride=3, symbol="SYN")
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge=ds.purge_samples)
    res = cpcv_factor_eval(ds, cv=cv)
    # We assert it *runs and is finite* — NOT that there is an edge (there isn't a
    # claim to make on synthetic data; the harness is the judge).
    assert np.isfinite(res.oos_ic_mean) and res.n_samples == len(ds)


def test_behavioral_dataset_under_survivorship_injection() -> None:
    from vpts.data import SurvivorshipInjector

    survivors = {f"S{i}": synthetic_survivor_ohlcv(420, seed=i) for i in range(3)}
    res = SurvivorshipInjector(n_delisted=2, seed=900).inject(survivors)
    built = [build_behavioral_dataset(f, horizon=20, stride=5, symbol=s)
             for s, f in res.frames.items()]
    assert len(built) == 5 and all(b.X.shape[0] > 0 for b in built)


def test_config_changes_feature_count() -> None:
    df = synthetic_survivor_ohlcv(300, seed=7)
    cfg = BehavioralConfig(short=3, med=10, long=30, grab_lookback=10)
    frame = behavioral_feature_frame(df, cfg)
    assert "rvol_3" in frame.columns and "accum_slope_30" in frame.columns


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} feature tests …\n")
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
