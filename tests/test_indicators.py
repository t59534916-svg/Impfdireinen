"""Tests for the classic-indicator feature family (vpts.features.build_indicator_dataset).

Load-bearing checks: (1) NO LOOK-AHEAD — perturbing future bars must not change a row's
features; (2) NULL-CLEARING — on a near-random-walk it reports no significant edge through
the honest harness. Offline and deterministic.

    python tests/test_indicators.py
    pytest tests/test_indicators.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import honest_backtest                              # noqa: E402
from vpts.data import synthetic_survivor_ohlcv               # noqa: E402
from vpts.features import build_indicator_dataset            # noqa: E402


def test_build_shape_and_feature_names() -> None:
    df = synthetic_survivor_ohlcv(500, seed=2)
    ds = build_indicator_dataset(df, lookback=120, horizon=20, stride=8, symbol="X")
    assert ds.X.shape[1] == 5 and len(ds) == ds.X.shape[0] == ds.y.shape[0]
    assert ds.feature_names == ("rsi", "macd", "ma_cross", "mom_12", "fib_pos")
    assert ds.horizon == 20 and ds.stride == 8 and np.isfinite(ds.X).all()


def test_no_look_ahead() -> None:
    df = synthetic_survivor_ohlcv(500, seed=1)
    ds = build_indicator_dataset(df, lookback=120, horizon=20, stride=8, symbol="X")
    t_ts = ds.timestamps[len(ds) // 3]                        # a sampled bar well before the end
    t_pos = df.index.get_loc(t_ts)
    rng = np.random.default_rng(0)
    scr = df.copy()
    fut = scr.iloc[t_pos + 1:]
    scr.iloc[t_pos + 1:] = (fut.to_numpy() * (1 + rng.normal(0, 0.1, fut.shape)))  # perturb FUTURE only
    ds2 = build_indicator_dataset(scr, lookback=120, horizon=20, stride=8, symbol="X")
    i1 = list(ds.timestamps).index(t_ts)
    i2 = list(ds2.timestamps).index(t_ts)
    assert np.allclose(ds.X[i1], ds2.X[i2]), "features at t changed when only future bars moved"


def _iid_frame(n: int, seed: int):
    """A pure GBM random walk — a genuine null (features ⊥ forward return)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close,
                         "Volume": rng.uniform(1e6, 2e6, n)},
                        index=pd.date_range("2010-01-04", periods=n, freq="B"))


def test_null_clearing_through_harness() -> None:
    # On a genuine i.i.d. null with ADEQUATE samples/name the indicators must NOT clear the
    # permutation null. (NB: the harness over-rejects in small-sample regimes — few bars per
    # name — so this uses long frames where it is calibrated; see CHANGELOG 2.1.0.)
    frames = {f"S{i}": _iid_frame(1500, seed=i) for i in range(6)}
    ds = [build_indicator_dataset(f, lookback=120, horizon=20, stride=8, symbol=s)
          for s, f in frames.items()]
    rep = honest_backtest(ds, perms=120, n_trials=11, seed=0)
    assert not rep.ic_significant and rep.block_perm_p > 0.05
    assert rep.verdict.startswith("NO EDGE")


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} indicator-family tests …\n")
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
