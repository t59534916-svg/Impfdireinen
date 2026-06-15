"""Tests for vpts.ml.regime — unsupervised regime detection / pattern discovery.

Load-bearing: walk-forward causality (a bar's regime uses only prior bars) and the
permutation evaluator separating real regime structure from noise.

    python tests/test_regime.py
    pytest tests/test_regime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.ml.regime import (  # noqa: E402
    RegimeClusterer,
    regime_forward_stats,
    walk_forward_regimes,
)


def _blobs(n_per=120, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [6, 6], [0, 8]], float)
    X = np.vstack([c + rng.normal(0, 0.4, (n_per, 2)) for c in centers])
    truth = np.repeat([0, 1, 2], n_per)
    return X, truth


# --------------------------------------------------------------------------- #
def test_clusterer_recovers_separated_blobs() -> None:
    X, truth = _blobs()
    pred = RegimeClusterer(n_regimes=3, seed=1).fit(X).predict(X)
    assert len(np.unique(pred)) == 3
    # Each true blob is dominated by a single predicted label (high purity).
    for t in (0, 1, 2):
        counts = np.bincount(pred[truth == t], minlength=3)
        assert counts.max() / counts.sum() > 0.95


def test_clusterer_predict_on_new_data_and_validation() -> None:
    X, _ = _blobs()
    model = RegimeClusterer(n_regimes=3, seed=0).fit(X)
    new = np.array([[0, 0], [6, 6], [0, 8]], float)
    assert len(model.predict(new)) == 3
    for bad in (dict(n_regimes=1),):
        try:
            RegimeClusterer(**bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_walk_forward_has_no_lookahead() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 3))
    min_train, refit_every = 100, 50
    labels = walk_forward_regimes(X, n_regimes=3, min_train=min_train, refit_every=refit_every)
    assert (labels[:min_train] == -1).all()             # warm-up unlabeled
    assert set(np.unique(labels[min_train:])) <= {0, 1, 2}
    # Changing FUTURE rows must not change already-assigned labels.
    cutoff = min_train + refit_every
    X2 = X.copy()
    X2[cutoff:] = rng.normal(size=(X.shape[0] - cutoff, 3))   # scramble the future
    labels2 = walk_forward_regimes(X2, n_regimes=3, min_train=min_train, refit_every=refit_every)
    assert np.array_equal(labels[:cutoff], labels2[:cutoff])  # past labels unchanged


def test_regime_forward_stats_detects_real_separation() -> None:
    rng = np.random.default_rng(5)
    n = 300
    labels = np.tile([0, 1, 2], n)
    fwd = np.where(labels == 0, rng.normal(0.02, 0.01, labels.size),
                   np.where(labels == 1, rng.normal(-0.02, 0.01, labels.size),
                            rng.normal(0.0, 0.01, labels.size)))
    res = regime_forward_stats(labels, fwd, n_permutations=300, seed=1)
    assert res.significant and res.p_value < 0.05         # regimes truly separate
    assert res.best_regime == 0 and res.worst_regime == 1
    assert res.spread > 0.03
    assert "verdict" in res.summary()


def test_regime_forward_stats_null_on_noise() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 3, 600)
    fwd = rng.normal(0, 0.01, 600)                        # labels independent of returns
    res = regime_forward_stats(labels, fwd, n_permutations=300, seed=2)
    assert not res.significant and res.p_value > 0.05     # no spurious separation


def test_integration_walk_forward_then_stats() -> None:
    rng = np.random.default_rng(9)
    X = np.cumsum(rng.normal(size=(500, 4)), axis=0)      # drifting features
    labels = walk_forward_regimes(X, n_regimes=3, min_train=150, refit_every=50)
    fwd = rng.normal(0, 0.01, 500)
    res = regime_forward_stats(labels, fwd, n_permutations=100)
    assert 0.0 < res.p_value <= 1.0 and res.n_samples > 0


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} regime tests …\n")
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
