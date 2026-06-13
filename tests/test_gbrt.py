"""Tests for the dependency-free gradient-boosted trees + its CPCV evaluation."""
import numpy as np
import pytest

from vpts.ml.gbrt import (
    GBRTCrossSectionalResult,
    GradientBoostedTrees,
    gbrt_cross_sectional_eval,
)
from vpts.ml.models import CrossSectionalPanel


def test_gbrt_validates_params():
    for bad in (dict(n_estimators=0), dict(learning_rate=0.0), dict(learning_rate=2.0),
                dict(max_depth=0), dict(min_samples_leaf=0), dict(subsample=0.0),
                dict(subsample=1.5)):
        with pytest.raises(ValueError):
            GradientBoostedTrees(**bad)


def test_gbrt_recovers_nonlinear_function():
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.uniform(-1.0, 1.0, size=(n, 4))
    # additive + a nonlinearity; features 2,3 are pure noise
    y = 1.5 * X[:, 0] + X[:, 1] ** 2 + 0.1 * rng.standard_normal(n)
    tr, te = slice(0, 1000), slice(1000, n)
    model = GradientBoostedTrees(n_estimators=80, learning_rate=0.1,
                                 max_depth=3, min_samples_leaf=20).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    # beats the mean baseline and tracks the target
    mse_model = float(np.mean((pred - y[te]) ** 2))
    mse_mean = float(np.mean((y[tr].mean() - y[te]) ** 2))
    assert mse_model < 0.5 * mse_mean
    assert np.corrcoef(pred, y[te])[0, 1] > 0.85


def test_gbrt_feature_importances():
    rng = np.random.default_rng(1)
    n = 1500
    X = rng.uniform(-1.0, 1.0, size=(n, 4))
    y = 1.5 * X[:, 0] + X[:, 1] ** 2 + 0.1 * rng.standard_normal(n)
    model = GradientBoostedTrees(n_estimators=80, min_samples_leaf=20).fit(X, y)
    imp = model.feature_importances_
    assert imp.shape == (4,)
    assert imp.sum() == pytest.approx(1.0, abs=1e-6)
    # the two informative features dominate the two noise features
    assert imp[0] + imp[1] > 0.7


def test_gbrt_null_no_spurious_fit():
    rng = np.random.default_rng(2)
    n = 1200
    X = rng.standard_normal((n, 4))
    y = rng.standard_normal(n)                 # independent of X
    tr, te = slice(0, 800), slice(800, n)
    model = GradientBoostedTrees(n_estimators=60, learning_rate=0.1,
                                 max_depth=3, min_samples_leaf=30).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    # no real out-of-sample correlation with a noise target
    assert abs(np.corrcoef(pred, y[te])[0, 1]) < 0.2


def test_gbrt_constant_target():
    X = np.random.default_rng(3).standard_normal((50, 3))
    y = np.full(50, 2.5)
    model = GradientBoostedTrees(n_estimators=10).fit(X, y)
    assert np.allclose(model.predict(X), 2.5)


# --------------------------------------------------------------------------- #
def _synthetic_panel(n_dates=80, n_names=30, predictive=True, seed=0):
    """A date×name panel: feature 0 predicts within-date returns iff *predictive*."""
    rng = np.random.default_rng(seed)
    feats = ("f0", "f1", "f2", "f3")
    X_rows, y_rows, d_rows = [], [], []
    for d in range(n_dates):
        raw = rng.standard_normal((n_names, len(feats)))
        ranks = np.empty_like(raw)
        for j in range(len(feats)):                       # centred cross-sectional rank
            order = np.argsort(raw[:, j])
            r = np.empty(n_names)
            r[order] = np.arange(n_names)
            ranks[:, j] = r / (n_names - 1) - 0.5
        beta = 0.6 if predictive else 0.0
        y = beta * ranks[:, 0] + 0.4 * rng.standard_normal(n_names)
        for i in range(n_names):
            X_rows.append(ranks[i].tolist())
            y_rows.append(float(y[i]))
            d_rows.append(d)
    return CrossSectionalPanel(
        X=np.array(X_rows, float), y=np.array(y_rows, float),
        date_id=np.array(d_rows, int), feature_names=feats,
        horizon=5, rebalance=5, n_dates=n_dates,
        symbols=tuple(f"N{i}" for i in range(n_names)),
    )


def test_gbrt_cross_sectional_detects_signal():
    res = gbrt_cross_sectional_eval(_synthetic_panel(predictive=True),
                                    n_estimators=50, learning_rate=0.1, min_samples_leaf=20)
    assert isinstance(res, GBRTCrossSectionalResult)
    assert res.oos_ic_mean > 0.10                  # genuine cross-sectional signal
    assert res.feature_importances[0] == max(res.feature_importances)  # f0 most important
    assert 0.0 <= res.oos_dir_accuracy <= 1.0
    assert res.majority_baseline == pytest.approx(max(res.up_rate, 1 - res.up_rate))
    # round-trips through as_dict/summary without error
    assert "oos_ic_mean" in res.as_dict()
    assert "GBRT cross-sectional" in res.summary()


def test_gbrt_cross_sectional_null_clears():
    res = gbrt_cross_sectional_eval(_synthetic_panel(predictive=False, seed=7),
                                    n_estimators=50, learning_rate=0.1, min_samples_leaf=20)
    # No signal ⇒ IC ≈ 0. A single CPCV run on a null panel still wobbles by a few
    # IC points from finite-sample noise + mild tree overfit (exactly why the
    # permutation test, not a point estimate, is the real gate) — so allow slack.
    assert abs(res.oos_ic_mean) < 0.05
    assert res.edge_over_baseline < 0.05           # no directional edge over the up-rate
