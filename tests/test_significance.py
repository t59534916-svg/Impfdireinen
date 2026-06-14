"""Tests for the significance gate (deflated Sharpe + up-rate directional skill)."""
import numpy as np
import pytest

from vpts.ml.significance import (
    DirectionalSkill,
    SharpeSignificance,
    deflated_sharpe_ratio,
    directional_skill,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def test_sharpe_ratio_basic():
    r = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
    assert sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1))
    assert np.isnan(sharpe_ratio(np.array([1.0])))


def test_expected_max_sharpe_monotonic_in_trials():
    assert expected_max_sharpe(1, 0.5) == 0.0          # one trial ⇒ no inflation
    vals = [expected_max_sharpe(n, 0.5) for n in (2, 10, 100, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))   # strictly increasing
    assert all(v > 0 for v in vals)


def test_psr_monotonic_in_sharpe():
    lo = probabilistic_sharpe_ratio(0.05, 250)
    hi = probabilistic_sharpe_ratio(0.20, 250)
    assert 0.0 <= lo <= hi <= 1.0
    # a clearly positive per-obs Sharpe over a long sample is significant vs zero
    assert probabilistic_sharpe_ratio(0.20, 1000) > 0.95


def test_deflated_sharpe_signal_detection():
    # A genuine edge: positive-drift returns, single trial ⇒ significant.
    rng = np.random.default_rng(0)
    r = 0.12 + rng.standard_normal(1000)       # per-obs Sharpe ≈ 0.12 over a long sample
    res = deflated_sharpe_ratio(r, n_trials=1)
    assert isinstance(res, SharpeSignificance)
    assert res.psr_vs_zero > 0.95
    assert res.deflated_sr > 0.95
    assert res.significant


def test_deflated_sharpe_kills_multiple_testing():
    # Same series, but declare it was the best of 1000 trials ⇒ deflated away.
    rng = np.random.default_rng(1)
    r = 0.10 + rng.standard_normal(500)
    single = deflated_sharpe_ratio(r, n_trials=1)
    many = deflated_sharpe_ratio(r, n_trials=1000, sr_std=0.05)
    assert many.sr_benchmark > single.sr_benchmark   # higher hurdle
    assert many.deflated_sr < single.deflated_sr      # deflated
    assert not many.significant                        # no longer significant


def test_deflated_sharpe_null_pure_noise():
    # Zero-mean noise should not look significant even at a single trial.
    rng = np.random.default_rng(2)
    r = rng.standard_normal(500)
    res = deflated_sharpe_ratio(r, n_trials=1)
    assert not res.significant
    assert res.deflated_sr < 0.95


def test_directional_skill_perfect_and_null():
    f = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.005])
    perfect = directional_skill(np.sign(f), f)
    assert perfect.directional_accuracy == pytest.approx(1.0)
    assert perfect.beats_baseline

    # Up-biased market, signal uncorrelated ⇒ should not beat the majority baseline.
    rng = np.random.default_rng(3)
    fwd = rng.standard_normal(2000) + 0.5     # ~69% up
    sig = rng.standard_normal(2000)            # noise
    ds = directional_skill(sig, fwd)
    assert isinstance(ds, DirectionalSkill)
    assert ds.up_rate > 0.5
    assert ds.majority_baseline == pytest.approx(max(ds.up_rate, 1 - ds.up_rate))
    assert ds.edge_over_baseline <= 0.02       # no real edge over always-predict-up


def test_directional_skill_validates_input():
    with pytest.raises(ValueError):
        directional_skill(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        directional_skill(np.array([1.0, 2.0]), np.array([1.0]))
