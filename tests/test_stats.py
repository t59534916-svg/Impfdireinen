"""Tests for vpts.stats — anti-overfitting statistics.

These assert *known analytic properties* (monotonicity, self-consistency, the
overfit-vs-signal contrast) rather than just "it runs", so a wrong formula fails.

    python tests/test_stats.py
    pytest tests/test_stats.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.stats import (  # noqa: E402
    adjusted_pvalues,
    annualize_sharpe,
    annualized_sharpe_lo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    haircut_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sample_skew_kurt,
    sharpe_ratio,
)


# --------------------------------------------------------------------------- #
# Sharpe basics
# --------------------------------------------------------------------------- #
def test_sharpe_ratio_basic() -> None:
    r = np.array([0.01, 0.02, -0.005, 0.015, 0.0])
    assert math.isclose(sharpe_ratio(r), (r.mean()) / r.std(ddof=1), rel_tol=1e-12)
    assert math.isnan(sharpe_ratio([0.01]))            # too few obs
    assert math.isnan(sharpe_ratio([0.01, 0.01, 0.01]))  # zero variance


def test_skew_kurt_normal_reference() -> None:
    rng = np.random.default_rng(0)
    skew, kurt = sample_skew_kurt(rng.normal(0, 1, 50_000))
    assert abs(skew) < 0.1 and abs(kurt - 3.0) < 0.2     # normal → (0, 3)


# --------------------------------------------------------------------------- #
# Probabilistic Sharpe Ratio
# --------------------------------------------------------------------------- #
def test_psr_at_benchmark_is_half() -> None:
    # When observed SR == benchmark, the statistic is 0 ⇒ PSR = 0.5 exactly.
    res = probabilistic_sharpe_ratio(sr=0.1, benchmark_sr=0.1, n=100, skew=0.0, kurt=3.0)
    assert math.isclose(res.psr, 0.5, abs_tol=1e-9)


def test_psr_monotonic_in_sr_and_n() -> None:
    lo = probabilistic_sharpe_ratio(sr=0.05, n=250).psr
    hi = probabilistic_sharpe_ratio(sr=0.15, n=250).psr
    assert hi > lo                                        # higher SR ⇒ higher PSR
    short = probabilistic_sharpe_ratio(sr=0.1, n=60).psr
    long = probabilistic_sharpe_ratio(sr=0.1, n=600).psr
    assert long > short                                   # longer record ⇒ more sure


def test_psr_negative_skew_penalised() -> None:
    base = probabilistic_sharpe_ratio(sr=0.12, n=250, skew=0.0, kurt=3.0).psr
    neg = probabilistic_sharpe_ratio(sr=0.12, n=250, skew=-1.5, kurt=6.0).psr
    assert neg < base       # negative skew + fat tails lower confidence for SR>0


# --------------------------------------------------------------------------- #
# Expected max Sharpe + Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
def test_expected_max_sharpe_grows_with_trials() -> None:
    assert expected_max_sharpe(1, 0.04) == 0.0
    e10 = expected_max_sharpe(10, 0.04)
    e1000 = expected_max_sharpe(1000, 0.04)
    assert 0.0 < e10 < e1000                              # more trials ⇒ higher bar


def test_deflated_sharpe_drops_with_more_trials() -> None:
    common = dict(sr=0.12, n=500, skew=0.0, kurt=3.0, var_trials_sr=0.01)
    d1 = deflated_sharpe_ratio(n_trials=1, **common).dsr
    d50 = deflated_sharpe_ratio(n_trials=50, **common).dsr
    d500 = deflated_sharpe_ratio(n_trials=500, **common).dsr
    assert d1 > d50 > d500                                # deflation increases
    # n_trials=1 with a 0 benchmark equals the plain PSR.
    psr = probabilistic_sharpe_ratio(sr=0.12, n=500, skew=0.0, kurt=3.0).psr
    assert math.isclose(d1, psr, abs_tol=1e-9)


def test_deflated_sharpe_from_returns() -> None:
    rng = np.random.default_rng(3)
    res = deflated_sharpe_ratio(rng.normal(0.001, 0.01, 750), n_trials=100)
    assert 0.0 <= res.dsr <= 1.0 and res.n == 750
    json.dumps(res.as_dict())
    assert "DSR" in res.summary()


# --------------------------------------------------------------------------- #
# Minimum Track Record Length
# --------------------------------------------------------------------------- #
def test_min_trl_self_consistent_with_psr() -> None:
    sr, skew, kurt, prob = 0.10, -0.3, 4.0, 0.95
    mtrl = min_track_record_length(sr=sr, benchmark_sr=0.0, skew=skew, kurt=kurt, prob=prob)
    n_star = int(math.ceil(mtrl.min_trl))
    psr = probabilistic_sharpe_ratio(sr=sr, benchmark_sr=0.0, n=n_star, skew=skew, kurt=kurt).psr
    assert abs(psr - prob) < 0.02                         # at n*, PSR ≈ target prob


def test_min_trl_infinite_when_no_excess() -> None:
    res = min_track_record_length(sr=0.05, benchmark_sr=0.05)
    assert math.isinf(res.min_trl)
    achieved = min_track_record_length(sr=0.2, benchmark_sr=0.0, prob=0.95, n_obs=10_000)
    assert achieved.achieved is True


# --------------------------------------------------------------------------- #
# Lo (2002) autocorrelation-corrected annualization
# --------------------------------------------------------------------------- #
def test_lo_matches_sqrt_time_when_iid() -> None:
    rng = np.random.default_rng(11)
    r = rng.normal(0.0005, 0.01, 2000)                    # i.i.d.
    naive = annualize_sharpe(sharpe_ratio(r), 12)
    lo = annualized_sharpe_lo(r, 12)
    assert abs(lo - naive) / abs(naive) < 0.25            # ≈ √q for i.i.d.


def test_lo_below_sqrt_time_with_positive_autocorr() -> None:
    rng = np.random.default_rng(5)
    n, phi = 3000, 0.5
    eps = rng.normal(0, 0.01, n)
    r = np.empty(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    r = r + 0.002                                         # positive drift ⇒ SR > 0
    naive = annualize_sharpe(sharpe_ratio(r), 12)
    lo = annualized_sharpe_lo(r, 12)
    assert lo < naive                                     # positive autocorr deflates


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------- #
def test_pbo_high_on_pure_noise() -> None:
    rng = np.random.default_rng(7)
    M = rng.normal(0, 1, size=(240, 20))                  # no config is truly better
    res = probability_of_backtest_overfitting(M, n_splits=10, metric="sharpe")
    assert 0.30 <= res.pbo <= 0.70                        # selecting noise ⇒ ~coin flip
    assert res.n_combinations == math.comb(10, 5)


def test_pbo_low_with_a_genuinely_dominant_config() -> None:
    rng = np.random.default_rng(9)
    T, N = 240, 12
    M = rng.normal(0, 1, size=(T, N))
    M[:, 0] += 0.8                                         # config 0 truly dominates
    res = probability_of_backtest_overfitting(M, n_splits=10)
    assert res.pbo < 0.10                                 # IS-best generalizes OOS
    json.dumps(res.as_dict())


def test_pbo_input_validation() -> None:
    for kwargs in (
        dict(perf_matrix=np.zeros((10, 1))),              # <2 configs
        dict(perf_matrix=np.zeros((10, 5)), n_splits=5),  # odd n_splits
        dict(perf_matrix=np.zeros((3, 5)), n_splits=8),   # too few periods
    ):
        try:
            probability_of_backtest_overfitting(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {kwargs}")


# --------------------------------------------------------------------------- #
# Multiple-testing corrections + haircut
# --------------------------------------------------------------------------- #
def test_adjusted_pvalues_properties() -> None:
    p = np.array([0.001, 0.01, 0.03, 0.5])
    bonf = adjusted_pvalues(p, "bonferroni")
    holm = adjusted_pvalues(p, "holm")
    bh = adjusted_pvalues(p, "bh")
    by = adjusted_pvalues(p, "by")
    assert np.allclose(bonf, np.minimum(1.0, p * 4))      # Bonferroni exact
    assert np.all(holm <= bonf + 1e-12)                   # Holm ≤ Bonferroni (FWER)
    assert np.all(bh <= by + 1e-12)                       # BH ≤ BY (BY conservative)
    for adj in (bonf, holm, bh, by):
        assert np.all(adj >= p - 1e-12) and np.all(adj <= 1.0)


def test_haircut_zero_when_single_test() -> None:
    # sr=2.0 is an *annualized* Sharpe over 10y of daily obs (annualization=252).
    res = haircut_sharpe(sr=2.0, n_obs=2520, n_tests=1, method="bonferroni",
                         annualization=252)
    assert res.haircut_ratio < 1e-6                       # M=1 ⇒ no haircut
    assert math.isclose(res.haircut_sr, 2.0, rel_tol=1e-6)


def test_haircut_grows_with_more_tests() -> None:
    kw = dict(sr=2.5, n_obs=2520, annualization=252)
    base = haircut_sharpe(n_tests=1, **kw).haircut_sr
    many = haircut_sharpe(n_tests=200, **kw).haircut_sr
    lots = haircut_sharpe(n_tests=5000, **kw).haircut_sr
    assert base > many > lots                              # more trials ⇒ smaller SR
    assert "haircut" in haircut_sharpe(n_tests=200, **kw).summary().lower()


def test_haircut_holm_needs_other_pvalues() -> None:
    try:
        haircut_sharpe(sr=1.5, n_obs=252, n_tests=10, method="holm")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError without other_pvalues")
    res = haircut_sharpe(sr=1.5, n_obs=252, n_tests=3, method="holm",
                         other_pvalues=[0.2, 0.4])
    assert 0.0 <= res.haircut_ratio <= 1.0


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} stats tests …\n")
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
