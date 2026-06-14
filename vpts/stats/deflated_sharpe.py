"""Sharpe-ratio significance under short samples, fat tails, and selection bias.

Implements the Bailey & López de Prado family of Sharpe-ratio statistics:

* :func:`probabilistic_sharpe_ratio` (PSR) — Bailey & López de Prado (2012),
  *The Sharpe Ratio Efficient Frontier*, J. of Risk.
* :func:`deflated_sharpe_ratio` (DSR) — Bailey & López de Prado (2014),
  *The Deflated Sharpe Ratio*, J. of Portfolio Management.
* :func:`min_track_record_length` (minTRL) — same papers.
* :func:`annualized_sharpe_lo` — Lo (2002), *The Statistics of Sharpe Ratios*,
  Financial Analysts Journal (autocorrelation-corrected annualization).

Why this exists
---------------
A raw Sharpe ratio is a point estimate that flatters short, skewed, fat-tailed,
*cherry-picked* track records — exactly the failure modes that produce false
"edges". These functions answer the honest questions instead: *given the sample,
the higher moments, and how many strategies I tried, what is the probability this
Sharpe is real?* Per López de Prado, the deflated Sharpe is the single most
important guard against backtest overfitting.

Convention
----------
Unless stated otherwise, ``sr`` and ``benchmark_sr`` are **per-observation**
(non-annualized) Sharpe ratios, matching the frequency of the return series. Use
:func:`annualize_sharpe` / :func:`annualized_sharpe_lo` to scale to a year.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from vpts.stats.models import DeflatedSharpeResult, MinTRLResult, PSRResult

#: Euler–Mascheroni constant, used in the expected-maximum-Sharpe estimator.
EULER_MASCHERONI: float = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Basic estimators
# --------------------------------------------------------------------------- #
def _clean(returns: Sequence[float]) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def sharpe_ratio(returns: Sequence[float], *, benchmark: float = 0.0, ddof: int = 1) -> float:
    """Per-observation Sharpe ratio ``(mean - benchmark) / std``.

    ``benchmark`` is a per-observation hurdle (e.g. the risk-free rate per bar).
    Returns ``nan`` for fewer than two finite observations or zero variance.
    """
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=ddof)
    if sd == 0:
        return float("nan")
    return float((r.mean() - benchmark) / sd)


def sample_skew_kurt(returns: Sequence[float]) -> tuple[float, float]:
    """Return ``(skewness, kurtosis)`` with **non-excess** kurtosis (normal → 3.0)."""
    r = _clean(returns)
    if r.size < 2:
        return 0.0, 3.0
    sd = r.std(ddof=0)
    if sd == 0:
        return 0.0, 3.0
    z = (r - r.mean()) / sd
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


def annualize_sharpe(sr_per_period: float, periods_per_year: float) -> float:
    """Naive √-time annualization (assumes i.i.d. returns)."""
    return sr_per_period * math.sqrt(periods_per_year)


# --------------------------------------------------------------------------- #
# Probabilistic Sharpe Ratio
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    returns: Optional[Sequence[float]] = None,
    *,
    benchmark_sr: float = 0.0,
    sr: Optional[float] = None,
    n: Optional[int] = None,
    skew: Optional[float] = None,
    kurt: Optional[float] = None,
    ddof: int = 1,
) -> PSRResult:
    """Probabilistic Sharpe Ratio: ``P(true SR > benchmark_sr)``.

    Pass a return series via *returns* (moments are estimated), **or** supply
    *sr*, *n*, *skew*, *kurt* directly (handy for testing / re-using cached
    moments). ``benchmark_sr`` and *sr* must be on the same per-observation scale.

    The estimator (Bailey & López de Prado 2012) is

        PSR(SR*) = Φ( (SR − SR*)·√(n−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) )

    with γ₃ skewness, γ₄ non-excess kurtosis, Φ the standard-normal CDF.
    """
    if returns is not None:
        r = _clean(returns)
        n = int(r.size)
        sr = sharpe_ratio(r, ddof=ddof)
        skew, kurt = sample_skew_kurt(r)
    if sr is None or n is None:
        raise ValueError("Provide either `returns` or all of `sr` and `n`.")
    if n < 2:
        raise ValueError("Need at least 2 observations for the PSR.")
    skew = 0.0 if skew is None else float(skew)
    kurt = 3.0 if kurt is None else float(kurt)
    if not math.isfinite(sr):
        return PSRResult(float("nan"), float("nan"), benchmark_sr, n, skew, kurt, float("nan"))

    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    # The variance term is positive for any real return distribution; clamp to
    # avoid a domain error from pathological/contrived moment inputs.
    denom = math.sqrt(max(var_term, 1e-12))
    stat = (sr - benchmark_sr) * math.sqrt(n - 1) / denom
    return PSRResult(float(norm.cdf(stat)), float(sr), float(benchmark_sr), int(n),
                     skew, kurt, float(stat))


# --------------------------------------------------------------------------- #
# Expected maximum Sharpe & the Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
def expected_max_sharpe(n_trials: int, var_trials_sr: float) -> float:
    """Expected maximum of ``n_trials`` i.i.d. Sharpe estimates with variance
    ``var_trials_sr`` (centred at zero) — the selection-adjusted benchmark.

    Uses the extreme-value approximation from López de Prado (2018), *Advances in
    Financial Machine Learning*, Ch. 8:

        E[max SR] ≈ √Var · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

    with γ the Euler–Mascheroni constant. ``n_trials == 1`` ⇒ benchmark 0.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if var_trials_sr < 0:
        raise ValueError("var_trials_sr must be >= 0.")
    if n_trials == 1 or var_trials_sr == 0:
        return 0.0
    sigma = math.sqrt(var_trials_sr)
    g = EULER_MASCHERONI
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma * ((1.0 - g) * a + g * b))


def deflated_sharpe_ratio(
    returns: Optional[Sequence[float]] = None,
    *,
    n_trials: int,
    sr: Optional[float] = None,
    n: Optional[int] = None,
    skew: Optional[float] = None,
    kurt: Optional[float] = None,
    var_trials_sr: Optional[float] = None,
    sr_trials: Optional[Sequence[float]] = None,
    ddof: int = 1,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio — PSR against the expected-maximum-Sharpe benchmark.

    Parameters
    ----------
    returns:
        The *selected* strategy's return series (or supply *sr*/*n*/moments).
    n_trials:
        Number of (effectively independent) strategy configurations tried.
    var_trials_sr / sr_trials:
        Variance of the trials' Sharpe estimates. Supply it directly, or pass the
        vector of trial Sharpes *sr_trials* and it is computed (with the same
        *ddof*). If neither is given, falls back to ``Var = 1/(n−1)`` — the
        null variance of a single Sharpe estimate under i.i.d. normal returns.

    Returns a :class:`DeflatedSharpeResult`; ``dsr ≥ 0.95`` is the usual bar.
    """
    if returns is not None:
        r = _clean(returns)
        n = int(r.size)
        sr = sharpe_ratio(r, ddof=ddof)
        skew, kurt = sample_skew_kurt(r)
    if sr is None or n is None:
        raise ValueError("Provide either `returns` or all of `sr` and `n`.")
    skew = 0.0 if skew is None else float(skew)
    kurt = 3.0 if kurt is None else float(kurt)

    if var_trials_sr is None:
        if sr_trials is not None:
            v = _clean(sr_trials)
            var_trials_sr = float(v.var(ddof=ddof)) if v.size > 1 else 0.0
        else:
            # Null per-observation variance of a Sharpe estimate (Lo 2002): 1/(n-1).
            var_trials_sr = 1.0 / max(n - 1, 1)

    smax = expected_max_sharpe(n_trials, var_trials_sr)
    psr = probabilistic_sharpe_ratio(benchmark_sr=smax, sr=sr, n=n, skew=skew, kurt=kurt)
    return DeflatedSharpeResult(
        dsr=psr.psr, sr=float(sr), expected_max_sr=smax, n_trials=int(n_trials),
        var_trials_sr=float(var_trials_sr), n=int(n), skew=skew, kurt=kurt,
    )


# --------------------------------------------------------------------------- #
# Minimum Track Record Length
# --------------------------------------------------------------------------- #
def min_track_record_length(
    *,
    sr: float,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
    prob: float = 0.95,
    n_obs: Optional[int] = None,
) -> MinTRLResult:
    """Minimum observations for ``PSR(benchmark_sr) = prob``.

    Inverts the PSR for *n*:

        minTRL = 1 + (1 − γ₃·SR + ((γ₄−1)/4)·SR²) · (Φ⁻¹(prob) / (SR − SR*))²

    Returns ``inf`` when ``sr <= benchmark_sr`` (no finite sample makes a
    non-positive excess Sharpe significant).
    """
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1).")
    if sr <= benchmark_sr:
        return MinTRLResult(float("inf"), float(sr), float(benchmark_sr),
                            float(skew), float(kurt), float(prob), n_obs)
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    z = norm.ppf(prob)
    min_trl = 1.0 + max(var_term, 1e-12) * (z / (sr - benchmark_sr)) ** 2
    return MinTRLResult(float(min_trl), float(sr), float(benchmark_sr),
                        float(skew), float(kurt), float(prob), n_obs)


# --------------------------------------------------------------------------- #
# Lo (2002) autocorrelation-corrected annualization
# --------------------------------------------------------------------------- #
def annualized_sharpe_lo(
    returns: Sequence[float],
    periods_per_year: int,
    *,
    max_lag: Optional[int] = None,
    ddof: int = 1,
) -> float:
    """Annualized Sharpe with Lo's serial-correlation correction.

    The naive annualization ``√q · SR`` assumes i.i.d. returns. Positive
    autocorrelation (smoothed/illiquid returns, trend strategies) makes √-time
    **overstate** the annual Sharpe. Lo (2002) replaces √q with

        η(q) = q / √( q + 2·Σ_{k=1}^{q−1} (q−k)·ρ_k )

    where ρ_k is the k-th autocorrelation of the return series. Reduces to √q when
    all ρ_k = 0. ``max_lag`` caps the number of lags actually estimated (defaults
    to ``min(q−1, n−1)``), since high-lag autocorrelations are noisy on short data.
    """
    r = _clean(returns)
    sr = sharpe_ratio(r, ddof=ddof)
    if not math.isfinite(sr):
        return float("nan")
    q = int(periods_per_year)
    if q <= 1:
        return sr
    lag_cap = min(q - 1, r.size - 1)
    if max_lag is not None:
        lag_cap = min(lag_cap, int(max_lag))
    if lag_cap < 1:
        return annualize_sharpe(sr, q)

    x = r - r.mean()
    denom0 = float(np.dot(x, x))
    if denom0 == 0:
        return annualize_sharpe(sr, q)
    acc = 0.0
    for k in range(1, lag_cap + 1):
        rho_k = float(np.dot(x[:-k], x[k:]) / denom0)
        acc += (q - k) * rho_k
    var_scale = q + 2.0 * acc
    if var_scale <= 0:  # pathological negative-autocorrelation edge: fall back to √q
        return annualize_sharpe(sr, q)
    eta = q / math.sqrt(var_scale)
    return float(eta * sr)
