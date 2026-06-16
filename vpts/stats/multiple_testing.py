"""Multiple-testing corrections and the Harvey–Liu–Zhu Sharpe haircut.

Reference: Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected
Returns*, Review of Financial Studies; Harvey & Liu (2015), *Backtesting*,
J. of Portfolio Management.

The problem
-----------
When a strategy is the survivor of many trials, its t-statistic overstates its
significance: with enough tries, *something* clears t > 2 by chance. The honest
fix is to control for the number of tests. This module provides the standard
adjusted-p-value procedures and maps the adjustment back onto the Sharpe ratio
("haircut"): how much of the reported Sharpe survives once you account for how
many strategies were tried.

Two interfaces
--------------
* :func:`adjusted_pvalues` — Bonferroni / Holm (FWER) and BH / BY (FDR)
  corrections on a **vector** of p-values. Textbook-exact; use when you have the
  p-values of *all* the strategies you tested.
* :func:`haircut_sharpe` — the single-strategy convenience: deflate one Sharpe
  given the number of trials ``n_tests`` (Bonferroni is exact from ``n_tests``
  alone; Holm/BHY additionally need the other p-values).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from vpts.stats.models import HaircutResult


# --------------------------------------------------------------------------- #
# Adjusted p-values on a vector of tests
# --------------------------------------------------------------------------- #
def _harmonic(m: int) -> float:
    return float(np.sum(1.0 / np.arange(1, m + 1)))


def adjusted_pvalues(pvalues: Sequence[float], method: str = "holm") -> np.ndarray:
    """Return multiplicity-adjusted p-values, aligned to the input order.

    Methods
    -------
    ``"bonferroni"``  FWER control: ``min(1, M·p)``.
    ``"holm"``        FWER control, step-down — uniformly more powerful than
                      Bonferroni.
    ``"bh"``          Benjamini–Hochberg FDR control (independence / PRDS).
    ``"by"``          Benjamini–Yekutieli FDR control (arbitrary dependence;
                      the conservative variant Harvey–Liu–Zhu favour for finance).
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvalues must be a non-empty 1-D sequence.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1].")
    m = p.size
    order = np.argsort(p)
    ranked = p[order]

    if method == "bonferroni":
        adj_sorted = np.minimum(1.0, ranked * m)
    elif method == "holm":
        adj_sorted = np.minimum(1.0, ranked * (m - np.arange(m)))
        adj_sorted = np.maximum.accumulate(adj_sorted)        # enforce monotone ↑
    elif method in ("bh", "by"):
        c = _harmonic(m) if method == "by" else 1.0
        factor = c * m / np.arange(1, m + 1)
        adj_sorted = np.minimum(1.0, ranked * factor)
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]  # step-up, monotone
    else:
        raise ValueError("method must be one of 'bonferroni', 'holm', 'bh', 'by'.")

    out = np.empty_like(adj_sorted)
    out[order] = adj_sorted
    return out


# --------------------------------------------------------------------------- #
# Single-Sharpe haircut
# --------------------------------------------------------------------------- #
def _sharpe_to_t(sr: float, n_obs: int) -> float:
    """t-statistic of a per-observation Sharpe over *n_obs* bars: ``t = SR·√n``."""
    return sr * math.sqrt(n_obs)


def _two_sided_p(t: float) -> float:
    # Use the survival function, not 1 - cdf: for large |t| the latter cancels to
    # exactly 0.0 in float64 (t ≈ 8 already underflows), which would then explode
    # to an infinite haircut. sf stays accurate deep into the tail.
    return float(2.0 * norm.sf(abs(t)))


def haircut_sharpe(
    sr: float,
    n_obs: int,
    n_tests: int,
    *,
    method: str = "bonferroni",
    other_pvalues: Optional[Sequence[float]] = None,
    annualization: float = 1.0,
) -> HaircutResult:
    """Haircut a Sharpe ratio for ``n_tests`` multiple comparisons.

    Parameters
    ----------
    sr:
        The observed Sharpe. If it is **annualized**, pass ``annualization`` equal
        to the periods-per-year used to annualize it so the t-stat is recovered on
        the per-observation scale (``t = (sr/√ann)·√n_obs``).
    n_obs:
        Number of return observations behind *sr*.
    n_tests:
        Number of strategies / configurations tried (the multiplicity ``M``).
    method:
        ``"bonferroni"`` (exact from ``n_tests`` alone), or ``"holm"`` / ``"bh"`` /
        ``"by"`` — the latter three require *other_pvalues* (the p-values of the
        other ``n_tests − 1`` strategies) to be well-defined.

    Returns
    -------
    HaircutResult
        With the single-test and adjusted p-values, the implied adjusted t-stat,
        and the haircut Sharpe ``sr · t_adjusted / t_observed``.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2.")
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1.")
    if annualization < 1.0:
        raise ValueError(
            "annualization must be >= 1 (periods per year; use 1 for an already "
            "per-period Sharpe). A value in (0, 1) was silently ignored before."
        )

    sr_per_period = sr / math.sqrt(annualization) if annualization > 1 else sr
    t_obs = _sharpe_to_t(sr_per_period, n_obs)
    p_single = _two_sided_p(t_obs)

    if method == "bonferroni":
        p_adj = min(1.0, p_single * n_tests)
    elif method in ("holm", "bh", "by"):
        if other_pvalues is None:
            raise ValueError(
                f"method='{method}' needs `other_pvalues` (the p-values of the "
                f"other {n_tests - 1} strategies). Use 'bonferroni' for an "
                "M-only haircut."
            )
        pool = np.concatenate([[p_single], np.asarray(other_pvalues, dtype=float)])
        p_adj = float(adjusted_pvalues(pool, method=method)[0])
    else:
        raise ValueError("method must be one of 'bonferroni', 'holm', 'bh', 'by'.")

    # Map the adjusted p-value back to a t-stat, then scale the Sharpe by t_adj/|t_obs|.
    # isf(p/2) is the stable inverse of the two-sided sf used above (no 1-x cancellation).
    # Use |t_obs| so a strongly-significant NEGATIVE Sharpe keeps its sign and shrinks
    # in magnitude, rather than being zeroed out.
    t_adj = float(norm.isf(p_adj / 2.0)) if p_adj < 1.0 else 0.0
    t_adj = max(t_adj, 0.0)
    ratio = t_adj / abs(t_obs) if t_obs != 0 else 0.0
    haircut_sr = sr * ratio
    return HaircutResult(
        method=method,
        sr=float(sr),
        n_tests=int(n_tests),
        p_single=float(p_single),
        p_adjusted=float(p_adj),
        t_observed=float(t_obs),
        t_adjusted=float(t_adj),
        haircut_sr=float(haircut_sr),
        haircut_ratio=float(max(0.0, 1.0 - ratio)),
    )
