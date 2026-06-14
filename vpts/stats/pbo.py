"""Probability of Backtest Overfitting (PBO) via Combinatorial Symmetric CV.

Reference: Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of
Backtest Overfitting*, Journal of Computational Finance.

The idea
--------
You ran ``N`` strategy configurations and you intend to deploy the one with the
best backtest. **Is "best in sample" predictive of "good out of sample", or are
you just selecting noise?** CSCV answers this without a held-out period being
cherry-picked:

1. Slice the per-period performance matrix (rows = time, cols = configs) into
   ``S`` disjoint blocks.
2. For every way of choosing ``S/2`` blocks as in-sample (IS) — the rest are OOS:
   pick the IS-best config, then look up its **OOS rank**.
3. PBO = the fraction of those splits in which the IS-best config falls **below
   the OOS median** (its rank-logit ≤ 0).

A PBO near 0 means selecting the backtest winner generalizes; a PBO near 0.5 (or
above) means the selection procedure is overfitting — the winner is winning on
luck. This is the test that exposes the classic "in-sample AUC 0.94, OOS 0.49"
trap directly, *as a property of the selection process* rather than one model.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np

from vpts.stats.models import PBOResult


def _config_sharpe(block: np.ndarray) -> np.ndarray:
    """Per-config Sharpe across the rows of *block* (shape ``(rows, n_configs)``)."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, 0.0)
    return np.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)


def _config_mean(block: np.ndarray) -> np.ndarray:
    return block.mean(axis=0)


_METRICS = {"sharpe": _config_sharpe, "mean": _config_mean}


def probability_of_backtest_overfitting(
    perf_matrix: Sequence[Sequence[float]],
    *,
    n_splits: int = 16,
    metric: str = "sharpe",
) -> PBOResult:
    """Compute the PBO of a strategy-selection procedure via CSCV.

    Parameters
    ----------
    perf_matrix:
        Per-period performance, shape ``(T_periods, N_configs)``. Entry ``[t, n]``
        is config *n*'s return (or P&L) in period *t*. These are the candidate
        strategies you would select among — e.g. one column per parameter set.
    n_splits:
        Number of disjoint time blocks ``S`` (must be even, ≥ 4). The number of
        IS/OOS combinations is ``C(S, S/2)``; ``S=16`` ⇒ 12 870 combos.
    metric:
        Per-config performance statistic per block: ``"sharpe"`` (default) or
        ``"mean"``.

    Returns
    -------
    PBOResult
        ``pbo`` plus the logit distribution, the OOS performance-degradation slope
        (IS rank vs OOS performance), and the probability the IS-best loses OOS.
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (periods x configs).")
    T, N = M.shape
    if N < 2:
        raise ValueError("Need at least 2 configurations to assess overfitting.")
    if n_splits < 4 or n_splits % 2 != 0:
        raise ValueError("n_splits must be an even integer >= 4.")
    if T < n_splits:
        raise ValueError(f"Need at least n_splits={n_splits} periods, got {T}.")
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {sorted(_METRICS)}.")
    stat = _METRICS[metric]

    blocks = np.array_split(np.arange(T), n_splits)
    block_ids = range(n_splits)

    logits: list[float] = []
    oos_losses = 0
    is_ranks: list[float] = []   # normalized IS rank of the IS-best (always ~1.0)
    oos_perf_best: list[float] = []

    for combo in combinations(block_ids, n_splits // 2):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in block_ids if b not in combo])

        r_is = stat(M[is_rows])
        r_oos = stat(M[oos_rows])

        n_star = int(np.argmax(r_is))
        # Relative OOS rank of the IS-best config in (0, 1): rank / (N + 1).
        # ``argsort().argsort()`` gives 0-based ranks (ties broken by position).
        oos_rank = int(np.argsort(np.argsort(r_oos))[n_star]) + 1
        omega = oos_rank / (N + 1)
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        logits.append(math.log(omega / (1.0 - omega)))

        if r_oos[n_star] < 0:
            oos_losses += 1
        is_ranks.append(1.0)  # the IS-best is, by definition, IS rank 1
        oos_perf_best.append(float(r_oos[n_star]))

    arr = np.asarray(logits, dtype=float)
    n_combos = arr.size
    pbo = float((arr <= 0).mean())

    # OOS performance-degradation slope: regress the IS-best's OOS performance on
    # its (normalized) IS performance rank. A steep negative slope corroborates
    # overfitting (the more you optimised IS, the worse it did OOS).
    is_best_perf = np.array([float(stat(M[np.concatenate([blocks[b] for b in combo])]).max())
                             for combo in combinations(block_ids, n_splits // 2)])
    slope = _ols_slope(is_best_perf, np.asarray(oos_perf_best))

    return PBOResult(
        pbo=pbo,
        n_configs=N,
        n_splits=n_splits,
        n_combinations=n_combos,
        median_logit=float(np.median(arr)),
        prob_oos_loss=float(oos_losses / n_combos),
        degradation_slope=slope,
        logits=tuple(round(float(x), 4) for x in arr),
    )


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        return 0.0
    return float(np.dot(xc, y - y.mean()) / denom)
