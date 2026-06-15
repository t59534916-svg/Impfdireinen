"""Block-permutation significance test — an honest null for autocorrelated targets.

Why it exists
-------------
The repo's standard test shuffles labels **per row**. That is correct when labels
are independent, but financial labels usually aren't: a forward ``horizon``-bar
return sampled every ``stride < horizon`` bars overlaps its neighbours, so adjacent
labels are strongly autocorrelated. Per-row shuffling destroys that autocorrelation,
which **shrinks the null's variance** and makes the test anti-conservative — it will
flag spurious edges (the exact effect seen in ``examples/behavioral_ai_demo.py`` at
small stride).

A **block permutation** reshuffles *contiguous blocks* of the target instead of
individual rows. Within-block autocorrelation is preserved, so the null's spread is
realistic and the p-value is honest. Reference: the moving/circular block bootstrap
(Künsch 1989; Politis & Romano 1994), adapted to a label-permutation test.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from vpts.stats.models import BlockPermutationResult


def recommend_block_size(horizon: int, stride: int) -> int:
    """Block length covering one label's autocorrelation span: ``ceil(horizon/stride)+1``.

    Two samples whose forward windows overlap are correlated; that span is roughly
    ``horizon/stride`` samples. A block at least that long keeps correlated labels
    together under the shuffle.
    """
    if horizon <= 0 or stride <= 0:
        raise ValueError("horizon and stride must be positive.")
    return int(math.ceil(horizon / stride)) + 1


def block_shuffle_indices(n: int, block_size: int, *, rng: np.random.Generator) -> np.ndarray:
    """Return a length-*n* permutation that keeps contiguous blocks intact.

    The series ``[0, n)`` is cut into contiguous blocks of ``block_size`` and the
    **block order** is shuffled; indices within a block keep their order. Each index
    appears exactly once (a true permutation), so applying it to a target array
    preserves within-block autocorrelation while breaking the target↔feature
    alignment at block granularity.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    block_size = max(1, min(int(block_size), n))
    starts = np.arange(0, n, block_size)
    rng.shuffle(starts)
    return np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])


def block_permutation_test(
    target: Sequence[float],
    stat_fn: Callable[[np.ndarray], float],
    *,
    block_size: int,
    n_permutations: int = 200,
    alternative: str = "greater",
    seed: int = 0,
) -> BlockPermutationResult:
    """Block-permutation p-value for a statistic that depends on *target*'s ordering.

    Parameters
    ----------
    target:
        The array whose alignment with the (fixed) features is being tested — e.g.
        forward returns ``y``. The statistic is recomputed on block-shuffled copies.
    stat_fn:
        ``stat_fn(target_array) -> float``. The real statistic is ``stat_fn(target)``;
        the null is ``stat_fn`` over block-shuffled targets. Capture the features in
        the closure (e.g. ``lambda y: spearman(signal, y)``).
    block_size:
        Contiguous block length (see :func:`recommend_block_size`).
    alternative:
        ``"greater"`` (default), ``"less"``, or ``"two-sided"``.

    Returns a :class:`~vpts.stats.models.BlockPermutationResult`. The p-value uses
    the conservative ``(#null at-least-as-extreme + 1) / (n + 1)`` estimator.
    """
    y = np.asarray(target, dtype=float)
    n = y.size
    if n < 2:
        raise ValueError("target must have at least 2 elements.")
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    real = float(stat_fn(y))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        null[i] = stat_fn(y[block_shuffle_indices(n, block_size, rng=rng)])
    null = null[np.isfinite(null)]
    m = null.size

    if alternative == "greater":
        exceed = int(np.sum(null >= real))
    elif alternative == "less":
        exceed = int(np.sum(null <= real))
    else:
        exceed = int(np.sum(np.abs(null) >= abs(real)))
    p_value = (exceed + 1) / (m + 1)

    return BlockPermutationResult(
        real_stat=real,
        null_mean=float(null.mean()) if m else float("nan"),
        null_std=float(null.std()) if m else float("nan"),
        p_value=float(p_value),
        n_permutations=int(m),
        block_size=int(max(1, min(block_size, n))),
        alternative=alternative,
    )
