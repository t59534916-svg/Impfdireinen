"""Unsupervised regime detection & pattern discovery (the one ML gap), walk-forward.

The supervised harness (``cpcv_factor_eval`` etc.) asks "do these features predict
the forward return?". This module asks the *unsupervised* questions the project was
missing:

* **Pattern discovery** — cluster bars by their behavioral/structural feature
  vectors (:class:`RegimeClusterer`, transparent k-means on standardized features —
  chosen over a GMM/HMM for explainability) to find recurring *setup types*.
* **Regime detection** — assign each bar a regime label using only **past** data
  (:func:`walk_forward_regimes`: fit on a trailing window, assign the next block,
  refit periodically, align labels across refits so a regime keeps its identity).
* **Honest evaluation** — :func:`regime_forward_stats` reports each discovered
  pattern's *out-of-sample* forward-return distribution and a **label-shuffle
  permutation test** for "do the regimes separate forward returns at all?". A
  pattern that cannot beat its shuffled null is reported as no signal.

Dependency-light (numpy only), no look-ahead, and judged by the same standard as
everything else: a discovered "pattern" is a hypothesis until it clears its null.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vpts.ml.regime_models import RegimeForwardStats


# --------------------------------------------------------------------------- #
# Clusterer (k-means++ on standardized features)
# --------------------------------------------------------------------------- #
class RegimeClusterer:
    """Transparent unsupervised regime/pattern clusterer (k-means).

    Features are standardized on ``fit`` (train statistics stored and reused on
    ``predict`` — no leakage). k-means++ initialization, best of ``n_init`` runs.

    Parameters
    ----------
    n_regimes:
        Number of clusters / regimes.
    seed, n_init, max_iter:
        Determinism and convergence controls.
    """

    def __init__(self, n_regimes: int = 3, *, seed: int = 0, n_init: int = 6,
                 max_iter: int = 100) -> None:
        if n_regimes < 2:
            raise ValueError("n_regimes must be >= 2.")
        self.n_regimes = int(n_regimes)
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None
        self.inertia_: float = float("nan")

    # -- internals ------------------------------------------------------- #
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    @staticmethod
    def _kpp_init(Xs: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        n = Xs.shape[0]
        centers = [Xs[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min([np.sum((Xs - c) ** 2, axis=1) for c in centers], axis=0)
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
            centers.append(Xs[rng.choice(n, p=probs)])
        return np.array(centers, dtype=float)

    @staticmethod
    def _assign(Xs: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # (n, k) squared distances → nearest center.
        d2 = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1)

    def _lloyd(self, Xs: np.ndarray, rng: np.random.Generator):
        centers = self._kpp_init(Xs, self.n_regimes, rng)
        labels = np.zeros(Xs.shape[0], dtype=int)
        for _ in range(self.max_iter):
            labels = self._assign(Xs, centers)
            new = np.array([Xs[labels == j].mean(axis=0) if np.any(labels == j)
                            else Xs[rng.integers(Xs.shape[0])] for j in range(self.n_regimes)])
            if np.allclose(new, centers):
                centers = new
                break
            centers = new
        inertia = float(((Xs - centers[labels]) ** 2).sum())
        return centers, inertia

    # -- public ---------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> "RegimeClusterer":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < self.n_regimes:
            raise ValueError("X must be 2-D with at least n_regimes rows.")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = self._standardize(X)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            centers, inertia = self._lloyd(Xs, rng)
            if best is None or inertia < best[1]:
                best = (centers, inertia)
        self.centers_, self.inertia_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("fit() must be called before predict().")
        return self._assign(self._standardize(np.asarray(X, dtype=float)), self.centers_)


def _align(new_centers: np.ndarray, ref_centers: np.ndarray) -> np.ndarray:
    """Return a permutation mapping new clusters to the nearest reference clusters.

    Keeps a regime's identity stable across refits (greedy nearest-center match), so
    'regime 2' means roughly the same thing through time.
    """
    k = new_centers.shape[0]
    d = ((new_centers[:, None, :] - ref_centers[None, :, :]) ** 2).sum(axis=2)
    mapping = -np.ones(k, dtype=int)
    used = set()
    for new_idx in np.argsort(d.min(axis=1)):
        for ref_idx in np.argsort(d[new_idx]):
            if ref_idx not in used:
                mapping[new_idx] = ref_idx
                used.add(int(ref_idx))
                break
    return mapping


# --------------------------------------------------------------------------- #
# Walk-forward regime assignment (no look-ahead)
# --------------------------------------------------------------------------- #
def walk_forward_regimes(
    X: np.ndarray,
    *,
    n_regimes: int = 3,
    min_train: int = 200,
    refit_every: int = 60,
    seed: int = 0,
) -> np.ndarray:
    """Assign a regime label to each row of *X* using only prior rows.

    Bars before ``min_train`` get label ``-1`` (warm-up). From there the clusterer
    is fit on all rows seen so far, used to label the next ``refit_every`` bars, then
    refit — with labels aligned to the previous fit so regime identities persist.
    No row is labeled using information from its own or later bars.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)
    if n <= min_train:
        return labels
    ref_centers = None
    t = min_train
    while t < n:
        model = RegimeClusterer(n_regimes=n_regimes, seed=seed).fit(X[:t])
        centers = model.centers_
        # remap[new_label] = stable identity (nearest previous-fit cluster).
        remap = np.arange(n_regimes) if ref_centers is None else _align(centers, ref_centers)
        block = slice(t, min(t + refit_every, n))
        labels[block] = remap[model.predict(X[block])]
        ref_centers = centers[np.argsort(remap)]   # centers reordered into aligned identity space
        t += refit_every
    return labels


# --------------------------------------------------------------------------- #
# Honest evaluation: do the discovered regimes separate forward returns?
# --------------------------------------------------------------------------- #
def regime_forward_stats(
    labels: np.ndarray,
    forward_returns: np.ndarray,
    *,
    n_permutations: int = 500,
    seed: int = 0,
) -> RegimeForwardStats:
    """Per-regime forward-return separation + a label-shuffle permutation test.

    The statistic is the **spread** between the best- and worst-mean-return regimes
    (a regime-conditional long/short). The null shuffles the regime labels against
    the forward returns, destroying any regime→return link; the p-value is the
    fraction of shuffles whose spread matches or beats the real one. A spread that
    cannot clear its shuffled null is **no signal**, regardless of how clean the
    in-sample separation looks.
    """
    labels = np.asarray(labels)
    fwd = np.asarray(forward_returns, dtype=float)
    mask = (labels >= 0) & np.isfinite(fwd)
    labels, fwd = labels[mask], fwd[mask]
    if fwd.size < 4:
        raise ValueError("Need at least 4 labeled, finite samples.")
    uniq = np.unique(labels)

    means = {int(r): float(fwd[labels == r].mean()) for r in uniq}
    counts = {int(r): int((labels == r).sum()) for r in uniq}
    real_spread = max(means.values()) - min(means.values())

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(labels)
        m = [fwd[perm == r].mean() for r in uniq]
        null[i] = max(m) - min(m)
    p_value = float((np.sum(null >= real_spread) + 1) / (n_permutations + 1))

    return RegimeForwardStats(
        regime_mean_return=means,
        regime_count=counts,
        best_regime=int(max(means, key=means.get)),
        worst_regime=int(min(means, key=means.get)),
        spread=float(real_spread),
        null_spread_mean=float(null.mean()),
        p_value=p_value,
        n_permutations=n_permutations,
        n_samples=int(fwd.size),
    )
