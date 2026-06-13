"""Gradient-boosted regression trees — dependency-free (numpy only).

The cross-sectional companion `RidgeFactorModel` is linear; this adds the
nonlinear, interaction-capturing learner that the survey literature consistently
finds is the *best-evidence* family on noisy, medium-N tabular financial data
(tree ensembles beat deep learning on this regime — Grinsztajn et al. 2022).
It is implemented in plain numpy to honour the repo's dependency-light ethos
(like `RidgeFactorModel` and `LogisticMetaModel`), so it stays runnable without
XGBoost/LightGBM.

Design follows the standard gradient-boosting recipe (Friedman 2001) for
squared-error loss: start from the mean, then iteratively fit shallow CART
regression trees to the residual (the negative gradient) and add a shrunk step.
Trees are scale-invariant, so no feature standardisation is needed.

`gbrt_cross_sectional_eval` plugs the model into the existing purged + embargoed
CPCV (`vpts.validation.cpcv`) over rebalance dates, and — per the directional
brief — scores it against the **unconditional up-rate baseline**, not 50%.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

from vpts.ml.models import CrossSectionalPanel
from vpts.validation.cpcv import CombinatorialPurgedCV


# --------------------------------------------------------------------------- #
# Regression tree (numpy, vectorised split search)
# --------------------------------------------------------------------------- #
def _best_split(
    X: np.ndarray, y: np.ndarray, min_leaf: int
) -> tuple[int, float, float]:
    """Best (feature, threshold, gain) by variance reduction; gain<=0 ⇒ no split."""
    n, p = X.shape
    sum_y = float(y.sum())
    parent_sse = float((y * y).sum()) - sum_y * sum_y / n
    best_feat, best_thr, best_gain = -1, 0.0, 0.0
    for f in range(p):
        xf = X[:, f]
        order = np.argsort(xf, kind="mergesort")
        xs = xf[order]
        ys = y[order]
        csum = np.cumsum(ys)
        csum2 = np.cumsum(ys * ys)
        nl = np.arange(1, n, dtype=float)            # left size for split after i
        sl, sl2 = csum[:-1], csum2[:-1]
        nr = n - nl
        sr, sr2 = csum[-1] - sl, csum2[-1] - sl2
        sse_l = sl2 - sl * sl / nl
        sse_r = sr2 - sr * sr / nr
        gain = parent_sse - (sse_l + sse_r)
        valid = (xs[:-1] != xs[1:]) & (nl >= min_leaf) & (nr >= min_leaf)
        gain = np.where(valid, gain, -np.inf)
        i = int(np.argmax(gain))
        if np.isfinite(gain[i]) and gain[i] > best_gain:
            best_feat, best_thr, best_gain = f, float((xs[i] + xs[i + 1]) / 2.0), float(gain[i])
    return best_feat, best_thr, best_gain


def _build_tree(
    X: np.ndarray, y: np.ndarray, depth: int, max_depth: int,
    min_leaf: int, min_gain: float, importances: np.ndarray,
) -> dict:
    node: dict = {"value": float(y.mean()) if y.size else 0.0}
    if depth >= max_depth or X.shape[0] < 2 * min_leaf:
        return node
    feat, thr, gain = _best_split(X, y, min_leaf)
    if feat < 0 or gain <= min_gain:
        return node
    mask = X[:, feat] <= thr
    if int(mask.sum()) < min_leaf or int((~mask).sum()) < min_leaf:
        return node
    importances[feat] += gain
    node["feature"], node["threshold"] = feat, thr
    node["left"] = _build_tree(X[mask], y[mask], depth + 1, max_depth, min_leaf, min_gain, importances)
    node["right"] = _build_tree(X[~mask], y[~mask], depth + 1, max_depth, min_leaf, min_gain, importances)
    return node


def _predict_tree(node: dict, X: np.ndarray) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=float)
    if "feature" not in node:
        out[:] = node["value"]
        return out
    mask = X[:, node["feature"]] <= node["threshold"]
    if mask.any():
        out[mask] = _predict_tree(node["left"], X[mask])
    if (~mask).any():
        out[~mask] = _predict_tree(node["right"], X[~mask])
    return out


class GradientBoostedTrees:
    """Squared-error gradient-boosted regression trees (numpy).

    Parameters
    ----------
    n_estimators: number of boosting rounds (trees).
    learning_rate: shrinkage applied to each tree's contribution.
    max_depth: maximum depth of each (CART) tree.
    min_samples_leaf: minimum samples in a leaf (a regulariser).
    subsample: row fraction sampled (without replacement) per tree (stochastic GB).
    random_state: seed for the subsample RNG (determinism).

    Attributes set after ``fit``: ``base_`` (initial mean), ``trees_`` (list of
    tree dicts), ``feature_importances_`` (gain-based, sums to 1).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        min_gain: float = 1e-12,
        random_state: int = 0,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1].")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.min_gain = float(min_gain)
        self.random_state = int(random_state)
        self.base_: float = 0.0
        self.trees_: list[dict] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedTrees":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be 2-D with one row per target.")
        n, p = X.shape
        self.base_ = float(y.mean())
        F = np.full(n, self.base_, dtype=float)
        self.trees_ = []
        imp = np.zeros(p, dtype=float)
        rng = np.random.default_rng(self.random_state)
        k = max(2 * self.min_samples_leaf, int(round(self.subsample * n)))
        for _ in range(self.n_estimators):
            residual = y - F
            if self.subsample < 1.0 and k < n:
                idx = rng.choice(n, size=k, replace=False)
                Xs, rs = X[idx], residual[idx]
            else:
                Xs, rs = X, residual
            tree = _build_tree(Xs, rs, 0, self.max_depth, self.min_samples_leaf, self.min_gain, imp)
            self.trees_.append(tree)
            F += self.learning_rate * _predict_tree(tree, X)
        tot = float(imp.sum())
        self.feature_importances_ = imp / tot if tot > 0 else np.full(p, 1.0 / p)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.base_, dtype=float)
        for tree in self.trees_:
            out += self.learning_rate * _predict_tree(tree, X)
        return out

    def signal(self, X: np.ndarray) -> np.ndarray:
        """Alias for :meth:`predict` (drop-in with the ridge model's API)."""
        return self.predict(X)


# --------------------------------------------------------------------------- #
# Cross-sectional CPCV evaluation, scored against the up-rate baseline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GBRTCrossSectionalResult:
    """OOS evaluation of a GBRT cross-sectional rank model under purged CPCV.

    ``oos_ic_*`` are per-test-date Spearman ICs of the GBRT score vs the forward
    return. The directional block compares pooled OOS accuracy to the
    **unconditional up-rate**, not 50%: ``edge_over_baseline`` is the only honest
    'skill' number (positive ⇒ beats always-predicting-the-majority-direction).
    """

    feature_names: tuple[str, ...]
    feature_importances: tuple[float, ...]
    oos_ic_mean: float
    oos_ic_median: float
    oos_ic_std: float
    pct_dates_positive_ic: float
    oos_dir_accuracy: float          # within-date "above median ⇒ up" hit-rate
    up_rate: float                   # unconditional P(forward return > 0)
    majority_baseline: float         # max(up_rate, 1 - up_rate)
    edge_over_baseline: float        # oos_dir_accuracy - majority_baseline
    oos_ics: tuple[float, ...]       # the raw per-test-date OOS IC stream (for the Sharpe gate)
    n_dates_oos: int
    n_rows: int
    n_names: int
    horizon: int
    rebalance: int
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_names": self.n_names, "n_rows": self.n_rows,
            "horizon": self.horizon, "rebalance": self.rebalance, "params": self.params,
            "feature_importances": {n: round(v, 4) for n, v
                                    in zip(self.feature_names, self.feature_importances)},
            "oos_ic_mean": round(self.oos_ic_mean, 4),
            "oos_ic_median": round(self.oos_ic_median, 4),
            "oos_ic_std": round(self.oos_ic_std, 4),
            "pct_dates_positive_ic": round(self.pct_dates_positive_ic, 1),
            "oos_dir_accuracy": round(self.oos_dir_accuracy, 4),
            "up_rate": round(self.up_rate, 4),
            "majority_baseline": round(self.majority_baseline, 4),
            "edge_over_baseline": round(self.edge_over_baseline, 4),
            "n_dates_oos": self.n_dates_oos,
        }

    def summary(self) -> str:
        imp = ", ".join(f"{n} {v:.2f}" for n, v in
                        zip(self.feature_names, self.feature_importances))
        verdict = (
            "no OOS edge (IC ~ 0)" if abs(self.oos_ic_mean) < 0.01
            else ("positive OOS signal" if self.oos_ic_mean > 0 else "anti-predictive"))
        edge = (
            "beats the up-rate baseline" if self.edge_over_baseline > 0.005
            else "does NOT beat the up-rate (coin-flip vs majority)")
        return "\n".join([
            f"GBRT cross-sectional CPCV — {self.n_names} names, {self.n_rows} rows, "
            f"{self.n_dates_oos} OOS dates (horizon {self.horizon}d, rebal {self.rebalance}d)",
            "-" * 64,
            f"  Importances    : {imp}",
            f"  Combined OOS IC: mean {self.oos_ic_mean:+.3f}  median {self.oos_ic_median:+.3f}  "
            f"σ {self.oos_ic_std:.3f}  ({self.pct_dates_positive_ic:.0f}% dates > 0)",
            f"  Dir. accuracy  : {self.oos_dir_accuracy * 100:.1f}%   "
            f"(up-rate {self.up_rate * 100:.1f}%, majority baseline {self.majority_baseline * 100:.1f}%)",
            f"  Edge vs base   : {self.edge_over_baseline * 100:+.2f} pp   -> {edge}",
            f"  Verdict        : {verdict}",
        ])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _date_groups(date_id: np.ndarray, n_dates: int) -> list[np.ndarray]:
    """Row indices grouped by date_id (groups[d] = rows on date d)."""
    groups: list[np.ndarray] = [np.empty(0, dtype=int)] * n_dates
    order = np.argsort(date_id, kind="mergesort")
    sid = date_id[order]
    for ch in np.split(order, np.flatnonzero(np.diff(sid)) + 1):
        if ch.size:
            groups[int(date_id[ch[0]])] = ch
    return groups


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.asarray(a).size < 3:
        return float("nan")
    rho = spearmanr(a, b).correlation
    return float(rho) if rho is not None and np.isfinite(rho) else float("nan")


def gbrt_cross_sectional_eval(
    panel: CrossSectionalPanel,
    cv: Optional[CombinatorialPurgedCV] = None,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    min_samples_leaf: int = 20,
    subsample: float = 0.8,
    random_state: int = 0,
) -> GBRTCrossSectionalResult:
    """Purged-CPCV OOS evaluation of a GBRT cross-sectional rank model.

    For every CPCV split, a GBRT is fit on the train dates' rows and scored on
    each purged/embargoed test date (per-date Spearman IC + a within-date
    above-median directional hit-rate). Directional accuracy is reported against
    the unconditional up-rate, the only honest baseline.
    """
    X, y, date_id = panel.X, panel.y, panel.date_id
    groups = _date_groups(date_id, panel.n_dates)
    cv = cv or CombinatorialPurgedCV(
        n_groups=6, n_test_groups=2, purge=panel.purge_dates, embargo_pct=0.01)

    oos_ic: list[float] = []
    importances: list[np.ndarray] = []
    correct = total = up = 0
    for split in cv.split(panel.n_dates):
        tr_dates, te_dates = split.train_idx, split.test_idx
        tr_rows = (np.concatenate([groups[d] for d in tr_dates])
                   if tr_dates.size else np.empty(0, dtype=int))
        if tr_rows.size < max(40, 4 * X.shape[1]):
            continue
        model = GradientBoostedTrees(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, subsample=subsample, random_state=random_state,
        ).fit(X[tr_rows], y[tr_rows])
        importances.append(model.feature_importances_)
        for d in te_dates:
            g = groups[d]
            if g.size < 3:
                continue
            pred = model.predict(X[g])
            real = y[g]
            ic = _spearman(pred, real)
            if np.isfinite(ic):
                oos_ic.append(ic)
            pred_up = pred > np.median(pred)        # within-date: top half ⇒ predicted up
            real_up = real > 0
            correct += int(np.sum(pred_up == real_up))
            total += int(real.size)
            up += int(np.sum(real_up))

    if not oos_ic or total == 0:
        raise ValueError(
            "CPCV produced no usable OOS dates for the GBRT model — "
            "increase the universe/history or reduce n_groups.")
    arr = np.array(oos_ic, dtype=float)
    imp_mean = (np.mean(np.vstack(importances), axis=0)
                if importances else np.full(X.shape[1], np.nan))
    up_rate = up / total
    dir_acc = correct / total
    majority = max(up_rate, 1.0 - up_rate)
    return GBRTCrossSectionalResult(
        feature_names=panel.feature_names,
        feature_importances=tuple(round(float(v), 4) for v in imp_mean),
        oos_ic_mean=float(arr.mean()),
        oos_ic_median=float(np.median(arr)),
        oos_ic_std=float(arr.std()),
        pct_dates_positive_ic=float((arr > 0).mean() * 100.0),
        oos_dir_accuracy=float(dir_acc),
        up_rate=float(up_rate),
        majority_baseline=float(majority),
        edge_over_baseline=float(dir_acc - majority),
        oos_ics=tuple(round(float(v), 6) for v in arr),
        n_dates_oos=int(arr.size),
        n_rows=len(panel),
        n_names=panel.n_names,
        horizon=panel.horizon,
        rebalance=panel.rebalance,
        params={
            "n_estimators": n_estimators, "learning_rate": learning_rate,
            "max_depth": max_depth, "min_samples_leaf": min_samples_leaf,
            "subsample": subsample,
        },
    )
