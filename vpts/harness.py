"""The honest backtest harness — one call, the whole skeptic's checklist.

`RESEARCH.md` concludes that the durable asset of this project is not any signal but
the **harness** that judges signals honestly. This module makes that asset a single
entry point. Hand :func:`honest_backtest` a ``(features → forward-return)`` dataset
(or several, one per name) and it returns, in one object, everything the research log
holds every claim to:

* the **purged-CPCV out-of-sample IC** distribution (mean / std / % folds positive);
* a **block-permutation p-value** that preserves label autocorrelation (the honest
  null for overlapping labels — a per-row shuffle over-rejects);
* the **conviction-bucket curve** and flat-middle long/short net of cost (does the
  signal monotonically rank forward returns, or does it invert?);
* the **Deflated Sharpe Ratio** of the tradeable book, selection-adjusted for the
  number of variants tried, and the **Probability of Backtest Overfitting** (CSCV);
* optionally, a **survivorship-injection sweep**: re-score as synthetic delisted
  names are mixed in, the stress that exposed the project's central mirage.

A third party can therefore judge their own idea in well under 20 lines — see
``examples/honest_harness_demo.py`` and the README quickstart. Every evaluator keeps
its **null-clearing** test (returns non-significant on random input); see
``tests/test_harness.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from vpts.ml.factor_model import cpcv_factor_eval, cpcv_factor_quantile_returns
from vpts.ml.models import FactorDataset
from vpts.stats import (
    block_shuffle_indices,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    recommend_block_size,
    sample_skew_kurt,
    sharpe_ratio,
)
from vpts.validation import CombinatorialPurgedCV

logger = logging.getLogger(__name__)

#: A signal clears this DSR bar (Bailey & López de Prado's usual 95% threshold).
DSR_BAR = 0.95
#: PBO at or above this means selecting the best "config" is overfitting (Bailey et al.).
PBO_BAR = 0.5


def _verdict(*, sig: bool, overfit: bool, inverts: bool, dsr_ok: bool, pbo: float) -> str:
    """The one-line verdict, derived purely from the computed gates.

    Order mirrors :func:`vpts.insight.guardrails.assess`: a high PBO disqualifies a
    significant result (the *selection* is noise) before the survivorship and
    selection-adjustment checks. Isolated so the gating is unit-testable.
    """
    if not sig:
        return "NO EDGE — IC does not clear its block-permutation null."
    if overfit:
        return (f"OVERFIT — significant IC but PBO {pbo:.0%} ≥ {PBO_BAR:.0%}: "
                "selecting the best config is noise.")
    if inverts:
        return "SURVIVORSHIP MIRAGE — significant on survivors but inverts under injection."
    if not dsr_ok:
        return "FAILS selection adjustment — significant IC but DSR below the 0.95 bar."
    return "PASSES the honest checklist (IC significant, PBO ok, DSR clears, no inversion)."


@dataclass(frozen=True)
class InjectionPoint:
    """One rung of the survivorship-injection sweep."""

    delisted_fraction: float
    pooled_ic: float
    long_short_net_pct: float
    top_bucket_pct: float


@dataclass(frozen=True)
class HonestReport:
    """Everything :func:`honest_backtest` measured, plus a one-line verdict."""

    n_datasets: int
    n_samples: int
    oos_ic_mean: float
    oos_ic_std: float
    pct_folds_positive: float
    block_perm_p: float
    perms: int
    bucket_curve: tuple[float, ...]
    long_short_net_pct: float
    deflated_sharpe: float
    n_trials: int
    pbo: float
    injection_sweep: Optional[tuple[InjectionPoint, ...]]
    verdict: str

    @property
    def ic_significant(self) -> bool:
        return np.isfinite(self.block_perm_p) and self.block_perm_p < 0.05

    @property
    def dsr_clears(self) -> bool:
        return np.isfinite(self.deflated_sharpe) and self.deflated_sharpe >= DSR_BAR

    @property
    def dsr_selection_adjusted(self) -> bool:
        """True iff the DSR actually deflates for selection (``n_trials > 1``).

        With ``n_trials <= 1`` the expected-max-Sharpe benchmark is 0, so the DSR is a
        plain PSR with no multiple-testing penalty — read it accordingly.
        """
        return self.n_trials > 1

    @property
    def pbo_overfit(self) -> Optional[bool]:
        return None if not np.isfinite(self.pbo) else bool(self.pbo >= 0.5)

    @property
    def inverts(self) -> Optional[bool]:
        """Does the conviction direction flip from positive (base) to negative (most
        injected) across the sweep? ``None`` if no sweep was run."""
        sweep = self.injection_sweep
        if not sweep or len(sweep) < 2:
            return None
        return sweep[0].top_bucket_pct > 0 and sweep[-1].top_bucket_pct < 0

    def to_dict(self) -> dict:
        d = {
            "n_datasets": self.n_datasets, "n_samples": self.n_samples,
            "oos_ic_mean": self.oos_ic_mean, "oos_ic_std": self.oos_ic_std,
            "pct_folds_positive": self.pct_folds_positive,
            "block_perm_p": self.block_perm_p, "ic_significant": self.ic_significant,
            "bucket_curve": list(self.bucket_curve),
            "long_short_net_pct": self.long_short_net_pct,
            "deflated_sharpe": self.deflated_sharpe, "dsr_clears": self.dsr_clears,
            "n_trials": self.n_trials, "dsr_selection_adjusted": self.dsr_selection_adjusted,
            "pbo": self.pbo, "pbo_overfit": self.pbo_overfit,
            "verdict": self.verdict,
        }
        if self.injection_sweep is not None:
            d["injection_sweep"] = [
                {"delisted_fraction": p.delisted_fraction, "pooled_ic": p.pooled_ic,
                 "long_short_net_pct": p.long_short_net_pct, "top_bucket_pct": p.top_bucket_pct}
                for p in self.injection_sweep
            ]
            d["inverts"] = self.inverts
        return d

    def summary(self) -> str:
        sig = "significant" if self.ic_significant else "n.s."
        curve = " ".join(f"{c:+.2f}" for c in self.bucket_curve) if self.bucket_curve else "n/a"
        pbo_s = "n/a" if not np.isfinite(self.pbo) else f"{self.pbo:.0%}"
        lines = [
            f"Honest backtest — {self.n_datasets} dataset(s), {self.n_samples} samples",
            f"  OOS IC        : {self.oos_ic_mean:+.3f} ± {self.oos_ic_std:.3f}  "
            f"({self.pct_folds_positive:.0f}% folds +)",
            f"  block-perm p  : {self.block_perm_p:.3f}  ({sig}, {self.perms} perms)",
            f"  bucket curve  : {curve}   L/S net {self.long_short_net_pct:+.3f}%/bet",
            f"  Deflated SR   : {self.deflated_sharpe:.3f}  "
            f"({'clears' if self.dsr_clears else 'FAILS'} {DSR_BAR:.2f}, n_trials={self.n_trials}"
            + ("" if self.dsr_selection_adjusted else ", NO selection adj") + ")",
            f"  PBO           : {pbo_s}"
            + ("" if self.pbo_overfit is None else f"  ({'overfit' if self.pbo_overfit else 'generalizes'})"),
        ]
        if self.injection_sweep is not None:
            lines.append("  survivorship sweep (delisted frac → IC, top-bucket %):")
            for p in self.injection_sweep:
                lines.append(f"    {p.delisted_fraction:>4.0%}  IC {p.pooled_ic:+.3f}  "
                             f"top {p.top_bucket_pct:+.2f}%")
            inv = self.inverts
            if inv is not None:
                lines.append(f"    → conviction direction {'INVERTS' if inv else 'does not invert'} "
                             "under injection")
        lines.append(f"  VERDICT       : {self.verdict}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _as_list(datasets) -> list[FactorDataset]:
    return [datasets] if isinstance(datasets, FactorDataset) else list(datasets)


def _cv_for(ds: FactorDataset, cv: Optional[CombinatorialPurgedCV],
            n_groups: int, n_test: int) -> CombinatorialPurgedCV:
    return cv or CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test,
                                       purge=ds.purge_samples, embargo_pct=0.01)


def _fold_ics(ds, cv, alpha) -> Optional[np.ndarray]:
    try:
        return np.array(cpcv_factor_eval(ds, cv=cv, alpha=alpha).fold_ics, dtype=float)
    except ValueError:
        return None


def _pooled_ic(datasets, cvs, alpha) -> tuple[float, float, float]:
    arrs = [a for ds, cv in zip(datasets, cvs) if (a := _fold_ics(ds, cv, alpha)) is not None]
    if not arrs:
        return float("nan"), float("nan"), float("nan")
    ics = np.concatenate(arrs)
    return float(ics.mean()), float(ics.std()), float((ics > 0).mean() * 100.0)


def _block_perm_p(datasets, cvs, alpha, real_ic, perms, seed) -> float:
    if not np.isfinite(real_ic):
        return float("nan")
    rng = np.random.default_rng(seed)
    null = np.empty(perms, dtype=float)
    for i in range(perms):
        arrs = []
        for ds, cv in zip(datasets, cvs):
            bs = recommend_block_size(ds.horizon, ds.stride)
            perm = block_shuffle_indices(len(ds), bs, rng=rng)
            shuf = FactorDataset(X=ds.X, y=ds.y[perm], baseline=ds.baseline,
                                 feature_names=ds.feature_names, horizon=ds.horizon,
                                 stride=ds.stride, timestamps=ds.timestamps, symbol=ds.symbol)
            a = _fold_ics(shuf, cv, alpha)
            if a is not None:
                arrs.append(a)
        null[i] = float(np.concatenate(arrs).mean()) if arrs else np.nan
    null = null[np.isfinite(null)]
    return float((np.sum(null >= real_ic) + 1) / (null.size + 1))


def _book(datasets, cvs, alpha, n_buckets, cost_bps):
    """Conviction-bucket curve (averaged), long/short net (mean), per-bet streams."""
    curves, ls, streams = [], [], []
    for ds, cv in zip(datasets, cvs):
        try:
            bres, stream = cpcv_factor_quantile_returns(
                ds, cv=cv, n_buckets=n_buckets, cost_bps=cost_bps, return_stream=True)
        except ValueError:
            continue
        curves.append(np.asarray(bres.bucket_returns_pct, float))
        ls.append(float(bres.long_short_net_pct))
        streams.append(np.asarray(stream, float))
    curve = tuple(np.mean(curves, axis=0)) if curves else ()
    ls_net = float(np.mean(ls)) if ls else float("nan")
    return curve, ls_net, streams


def _effective_n(n_raw: int, ds_list) -> int:
    """Deflate the per-bet sample count for overlapping-label autocorrelation.

    With ``stride < horizon`` consecutive bets share most of their forward window, so
    the raw count over-states the number of *independent* observations and would
    inflate the PSR/DSR confidence (which scales with ``√(n−1)``). Deflate by the
    autocorrelation span — the same block size the permutation null uses — so the bar
    isn't cleared on optimism. Always ≥ 8.
    """
    spans = [recommend_block_size(ds.horizon, ds.stride) for ds in ds_list] or [1]
    span = max(1, int(np.median(spans)))
    return max(8, n_raw // span)


def _dsr(streams, ds_list, n_trials: int) -> float:
    """Deflated Sharpe of the pooled tails-only book, on an overlap-adjusted sample.

    The Sharpe/skew/kurtosis are estimated from the pooled per-bet stream, but the
    sample size handed to the PSR is :func:`_effective_n` (overlap-deflated), not the
    raw count. Returns NaN when there are too few bets to assess.
    """
    pooled = np.concatenate([s for s in streams if s.size]) if streams else np.array([])
    if pooled.size < 8:
        return float("nan")
    sr = sharpe_ratio(pooled, ddof=1)
    skew, kurt = sample_skew_kurt(pooled)
    eff_n = _effective_n(pooled.size, ds_list)
    return float(deflated_sharpe_ratio(
        sr=sr, n=eff_n, skew=skew, kurt=kurt, n_trials=int(n_trials)).dsr)


def _pbo(streams) -> float:
    usable = [s for s in streams if s.size]
    if len(usable) < 2:
        return float("nan")
    T = min(s.size for s in usable)
    if T < 8:
        return float("nan")
    mat = np.column_stack([s[:T] for s in usable])      # names as 'configs'
    n_splits = max(4, min(10, (T // 2) * 2))
    try:
        return float(probability_of_backtest_overfitting(mat, n_splits=n_splits, metric="mean").pbo)
    except ValueError:
        return float("nan")


def _injection_sweep(frames, feature_builder, fractions, alpha, n_groups, n_test,
                     n_buckets, cost_bps, terminal_frac, rally, seed):
    from vpts.data import SurvivorshipInjector

    points = []
    for f in sorted(fractions):
        if f <= 0:
            built = {s: feature_builder(fr, s) for s, fr in frames.items()}
        else:
            inj = SurvivorshipInjector(delisted_fraction=f, seed=seed,
                                       terminal_frac=terminal_frac, rally=rally).inject(frames)
            built = {s: feature_builder(fr, s) for s, fr in inj.frames.items()}
        datasets = [d for d in built.values() if d is not None and len(d) > 0]
        cvs = [_cv_for(d, None, n_groups, n_test) for d in datasets]
        ic, _std, _pct = _pooled_ic(datasets, cvs, alpha)
        curve, ls_net, _streams = _book(datasets, cvs, alpha, n_buckets, cost_bps)
        actual = inj.delisted_fraction if f > 0 else 0.0
        points.append(InjectionPoint(delisted_fraction=float(actual), pooled_ic=ic,
                                     long_short_net_pct=ls_net,
                                     top_bucket_pct=float(curve[-1]) if curve else float("nan")))
    return tuple(points)


def honest_backtest(
    datasets,
    *,
    cv: Optional[CombinatorialPurgedCV] = None,
    alpha: float = 1.0,
    n_groups: int = 6,
    n_test: int = 2,
    n_buckets: int = 5,
    cost_bps: float = 10.0,
    n_trials: int = 1,
    perms: int = 200,
    seed: int = 0,
    # optional survivorship-injection sweep (needs price frames + a feature builder)
    frames: Optional[dict] = None,
    feature_builder: Optional[Callable[[pd.DataFrame, str], FactorDataset]] = None,
    injection_fractions: Sequence[float] = (0.0, 0.2, 0.5),
    terminal_frac: Optional[float] = None,
    rally: str = "off",
) -> HonestReport:
    """Run the full honest-backtest checklist on a ``(features → forward-return)`` dataset.

    Parameters
    ----------
    datasets:
        A single :class:`~vpts.ml.models.FactorDataset` or a sequence of them (one per
        name; their fold ICs are pooled). Build one from OHLCV with
        :func:`~vpts.structure.build_structural_dataset`, or from any feature matrix
        by constructing ``FactorDataset(X, y, baseline, feature_names, horizon, stride)``.
    n_trials:
        Total number of strategy variants you have tried — **including this run** —
        the multiple-testing penalty for the Deflated Sharpe Ratio. **Be honest
        here**: ``n_trials <= 1`` sets the expected-max-Sharpe benchmark to 0, i.e.
        **no** selection adjustment at all (a plain PSR), and emits a warning. The DSR
        is otherwise computed on a sample size deflated for overlapping-label
        autocorrelation (``stride < horizon``), so it isn't inflated by overlap.
    perms:
        Block-permutation shuffles for the IC p-value.
    frames, feature_builder:
        Supply both to run the **survivorship-injection sweep**: ``frames`` is a
        ``{symbol: OHLCV}`` survivor mapping and ``feature_builder(frame, symbol)``
        returns a :class:`FactorDataset`. Synthetic delisted names are mixed in at each
        ``injection_fractions`` rung and the book is re-scored.

    Returns a :class:`HonestReport` (``.summary()`` / ``.to_dict()``).
    """
    ds_list = _as_list(datasets)
    if not ds_list:
        raise ValueError("honest_backtest needs at least one FactorDataset.")
    cvs = [_cv_for(ds, cv, n_groups, n_test) for ds in ds_list]

    ic_mean, ic_std, pct_pos = _pooled_ic(ds_list, cvs, alpha)
    p = _block_perm_p(ds_list, cvs, alpha, ic_mean, perms, seed)
    curve, ls_net, streams = _book(ds_list, cvs, alpha, n_buckets, cost_bps)

    dsr = _dsr(streams, ds_list, n_trials)
    pbo = _pbo(streams)
    if np.isfinite(dsr) and n_trials <= 1:
        logger.warning(
            "honest_backtest: n_trials=%d applies NO selection adjustment to the DSR "
            "(expected-max-Sharpe = 0). Pass the total number of strategy variants you "
            "have tried — including this run — for the deflation to mean anything.",
            n_trials,
        )

    sweep = None
    if frames is not None and feature_builder is not None:
        sweep = _injection_sweep(frames, feature_builder, injection_fractions, alpha,
                                 n_groups, n_test, n_buckets, cost_bps,
                                 terminal_frac, rally, seed)

    # ---- verdict ---------------------------------------------------------- #
    n_samples = sum(len(ds) for ds in ds_list)
    sig = bool(np.isfinite(p) and p < 0.05)
    overfit = bool(np.isfinite(pbo) and pbo >= PBO_BAR)
    dsr_ok = bool(np.isfinite(dsr) and dsr >= DSR_BAR)
    inverts = bool(sweep is not None and len(sweep) >= 2
                   and sweep[0].top_bucket_pct > 0 and sweep[-1].top_bucket_pct < 0)
    verdict = _verdict(sig=sig, overfit=overfit, inverts=inverts, dsr_ok=dsr_ok, pbo=pbo)

    return HonestReport(
        n_datasets=len(ds_list), n_samples=n_samples,
        oos_ic_mean=ic_mean, oos_ic_std=ic_std, pct_folds_positive=pct_pos,
        block_perm_p=p, perms=perms, bucket_curve=tuple(round(float(c), 4) for c in curve),
        long_short_net_pct=ls_net, deflated_sharpe=float(dsr), n_trials=int(n_trials),
        pbo=float(pbo), injection_sweep=sweep, verdict=verdict,
    )
