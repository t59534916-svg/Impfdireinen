"""``vpts.stats`` — anti-overfitting / data-snooping statistics.

The harness in :mod:`vpts.validation` and :mod:`vpts.ml` tells you whether *one*
signal clears its shuffled null. This package answers the harder, selection-aware
questions that decide whether a *chosen* strategy is real:

* **Was the Sharpe long/clean enough to believe?** — Probabilistic Sharpe Ratio
  (:func:`probabilistic_sharpe_ratio`) and the minimum track-record length
  (:func:`min_track_record_length`).
* **Did selecting the best of many trials manufacture it?** — the Deflated Sharpe
  Ratio (:func:`deflated_sharpe_ratio`) and the Probability of Backtest
  Overfitting (:func:`probability_of_backtest_overfitting`).
* **How much Sharpe survives the multiple comparisons?** — the Harvey–Liu–Zhu
  haircut (:func:`haircut_sharpe`) and standard adjusted p-values
  (:func:`adjusted_pvalues`).
* **Is the Sharpe inflated by serial correlation?** — Lo's annualization
  (:func:`annualized_sharpe_lo`).

All functions are pure (numpy/scipy only), deterministic, and unit-tested. They
take either a raw return series or pre-computed statistics, and return immutable
result objects with ``as_dict()`` / ``summary()``.
"""
from __future__ import annotations

from vpts.stats.deflated_sharpe import (
    EULER_MASCHERONI,
    annualize_sharpe,
    annualized_sharpe_lo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sample_skew_kurt,
    sharpe_ratio,
)
from vpts.stats.models import (
    DeflatedSharpeResult,
    HaircutResult,
    MinTRLResult,
    PBOResult,
    PSRResult,
)
from vpts.stats.multiple_testing import adjusted_pvalues, haircut_sharpe
from vpts.stats.pbo import probability_of_backtest_overfitting

__all__ = [
    # Sharpe estimators
    "sharpe_ratio",
    "sample_skew_kurt",
    "annualize_sharpe",
    "annualized_sharpe_lo",
    # PSR / DSR / minTRL
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "EULER_MASCHERONI",
    # PBO
    "probability_of_backtest_overfitting",
    # multiple testing
    "adjusted_pvalues",
    "haircut_sharpe",
    # results
    "PSRResult",
    "DeflatedSharpeResult",
    "MinTRLResult",
    "PBOResult",
    "HaircutResult",
]
