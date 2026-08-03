"""Act V — data analysis: financial time series **and** fundamental data.

Two halves, one rule.

**Time series** (:mod:`vpts.analysis.timeseries`) — the descriptive diagnostic of
a price series: return/risk, distribution shape and tails, memory
(autocorrelation, Ljung-Box, Lo-MacKinlay variance ratio, Hurst, ADF),
volatility structure (clustering, ARCH-LM, Parkinson/Garman-Klass) and drawdown.
:func:`analyze_timeseries` runs the battery and returns a frozen
:class:`~vpts.analysis.models.TimeSeriesReport`.

**Fundamentals** (:mod:`vpts.analysis.fundamentals`) — point-in-time statements
behind a provider-agnostic :class:`FundamentalsSource` with an injectable HTTP
transport, turned into valuation/quality/leverage/growth ratios plus the
Piotroski F-score and Altman Z-score, and aligned onto a price index **as-of the
filing date**.

The rule: neither half is allowed to claim an edge. Descriptive statistics
describe, and are in-sample by construction. Anything that wants to be called a
signal goes through :mod:`vpts.analysis.dataset` — which emits the same
:class:`~vpts.ml.models.FactorDataset` / :class:`~vpts.ml.models.CrossSectionalPanel`
every other feature family in this repo emits — and is then judged by
:func:`vpts.harness.honest_backtest`: purged CPCV, block-permutation null,
Deflated Sharpe / PBO, survivorship injection. Same bar for everyone.
"""
from __future__ import annotations

from vpts.analysis.models import (
    FUNDAMENTAL_FEATURES,
    LINE_ITEMS,
    FundamentalRatios,
    FundamentalSeries,
    FundamentalSnapshot,
    TimeSeriesReport,
)
from vpts.analysis.timeseries import (
    ADF_CRITICAL,
    adf_test,
    analyze_timeseries,
    arch_lm_test,
    autocorrelation,
    distribution_stats,
    drawdown_curve,
    drawdown_stats,
    garman_klass_vol,
    hurst_exponent,
    jump_fraction,
    ljung_box,
    log_returns,
    parkinson_vol,
    realized_vol,
    rolling_correlation,
    tail_risk,
    variance_ratio,
)
from vpts.analysis.fundamentals import (
    DEFAULT_REPORTING_LAG_DAYS,
    FMPFundamentalsSource,
    FundamentalsSource,
    SyntheticFundamentalsSource,
    align_asof,
    align_fundamentals,
    altman_z_score,
    audit_point_in_time,
    compute_ratios,
    fundamental_ratio_frame,
    piotroski_f_score,
)
from vpts.analysis.dataset import (
    build_combined_dataset,
    build_fundamental_dataset,
    build_fundamental_panel,
    fundamental_feature_frame,
)

__all__ = [
    # models
    "FUNDAMENTAL_FEATURES", "LINE_ITEMS", "FundamentalRatios", "FundamentalSeries",
    "FundamentalSnapshot", "TimeSeriesReport",
    # time series
    "ADF_CRITICAL", "adf_test", "analyze_timeseries", "arch_lm_test", "autocorrelation",
    "distribution_stats", "drawdown_curve", "drawdown_stats", "garman_klass_vol",
    "hurst_exponent", "jump_fraction", "ljung_box", "log_returns", "parkinson_vol",
    "realized_vol", "rolling_correlation", "tail_risk", "variance_ratio",
    # fundamentals
    "DEFAULT_REPORTING_LAG_DAYS", "FMPFundamentalsSource", "FundamentalsSource",
    "SyntheticFundamentalsSource", "align_asof", "align_fundamentals", "altman_z_score",
    "audit_point_in_time", "compute_ratios", "fundamental_ratio_frame", "piotroski_f_score",
    # the evaluation contract
    "build_combined_dataset", "build_fundamental_dataset", "build_fundamental_panel",
    "fundamental_feature_frame",
]
