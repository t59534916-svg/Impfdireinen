"""Immutable results for the anti-overfitting statistics suite (:mod:`vpts.stats`).

Every estimator in this package returns one of these frozen dataclasses. They all
expose ``as_dict()`` (JSON-friendly) and ``summary()`` (human/LLM-friendly) so the
numbers can flow straight into a report, a test assertion, or the insight layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Sharpe-significance family (Bailey & López de Prado)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PSRResult:
    """Probabilistic Sharpe Ratio — P(true SR > benchmark SR) given the sample.

    The PSR deflates the observed Sharpe for **track-record length, skew and fat
    tails**: a high Sharpe on a short, negatively-skewed, kurtotic return stream
    is *not* the same evidence as the same Sharpe on a long, well-behaved one.
    """

    psr: float
    sr: float
    benchmark_sr: float
    n: int
    skew: float
    kurt: float
    statistic: float

    def is_significant(self, confidence: float = 0.95) -> bool:
        return self.psr >= confidence

    def as_dict(self) -> dict:
        return {
            "psr": round(self.psr, 4),
            "sr_per_period": round(self.sr, 4),
            "benchmark_sr": round(self.benchmark_sr, 4),
            "n": self.n,
            "skew": round(self.skew, 3),
            "kurt": round(self.kurt, 3),
        }

    def summary(self) -> str:
        return (
            f"PSR = {self.psr:.1%}  (SR={self.sr:.3f}/period vs SR*={self.benchmark_sr:.3f}, "
            f"n={self.n}, skew={self.skew:+.2f}, kurt={self.kurt:.2f})"
        )


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Deflated Sharpe Ratio — PSR with the benchmark set to the *expected maximum*
    Sharpe under ``n_trials`` independent backtests.

    This is the headline overfitting control: if you tried 200 strategy
    configurations and kept the best, the bar it must clear is not zero — it is the
    Sharpe you would *expect* to see from luck alone across 200 trials. The DSR is
    the probability the selected strategy beats that selection-adjusted bar.
    """

    dsr: float
    sr: float
    expected_max_sr: float
    n_trials: int
    var_trials_sr: float
    n: int
    skew: float
    kurt: float

    def is_significant(self, confidence: float = 0.95) -> bool:
        return self.dsr >= confidence

    def as_dict(self) -> dict:
        return {
            "dsr": round(self.dsr, 4),
            "sr_per_period": round(self.sr, 4),
            "expected_max_sr": round(self.expected_max_sr, 4),
            "n_trials": self.n_trials,
            "var_trials_sr": round(self.var_trials_sr, 6),
            "n": self.n,
            "skew": round(self.skew, 3),
            "kurt": round(self.kurt, 3),
        }

    def summary(self) -> str:
        verdict = "significant" if self.is_significant() else "NOT significant"
        return (
            f"DSR = {self.dsr:.1%} ({verdict} @95%)  — SR {self.sr:.3f} vs "
            f"selection-adjusted bar {self.expected_max_sr:.3f} from {self.n_trials} trials"
        )


@dataclass(frozen=True)
class MinTRLResult:
    """Minimum Track Record Length — observations needed for PSR(SR*) = ``prob``.

    Answers "how long must this track record be before its Sharpe is statistically
    convincing?". If ``min_trl`` exceeds the data you have, the result is *not yet*
    significant no matter how pretty the point estimate.
    """

    min_trl: float
    sr: float
    benchmark_sr: float
    skew: float
    kurt: float
    prob: float
    n_obs: Optional[int] = None

    @property
    def achieved(self) -> Optional[bool]:
        if self.n_obs is None:
            return None
        return self.n_obs >= self.min_trl

    def as_dict(self) -> dict:
        return {
            "min_trl": round(self.min_trl, 1),
            "sr_per_period": round(self.sr, 4),
            "benchmark_sr": round(self.benchmark_sr, 4),
            "prob": self.prob,
            "n_obs": self.n_obs,
            "achieved": self.achieved,
        }

    def summary(self) -> str:
        body = (
            f"minTRL = {self.min_trl:.0f} obs for {self.prob:.0%} confidence "
            f"(SR={self.sr:.3f} vs {self.benchmark_sr:.3f})"
        )
        if self.n_obs is not None:
            body += f"  — have {self.n_obs} → {'OK' if self.achieved else 'TOO SHORT'}"
        return body


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PBOResult:
    """Probability of Backtest Overfitting via Combinatorial Symmetric CV.

    ``pbo`` is the fraction of in-sample/out-of-sample splits in which the
    *in-sample best* configuration lands **below the OOS median** — i.e. the share
    of the time that picking the backtest winner is no better than (or worse than)
    a coin flip out of sample. High PBO (≳0.5) means the selection procedure is
    overfitting: the "best" strategy is best because of luck, not signal.
    """

    pbo: float
    n_configs: int
    n_splits: int
    n_combinations: int
    median_logit: float
    prob_oos_loss: float
    degradation_slope: float
    logits: tuple[float, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "pbo": round(self.pbo, 4),
            "n_configs": self.n_configs,
            "n_splits": self.n_splits,
            "n_combinations": self.n_combinations,
            "median_logit": round(self.median_logit, 4),
            "prob_oos_loss": round(self.prob_oos_loss, 4),
            "degradation_slope": round(self.degradation_slope, 4),
        }

    def summary(self) -> str:
        return (
            f"PBO = {self.pbo:.1%}  (N={self.n_configs} configs, S={self.n_splits} "
            f"splits, {self.n_combinations} combos)\n"
            f"  OOS degradation slope {self.degradation_slope:+.3f}  ·  "
            f"P(OOS loss) {self.prob_oos_loss:.1%}  ·  median logit {self.median_logit:+.2f}"
        )


# --------------------------------------------------------------------------- #
# Multiple-testing haircut (Harvey, Liu & Zhu)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HaircutResult:
    """Multiple-testing haircut of a Sharpe ratio (Harvey, Liu & Zhu 2016).

    Given that a strategy was the survivor of ``n_tests`` trials, its t-statistic
    is inflated. This deflates it: the adjusted p-value (Bonferroni / Holm / BHY)
    is mapped back to an adjusted t-stat, and the Sharpe is scaled by the same
    ratio. ``haircut_ratio`` is the fraction of Sharpe shaved off by honesty.
    """

    method: str
    sr: float
    n_tests: int
    p_single: float
    p_adjusted: float
    t_observed: float
    t_adjusted: float
    haircut_sr: float
    haircut_ratio: float

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "sr": round(self.sr, 4),
            "n_tests": self.n_tests,
            "p_single": round(self.p_single, 6),
            "p_adjusted": round(self.p_adjusted, 6),
            "t_observed": round(self.t_observed, 4),
            "t_adjusted": round(self.t_adjusted, 4),
            "haircut_sr": round(self.haircut_sr, 4),
            "haircut_ratio": round(self.haircut_ratio, 4),
        }

    def summary(self) -> str:
        return (
            f"{self.method.title()} haircut: SR {self.sr:.3f} → {self.haircut_sr:.3f} "
            f"({self.haircut_ratio:.0%} shaved)  "
            f"[t {self.t_observed:.2f}→{self.t_adjusted:.2f}, p {self.p_single:.1e}→{self.p_adjusted:.1e}, "
            f"M={self.n_tests}]"
        )
