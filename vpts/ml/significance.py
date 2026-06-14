"""Significance gate — the discipline that turns a backtest number into evidence.

Three tools the directional brief insists on (and that a permutation test alone
does not cover):

* **Deflated / Probabilistic Sharpe ratio** (Bailey & López de Prado 2012, 2014)
  — corrects a Sharpe for non-normal returns, sample length, and — critically —
  the number of strategy *trials*, since the maximum of many noisy Sharpe ratios
  is inflated even when every candidate is pure noise.
* **Directional skill vs the up-rate baseline** — directional accuracy is only
  meaningful relative to the *unconditional* up-rate (~53–55% for equities,
  because markets rise more often than they fall), never relative to 50%.

All per-observation Sharpe inputs are in the SAME (non-annualised) units as the
sample length ``n_obs``, following López de Prado's convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_GAMMA_E = 0.5772156649015329   # Euler–Mascheroni constant


def sharpe_ratio(returns: np.ndarray, *, ddof: int = 1) -> float:
    """Per-observation Sharpe ratio (mean / std) of a return series."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2:
        return float("nan")
    sd = float(r.std(ddof=ddof))
    return float(r.mean() / sd) if sd > 0 else float("nan")


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` i.i.d. noise strategies.

    The benchmark ``SR0`` in the deflated Sharpe ratio: the Sharpe you would see
    from the *luckiest* of ``n_trials`` independent strategies whose true Sharpe
    is zero and whose Sharpe-estimate dispersion is ``sr_std`` (Bailey & López de
    Prado 2014). Monotonically increasing in the number of trials.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if n_trials == 1 or sr_std <= 0:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1.0 - _GAMMA_E) * a + _GAMMA_E * b))


def probabilistic_sharpe_ratio(
    sr: float, n_obs: int, *, skew: float = 0.0, kurt: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """P(true Sharpe > benchmark) given the estimate, sample length, and moments.

    ``sr`` and ``sr_benchmark`` are per-observation; ``kurt`` is Pearson's
    (normal = 3). Non-normality (negative skew, excess kurtosis) widens the
    estimator's standard error and lowers the PSR.
    """
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / denom
    return float(norm.cdf(z))


@dataclass(frozen=True)
class SharpeSignificance:
    """Deflated/probabilistic Sharpe verdict for a strategy."""

    sharpe: float
    n_obs: int
    n_trials: int
    sr_benchmark: float        # expected max Sharpe across the trials (SR0)
    psr_vs_zero: float         # P(true SR > 0), single-trial
    deflated_sr: float         # P(true SR > SR0), i.e. corrected for multiple testing
    skew: float
    kurt: float

    @property
    def significant(self) -> bool:
        """True iff the deflated Sharpe clears the conventional 0.95 bar."""
        return self.deflated_sr > 0.95

    def as_dict(self) -> dict:
        return {
            "sharpe": round(self.sharpe, 4), "n_obs": self.n_obs, "n_trials": self.n_trials,
            "sr_benchmark": round(self.sr_benchmark, 4),
            "psr_vs_zero": round(self.psr_vs_zero, 4),
            "deflated_sr": round(self.deflated_sr, 4),
            "significant": self.significant,
        }

    def summary(self) -> str:
        return "\n".join([
            f"Sharpe significance  (n={self.n_obs}, trials={self.n_trials})",
            "-" * 56,
            f"  Sharpe (per obs)  : {self.sharpe:+.3f}  (skew {self.skew:+.2f}, kurt {self.kurt:.2f})",
            f"  PSR vs 0          : {self.psr_vs_zero:.3f}  (single-trial, P[true SR > 0])",
            f"  Benchmark SR0     : {self.sr_benchmark:.3f}  (expected max of {self.n_trials} trials)",
            f"  Deflated SR       : {self.deflated_sr:.3f}  -> "
            f"{'SIGNIFICANT' if self.significant else 'NOT significant'} after multiple testing",
        ])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def deflated_sharpe_ratio(
    returns: np.ndarray, *, n_trials: int = 1, sr_std: float = 0.0,
) -> SharpeSignificance:
    """Deflated Sharpe ratio of a return series, correcting for ``n_trials``.

    ``sr_std`` is the dispersion of Sharpe estimates across the trials that were
    run (the variance of the multiple-testing search). With ``n_trials=1`` the
    deflated SR reduces to the probabilistic SR vs zero.
    """
    r = np.asarray(returns, dtype=float).ravel()
    n = int(r.size)
    sr = sharpe_ratio(r)
    if not np.isfinite(sr):
        return SharpeSignificance(float("nan"), n, n_trials, 0.0, float("nan"),
                                  float("nan"), 0.0, 3.0)
    mu, sd = r.mean(), r.std()
    sk = float(((r - mu) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    ku = float(((r - mu) ** 4).mean() / sd ** 4) if sd > 0 else 3.0
    sr0 = expected_max_sharpe(n_trials, sr_std)
    psr0 = probabilistic_sharpe_ratio(sr, n, skew=sk, kurt=ku, sr_benchmark=0.0)
    dsr = probabilistic_sharpe_ratio(sr, n, skew=sk, kurt=ku, sr_benchmark=sr0)
    return SharpeSignificance(sr, n, n_trials, sr0, psr0, dsr, sk, ku)


@dataclass(frozen=True)
class DirectionalSkill:
    """Directional accuracy measured against the unconditional up-rate."""

    directional_accuracy: float
    up_rate: float
    majority_baseline: float        # max(up_rate, 1 - up_rate)
    edge_over_baseline: float        # accuracy - majority_baseline (the honest skill)
    n: int

    @property
    def beats_baseline(self) -> bool:
        return self.edge_over_baseline > 0.0

    def as_dict(self) -> dict:
        return {
            "directional_accuracy": round(self.directional_accuracy, 4),
            "up_rate": round(self.up_rate, 4),
            "majority_baseline": round(self.majority_baseline, 4),
            "edge_over_baseline": round(self.edge_over_baseline, 4),
            "beats_baseline": self.beats_baseline, "n": self.n,
        }

    def summary(self) -> str:
        return (f"Directional skill (n={self.n}): acc {self.directional_accuracy * 100:.1f}%  "
                f"vs up-rate {self.up_rate * 100:.1f}% / majority {self.majority_baseline * 100:.1f}%  "
                f"-> edge {self.edge_over_baseline * 100:+.2f}pp "
                f"({'beats' if self.beats_baseline else 'does NOT beat'} baseline)")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def directional_skill(signal: np.ndarray, forward_return: np.ndarray) -> DirectionalSkill:
    """Directional accuracy of ``sign(signal)`` vs ``sign(forward_return)``, vs the up-rate.

    The headline number is ``edge_over_baseline``: accuracy minus the accuracy of
    always predicting the majority direction. Beating 50% is trivial when the
    market rises ~55% of the time; beating the up-rate is the real bar.
    """
    s = np.asarray(signal, dtype=float).ravel()
    f = np.asarray(forward_return, dtype=float).ravel()
    if s.size != f.size or s.size == 0:
        raise ValueError("signal and forward_return must be non-empty and equal length.")
    pred_up = s > 0
    real_up = f > 0
    acc = float(np.mean(pred_up == real_up))
    up_rate = float(np.mean(real_up))
    majority = max(up_rate, 1.0 - up_rate)
    return DirectionalSkill(acc, up_rate, majority, acc - majority, int(s.size))
