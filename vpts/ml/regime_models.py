"""Immutable result for the unsupervised regime/pattern evaluation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeForwardStats:
    """Do the discovered regimes separate forward returns — beyond chance?

    ``spread`` is the best-minus-worst regime mean forward return (a regime-
    conditional long/short). ``p_value`` is a label-shuffle permutation test of that
    spread: a regime structure that cannot beat its shuffled null is **no signal**,
    however clean the in-sample separation looks.
    """

    regime_mean_return: dict
    regime_count: dict
    best_regime: int
    worst_regime: int
    spread: float
    null_spread_mean: float
    p_value: float
    n_permutations: int
    n_samples: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict:
        return {
            "regime_mean_return": {int(k): round(v, 5) for k, v in self.regime_mean_return.items()},
            "regime_count": {int(k): int(v) for k, v in self.regime_count.items()},
            "best_regime": self.best_regime,
            "worst_regime": self.worst_regime,
            "spread": round(self.spread, 5),
            "null_spread_mean": round(self.null_spread_mean, 5),
            "p_value": round(self.p_value, 4),
            "n_samples": self.n_samples,
        }

    def summary(self) -> str:
        verdict = ("regimes SEPARATE forward returns" if self.significant
                   else "no regime separation (within the shuffled null)")
        curve = "  ".join(f"r{k}:{v * 100:+.2f}%(n={self.regime_count[k]})"
                          for k, v in sorted(self.regime_mean_return.items()))
        return "\n".join([
            f"Regime forward-return separation ({self.n_samples} OOS samples, "
            f"{self.n_permutations} shuffles)",
            "-" * 60,
            f"  per-regime mean fwd : {curve}",
            f"  best−worst spread   : {self.spread * 100:+.3f}%  "
            f"(best r{self.best_regime} − worst r{self.worst_regime})",
            f"  null spread (mean)  : {self.null_spread_mean * 100:+.3f}%   "
            f"-> p = {self.p_value:.3f}",
            f"  verdict             : {verdict}",
        ])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()
