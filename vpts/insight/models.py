"""Data structures for the LLM insight layer.

The design separates **what the harness found** (:class:`Evidence`) from **what may
be claimed about it** (:class:`Verdict`, computed in code, not by the LLM) and the
**rendered explanation** (:class:`Insight`). This separation is the honesty
mechanism: the model narrates a verdict it is handed; it does not get to decide
whether an edge exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    """The only conclusions the system permits, ranked by how much they license.

    Computed deterministically from the statistics (see :mod:`vpts.insight.guardrails`),
    never by the LLM. The narration must stay within the verdict's bounds.
    """

    OVERFIT = "overfit"                          # PBO high / huge IS-OOS gap
    NO_EDGE = "no_edge"                          # fails its permutation null
    SURVIVORSHIP_FRAGILE = "survivorship_fragile"  # significant on survivors, dies/inverts injected
    WEAK_UNVALIDATED = "weak_unvalidated"        # suggestive but not selection-proof
    VALIDATED = "validated"                      # clears every bar we can test here

    @property
    def permits_edge_claim(self) -> bool:
        """Only a VALIDATED verdict may be described as a (provisional) edge."""
        return self is Verdict.VALIDATED


@dataclass(frozen=True)
class VerdictResult:
    """A verdict plus the machine-checked reasons behind it."""

    verdict: Verdict
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"verdict": self.verdict.value, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class Evidence:
    """Validated harness output for one signal — the *only* input the LLM may narrate.

    Every field is a number the harness produced. Optional fields are ``None`` when
    that test was not run. The narration layer is forbidden from introducing any
    quantity not present here.
    """

    name: str
    oos_ic: Optional[float] = None              # out-of-sample information coefficient
    p_value: Optional[float] = None             # label-shuffle permutation p-value
    deflated_sharpe: Optional[float] = None     # DSR (selection-adjusted)
    pbo: Optional[float] = None                 # probability of backtest overfitting
    survives_injection: Optional[bool] = None   # holds under survivorship injection?
    inverts_under_injection: Optional[bool] = None  # edge flips sign off survivors?
    n_samples: Optional[int] = None
    horizon: Optional[int] = None
    net_return_per_bet_bps: Optional[float] = None  # net of costs, if traded
    notes: str = ""

    def as_dict(self) -> dict:
        d = {
            "name": self.name,
            "oos_ic": self.oos_ic,
            "p_value": self.p_value,
            "deflated_sharpe": self.deflated_sharpe,
            "pbo": self.pbo,
            "survives_injection": self.survives_injection,
            "inverts_under_injection": self.inverts_under_injection,
            "n_samples": self.n_samples,
            "horizon": self.horizon,
            "net_return_per_bet_bps": self.net_return_per_bet_bps,
            "notes": self.notes,
        }
        return {k: v for k, v in d.items() if v is not None and v != ""}


@dataclass(frozen=True)
class Insight:
    """A rendered explanation, its verdict, and any guardrail warnings."""

    text: str
    verdict: Verdict
    evidence: Evidence
    used_llm: bool
    model: Optional[str] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_clean(self) -> bool:
        """True if the rendered text raised no overclaim guardrail warnings."""
        return not self.warnings

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "used_llm": self.used_llm,
            "model": self.model,
            "warnings": list(self.warnings),
            "evidence": self.evidence.as_dict(),
            "text": self.text,
        }
