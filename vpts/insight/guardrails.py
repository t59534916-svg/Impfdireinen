"""The honesty guardrails — code that decides the verdict and polices the prose.

Two pure functions, both independent of any LLM:

* :func:`assess` maps :class:`~vpts.insight.models.Evidence` to a
  :class:`~vpts.insight.models.Verdict` using the same bars the rest of the repo
  uses (permutation significance, survivorship injection, PBO, deflated Sharpe).
  **The model never decides this.**
* :func:`scan_for_overclaims` flags language that asserts an edge the verdict
  doesn't license — a structural backstop that does not depend on the model
  obeying its instructions.

Thresholds match `RESEARCH.md` (significance at p < 0.05; PBO ≳ 0.5 is overfit;
DSR ≥ 0.95 is the selection-adjusted bar).
"""
from __future__ import annotations

import re
from typing import Iterable

from vpts.insight.models import Evidence, Verdict, VerdictResult

P_SIGNIFICANT = 0.05
PBO_OVERFIT = 0.5
DSR_BAR = 0.95


def assess(evidence: Evidence) -> VerdictResult:
    """Derive the permitted :class:`Verdict` from the evidence. Deterministic."""
    reasons: list[str] = []
    e = evidence

    # 1) Overfitting dominates everything: a high PBO means the *selection* is noise.
    if e.pbo is not None and e.pbo >= PBO_OVERFIT:
        reasons.append(f"PBO {e.pbo:.0%} ≥ {PBO_OVERFIT:.0%}: selecting the best config is overfitting.")
        return VerdictResult(Verdict.OVERFIT, tuple(reasons))

    # 2) Must clear its own permutation null at all.
    if e.p_value is not None and e.p_value >= P_SIGNIFICANT:
        reasons.append(f"permutation p = {e.p_value:.3f} ≥ {P_SIGNIFICANT}: does not beat its shuffled null.")
        return VerdictResult(Verdict.NO_EDGE, tuple(reasons))

    significant = e.p_value is not None and e.p_value < P_SIGNIFICANT

    # 3) Survivorship is the binding constraint in this repo — check it before celebrating.
    if e.inverts_under_injection:
        reasons.append("conviction edge inverts sign once delisted names are injected — a survivorship mirage.")
        return VerdictResult(Verdict.SURVIVORSHIP_FRAGILE, tuple(reasons))
    if significant and e.survives_injection is False:
        reasons.append("significant on survivors but loses significance under survivorship injection.")
        return VerdictResult(Verdict.SURVIVORSHIP_FRAGILE, tuple(reasons))

    # 4) Significant and survivorship-robust — but still gate on selection-adjusted Sharpe.
    if significant and (e.survives_injection is True):
        if e.deflated_sharpe is not None and e.deflated_sharpe < DSR_BAR:
            reasons.append(f"survives injection but DSR {e.deflated_sharpe:.2f} < {DSR_BAR}: not selection-proof.")
            return VerdictResult(Verdict.WEAK_UNVALIDATED, tuple(reasons))
        reasons.append("clears permutation, survivorship injection, and (where tested) the deflated-Sharpe bar.")
        return VerdictResult(Verdict.VALIDATED, tuple(reasons))

    # 5) Anything else is suggestive at best.
    if significant:
        reasons.append("significant on this sample but survivorship was not (or could not be) tested.")
    else:
        reasons.append("no decisive significance test available; treat as a hypothesis only.")
    return VerdictResult(Verdict.WEAK_UNVALIDATED, tuple(reasons))


# Phrases that assert a tradeable/real edge. Forbidden unless the verdict is VALIDATED.
_OVERCLAIM_PATTERNS: tuple[str, ...] = (
    r"\bprofitabl\w*\b",
    r"\b(strong|real|genuine|reliable|robust|proven|clear)\s+(edge|signal|alpha)\b",
    r"\btradeable\s+edge\b",
    r"\b(guarantee\w*|certain(ly)?|sure[- ]?thing)\b",
    r"\b(will|should)\s+(outperform|beat the market|make money|be profitable)\b",
    r"\b(buy|long|short)\s+(signal|now|this)\b",
    r"\bconsistently\s+(profitable|outperform\w*|wins?)\b",
    r"\bpositive\s+expectancy\b",
    r"\bgenerates?\s+alpha\b",
)
_OVERCLAIM_RE = [re.compile(p, re.IGNORECASE) for p in _OVERCLAIM_PATTERNS]


def scan_for_overclaims(text: str, verdict: Verdict) -> tuple[str, ...]:
    """Return the overclaim phrases present in *text* that the *verdict* forbids.

    Empty when the verdict is VALIDATED (claims are licensed) or no forbidden
    phrasing is found. This is a backstop against the LLM ignoring instructions —
    it operates on the output, not on the model's good behavior.

    Deliberately conservative: it is context-blind, so a *negated* phrase ("this is
    not a real edge") is also flagged. That is the safe failure direction — a
    redundant correction banner on honest text is harmless; letting a genuine
    overclaim through is not. Word honest narration to avoid the trigger phrases.
    """
    if verdict.permits_edge_claim:
        return ()
    hits: list[str] = []
    for rx in _OVERCLAIM_RE:
        for m in rx.finditer(text):
            hits.append(m.group(0))
    return tuple(dict.fromkeys(hits))  # dedupe, preserve order


def correction_banner(verdict: VerdictResult) -> str:
    """A blunt, machine-generated correction prepended when overclaims are caught."""
    return (
        "⚠️ AUTOMATED CORRECTION: the explanation above contained language asserting an "
        f"edge that the evidence does NOT support (verdict: {verdict.verdict.value}). "
        f"Reason: {' '.join(verdict.reasons)} Treat any such claim as unsupported."
    )
