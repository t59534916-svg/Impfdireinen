"""The honesty guardrails — code that decides the verdict and polices the prose.

Two pure functions, both independent of any LLM:

* :func:`assess` maps :class:`~vpts.insight.models.Evidence` to a
  :class:`~vpts.insight.models.Verdict` using the same bars the rest of the repo
  uses (permutation significance, survivorship injection, PBO, deflated Sharpe).
  **The model never decides this.**
* :func:`scan_for_overclaims` flags language that asserts an edge the verdict
  doesn't license. It is a **best-effort, defense-in-depth heuristic, not an
  exhaustive bar**: a regex over common edge-claim phrasings, so a determined or
  unlucky paraphrase can slip through. The authoritative honesty gate is the
  code-computed verdict from :func:`assess` (which the model never influences) and
  the verdict-locked template fallback — the scanner is the second line, not the
  first.

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
        reasons.append("signal inverts sign under survivorship injection — a survivorship mirage "
                       "(the survivor pattern reverses on delisted names).")
        return VerdictResult(Verdict.SURVIVORSHIP_FRAGILE, tuple(reasons))
    if significant and e.survives_injection is False:
        reasons.append("significant on survivors but loses significance under survivorship injection.")
        return VerdictResult(Verdict.SURVIVORSHIP_FRAGILE, tuple(reasons))

    # 4) Significant and survivorship-robust — but VALIDATED requires the
    # selection-adjusted bar to have been *tested and passed*, not merely skipped.
    if significant and (e.survives_injection is True):
        if e.deflated_sharpe is None:
            reasons.append("survives injection, but the selection-adjusted Sharpe (DSR) "
                           "was not computed — cannot validate without it.")
            return VerdictResult(Verdict.WEAK_UNVALIDATED, tuple(reasons))
        if e.deflated_sharpe < DSR_BAR:
            reasons.append(f"survives injection but DSR {e.deflated_sharpe:.2f} < {DSR_BAR}: not selection-proof.")
            return VerdictResult(Verdict.WEAK_UNVALIDATED, tuple(reasons))
        reasons.append("clears permutation, survivorship injection, and the deflated-Sharpe bar.")
        return VerdictResult(Verdict.VALIDATED, tuple(reasons))

    # 5) Anything else is suggestive at best.
    if significant:
        reasons.append("significant on this sample but survivorship was not (or could not be) tested.")
    else:
        reasons.append("no decisive significance test available; treat as a hypothesis only.")
    return VerdictResult(Verdict.WEAK_UNVALIDATED, tuple(reasons))


# Phrases that assert a tradeable/real edge. Forbidden unless the verdict is VALIDATED.
# NOTE: this list is illustrative, not exhaustive — see the module docstring. Paraphrases
# that avoid these tokens are caught (if at all) by the code-computed verdict, not here.
_OVERCLAIM_PATTERNS: tuple[str, ...] = (
    r"\bprofitabl\w*\b",
    r"\b(strong|real|genuine|reliable|robust|proven|clear)\s+(edge|signal|alpha)\b",
    r"\b(edge|signal|alpha)\s+is\s+(real|genuine|reliable|durable|robust|proven)\b",
    r"\btradeable\s+edge\b",
    r"\b(guarantee\w*|certain(ly)?|sure[- ]?thing)\b",
    r"\b(will|should)\s+(outperform|beat the market|make money|be profitable)\b",
    r"\b(buy|sell|long|short)\s+(signal|now|this|when|here)\b",
    r"\bconsistently\s+(profitable|outperform\w*|wins?)\b",
    r"\b(reliabl\w*\s+profit\w*|profit\s+from|makes?\s+money)\b",
    r"\bhighly\s+predictive\b",
    r"\b(excess\s+returns?|positive\s+expectancy)\b",
    r"\bexploit\w*\s+(inefficienc\w*|edge|pattern|signal)\b",
    r"\bgenerates?\s+alpha\b",
)
_OVERCLAIM_RE = [re.compile(p, re.IGNORECASE) for p in _OVERCLAIM_PATTERNS]


def scan_for_overclaims(text: str, verdict: Verdict) -> tuple[str, ...]:
    """Return the overclaim phrases present in *text* that the *verdict* forbids.

    Empty when the verdict is VALIDATED (claims are licensed) or no forbidden
    phrasing is found. A **best-effort** backstop against the LLM ignoring
    instructions — it pattern-matches common edge-claim phrasings, but is not
    exhaustive (a novel paraphrase can pass). It is the second line of defense; the
    first is the code-computed verdict, which the model never decides.

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
