"""Bridge the harness output to the insight layer — the missing runtime link.

:func:`~vpts.harness.honest_backtest` returns a :class:`~vpts.harness.HonestReport`
(metrics); the insight layer narrates an :class:`~vpts.insight.models.Evidence` (the
numbers a verdict is computed from). :func:`evidence_from_report` maps one to the
other faithfully, so a harness run can be explained in a single step
(:func:`explain_report`).

The verdict the insight layer recomputes via :func:`~vpts.insight.guardrails.assess`
is intentionally the **stricter** of the two: it will not say ``VALIDATED`` unless a
survivorship-injection sweep was actually run *and* survived, whereas
``HonestReport.verdict`` can read "PASSES" without a sweep. That is the safe
direction; the report's own one-liner is preserved verbatim in ``Evidence.notes``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vpts.harness import HonestReport
from vpts.insight.generator import InsightGenerator
from vpts.insight.llm import LLMClient
from vpts.insight.models import Evidence, Insight


def _finite(x) -> Optional[float]:
    """Float or ``None`` — the insight layer treats ``None`` as 'not measured'."""
    return float(x) if x is not None and np.isfinite(x) else None


def evidence_from_report(
    report: HonestReport,
    name: str,
    *,
    horizon: Optional[int] = None,
    notes: str = "",
) -> Evidence:
    """Map a :class:`~vpts.harness.HonestReport` to an :class:`Evidence`.

    Survivorship fields derive from the injection sweep (``None`` when no sweep ran):
    ``inverts_under_injection`` is the report's sign-flip flag, and
    ``survives_injection`` is ``True`` only if it does *not* invert **and** the pooled
    IC at the most-injected rung is still positive — conservative by design. The
    report's own verdict (and any DSR-not-adjusted caveat) is recorded in ``notes``.
    """
    sweep = report.injection_sweep
    inverts = report.inverts                       # None if no/short sweep
    survives: Optional[bool] = None
    if sweep is not None and inverts is not None:
        final_ic = sweep[-1].pooled_ic
        survives = bool(not inverts and np.isfinite(final_ic) and final_ic > 0)

    net_bps = (report.long_short_net_pct * 100.0          # %/bet → bps/bet
               if np.isfinite(report.long_short_net_pct) else None)

    auto = f"harness verdict: {report.verdict}"
    if not report.dsr_selection_adjusted:
        auto += " | DSR not selection-adjusted (n_trials<=1)"
    full_notes = f"{notes} ({auto})" if notes else auto

    return Evidence(
        name=name,
        oos_ic=_finite(report.oos_ic_mean),
        p_value=_finite(report.block_perm_p),
        deflated_sharpe=_finite(report.deflated_sharpe),
        pbo=_finite(report.pbo),
        survives_injection=survives,
        inverts_under_injection=inverts,
        n_samples=report.n_samples,
        horizon=horizon,
        net_return_per_bet_bps=net_bps,
        notes=full_notes,
    )


def explain_report(
    report: HonestReport,
    name: str,
    *,
    client: Optional[LLMClient] = None,
    horizon: Optional[int] = None,
    notes: str = "",
    fallback_to_template: bool = True,
) -> Insight:
    """One step: ``HonestReport`` → ``Evidence`` → narrated :class:`Insight`.

    With no *client* (or on any LLM failure, if *fallback_to_template*) a deterministic
    template is rendered — honest by construction.
    """
    evidence = evidence_from_report(report, name, horizon=horizon, notes=notes)
    return InsightGenerator(
        client=client, fallback_to_template=fallback_to_template
    ).explain(evidence)
