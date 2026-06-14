"""Turn validated harness evidence into a human-readable explanation — honestly.

:class:`InsightGenerator` is the public entry point. Its contract:

1. The **verdict is computed in code** (:func:`~vpts.insight.guardrails.assess`),
   not by the model.
2. The model is asked only to *explain the behavioral story* behind that verdict,
   under a system prompt that forbids asserting an unvalidated edge or inventing
   numbers.
3. The output is **scanned** (:func:`~vpts.insight.guardrails.scan_for_overclaims`);
   if the model overclaims anyway, a correction banner is prepended and the
   violation is recorded on the :class:`~vpts.insight.models.Insight`.
4. With **no LLM client** (or on any backend failure, if ``fallback_to_template``),
   a deterministic, faithful template is rendered instead — so the layer is always
   usable and never fabricates.
"""
from __future__ import annotations

from typing import Optional

from vpts.insight.guardrails import assess, correction_banner, scan_for_overclaims
from vpts.insight.llm import InsightLLMError, LLMClient
from vpts.insight.models import Evidence, Insight, Verdict, VerdictResult

SYSTEM_PROMPT = """\
You are the explanation layer of `vpts`, a rigorously validated, anti-data-snooping
trading-research system. Your ONLY job is to explain the *behavioral-finance story*
behind statistics that an out-of-sample, survivorship-aware validation harness has
already produced.

Non-negotiable rules:
1. A VERDICT has already been computed in code from the statistics. You must respect
   it exactly. You may NEVER claim a tradeable, profitable, or reliable edge unless
   the verdict is "validated". For any other verdict, be explicit that this is NOT a
   usable edge and explain why (e.g. survivorship fragility, overfitting, no
   significance).
2. Use ONLY the numbers provided. Do NOT invent statistics, returns, Sharpe ratios,
   dates, or universe sizes. If a number is not given, do not state one.
3. Explain the human/institutional behavior the feature is meant to capture
   (accumulation, absorption, liquidity grabs, FOMO, distribution, conviction
   shifts) and how that connects to the measured result — including, when the
   verdict is negative, WHY a plausible behavioral story still failed validation.
4. Be concise (a short paragraph or two), precise, and intellectually honest.
   Survivorship bias and overfitting are the default explanations for apparent
   edges; name them when relevant.

You are writing for a skeptical quant. Impressive-sounding but unsupported claims
are failures, not features."""


def _fmt(v, fmt: str) -> str:
    return format(v, fmt) if v is not None else "n/a"


def render_template(evidence: Evidence, vr: VerdictResult) -> str:
    """Deterministic, faithful explanation from the numbers — no model required.

    Never overclaims by construction: it states the verdict, the figures, and the
    standard caveats, full stop.
    """
    e = evidence
    headline = {
        Verdict.OVERFIT: "OVERFIT — the selection is fitting noise, not signal.",
        Verdict.NO_EDGE: "NO EDGE — fails its own permutation null.",
        Verdict.SURVIVORSHIP_FRAGILE: "SURVIVORSHIP MIRAGE — real on survivors, not survivorship-robust.",
        Verdict.WEAK_UNVALIDATED: "UNVALIDATED — suggestive at best; treat as a hypothesis.",
        Verdict.VALIDATED: "VALIDATED on this data — clears the bars we can test here (not a forward guarantee).",
    }[vr.verdict]

    stats = []
    if e.oos_ic is not None:
        stats.append(f"OOS IC {e.oos_ic:+.3f}")
    if e.p_value is not None:
        stats.append(f"permutation p = {e.p_value:.3f}")
    if e.deflated_sharpe is not None:
        stats.append(f"DSR {e.deflated_sharpe:.2f}")
    if e.pbo is not None:
        stats.append(f"PBO {e.pbo:.0%}")
    if e.net_return_per_bet_bps is not None:
        stats.append(f"{e.net_return_per_bet_bps:+.1f} bps/bet net")
    if e.survives_injection is not None:
        stats.append("survives injection" if e.survives_injection else "fails under injection")
    if e.inverts_under_injection:
        stats.append("INVERTS off survivors")
    stat_line = "; ".join(stats) if stats else "no summary statistics provided"

    lines = [
        f"Signal: {e.name}",
        f"Verdict: {headline}",
        f"Evidence: {stat_line}"
        + (f" (n={e.n_samples})" if e.n_samples else "")
        + (f", horizon {e.horizon} bars" if e.horizon else "")
        + ".",
        "Why: " + " ".join(vr.reasons),
    ]
    if e.notes:
        lines.append(f"Notes: {e.notes}")
    if vr.verdict is not Verdict.VALIDATED:
        lines.append(
            "This is a research finding, not a tradeable signal or financial advice. "
            "On survivorship-biased data, apparent edges routinely vanish or invert "
            "once delisted names are present."
        )
    return "\n".join(lines)


class InsightGenerator:
    """Generate honest behavioral-finance insight from validated evidence.

    Parameters
    ----------
    client:
        An :class:`~vpts.insight.llm.LLMClient`. If ``None``, only the deterministic
        template is used (fully offline).
    fallback_to_template:
        If the LLM call fails, fall back to the template instead of raising.
    """

    def __init__(self, client: Optional[LLMClient] = None, *, fallback_to_template: bool = True) -> None:
        self.client = client
        self.fallback_to_template = bool(fallback_to_template)

    def _build_user_prompt(self, evidence: Evidence, vr: VerdictResult) -> str:
        import json

        return (
            "Explain the following validated result for a skeptical quant.\n\n"
            f"VERDICT (computed in code — you must respect it): {vr.verdict.value}\n"
            f"Verdict reasons: {' '.join(vr.reasons)}\n\n"
            f"EVIDENCE (use only these numbers):\n{json.dumps(evidence.as_dict(), indent=2)}\n\n"
            "Write the behavioral-finance explanation now, obeying every rule in your "
            "instructions. If the verdict is not 'validated', make the absence of a "
            "usable edge unmistakable."
        )

    def explain(self, evidence: Evidence) -> Insight:
        """Return an :class:`Insight` for *evidence*, honest by construction."""
        vr = assess(evidence)

        if self.client is None:
            text = render_template(evidence, vr)
            return Insight(text=text, verdict=vr.verdict, evidence=evidence,
                           used_llm=False, warnings=scan_for_overclaims(text, vr.verdict))

        try:
            raw = self.client.complete(SYSTEM_PROMPT, self._build_user_prompt(evidence, vr))
            model = getattr(self.client, "model", None)
        except InsightLLMError:
            if not self.fallback_to_template:
                raise
            text = render_template(evidence, vr)
            return Insight(text=text, verdict=vr.verdict, evidence=evidence,
                           used_llm=False, model=None,
                           warnings=("LLM backend failed; rendered deterministic template.",))

        warnings = scan_for_overclaims(raw, vr.verdict)
        if warnings:
            # The model overclaimed despite instructions — neutralize it visibly.
            text = correction_banner(vr) + "\n\n" + raw
        else:
            text = raw
        return Insight(text=text, verdict=vr.verdict, evidence=evidence,
                       used_llm=True, model=model, warnings=warnings)
