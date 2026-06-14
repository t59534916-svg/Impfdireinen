"""``vpts.insight`` — an LLM explanation layer that cannot fabricate an edge.

The harness produces statistics; this layer turns them into a human-readable
behavioral-finance story — *without* ever asserting an edge the statistics don't
support. The guarantee is structural, not a matter of trusting the model:

1. The **verdict** (no-edge / survivorship-fragile / overfit / validated …) is
   computed in code from the numbers (:func:`~vpts.insight.guardrails.assess`).
2. The LLM is asked only to *explain* that verdict and the behavior behind it.
3. The output is **scanned** for edge-claims the verdict forbids; violations are
   corrected and recorded (:func:`~vpts.insight.guardrails.scan_for_overclaims`).
4. With no LLM (or on failure) a faithful, non-overclaiming **template** is used.

>>> from vpts.insight import Evidence, InsightGenerator
>>> ev = Evidence(name="structural delta", oos_ic=0.035, p_value=0.005,
...               survives_injection=False, inverts_under_injection=True)
>>> InsightGenerator().explain(ev).verdict.value      # offline, deterministic
'survivorship_fragile'

Pass an :class:`~vpts.insight.llm.AnthropicClient` to ``InsightGenerator`` for an
LLM-written explanation (Claude); the guardrails apply identically.
"""
from __future__ import annotations

from vpts.insight.generator import SYSTEM_PROMPT, InsightGenerator, render_template
from vpts.insight.guardrails import assess, scan_for_overclaims
from vpts.insight.llm import AnthropicClient, InsightLLMError, LLMClient, MockLLMClient
from vpts.insight.models import Evidence, Insight, Verdict, VerdictResult

__all__ = [
    "Evidence",
    "Insight",
    "Verdict",
    "VerdictResult",
    "InsightGenerator",
    "render_template",
    "SYSTEM_PROMPT",
    "assess",
    "scan_for_overclaims",
    "LLMClient",
    "AnthropicClient",
    "MockLLMClient",
    "InsightLLMError",
]
