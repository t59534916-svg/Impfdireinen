"""Tests for vpts.insight — the LLM layer that cannot fabricate an edge.

The load-bearing tests: (1) the verdict is computed correctly in code, and (2) an
adversarial model that *tries* to claim an edge is caught and corrected. No network
or `anthropic` package needed — the real client imports the SDK only inside
.complete(); everything here uses MockLLMClient.

    python tests/test_insight.py
    pytest tests/test_insight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.insight import (  # noqa: E402
    AnthropicClient,
    Evidence,
    InsightGenerator,
    MockLLMClient,
    Verdict,
    assess,
    render_template,
    scan_for_overclaims,
)
from vpts.insight.llm import InsightLLMError  # noqa: E402


# --------------------------------------------------------------------------- #
# Verdict logic — computed in code, never by the model
# --------------------------------------------------------------------------- #
def test_verdict_overfit_dominates() -> None:
    v = assess(Evidence("x", p_value=0.001, pbo=0.7, survives_injection=True)).verdict
    assert v is Verdict.OVERFIT                       # high PBO overrides significance


def test_verdict_no_edge_when_insignificant() -> None:
    assert assess(Evidence("x", oos_ic=0.01, p_value=0.35)).verdict is Verdict.NO_EDGE


def test_verdict_survivorship_fragile_on_inversion() -> None:
    e = Evidence("structural", oos_ic=0.035, p_value=0.005,
                 survives_injection=False, inverts_under_injection=True)
    assert assess(e).verdict is Verdict.SURVIVORSHIP_FRAGILE


def test_verdict_survivorship_fragile_when_injection_kills_significance() -> None:
    e = Evidence("meta", p_value=0.02, survives_injection=False)
    assert assess(e).verdict is Verdict.SURVIVORSHIP_FRAGILE


def test_verdict_validated_only_when_everything_clears() -> None:
    e = Evidence("clean", oos_ic=0.05, p_value=0.001,
                 survives_injection=True, deflated_sharpe=0.98)
    assert assess(e).verdict is Verdict.VALIDATED
    # Same but DSR below bar ⇒ downgraded.
    weak = Evidence("clean", oos_ic=0.05, p_value=0.001,
                    survives_injection=True, deflated_sharpe=0.80)
    assert assess(weak).verdict is Verdict.WEAK_UNVALIDATED
    # Significant + survives but DSR NOT tested ⇒ must NOT validate (selection bar
    # skipped); otherwise edge-claims get licensed with no selection control.
    no_dsr = Evidence("clean", oos_ic=0.05, p_value=0.001, survives_injection=True)
    assert assess(no_dsr).verdict is Verdict.WEAK_UNVALIDATED
    assert assess(no_dsr).verdict.permits_edge_claim is False


def test_verdict_weak_when_no_significance_test() -> None:
    assert assess(Evidence("hypo", oos_ic=0.02)).verdict is Verdict.WEAK_UNVALIDATED


# --------------------------------------------------------------------------- #
# Overclaim guardrail
# --------------------------------------------------------------------------- #
def test_scan_flags_overclaims_when_not_validated() -> None:
    text = "This is a strong, profitable edge that will outperform the market."
    hits = scan_for_overclaims(text, Verdict.SURVIVORSHIP_FRAGILE)
    assert hits                                       # caught the overclaim
    assert scan_for_overclaims(text, Verdict.VALIDATED) == ()  # licensed when validated


def test_scan_passes_clean_text() -> None:
    text = "On survivors this looks predictive, but it inverts under injection — no usable edge."
    assert scan_for_overclaims(text, Verdict.SURVIVORSHIP_FRAGILE) == ()


# --------------------------------------------------------------------------- #
# Template (deterministic, offline)
# --------------------------------------------------------------------------- #
def test_template_is_faithful_and_not_overclaiming() -> None:
    e = Evidence("structural delta", oos_ic=0.035, p_value=0.005,
                 survives_injection=False, inverts_under_injection=True,
                 n_samples=1308, net_return_per_bet_bps=-107.0)
    vr = assess(e)
    text = render_template(e, vr)
    assert "survivorship" in text.lower() and "+0.035" in text
    assert "research finding, not a tradeable signal" in text
    assert scan_for_overclaims(text, vr.verdict) == ()


# --------------------------------------------------------------------------- #
# Generator — offline, adversarial model, behaving model, failure fallback
# --------------------------------------------------------------------------- #
def test_generator_offline_uses_template() -> None:
    e = Evidence("x", oos_ic=0.01, p_value=0.4)
    ins = InsightGenerator().explain(e)
    assert ins.used_llm is False and ins.verdict is Verdict.NO_EDGE
    assert ins.is_clean and "NO EDGE" in ins.text


def test_generator_catches_adversarial_overclaim() -> None:
    # A model that ignores instructions and asserts a tradeable edge.
    rogue = MockLLMClient("Clear, reliable edge — this is profitable and will outperform.")
    e = Evidence("structural", p_value=0.005, survives_injection=False,
                 inverts_under_injection=True)
    ins = InsightGenerator(client=rogue).explain(e)
    assert ins.used_llm is True
    assert ins.warnings                               # overclaim detected
    assert "AUTOMATED CORRECTION" in ins.text         # neutralized in the output
    assert ins.is_clean is False


def test_generator_accepts_honest_model_output() -> None:
    honest = MockLLMClient(
        "The dip-buying footprint looks predictive on survivors, but it is a "
        "survivorship artifact: it loses significance once delisted names appear. "
        "Not a usable signal."
    )
    e = Evidence("structural", p_value=0.005, survives_injection=False)
    ins = InsightGenerator(client=honest).explain(e)
    assert ins.used_llm is True and ins.is_clean and not ins.warnings


def test_generator_falls_back_on_llm_failure() -> None:
    def boom(system: str, user: str) -> str:
        raise InsightLLMError("simulated outage")

    e = Evidence("x", p_value=0.001, survives_injection=True, deflated_sharpe=0.99)
    ins = InsightGenerator(client=MockLLMClient(boom)).explain(e)
    assert ins.used_llm is False                      # degraded to template
    assert ins.verdict is Verdict.VALIDATED and ins.warnings


def test_generator_propagates_failure_when_no_fallback() -> None:
    def boom(system: str, user: str) -> str:
        raise InsightLLMError("simulated outage")

    gen = InsightGenerator(client=MockLLMClient(boom), fallback_to_template=False)
    try:
        gen.explain(Evidence("x", p_value=0.001))
    except InsightLLMError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected InsightLLMError to propagate")


def test_mock_client_receives_verdict_in_prompt() -> None:
    captured = {}

    def capture(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return "ok, no edge here."

    e = Evidence("x", p_value=0.005, survives_injection=False, inverts_under_injection=True)
    InsightGenerator(client=MockLLMClient(capture)).explain(e)
    assert "survivorship_fragile" in captured["user"]   # verdict handed to the model
    assert "never claim" in captured["system"].lower()  # honesty contract present


def test_anthropic_client_constructs_without_sdk() -> None:
    # Construction must not import the anthropic package (only .complete() does),
    # so this works in CI where the SDK isn't installed.
    c = AnthropicClient(model="claude-opus-4-8", max_tokens=800)
    assert c.model == "claude-opus-4-8" and c.max_tokens == 800


def test_anthropic_client_failure_surfaces_as_insight_llm_error() -> None:
    # A backend failure — missing package (CI) OR unresolved auth (here) — must be
    # wrapped as InsightLLMError, NOT raised raw, so the template fallback can catch
    # it. (Regression: construction was previously outside the try/except.)
    from vpts.insight.llm import InsightLLMError

    try:
        AnthropicClient(max_tokens=50).complete("system", "user")
    except InsightLLMError:
        pass
    else:  # pragma: no cover - would mean a live key resolved in CI
        raise AssertionError("expected InsightLLMError on backend failure")


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} insight tests …\n")
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
