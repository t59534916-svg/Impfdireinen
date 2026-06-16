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
    evidence_from_report,
    explain_report,
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


def test_scan_catches_common_paraphrases() -> None:
    # Best-effort backstop (NOT exhaustive): these common paraphrased edge-claims must
    # be flagged. The authoritative honesty gate is the code-computed verdict, not this.
    for text in (
        "This edge is real and durable.",
        "You can reliably profit from this pattern going forward.",
        "It is an exploitable inefficiency that produces excess returns.",
        "You should buy when this triggers.",
        "This signal is highly predictive and consistently makes money.",
    ):
        assert scan_for_overclaims(text, Verdict.SURVIVORSHIP_FRAGILE), text


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
# Bridge: HonestReport -> Evidence -> Insight
# --------------------------------------------------------------------------- #
def _mirage_report():
    from vpts.harness import HonestReport, InjectionPoint
    return HonestReport(
        n_datasets=2, n_samples=1000, oos_ic_mean=0.04, oos_ic_std=0.10,
        pct_folds_positive=60.0, block_perm_p=0.01, perms=200,
        bucket_curve=(0.10, 0.0, -0.10), long_short_net_pct=0.26,
        deflated_sharpe=0.80, n_trials=5, pbo=0.20,
        injection_sweep=(InjectionPoint(0.0, 0.04, 0.26, 0.30),
                         InjectionPoint(0.5, -0.01, -1.07, -0.20)),
        verdict="SURVIVORSHIP MIRAGE — significant on survivors but inverts under injection.",
    )


def test_evidence_from_report_maps_metrics_and_survivorship() -> None:
    ev = evidence_from_report(_mirage_report(), "structural", horizon=20)
    assert ev.oos_ic == 0.04 and ev.p_value == 0.01 and ev.pbo == 0.20
    assert ev.net_return_per_bet_bps == 26.0           # 0.26%/bet -> 26 bps
    assert ev.inverts_under_injection is True           # top bucket +0.30 -> -0.20
    assert ev.survives_injection is False
    assert ev.horizon == 20 and "harness verdict" in ev.notes
    assert assess(ev).verdict is Verdict.SURVIVORSHIP_FRAGILE


def test_evidence_from_report_survives_when_no_inversion() -> None:
    from vpts.harness import HonestReport, InjectionPoint
    rep = HonestReport(
        n_datasets=1, n_samples=500, oos_ic_mean=0.05, oos_ic_std=0.10,
        pct_folds_positive=70.0, block_perm_p=0.002, perms=200,
        bucket_curve=(0.2, 0.0, -0.1), long_short_net_pct=0.10,
        deflated_sharpe=0.97, n_trials=8, pbo=0.10,
        injection_sweep=(InjectionPoint(0.0, 0.05, 0.10, 0.20),
                         InjectionPoint(0.5, 0.03, 0.05, 0.06)),
        verdict="PASSES the honest checklist.",
    )
    ev = evidence_from_report(rep, "x")
    assert ev.inverts_under_injection is False and ev.survives_injection is True
    assert assess(ev).verdict is Verdict.VALIDATED      # sig + survives + DSR>=0.95


def test_evidence_from_report_no_sweep_leaves_survivorship_none() -> None:
    from vpts.harness import HonestReport
    rep = HonestReport(
        n_datasets=3, n_samples=1500, oos_ic_mean=float("nan"), oos_ic_std=float("nan"),
        pct_folds_positive=float("nan"), block_perm_p=0.6, perms=120,
        bucket_curve=(), long_short_net_pct=float("nan"),
        deflated_sharpe=float("nan"), n_trials=1, pbo=float("nan"),
        injection_sweep=None,
        verdict="NO EDGE — IC does not clear its block-permutation null.",
    )
    ev = evidence_from_report(rep, "x")
    assert ev.survives_injection is None and ev.inverts_under_injection is None
    assert ev.oos_ic is None and ev.net_return_per_bet_bps is None      # NaNs -> None
    assert "DSR not selection-adjusted" in ev.notes                     # n_trials<=1 caveat
    assert assess(ev).verdict is Verdict.NO_EDGE


def test_explain_report_offline_uses_template() -> None:
    ins = explain_report(_mirage_report(), "structural")
    assert ins.used_llm is False and ins.is_clean
    assert ins.verdict is Verdict.SURVIVORSHIP_FRAGILE and "structural" in ins.text


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
