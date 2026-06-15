"""Tests for the survivorship-coverage audit (vpts.data.coverage).

All offline and deterministic — the audit takes an injected fetch callable, so a
mock 'survivor-only feed' exercises the full classification with zero network. The
key assertion mirrors the project's thesis: a survivor-only source returns **0%**
of the bankruptcy (death) names, which is the survivorship signature the audit
exists to surface.

    python tests/test_coverage.py
    pytest tests/test_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data import (  # noqa: E402
    KNOWN_DELISTED,
    DataFetchError,
    DelistedTicker,
    audit_coverage,
    audit_known_delisted,
)
from vpts.data.coverage import (  # noqa: E402
    ACQUISITION,
    BANKRUPTCY,
    DROPPED,
    ERROR,
    REACHABLE,
)


def _frame(n: int) -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100.0}, index=idx
    )


def _survivor_only_feed(served: set[str]):
    """A feed that serves only *served* names and drops everything else (yfinance-like)."""

    def fetch(sym: str) -> pd.DataFrame:
        if sym in served:
            return _frame(252)
        raise DataFetchError(f"{sym}: no data (delisted/dropped)")

    return fetch


# --------------------------------------------------------------------------- #
def test_audit_classifies_reachable_dropped_error() -> None:
    served = {"AAA"}

    def fetch(sym: str) -> pd.DataFrame:
        if sym == "AAA":
            return _frame(252)
        if sym == "BBB":
            raise DataFetchError("dropped")
        raise TimeoutError("flaky network")            # -> ERROR, not a clean verdict

    rep = audit_coverage(["AAA", "BBB", "CCC"], fetch, min_bars=60)
    by = {r.symbol: r.status for r in rep.results}
    assert by == {"AAA": REACHABLE, "BBB": DROPPED, "CCC": ERROR}
    # Errors are excluded from the denominator: 1 reachable of 2 clean = 0.5.
    assert abs(rep.reachable_fraction - 0.5) < 1e-9
    assert len(rep.reachable) == 1 and len(rep.dropped) == 1 and len(rep.errored) == 1


def test_audit_min_bars_threshold_marks_thin_history_dropped() -> None:
    def fetch(sym: str) -> pd.DataFrame:
        return _frame(10)                              # below min_bars

    rep = audit_coverage(["X"], fetch, min_bars=60)
    assert rep.results[0].status == DROPPED and rep.results[0].bars == 10


def test_audit_reachable_records_span() -> None:
    rep = audit_coverage(["X"], lambda s: _frame(252), min_bars=60)
    r = rep.results[0]
    assert r.status == REACHABLE and r.bars == 252
    assert r.first == "2015-01-01" and r.last is not None


def test_survivor_only_feed_has_zero_death_name_coverage() -> None:
    # THE survivorship signature: a survivor-only feed serves 0% of bankruptcy names.
    # Give it the (quirky) acquisition tickers but none of the death names.
    served = {t.symbol for t in KNOWN_DELISTED if t.category == ACQUISITION}
    rep = audit_known_delisted(_survivor_only_feed(served))
    by_cat = rep.fraction_by_category()
    assert by_cat[BANKRUPTCY] == 0.0                   # every death name dropped
    assert by_cat[ACQUISITION] == 1.0                  # the served set all reachable
    # Overall fraction is strictly between (mixed catalogue), and bankruptcies dropped.
    assert all(r.status == DROPPED for r in rep.results if r.category == BANKRUPTCY)


def test_audit_accepts_plain_strings_with_category_map() -> None:
    rep = audit_coverage(
        ["AAA", "BBB"], _survivor_only_feed({"AAA"}),
        categories={"AAA": ACQUISITION, "BBB": BANKRUPTCY},
    )
    assert rep.fraction_by_category() == {ACQUISITION: 1.0, BANKRUPTCY: 0.0}


def test_coverage_report_to_frame_and_summary() -> None:
    rep = audit_known_delisted(_survivor_only_feed(set()))   # nothing served
    df = rep.to_frame()
    assert len(df) == len(KNOWN_DELISTED)
    assert set(df.columns) == {"symbol", "category", "status", "bars", "first", "last", "detail"}
    assert "reachable" in rep.summary()
    assert rep.reachable_fraction == 0.0                # all dropped


def test_all_errored_fraction_is_nan() -> None:
    import math

    def fetch(sym: str) -> pd.DataFrame:
        raise RuntimeError("everything is on fire")

    rep = audit_coverage(["A", "B"], fetch)
    assert len(rep.errored) == 2 and math.isnan(rep.reachable_fraction)


def test_known_delisted_catalogue_is_wellformed() -> None:
    syms = [t.symbol for t in KNOWN_DELISTED]
    assert len(syms) == len(set(syms))                          # unique tickers
    assert all(isinstance(t, DelistedTicker) for t in KNOWN_DELISTED)
    assert all(t.category in (BANKRUPTCY, ACQUISITION) for t in KNOWN_DELISTED)
    assert all(1990 <= t.year <= 2025 and t.name for t in KNOWN_DELISTED)
    # Both legs are represented (the bias-driving deaths and the benign acquisitions).
    cats = {t.category for t in KNOWN_DELISTED}
    assert cats == {BANKRUPTCY, ACQUISITION}


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} coverage-audit tests …\n")
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
