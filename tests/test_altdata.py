"""Tests for vpts.altdata — options-flow / sentiment integration points.

Interfaces + null + static provider + causal merge. No live data.

    python tests/test_altdata.py
    pytest tests/test_altdata.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.altdata import (  # noqa: E402
    AltSignalSource,
    NullAltSource,
    StaticAltSource,
    merge_alt_features,
)


def _frame(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"rvol_5": np.arange(n, dtype=float),
                         "clv_pressure_5": np.linspace(-1, 1, n)}, index=idx)


def test_null_source_returns_nan_and_no_capabilities() -> None:
    src = NullAltSource()
    idx = _frame().index
    assert src.capabilities.has_options_flow is False
    assert src.capabilities.has_sentiment is False
    of = src.options_flow("AAPL", idx)
    assert len(of) == len(idx) and of.isna().all()


def test_merge_with_null_source_is_passthrough() -> None:
    frame = _frame()
    merged = merge_alt_features(frame, NullAltSource(), "AAPL")
    pd.testing.assert_frame_equal(merged, frame)       # nothing added


def test_static_source_adds_columns() -> None:
    frame = _frame(30)
    of = pd.Series(np.linspace(0, 2, 30), index=frame.index)
    sent = pd.Series(np.cos(np.arange(30) / 5.0), index=frame.index)
    src = StaticAltSource(options_flow={"AAPL": of}, sentiment={"AAPL": sent})
    assert src.capabilities.has_options_flow and src.capabilities.has_sentiment

    merged = merge_alt_features(frame, src, "AAPL")
    assert "alt_options_flow" in merged.columns and "alt_sentiment" in merged.columns
    assert len(merged) == len(frame)
    assert np.allclose(merged["alt_options_flow"].to_numpy(), of.to_numpy())


def test_static_source_skips_unknown_symbol() -> None:
    frame = _frame()
    src = StaticAltSource(options_flow={"AAPL": pd.Series(1.0, index=frame.index)})
    merged = merge_alt_features(frame, src, "MSFT")    # no data for MSFT
    assert "alt_options_flow" not in merged.columns    # all-NaN column omitted


def test_merge_does_not_look_ahead() -> None:
    # The merged alt column at bar t must equal the provider's value at t (no shift).
    frame = _frame(40)
    of = pd.Series(np.arange(40, dtype=float), index=frame.index)
    src = StaticAltSource(options_flow={"X": of})
    merged = merge_alt_features(frame, src, "X")
    for t in (5, 20, 39):
        assert merged["alt_options_flow"].iloc[t] == of.iloc[t]


def test_partial_provider_via_subclass() -> None:
    class SentimentOnly(AltSignalSource):
        @property
        def capabilities(self):
            from vpts.altdata import AltDataCapabilities
            return AltDataCapabilities(name="sent", has_sentiment=True)

        def sentiment(self, symbol, index):
            return pd.Series(0.5, index=index)

    frame = _frame()
    merged = merge_alt_features(frame, SentimentOnly(), "ANY")
    assert "alt_sentiment" in merged.columns and "alt_options_flow" not in merged.columns


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} altdata tests …\n")
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
