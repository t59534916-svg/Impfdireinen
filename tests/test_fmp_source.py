"""Tests for FMPSource — offline, via an injected HTTP stub (no key, no network).

Exercises EOD parsing, the plan-gated error surface, the key-required guard, the
opt-in delisted capability, and registry routing. The real network path
(``_urllib_get``) is not exercised here.

    python tests/test_fmp_source.py
    pytest tests/test_fmp_source.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data import DataFetchError, FMPSource, SourceRegistry  # noqa: E402


def _eod_payload(n: int = 6) -> list:
    # FMP returns newest-first; OHLCV per bar.
    base = pd.Timestamp("2017-01-03")
    rows = []
    for i in range(n):
        px = 29.0 + i * 0.1
        rows.append({"symbol": "TEST", "date": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                     "open": px, "high": px + 0.5, "low": px - 0.5, "close": px + 0.2,
                     "volume": 100_000 + i, "vwap": px})
    return list(reversed(rows))


def _stub(payload):
    captured = {}

    def http_get(url: str):
        captured["url"] = url
        return payload

    http_get.captured = captured  # type: ignore[attr-defined]
    return http_get


# --------------------------------------------------------------------------- #
def test_parses_eod_into_ohlcv() -> None:
    src = FMPSource(api_key="dummy", http_get=_stub(_eod_payload(6)))
    df = src.get_bars("TEST", period="1y", interval="1d")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 6 and isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing and df.index.tz is None     # ascending, tz-naive
    assert (df["High"] >= df["Low"]).all() and (df["Volume"] > 0).all()


def test_plan_gated_error_is_surfaced() -> None:
    # Delisted history on a free plan returns an error dict, not a list of bars.
    payload = {"Error Message": "This endpoint requires a higher plan. Upgrade your plan."}
    src = FMPSource(api_key="d", http_get=_stub(payload))
    try:
        src.get_bars("LEH", start="2008-01-01", end="2008-09-15")
    except DataFetchError as exc:
        assert "higher plan" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError surfacing FMP's plan message")


def test_requires_api_key() -> None:
    import os

    saved = os.environ.pop("FMP_API_KEY", None)
    try:
        src = FMPSource(api_key=None, http_get=_stub(_eod_payload()))
        try:
            src.get_bars("X")
        except DataFetchError as exc:
            assert "key" in str(exc).lower()
        else:  # pragma: no cover
            raise AssertionError("expected DataFetchError without an API key")
    finally:
        if saved is not None:
            os.environ["FMP_API_KEY"] = saved


def test_delisted_capability_is_opt_in() -> None:
    assert FMPSource(api_key="d").capabilities.provides_delisted is False
    paid = FMPSource(api_key="d", delisted_capable=True)
    assert paid.capabilities.provides_delisted is True and paid.capabilities.requires_api_key is True


def test_url_includes_symbol_range_and_key() -> None:
    stub = _stub(_eod_payload())
    FMPSource(api_key="KEY123", http_get=stub).get_bars(
        "MSFT", start="2015-01-01", end="2017-12-31", interval="1d")
    url = stub.captured["url"]
    assert "symbol=MSFT" in url and "from=2015-01-01" in url and "to=2017-12-31" in url
    assert "apikey=KEY123" in url and "historical-price-eod/full" in url


def test_adjusted_uses_dividend_adjusted_endpoint() -> None:
    stub = _stub(_eod_payload())
    FMPSource(api_key="d", http_get=stub, adjusted=True).get_bars("X", period="1y")
    assert "historical-price-eod/dividend-adjusted" in stub.captured["url"]


def test_empty_result_raises() -> None:
    src = FMPSource(api_key="d", http_get=_stub([]))
    try:
        src.get_bars("DEAD", period="5y")
    except DataFetchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError on empty result")


def test_intraday_interval_rejected() -> None:
    src = FMPSource(api_key="d", http_get=_stub(_eod_payload()))
    try:
        src.get_bars("X", interval="5m")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-daily interval")


def test_registry_routes_survivor_to_fmp() -> None:
    fmp = FMPSource(api_key="d", http_get=_stub(_eod_payload(8)))
    reg = SourceRegistry([fmp])
    df = reg.get_bars("ANY", period="2y")
    assert len(df) == 8 and df.attrs.get("source") == "fmp"
    # Default FMP is survivor-only, so an FMP-only chain is not survivorship-capable.
    assert reg.has_delisted_source is False


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} FMP tests …\n")
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
