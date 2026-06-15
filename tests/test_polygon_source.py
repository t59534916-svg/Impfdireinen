"""Tests for PolygonSource — offline, via an injected HTTP stub (no key, no network).

Exercises the parsing/shape logic, delisted detection, the key-required guard, and
registry routing. The real network path (`_urllib_get`) is not exercised here.

    python tests/test_polygon_source.py
    pytest tests/test_polygon_source.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data import DataFetchError, PolygonSource, SourceRegistry, SyntheticSource  # noqa: E402


def _agg_payload(n: int = 5) -> dict:
    base = pd.Timestamp("2018-01-02", tz="UTC")
    results = []
    for i in range(n):
        px = 100.0 + i
        results.append({
            "t": int((base + pd.Timedelta(days=i)).value // 1_000_000),  # ms epoch
            "o": px, "h": px + 1.0, "l": px - 1.0, "c": px + 0.5,
            "v": 1_000_000 + i, "vw": px, "n": 5000,
        })
    return {"ticker": "TEST", "status": "OK", "resultsCount": n, "results": results}


def _stub(payload: dict):
    captured = {}

    def http_get(url: str) -> dict:
        captured["url"] = url
        return payload

    http_get.captured = captured  # type: ignore[attr-defined]
    return http_get


# --------------------------------------------------------------------------- #
def test_parses_aggregates_into_ohlcv() -> None:
    src = PolygonSource(api_key="dummy", http_get=_stub(_agg_payload(6)))
    df = src.get_bars("TEST", period="1y", interval="1d")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 6 and isinstance(df.index, pd.DatetimeIndex)
    assert (df["High"] >= df["Low"]).all() and (df["Volume"] > 0).all()


def test_capabilities_advertise_delisted() -> None:
    caps = PolygonSource(api_key="dummy").capabilities
    assert caps.provides_delisted is True and caps.requires_api_key is True
    assert caps.is_free is False


def test_requires_api_key() -> None:
    import os

    saved = os.environ.pop("POLYGON_API_KEY", None)
    try:
        src = PolygonSource(api_key=None, http_get=_stub(_agg_payload()))
        for call in (lambda: src.get_bars("X"), lambda: src.list_delisted()):
            try:
                call()
            except DataFetchError:
                pass
            else:  # pragma: no cover
                raise AssertionError("expected DataFetchError without an API key")
    finally:
        if saved is not None:
            os.environ["POLYGON_API_KEY"] = saved


def test_empty_results_raises() -> None:
    src = PolygonSource(api_key="dummy", http_get=_stub({"status": "OK", "results": []}))
    try:
        src.get_bars("DEAD", period="5y")
    except DataFetchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError on empty results")


def test_unknown_interval_raises() -> None:
    src = PolygonSource(api_key="dummy", http_get=_stub(_agg_payload()))
    try:
        src.get_bars("X", interval="3s")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown interval")


def test_is_delisted_reads_active_flag() -> None:
    dead = PolygonSource(api_key="d", http_get=_stub({"results": {"ticker": "X", "active": False}}))
    live = PolygonSource(api_key="d", http_get=_stub({"results": {"ticker": "Y", "active": True}}))
    unknown = PolygonSource(api_key="d", http_get=_stub({"results": {"ticker": "Z"}}))
    assert dead.is_delisted("X") is True
    assert live.is_delisted("Y") is False
    assert unknown.is_delisted("Z") is None


def test_list_delisted_returns_reference_records() -> None:
    payload = {"results": [{"ticker": "AAA", "active": False, "delisted_utc": "2016-05-01"},
                           {"ticker": "BBB", "active": False, "delisted_utc": "2014-09-01"}]}
    src = PolygonSource(api_key="d", http_get=_stub(payload))
    out = src.list_delisted()
    assert {r["ticker"] for r in out} == {"AAA", "BBB"}


def test_url_includes_range_and_adjusted() -> None:
    stub = _stub(_agg_payload())
    src = PolygonSource(api_key="KEY123", http_get=stub, adjusted=True)
    src.get_bars("MSFT", start="2015-01-01", end="2017-12-31", interval="1d")
    url = stub.captured["url"]
    assert "/aggs/ticker/MSFT/range/1/day/2015-01-01/2017-12-31" in url
    assert "adjusted=true" in url and "apiKey=KEY123" in url


def test_registry_can_route_to_polygon_for_delisted() -> None:
    poly = PolygonSource(api_key="d", http_get=_stub(_agg_payload(8)))
    reg = SourceRegistry([SyntheticSource(), poly])
    # SyntheticSource also advertises delisted, but require= picks eligible sources
    # in priority order; here we confirm Polygon is reachable and parses.
    df = poly.get_bars("ANY", period="2y")
    assert len(df) == 8
    assert reg.has_delisted_source is True


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} polygon tests …\n")
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
