"""Tests for StooqSource — offline via an injected HTTP layer and a temp bulk export.

The load-bearing checks: it parses Stooq CSV correctly, **refuses to treat an
anti-bot / JS-challenge page as data** (raises DataFetchError), and reads a local
bulk export in both the simple and the bulk column layouts — all with zero network.

    python tests/test_stooq_source.py
    pytest tests/test_stooq_source.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data import DataFetchError, StooqSource  # noqa: E402

_SIMPLE_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2015-01-02,100.0,101.0,99.0,100.5,1000\n"
    "2015-01-05,100.5,102.0,100.0,101.5,1200\n"
    "2015-01-06,101.5,103.0,101.0,102.5,1500\n"
)
# Stooq bulk DB layout (semicolon-free CSV with <BRACKET> headers, YYYYMMDD dates).
_BULK_CSV = (
    "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
    "LEH.US,D,20080101,000000,60.0,61.0,59.0,60.5,5000,0\n"
    "LEH.US,D,20080102,000000,60.5,62.0,60.0,61.5,5200,0\n"
    "LEH.US,D,20080103,000000,61.5,63.0,61.0,62.5,5500,0\n"
)
_JS_WALL = ('<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
            "<noscript>This site requires JavaScript to verify your browser.</noscript>"
            "<script>verify()</script></body></html>")


def test_symbol_mapping_adds_us_suffix() -> None:
    src = StooqSource()
    assert src.stooq_symbol("AAPL") == "aapl.us"
    assert src.stooq_symbol("LEH") == "leh.us"
    assert src.stooq_symbol("MSFT.US") == "msft.us"      # already qualified → just lower-cased


def test_live_csv_parses_to_ohlcv() -> None:
    captured = {}

    def http_get(url: str) -> str:
        captured["url"] = url
        return _SIMPLE_CSV

    src = StooqSource(http_get=http_get)
    df = src.get_bars("AAPL")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 3 and df.index.is_monotonic_increasing
    assert df["Close"].iloc[-1] == 102.5
    assert "s=aapl.us" in captured["url"]                 # mapped symbol hit the endpoint
    assert src.capabilities.provides_delisted is True


def test_anti_bot_wall_raises_not_parsed() -> None:
    src = StooqSource(http_get=lambda url: _JS_WALL)
    try:
        src.get_bars("AAPL")
    except DataFetchError as exc:
        assert "anti-bot" in str(exc).lower() or "challenge" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError on the JS-challenge page")


def test_empty_response_raises() -> None:
    src = StooqSource(http_get=lambda url: "No data\n")
    try:
        src.get_bars("NOPE")
    except DataFetchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError on empty response")


def test_start_end_slicing() -> None:
    src = StooqSource(http_get=lambda url: _SIMPLE_CSV)
    df = src.get_bars("AAPL", start="2015-01-05")
    assert df.index.min() == pd.Timestamp("2015-01-05") and len(df) == 2


def test_bulk_root_reads_bulk_layout(tmp_path) -> None:
    (tmp_path / "leh.us.txt").write_text(_BULK_CSV)
    src = StooqSource(bulk_root=tmp_path, delisted=["LEH"])
    df = src.get_bars("LEH")
    assert len(df) == 3 and df["Close"].iloc[0] == 60.5
    assert df.index[0] == pd.Timestamp("2008-01-01")      # YYYYMMDD parsed
    assert src.is_delisted("LEH") is True and src.is_delisted("AAPL") is False


def test_bulk_root_missing_symbol_raises(tmp_path) -> None:
    (tmp_path / "aapl.us.csv").write_text(_SIMPLE_CSV)
    src = StooqSource(bulk_root=tmp_path)
    df = src.get_bars("AAPL")                              # simple layout under bulk_root works too
    assert len(df) == 3
    try:
        src.get_bars("MISSING")
    except DataFetchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError for a symbol absent from the bulk root")


def test_intraday_interval_rejected() -> None:
    src = StooqSource(http_get=lambda url: _SIMPLE_CSV)
    try:
        src.get_bars("AAPL", interval="5m")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-daily interval")


def test_missing_bulk_root_raises() -> None:
    try:
        StooqSource(bulk_root="/no/such/dir/xyz")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing bulk_root")


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} StooqSource tests …\n")
    import tempfile

    for t in tests:
        try:
            if "tmp_path" in t.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
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
