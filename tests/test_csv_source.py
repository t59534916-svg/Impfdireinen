"""Tests for the CSV ingester (vpts.data.read_ohlcv_csv / CsvSource).

The load-bearing case is a real-world **German** export (ariva-style): ';' delimiter,
decimal comma, dot thousands, DD.MM.YY dates, German column names — it must parse to the
same clean OHLCV as a plain English comma CSV. Plus a round-trip through materialize_lake
+ DataLakeSource (the whole ingestion backbone) when a parquet engine is present.

    python tests/test_csv_source.py
    pytest tests/test_csv_source.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data import CsvSource, DataFetchError, read_ohlcv_csv  # noqa: E402

_US_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2008-09-10,7.79,8.20,6.10,7.25,471000000\n"
    "2008-09-11,6.40,6.99,4.20,4.22,378000000\n"
    "2008-09-12,4.50,4.55,3.10,3.65,290000000\n"
)
# ariva-style German export: ';'-delimited, comma decimal, dot thousands, DD.MM.YY.
_DE_CSV = (
    "Datum;Erster;Hoch;Tief;Schlusskurs;Stücke;Volumen\n"
    "10.09.08;7,79;8,20;6,10;7,25;1.234;9.000\n"
    "11.09.08;6,40;6,99;4,20;4,22;2.000;8.500\n"
    "12.09.08;4,50;4,55;3,10;3,65;3.500;12.000\n"
)


def test_us_comma_csv_parses(tmp_path) -> None:
    p = tmp_path / "LEH.csv"
    p.write_text(_US_CSV)
    df = read_ohlcv_csv(p)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 3 and df.index.is_monotonic_increasing
    assert df["Close"].iloc[-1] == 3.65 and df.index[0] == pd.Timestamp("2008-09-10")


def test_german_ariva_style_csv_parses(tmp_path) -> None:
    p = tmp_path / "LEH.csv"
    p.write_text(_DE_CSV)
    df = read_ohlcv_csv(p)                                  # auto: ';', comma-decimal, dayfirst
    assert len(df) == 3 and list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index[0] == pd.Timestamp("2008-09-10")        # DD.MM.YY parsed day-first
    assert abs(df["Open"].iloc[0] - 7.79) < 1e-9            # comma decimal
    assert df["Volume"].iloc[0] == 1234                     # "Stücke" (share count), dot thousands
    assert df["Close"].iloc[-1] == 3.65


def test_german_and_us_agree(tmp_path) -> None:
    (tmp_path / "us.csv").write_text(_US_CSV)
    (tmp_path / "de.csv").write_text(_DE_CSV)
    us = read_ohlcv_csv(tmp_path / "us.csv")
    de = read_ohlcv_csv(tmp_path / "de.csv")
    # Same prices/dates regardless of locale (volume differs: US has real volume, DE Stücke).
    assert (us[["Open", "High", "Low", "Close"]].round(2).values
            == de[["Open", "High", "Low", "Close"]].round(2).values).all()
    assert list(us.index) == list(de.index)


def test_column_map_override(tmp_path) -> None:
    p = tmp_path / "X.csv"
    p.write_text("d,o,h,l,settle,qty\n2020-01-02,10,11,9,10.5,100\n2020-01-03,10.5,12,10,11.5,120\n")
    df = read_ohlcv_csv(p, column_map={"Date": "d", "Open": "o", "High": "h",
                                       "Low": "l", "Close": "settle", "Volume": "qty"})
    assert len(df) == 2 and df["Close"].iloc[-1] == 11.5


def test_missing_close_raises(tmp_path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("Date,Open,High,Low,Volume\n2020-01-02,10,11,9,100\n")
    try:
        read_ohlcv_csv(p)
    except DataFetchError as exc:
        assert "missing" in str(exc).lower() and "column_map" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError when Close is absent")


def test_csv_source_serves_and_slices(tmp_path) -> None:
    (tmp_path / "LEH.csv").write_text(_US_CSV)
    src = CsvSource(tmp_path, delisted=["LEH"])
    assert src.capabilities.provides_delisted is True
    assert src.available_symbols() == ["LEH"]
    df = src.get_bars("LEH", start="2008-09-11")
    assert len(df) == 2 and df.index.min() == pd.Timestamp("2008-09-11")
    assert src.is_delisted("LEH") is True and src.is_delisted("AAPL") is False
    try:
        src.get_bars("NOPE")
    except DataFetchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError for a missing symbol")


def test_csv_dir_round_trips_through_lake(tmp_path) -> None:
    """The whole backbone: CSV dir → materialize_lake → DataLakeSource."""
    try:
        import pyarrow  # noqa: F401
    except Exception:  # pragma: no cover
        print("  (no parquet engine — skipped)"); return
    from vpts.data import DataLakeSource, materialize_lake

    (tmp_path / "csv").mkdir()
    (tmp_path / "csv" / "LEH.csv").write_text(_DE_CSV)
    lake = tmp_path / "lake"
    rep = materialize_lake(lake, CsvSource(tmp_path / "csv"), ["LEH"], delisted=["LEH"])
    assert rep.n_written == 1 and rep.delisted_written == ("LEH",)
    back = DataLakeSource(lake).get_bars("LEH", period="max")
    assert len(back) == 3 and back["Close"].iloc[-1] == 3.65


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import tempfile

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} CSV-source tests …\n")
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
