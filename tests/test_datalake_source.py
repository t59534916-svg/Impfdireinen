"""Tests for DataLakeSource — reads a Hive-partitioned parquet lake (survivor + delisted).

Builds a tiny synthetic parquet lake in the real layout (`{ticker}/year=YYYY/data.parquet`)
in a temp dir, so the reader + delist inference are proven without the real data.
Skips cleanly if no parquet engine is installed.

    python tests/test_datalake_source.py
    pytest tests/test_datalake_source.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data import DataFetchError, synthetic_delisted_ohlcv, synthetic_survivor_ohlcv  # noqa: E402
from vpts.data.sources import DataLakeSource  # noqa: E402

_HAS_PARQUET = True
try:
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover
    try:
        import fastparquet  # noqa: F401
    except Exception:
        _HAS_PARQUET = False


def _write_lake(root: Path, frames: dict) -> None:
    """Write {ticker}/year=YYYY/data.parquet for each frame (Date as index column)."""
    for sym, df in frames.items():
        for year, g in df.groupby(df.index.year):
            d = root / sym / f"year={year}"
            d.mkdir(parents=True, exist_ok=True)
            out = g.copy()
            out.index.name = "Date"
            out.reset_index().to_parquet(d / "data.parquet", index=False)


def _build_lake():
    tmp = Path(tempfile.mkdtemp())
    # 3 survivors (end recent) + 1 delisted (ends ~3y earlier).
    surv = {f"S{i}": synthetic_survivor_ohlcv(900, seed=i, start_date="2014-01-01")
            for i in range(3)}
    dead = {"DEADCO": synthetic_delisted_ohlcv(400, seed=99, start_date="2014-01-01",
                                               terminal_frac=0.08)}
    _write_lake(tmp, {**surv, **dead})
    return tmp


# --------------------------------------------------------------------------- #
def test_reads_partitioned_parquet_and_concats_years() -> None:
    if not _HAS_PARQUET:
        print("  (no parquet engine — skipped)"); return
    root = _build_lake()
    src = DataLakeSource(root)
    df = src.get_bars("S0", period="max")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing
    assert len(df) > 800                                  # multiple year-partitions concatenated
    assert src.capabilities.provides_delisted is True


def test_available_symbols_and_missing() -> None:
    if not _HAS_PARQUET:
        return
    src = DataLakeSource(_build_lake())
    assert set(src.available_symbols()) == {"S0", "S1", "S2", "DEADCO"}
    try:
        src.get_bars("NOPE")
    except DataFetchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError for missing symbol")


def test_build_universe_infers_delisted() -> None:
    if not _HAS_PARQUET:
        return
    src = DataLakeSource(_build_lake(), active_gap_days=180)
    u = src.build_universe()
    assert set(u.symbols) == {"S0", "S1", "S2", "DEADCO"}
    assert u.delisted() == ["DEADCO"]                     # ends ~3y before the survivors
    assert set(u.survivors()) == {"S0", "S1", "S2"}
    assert u.survivorship_free is True
    assert src.is_delisted("DEADCO") is True and src.is_delisted("S0") is False


def test_explicit_delisted_overrides_inference() -> None:
    if not _HAS_PARQUET:
        return
    src = DataLakeSource(_build_lake(), delisted=["S1"])
    assert src.is_delisted("S1") is True and src.is_delisted("DEADCO") is False
    assert src.build_universe().delisted() == ["S1"]


def test_date_range_slice() -> None:
    if not _HAS_PARQUET:
        return
    src = DataLakeSource(_build_lake())
    df = src.get_bars("S0", start="2015-01-01", end="2015-12-31")
    assert df.index.min() >= pd.Timestamp("2015-01-01")
    assert df.index.max() <= pd.Timestamp("2015-12-31")


def test_is_delisted_works_before_build_universe() -> None:
    if not _HAS_PARQUET:
        return
    src = DataLakeSource(_build_lake(), active_gap_days=180)
    # No build_universe() call first — inference must still work (ordering-hazard fix).
    assert src.is_delisted("DEADCO") is True
    assert src.is_delisted("S0") is False


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} datalake tests (parquet engine: {_HAS_PARQUET}) …\n")
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
