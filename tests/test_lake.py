"""Tests for the lake materializer (vpts.data.materialize_lake).

Proves the ingestion backbone end-to-end with NO network: materialize a parquet lake
from a SyntheticSource (survivors + a delisted name), then read it back through
DataLakeSource and confirm the survivorship-free universe. Skips cleanly if no parquet
engine is installed.

    python tests/test_lake.py
    pytest tests/test_lake.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data import DataFetchError, materialize_lake  # noqa: E402
from vpts.data.base import DataSource, DataSourceCapabilities  # noqa: E402
from vpts.data.sources import DataLakeSource, SyntheticSource  # noqa: E402

_HAS_PARQUET = True
try:
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover
    try:
        import fastparquet  # noqa: F401
    except Exception:
        _HAS_PARQUET = False


class _PartialSource(DataSource):
    """SyntheticSource that 'delists' some symbols by refusing them (to test coverage)."""

    def __init__(self, refuse: set[str]) -> None:
        self._syn = SyntheticSource()
        self._refuse = refuse

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return self._syn.capabilities

    def get_bars(self, symbol, *, period="max", interval="1d", start=None, end=None):
        if symbol in self._refuse:
            raise DataFetchError(f"{symbol}: not served (simulated).")
        return self._syn.get_bars(symbol, period=period, interval=interval, start=start, end=end)


# --------------------------------------------------------------------------- #
def test_materialize_round_trips_through_datalake() -> None:
    if not _HAS_PARQUET:
        print("  (no parquet engine — skipped)"); return
    root = Path(tempfile.mkdtemp())
    syms = ["S0", "S1", "S2", "DEADCO"]
    rep = materialize_lake(root, SyntheticSource(), syms,
                           start="2014-01-01", end="2018-12-31",
                           delist_caps={"DEADCO": "2016-06-30"}, delisted=["DEADCO"])
    assert rep.n_written == 4 and rep.coverage_pct == 100.0
    assert rep.delisted_written == ("DEADCO",)

    # Read it back through the real DataLakeSource.
    src = DataLakeSource(root, active_gap_days=180)
    df = src.get_bars("S0", period="max")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    u = src.build_universe()
    assert set(u.symbols) == {"S0", "S1", "S2", "DEADCO"}
    assert u.delisted() == ["DEADCO"]                 # capped at 2016 → inferred dead
    assert src.is_delisted("DEADCO") is True


def test_delist_cap_truncates_history() -> None:
    if not _HAS_PARQUET:
        return
    root = Path(tempfile.mkdtemp())
    materialize_lake(root, SyntheticSource(), ["DEADCO"],
                     start="2014-01-01", end="2018-12-31", delist_caps={"DEADCO": "2016-06-30"})
    last = DataLakeSource(root).get_bars("DEADCO", period="max").index.max()
    assert last <= pd.Timestamp("2016-06-30")          # only pre-delisting history stored


def test_unreachable_symbols_recorded_not_fatal() -> None:
    if not _HAS_PARQUET:
        return
    root = Path(tempfile.mkdtemp())
    src = _PartialSource(refuse={"GONE1", "GONE2"})
    rep = materialize_lake(root, src, ["S0", "GONE1", "S1", "GONE2"])
    assert set(rep.written) == {"S0", "S1"} and set(rep.missing) == {"GONE1", "GONE2"}
    assert 0.0 < rep.coverage_pct < 100.0 and "unreachable" in rep.summary()


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} lake tests (parquet engine: {_HAS_PARQUET}) …\n")
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
