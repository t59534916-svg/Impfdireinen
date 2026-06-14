"""Tests for the provider-agnostic data layer: sources, registry, universe, injector.

All offline and deterministic — the synthetic source means the full survivorship
pipeline is exercised with zero network.

    python tests/test_data_layer.py
    pytest tests/test_data_layer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data import (  # noqa: E402
    DataFetchError,
    DataSource,
    DataSourceCapabilities,
    Membership,
    SourceRegistry,
    SurvivorshipInjector,
    SyntheticSource,
    Universe,
    synthetic_delisted_ohlcv,
    synthetic_survivor_ohlcv,
)
from vpts.data.base import validate_ohlcv  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic source + generators
# --------------------------------------------------------------------------- #
def test_synthetic_source_survivor_vs_delisted() -> None:
    src = SyntheticSource(delisted=["DEADCO"])
    surv = src.get_bars("GOODCO", period="2y")
    dead = src.get_bars("DEADCO", period="2y")
    assert list(surv.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert not src.is_delisted("GOODCO") and src.is_delisted("DEADCO")
    # The dead name ends far below where it started; the survivor does not collapse.
    assert dead["Close"].iloc[-1] < 0.5 * dead["Close"].iloc[0]
    assert surv["Close"].iloc[-1] > 0.5 * surv["Close"].iloc[0]
    assert src.capabilities.provides_delisted is True


def test_synthetic_is_deterministic() -> None:
    a = synthetic_survivor_ohlcv(200, seed=42)
    b = synthetic_survivor_ohlcv(200, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(synthetic_survivor_ohlcv(200, seed=43))


def test_ohlc_consistency() -> None:
    for gen in (synthetic_survivor_ohlcv, synthetic_delisted_ohlcv):
        df = gen(300, seed=1)
        assert (df["High"] >= df["Low"]).all()
        assert (df["High"] >= df["Close"]).all() and (df["High"] >= df["Open"]).all()
        assert (df["Low"] <= df["Close"]).all() and (df["Low"] <= df["Open"]).all()
        assert (df["Volume"] > 0).all()


# --------------------------------------------------------------------------- #
# Registry fallback + capability routing
# --------------------------------------------------------------------------- #
class _FailingSource(DataSource):
    """A source that always fails — to exercise the fallback path."""

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(name="always-fails", provides_intraday=True)

    def get_bars(self, symbol, *, period="6mo", interval="1d", start=None, end=None):
        raise DataFetchError("simulated outage")


def test_registry_falls_back_to_working_source() -> None:
    reg = SourceRegistry([_FailingSource(), SyntheticSource()])
    df = reg.get_bars("ANY", period="1y")
    assert len(df) > 0
    assert df.attrs["source"] == "synthetic"          # primary failed, fell back


def test_registry_all_fail_raises() -> None:
    reg = SourceRegistry([_FailingSource(), _FailingSource()])
    try:
        reg.get_bars("ANY")
    except DataFetchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError when all sources fail")


def test_registry_capability_gating() -> None:
    reg = SourceRegistry([SyntheticSource()])
    assert reg.has_delisted_source is True
    # No registered source advertises fundamentals → require= raises.
    try:
        reg.get_bars("ANY", require="provides_fundamentals")
    except DataFetchError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DataFetchError for unmet capability")
    # _FailingSource advertises intraday; routing to it then fails over.
    reg2 = SourceRegistry([SyntheticSource(), _FailingSource()])
    assert len(reg2.with_capability("provides_intraday")) == 1


def test_registry_requires_sources() -> None:
    try:
        SourceRegistry([])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty registry")


# --------------------------------------------------------------------------- #
# Point-in-time universe
# --------------------------------------------------------------------------- #
def test_universe_point_in_time_membership() -> None:
    u = Universe([
        Membership("ALIVE"),
        Membership("LATE", start=pd.Timestamp("2016-01-01")),
        Membership("DEAD", delist=pd.Timestamp("2015-06-30"), delisted=True),
    ])
    early = u.members_asof("2015-03-01")
    late = u.members_asof("2017-01-01")
    assert set(early) == {"ALIVE", "DEAD"}            # LATE not yet listed
    assert set(late) == {"ALIVE", "LATE"}             # DEAD already delisted
    assert u.survivors() == ["ALIVE", "LATE"] or set(u.survivors()) == {"ALIVE", "LATE"}
    assert u.delisted() == ["DEAD"]
    assert u.survivorship_free is True                # contains a delisted name


def test_universe_from_symbols_is_survivor_only() -> None:
    u = Universe.from_symbols(["A", "B", "C"])
    assert len(u) == 3 and u.survivorship_free is False
    assert set(u.members_asof("2020-01-01")) == {"A", "B", "C"}


def test_universe_roundtrip_frame() -> None:
    u = Universe([Membership("X"), Membership("Y", delist=pd.Timestamp("2018-01-01"),
                                              delisted=True, reason="merger")])
    u2 = Universe.from_frame(u.to_frame())
    assert set(u2.symbols) == {"X", "Y"} and u2.is_delisted("Y")


# --------------------------------------------------------------------------- #
# Survivorship injector
# --------------------------------------------------------------------------- #
def test_injector_augments_frames_and_universe() -> None:
    survivors = {f"S{i}": synthetic_survivor_ohlcv(300, seed=i) for i in range(4)}
    res = SurvivorshipInjector(n_delisted=3, seed=500).inject(survivors)
    assert res.n_survivors == 4 and res.n_delisted == 3
    assert len(res.frames) == 7
    assert set(res.universe.delisted()) == {"DEAD0", "DEAD1", "DEAD2"}
    assert res.universe.survivorship_free is True
    assert abs(res.delisted_fraction - 3 / 7) < 1e-9
    # Injected dead names carry a delist date at their last bar.
    dead0 = res.universe.membership("DEAD0")
    assert dead0.delisted and dead0.delist == res.frames["DEAD0"].index[-1]


def test_injector_zero_is_noop_universe_survivor_only() -> None:
    survivors = {"S0": synthetic_survivor_ohlcv(120, seed=0)}
    res = SurvivorshipInjector(n_delisted=0).inject(survivors)
    assert res.n_delisted == 0 and res.universe.survivorship_free is False


def test_injection_runs_through_structural_harness() -> None:
    """End-to-end: the injected universe flows into the existing ML harness."""
    from vpts.structure.dataset import build_structural_dataset

    survivors = {f"S{i}": synthetic_survivor_ohlcv(360, seed=i) for i in range(3)}
    res = SurvivorshipInjector(n_delisted=2, seed=700).inject(survivors)
    built = 0
    for sym, frame in res.frames.items():
        ds = build_structural_dataset(frame, lookback=120, horizon=20, stride=5, symbol=sym)
        assert ds.X.shape[0] == ds.y.shape[0] and ds.X.shape[1] == len(ds.feature_names)
        built += 1
    assert built == 5                                 # 3 survivors + 2 delisted


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} data-layer tests …\n")
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
