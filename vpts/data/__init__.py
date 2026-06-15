"""Data access package (free sources only).

Phase 1 shipped a robust :class:`~vpts.data.fetcher.MarketDataFetcher` around
``yfinance``. This package now also provides a **provider-agnostic layer** so the
pipeline is no longer welded to one feed and can express the question that
actually matters — *point-in-time / delisted data*:

* :class:`~vpts.data.base.DataSource` — the one-method provider contract, with
  honest :class:`~vpts.data.base.DataSourceCapabilities` (notably
  ``provides_delisted``).
* :class:`~vpts.data.sources.YFinanceSource` / :class:`~vpts.data.sources.SyntheticSource`
  — a free survivor-only feed and a deterministic offline feed that *can* mint
  delisted paths.
* :class:`~vpts.data.registry.SourceRegistry` — ordered, capability-aware
  fallback chain (:func:`~vpts.data.registry.default_registry`).
* :class:`~vpts.data.universe.Universe` / :class:`~vpts.data.universe.SurvivorshipInjector`
  — point-in-time membership and the survivorship-injection stress harness.
"""
from __future__ import annotations

from vpts.data.base import (
    OHLCV_COLUMNS,
    DataSource,
    DataSourceCapabilities,
    validate_ohlcv,
)
from vpts.data.coverage import (
    KNOWN_DELISTED,
    CoverageReport,
    DelistedTicker,
    SymbolCoverage,
    audit_coverage,
    audit_known_delisted,
)
from vpts.data.fetcher import (
    DataFetchError,
    InsufficientDataError,
    MarketDataFetcher,
    NoVolumeError,
)
from vpts.data.registry import SourceRegistry, default_registry
from vpts.data.sources import DataLakeSource, PolygonSource, SyntheticSource, YFinanceSource
from vpts.data.synthetic import synthetic_delisted_ohlcv, synthetic_survivor_ohlcv
from vpts.data.universe import (
    InjectionResult,
    Membership,
    SurvivorshipInjector,
    Universe,
)

__all__ = [
    # Phase-1 fetcher
    "MarketDataFetcher",
    "DataFetchError",
    "InsufficientDataError",
    "NoVolumeError",
    # source abstraction
    "DataSource",
    "DataSourceCapabilities",
    "OHLCV_COLUMNS",
    "validate_ohlcv",
    "YFinanceSource",
    "SyntheticSource",
    "PolygonSource",
    "DataLakeSource",
    "SourceRegistry",
    "default_registry",
    # survivorship-coverage audit (measure the bias on a live feed)
    "audit_coverage",
    "audit_known_delisted",
    "CoverageReport",
    "SymbolCoverage",
    "DelistedTicker",
    "KNOWN_DELISTED",
    # synthetic generators
    "synthetic_survivor_ohlcv",
    "synthetic_delisted_ohlcv",
    # point-in-time universe + survivorship
    "Universe",
    "Membership",
    "SurvivorshipInjector",
    "InjectionResult",
]
