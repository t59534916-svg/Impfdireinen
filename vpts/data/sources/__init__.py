"""Concrete :class:`~vpts.data.base.DataSource` implementations.

* :class:`~vpts.data.sources.yfinance_source.YFinanceSource` — free, survivor-only
  (wraps the Phase-1 :class:`~vpts.data.fetcher.MarketDataFetcher`).
* :class:`~vpts.data.sources.synthetic_source.SyntheticSource` — deterministic,
  offline, and able to mint *delisted* paths, so the whole stack (including the
  survivorship machinery) is testable with zero network.
"""
from __future__ import annotations

from vpts.data.sources.polygon_source import PolygonSource
from vpts.data.sources.synthetic_source import SyntheticSource
from vpts.data.sources.yfinance_source import YFinanceSource

__all__ = ["YFinanceSource", "SyntheticSource", "PolygonSource"]
