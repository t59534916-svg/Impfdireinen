"""Free Yahoo-Finance source — a thin :class:`DataSource` over the Phase-1 fetcher.

Capabilities are honest about the limitation that matters most: Yahoo serves
**survivors only**. A name that has delisted disappears from its history, so
``provides_delisted=False``. This is *why* the project's research wall exists, and
the registry/universe layers use this flag to warn rather than silently bias.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from vpts.data.base import DataSource, DataSourceCapabilities, validate_ohlcv
from vpts.data.fetcher import MarketDataFetcher


class YFinanceSource(DataSource):
    """Adapt :class:`~vpts.data.fetcher.MarketDataFetcher` to the source contract.

    Parameters are forwarded to the underlying fetcher, so caching, retries and
    interval clamping all still apply. Pass an existing fetcher via *fetcher* to
    share a cache directory.
    """

    def __init__(self, fetcher: Optional[MarketDataFetcher] = None, **fetcher_kwargs) -> None:
        self._fetcher = fetcher or MarketDataFetcher(**fetcher_kwargs)

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(
            name="yfinance",
            is_free=True,
            requires_api_key=False,
            provides_delisted=False,   # survivors only — the core confound
            provides_intraday=True,
            provides_fundamentals=False,
            provides_point_in_time_universe=False,
        )

    def get_bars(
        self,
        symbol: str,
        *,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self._fetcher.fetch(
            symbol, period=period, interval=interval, start=start, end=end
        )
        return validate_ohlcv(df, symbol=symbol, min_bars=self._fetcher.min_bars)
