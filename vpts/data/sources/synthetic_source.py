"""Deterministic, offline data source — survivors *and* delisted names.

Unlike every free live feed, :class:`SyntheticSource` can serve **delisted**
history (``provides_delisted=True``), because it generates it. That makes it the
one source you can use to exercise the entire survivorship-aware pipeline without
a paid provider — invaluable for tests, demos, and the injection stress harness.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import pandas as pd

from vpts.data.base import DataSource, DataSourceCapabilities, validate_ohlcv
from vpts.data.synthetic import synthetic_delisted_ohlcv, synthetic_survivor_ohlcv

#: Approximate business-day counts per ``period`` string (for length only).
_PERIOD_BARS = {
    "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504,
    "5y": 1260, "10y": 2520, "max": 1260, "ytd": 180,
}


class SyntheticSource(DataSource):
    """Generate reproducible OHLCV for arbitrary symbols.

    Parameters
    ----------
    delisted:
        Symbols to serve as *delisted* (declining-to-pennies) paths. Any symbol
        not listed is served as a survivor path.
    start_date:
        First bar date for every generated series.
    """

    def __init__(
        self,
        delisted: Optional[Iterable[str]] = None,
        *,
        start_date: str = "2015-01-02",
    ) -> None:
        self._delisted = {s.upper() for s in (delisted or ())}
        self._start_date = start_date

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(
            name="synthetic",
            is_free=True,
            requires_api_key=False,
            provides_delisted=True,
            provides_intraday=False,
            provides_fundamentals=False,
            provides_point_in_time_universe=True,
        )

    def is_delisted(self, symbol: str) -> bool:
        return symbol.upper() in self._delisted

    @staticmethod
    def _seed_for(symbol: str) -> int:
        # Stable per-symbol seed so the same ticker is reproducible across *processes*.
        # NB: the builtin hash() is salted per-process (PYTHONHASHSEED); use a fixed
        # digest so the "deterministic" guarantee actually holds in CI and elsewhere.
        digest = hashlib.sha1(symbol.upper().encode()).hexdigest()
        return int(digest[:8], 16)

    def get_bars(
        self,
        symbol: str,
        *,
        period: str = "2y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        if start and end:
            n = max(int(len(pd.bdate_range(start, end))), 30)
            start_date = start
        else:
            n = _PERIOD_BARS.get(period.lower(), 504)
            start_date = self._start_date
        gen = synthetic_delisted_ohlcv if self.is_delisted(symbol) else synthetic_survivor_ohlcv
        df = gen(n, seed=self._seed_for(symbol), start_date=start_date)
        df.attrs["symbol"] = symbol
        df.attrs["synthetic_delisted"] = self.is_delisted(symbol)
        return validate_ohlcv(df, symbol=symbol, min_bars=30)
