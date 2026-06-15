"""An ordered, capability-aware data-source registry with fallback.

Production data access is never one provider — it is a *chain*: try the primary,
fall back to the next on failure, and know which sources can answer which
questions (intraday? delisted?). :class:`SourceRegistry` is that chain. It catches
the single :class:`~vpts.data.fetcher.DataFetchError` family that every conforming
source raises, so a flaky or rate-limited provider degrades gracefully instead of
crashing the run.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from vpts.data.base import DataSource
from vpts.data.fetcher import DataFetchError

logger = logging.getLogger(__name__)


class SourceRegistry:
    """Hold data sources in priority order and serve bars with fallback.

    Parameters
    ----------
    sources:
        Sources in priority order (first = primary). Later ones are tried only if
        earlier ones fail.
    """

    def __init__(self, sources: list[DataSource]) -> None:
        if not sources:
            raise ValueError("SourceRegistry needs at least one DataSource.")
        self._sources = list(sources)

    # ------------------------------------------------------------------ #
    @property
    def sources(self) -> tuple[DataSource, ...]:
        return tuple(self._sources)

    def add(self, source: DataSource, *, primary: bool = False) -> "SourceRegistry":
        self._sources.insert(0, source) if primary else self._sources.append(source)
        return self

    def with_capability(self, attr: str) -> list[DataSource]:
        """Return sources whose capability flag *attr* is truthy (priority order)."""
        return [s for s in self._sources if getattr(s.capabilities, attr, False)]

    @property
    def has_delisted_source(self) -> bool:
        """Whether any registered source can serve delisted history.

        If this is ``False``, every backtest over this registry is survivorship-
        biased — the project's documented binding constraint. Callers should warn.
        """
        return any(s.capabilities.provides_delisted for s in self._sources)

    # ------------------------------------------------------------------ #
    def get_bars(
        self,
        symbol: str,
        *,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        require: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return bars for *symbol* from the first source that succeeds.

        Parameters
        ----------
        require:
            Optional capability flag name (e.g. ``"provides_delisted"``,
            ``"provides_intraday"``); only sources advertising it are tried.

        Raises
        ------
        DataFetchError
            If no eligible source returns valid data. The last error is chained.
        """
        candidates = self.with_capability(require) if require else self._sources
        if not candidates:
            raise DataFetchError(
                f"{symbol}: no registered source advertises capability {require!r}."
            )
        last_err: Optional[Exception] = None
        for src in candidates:
            try:
                df = src.get_bars(
                    symbol, period=period, interval=interval, start=start, end=end
                )
                df.attrs["source"] = src.name
                logger.info("Served %s from source %r (%d bars).", symbol, src.name, len(df))
                return df
            except DataFetchError as exc:
                last_err = exc
                logger.info("Source %r failed for %s: %s", src.name, symbol, exc)
            except Exception as exc:  # noqa: BLE001 - isolate a misbehaving source
                last_err = exc
                logger.warning("Source %r raised for %s: %s", src.name, symbol, exc)
        raise DataFetchError(
            f"{symbol}: all {len(candidates)} source(s) failed. Last error: {last_err}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        names = " -> ".join(s.name for s in self._sources)
        return f"<SourceRegistry [{names}] delisted={self.has_delisted_source}>"


def default_registry(cache_dir: str = ".cache/vpts") -> SourceRegistry:
    """Production-sensible default chain.

    If ``POLYGON_API_KEY`` is set, a :class:`~vpts.data.sources.PolygonSource` is
    placed **first** — it is the only source that can serve delisted / point-in-time
    history, so it makes the chain survivorship-capable. Otherwise the chain is
    yfinance (free, survivor-only) → synthetic (offline fallback for demos/tests),
    which is **not** survivorship-free for real tickers.
    """
    import os

    from vpts.data.sources import PolygonSource, SyntheticSource, YFinanceSource

    sources: list = []
    if os.environ.get("POLYGON_API_KEY"):
        sources.append(PolygonSource())          # delisted-capable → goes first
    sources.append(YFinanceSource(cache_dir=cache_dir))
    sources.append(SyntheticSource())
    return SourceRegistry(sources)
