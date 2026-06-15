"""Concrete alt-signal providers: a null default and a static (test/injection) one.

No live providers ship here — wiring a real options-flow or sentiment API is a
matter of subclassing :class:`~vpts.altdata.base.AltSignalSource` and overriding
the relevant method to hit that feed (causally). These two cover the offline cases.
"""
from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from vpts.altdata.base import AltDataCapabilities, AltSignalSource


class NullAltSource(AltSignalSource):
    """The default: advertises nothing and returns all-NaN series.

    Lets the pipeline always *ask* for options flow / sentiment and degrade cleanly
    when no provider is configured — the merge helper simply adds no columns.
    """

    @property
    def capabilities(self) -> AltDataCapabilities:
        return AltDataCapabilities(name="null", has_options_flow=False, has_sentiment=False,
                                   is_free=True, requires_api_key=False)


class StaticAltSource(AltSignalSource):
    """Serve pre-built per-symbol series — for tests, backfills, or injected studies.

    Parameters
    ----------
    options_flow, sentiment:
        Optional ``{symbol: Series}`` maps. The series are assumed **causal** and
        are reindexed onto the requested index (forward-filled within the provided
        span; values outside it stay NaN).
    """

    def __init__(
        self,
        *,
        options_flow: Optional[Mapping[str, pd.Series]] = None,
        sentiment: Optional[Mapping[str, pd.Series]] = None,
        name: str = "static",
    ) -> None:
        self._of = dict(options_flow or {})
        self._sent = dict(sentiment or {})
        self._name = name

    @property
    def capabilities(self) -> AltDataCapabilities:
        return AltDataCapabilities(
            name=self._name,
            has_options_flow=bool(self._of),
            has_sentiment=bool(self._sent),
            is_free=True,
            requires_api_key=False,
        )

    def _serve(self, store: dict, symbol: str, index: pd.Index) -> pd.Series:
        if symbol not in store:
            return self._nan_series(index)
        # reindex onto the target index; ffill only within the provided span (causal).
        return store[symbol].reindex(index).ffill()

    def options_flow(self, symbol: str, index: pd.Index) -> pd.Series:
        return self._serve(self._of, symbol, index)

    def sentiment(self, symbol: str, index: pd.Index) -> pd.Series:
        return self._serve(self._sent, symbol, index)
