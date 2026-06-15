"""Integration points for alternative behavioral signals (options flow, sentiment).

These are **interfaces, not data**. The repo's free OHLCV layer can only *proxy*
order flow (synthetic delta) and crowd psychology (FOMO extension); the real
versions live in paid/alt feeds. This module defines where they plug in so the
behavioral feature set can be extended without rewiring the harness — and so a
provider that *does* have the data drops straight in.

Behavioral-finance rationale
----------------------------
* **Options flow** — dealer gamma/vanna positioning and put-call skew drive
  *mechanical* hedging flows (gamma squeezes, charm/vanna into expiry) and reveal
  where institutions are paying for protection vs. chasing upside. This is real
  positioning, not a price proxy.
* **Sentiment** — news/social tone and retail positioning proxy the crowd's
  emotional state (the FOMO and capitulation the price-based features only infer).

A provider returns **causal, per-bar** series aligned to a requested index; the
default :class:`AltSignalSource` methods return all-NaN, so a partial provider only
overrides what it actually has.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AltDataCapabilities:
    """What an alternative-signal provider supplies."""

    name: str
    has_options_flow: bool = False
    has_sentiment: bool = False
    is_free: bool = False
    requires_api_key: bool = True

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "has_options_flow": self.has_options_flow,
            "has_sentiment": self.has_sentiment,
            "is_free": self.is_free,
            "requires_api_key": self.requires_api_key,
        }


class AltSignalSource(ABC):
    """Base class for an alternative behavioral-signal provider.

    Subclasses must declare :meth:`capabilities` and override whichever of
    :meth:`options_flow` / :meth:`sentiment` they actually serve. Both default to a
    causal all-NaN series so callers can always request either and get a
    correctly-shaped result. Implementations MUST return series that use only
    information available at or before each bar (no look-ahead).
    """

    @property
    @abstractmethod
    def capabilities(self) -> AltDataCapabilities:
        ...

    @staticmethod
    def _nan_series(index: pd.Index) -> pd.Series:
        return pd.Series(np.nan, index=index, dtype=float)

    def options_flow(self, symbol: str, index: pd.Index) -> pd.Series:
        """Per-bar options-positioning signal (e.g. net dealer gamma / skew). NaN by default."""
        return self._nan_series(index)

    def sentiment(self, symbol: str, index: pd.Index) -> pd.Series:
        """Per-bar sentiment signal (news/social/retail positioning). NaN by default."""
        return self._nan_series(index)

    @property
    def name(self) -> str:
        return self.capabilities.name
