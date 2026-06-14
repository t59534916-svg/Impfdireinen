"""A provider-agnostic data-source interface.

Phase 1 hard-wired the pipeline to ``yfinance``. That is fine for free survivor
data, but it makes the single most important data question — *can I get
point-in-time / delisted history?* — impossible to even express. This module
introduces a thin abstraction so any provider (free or paid, live or synthetic)
plugs in behind one contract, and so the rest of the system can *ask* a source
what it can and cannot do.

The contract
------------
A :class:`DataSource` returns a clean OHLCV frame (``Open, High, Low, Close,
Volume``; timestamp index; sorted; no all-NaN rows) for a symbol, and advertises
its :class:`DataSourceCapabilities`. The crucial flag is
:attr:`DataSourceCapabilities.provides_delisted`: yfinance and most free feeds
silently drop names that have since delisted, which is the survivorship confound
the project's `RESEARCH.md` identifies as the binding constraint. A source that
sets this flag ``True`` is one you could actually use to settle the question.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

#: The canonical column set every source must return (in this order).
OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class DataSourceCapabilities:
    """What a data source can and cannot do — queried by the registry and the
    universe machinery to route requests and to flag survivorship risk."""

    name: str
    is_free: bool = True
    requires_api_key: bool = False
    #: True only if the source serves history for names that have since delisted.
    #: This is the survivorship-bias escape hatch; free feeds are almost all False.
    provides_delisted: bool = False
    provides_intraday: bool = False
    provides_fundamentals: bool = False
    provides_point_in_time_universe: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "is_free": self.is_free,
            "requires_api_key": self.requires_api_key,
            "provides_delisted": self.provides_delisted,
            "provides_intraday": self.provides_intraday,
            "provides_fundamentals": self.provides_fundamentals,
            "provides_point_in_time_universe": self.provides_point_in_time_universe,
        }


class DataSource(ABC):
    """Abstract base class for an OHLCV provider.

    Implementations must override :meth:`capabilities` and :meth:`get_bars`. The
    return value of :meth:`get_bars` is validated by :func:`validate_ohlcv`, so a
    conforming implementation only has to fetch and roughly shape the data.
    """

    @property
    @abstractmethod
    def capabilities(self) -> DataSourceCapabilities:
        """Static description of what this source supports."""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        *,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a clean OHLCV frame for *symbol* (see :data:`OHLCV_COLUMNS`)."""

    # Optional capability hook — sources that know their universe override this.
    def is_delisted(self, symbol: str) -> Optional[bool]:  # noqa: D401
        """Return whether *symbol* is delisted, or ``None`` if the source can't say."""
        return None

    @property
    def name(self) -> str:
        return self.capabilities.name

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        c = self.capabilities
        return f"<{type(self).__name__} name={c.name!r} delisted={c.provides_delisted}>"


def validate_ohlcv(df: pd.DataFrame, *, symbol: str, min_bars: int = 1) -> pd.DataFrame:
    """Validate/normalise a source's frame to the OHLCV contract.

    Raises :class:`~vpts.data.fetcher.DataFetchError` (or a subclass) on a frame
    that is empty, too short, missing required columns, or volume-less — so the
    registry's fallback logic can catch one exception type from every source.
    """
    from vpts.data.fetcher import DataFetchError, InsufficientDataError, NoVolumeError

    if df is None or len(df) == 0:
        raise DataFetchError(f"{symbol}: source returned no data.")
    missing = [c for c in ("High", "Low", "Close", "Volume") if c not in df.columns]
    if missing:
        raise DataFetchError(f"{symbol}: missing columns {missing}.")
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if len(out) < min_bars:
        raise InsufficientDataError(
            f"{symbol}: only {len(out)} bar(s) (need >= {min_bars})."
        )
    if float(pd.to_numeric(out["Volume"], errors="coerce").fillna(0).sum()) <= 0:
        raise NoVolumeError(f"{symbol}: data has no usable volume.")
    out.attrs.setdefault("symbol", symbol)
    return out
