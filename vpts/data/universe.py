"""Point-in-time universe + survivorship-injection — the survivorship escape hatch.

`RESEARCH.md`'s central finding is that the binding constraint is **survivorship
bias**: a universe assembled from names that exist *today* manufactures an edge
that inverts once delisted names are present. This module provides the two pieces
needed to confront that honestly:

* :class:`Universe` — point-in-time membership with **delist dates**, so a
  backtest can ask "who was tradable *as of* date *t*?" instead of "who survived
  to today?". With real point-in-time data this removes the bias outright; the
  abstraction exists so the rest of the system is ready for that data.
* :class:`SurvivorshipInjector` — since free feeds are survivor-only, this mixes
  synthetic *delisted* paths into a survivor set (the canonical `RESEARCH.md`
  stress), yielding an augmented ``{symbol: frame}`` mapping **and** a matching
  point-in-time :class:`Universe`. Any experiment can then be re-run "with
  injection" to see whether its edge is real or a survivorship artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from vpts.data.synthetic import synthetic_delisted_ohlcv


@dataclass(frozen=True)
class Membership:
    """One symbol's point-in-time membership window.

    ``start`` is the first date the name is tradable (``None`` = unbounded past);
    ``delist`` is the last tradable date (``None`` = still listed). ``delisted``
    is a convenience flag; ``reason`` is free-text provenance.
    """

    symbol: str
    start: Optional[pd.Timestamp] = None
    delist: Optional[pd.Timestamp] = None
    delisted: bool = False
    reason: str = ""

    def active_asof(self, date: pd.Timestamp) -> bool:
        if self.start is not None and date < self.start:
            return False
        if self.delist is not None and date > self.delist:
            return False
        return True


class Universe:
    """A collection of point-in-time :class:`Membership` records."""

    def __init__(self, memberships: Iterable[Membership]) -> None:
        self._members: dict[str, Membership] = {}
        for m in memberships:
            self._members[m.symbol] = m

    # -- construction ---------------------------------------------------- #
    @classmethod
    def from_symbols(cls, symbols: Iterable[str]) -> "Universe":
        """A survivor universe: every name listed for all time (no delistings)."""
        return cls(Membership(symbol=s) for s in symbols)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Universe":
        """Build from a DataFrame with columns ``symbol`` and optional
        ``start`` / ``delist`` / ``reason``."""
        if "symbol" not in df.columns:
            raise ValueError("Universe.from_frame needs a 'symbol' column.")
        out = []
        for _, row in df.iterrows():
            start = pd.Timestamp(row["start"]) if row.get("start") not in (None, "") and pd.notna(row.get("start")) else None
            delist = pd.Timestamp(row["delist"]) if row.get("delist") not in (None, "") and pd.notna(row.get("delist")) else None
            out.append(Membership(
                symbol=str(row["symbol"]), start=start, delist=delist,
                delisted=delist is not None, reason=str(row.get("reason", "")),
            ))
        return cls(out)

    # -- queries --------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._members)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._members

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._members)

    def membership(self, symbol: str) -> Membership:
        return self._members[symbol]

    def members_asof(self, date) -> list[str]:
        """Symbols tradable **as of** *date* — the point-in-time, bias-free list."""
        ts = pd.Timestamp(date)
        return [s for s, m in self._members.items() if m.active_asof(ts)]

    def survivors(self) -> list[str]:
        return [s for s, m in self._members.items() if not m.delisted]

    def delisted(self) -> list[str]:
        return [s for s, m in self._members.items() if m.delisted]

    def is_delisted(self, symbol: str) -> bool:
        return self._members[symbol].delisted

    @property
    def survivorship_free(self) -> bool:
        """True if the universe contains any delisted names (i.e. is not purely
        survivors). A purely-survivor universe is the documented confound."""
        return any(m.delisted for m in self._members.values())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"symbol": m.symbol, "start": m.start, "delist": m.delist,
              "delisted": m.delisted, "reason": m.reason}
             for m in self._members.values()]
        )


@dataclass(frozen=True)
class InjectionResult:
    """Output of :meth:`SurvivorshipInjector.inject` — augmented data + universe."""

    frames: dict          # {symbol: OHLCV DataFrame}, survivors + synthetic dead
    universe: Universe
    n_survivors: int
    n_delisted: int

    @property
    def delisted_fraction(self) -> float:
        total = self.n_survivors + self.n_delisted
        return self.n_delisted / total if total else 0.0


class SurvivorshipInjector:
    """Mix synthetic *delisted* paths into a survivor set for stress testing.

    Parameters
    ----------
    n_delisted:
        How many synthetic dead names to add.
    seed:
        Base seed; the k-th dead name uses ``seed + k`` (reproducible).
    label:
        Prefix for the synthetic symbols (default ``"DEAD"``).
    """

    def __init__(
        self,
        n_delisted: int = 10,
        *,
        seed: int = 100,
        label: str = "DEAD",
        delisted_fraction: Optional[float] = None,
        terminal_frac: Optional[float] = None,
        rally: str = "off",
    ) -> None:
        if n_delisted < 0:
            raise ValueError("n_delisted must be >= 0.")
        if delisted_fraction is not None and not 0.0 <= delisted_fraction < 1.0:
            raise ValueError("delisted_fraction must be in [0, 1).")
        self.n_delisted = int(n_delisted)
        self.seed = int(seed)
        self.label = str(label)
        self.delisted_fraction = delisted_fraction
        self.terminal_frac = terminal_frac
        self.rally = rally

    def _count(self, n_survivors: int) -> int:
        """How many dead names to add — from a target fraction if set, else n_delisted.

        ``delisted_fraction`` is the share of the *augmented* universe that should be
        delisted, so ``dead = round(f/(1-f) · survivors)`` — which can exceed the
        survivor count (a **loser-heavy** population), matching the empirical reality
        that most names underperform/delist over their lifetime (Bessembinder 2018).
        """
        if self.delisted_fraction is None:
            return self.n_delisted
        f = self.delisted_fraction
        return int(round(f / (1.0 - f) * n_survivors))

    def inject(self, survivors: dict[str, pd.DataFrame]) -> InjectionResult:
        """Return survivors + synthetic dead names and a point-in-time universe.

        Each dead name's length and calendar are matched to the *median* survivor
        length so the injection is comparable to the real data, and its
        :class:`Membership` carries a ``delist`` date at its last bar. The number of
        dead names is ``n_delisted`` (or derived from ``delisted_fraction``, which may
        make the population loser-heavy); ``terminal_frac`` sets the death severity.
        """
        if not survivors:
            raise ValueError("Need at least one survivor frame to inject into.")
        frames: dict[str, pd.DataFrame] = dict(survivors)
        n_bars = int(np.median([len(v) for v in survivors.values()]))   # true median length
        start_date = str(min(v.index[0] for v in survivors.values()).date())
        n_dead = self._count(len(survivors))

        members = [Membership(symbol=s) for s in survivors]  # survivors: no delist
        for k in range(n_dead):
            sym = f"{self.label}{k}"
            dead = synthetic_delisted_ohlcv(n_bars, seed=self.seed + k, start_date=start_date,
                                            terminal_frac=self.terminal_frac, rally=self.rally)
            frames[sym] = dead
            members.append(Membership(
                symbol=sym, start=pd.Timestamp(dead.index[0]),
                delist=pd.Timestamp(dead.index[-1]), delisted=True,
                reason="synthetic survivorship injection",
            ))
        return InjectionResult(
            frames=frames, universe=Universe(members),
            n_survivors=len(survivors), n_delisted=n_dead,
        )
