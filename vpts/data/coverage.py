"""Measure how survivorship-biased a data source actually is.

`RESEARCH.md`'s central finding is that the binding constraint on this whole
project is **survivorship bias**: free feeds silently drop names that have since
delisted, so any universe assembled from them is survivor-only and manufactures
an edge that inverts once dead names are present. The codebase already *flags*
that risk (:attr:`~vpts.data.base.DataSourceCapabilities.provides_delisted`) and
*simulates* it (:class:`~vpts.data.universe.SurvivorshipInjector`) — but it never
**measured** it on a live feed. Until you do, ``provides_delisted=False`` is an
assertion; this module turns it into a number.

Given a list of tickers *known* to have delisted and a fetch callable for any
source, :func:`audit_coverage` reports how many the source can still serve — the
source's empirical **survivorship signature**. Point it at a free feed and the
answer is stark and reproducible: ~0 of the bankruptcy/liquidation names come
back (they were the ones that went to zero — exactly the names that matter), which
is *why* `RESEARCH.md` must fall back to synthetic injection to probe the death
leg at all.

The fetch callable is injected, so the core logic is provider-agnostic and unit
-testable offline with a mock source; only the example wires it to a live feed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

import pandas as pd

from vpts.data.fetcher import DataFetchError

#: Categories used by the curated catalogue below.
BANKRUPTCY = "bankruptcy"      # went to (near) zero — liquidation / Ch.7/Ch.11 wipeout
ACQUISITION = "acquisition"    # left the universe via merger/buyout (often at a premium)


@dataclass(frozen=True)
class DelistedTicker:
    """One historically-delisted US name, with provenance for the audit."""

    symbol: str
    category: str   # BANKRUPTCY | ACQUISITION
    year: int
    name: str


#: A curated, verifiable catalogue of well-known US delistings, split into the two
#: economically distinct kinds. The **bankruptcy** leg is what drives the
#: survivorship mirage (those names went to zero); the **acquisition** leg left the
#: universe without crashing. A survivorship-free feed must serve both; a free feed
#: serves neither (acquisitions) or, at best, a quirky handful whose symbol persisted.
KNOWN_DELISTED: tuple[DelistedTicker, ...] = (
    # --- bankruptcy / liquidation (the names that went to ~0) ---
    DelistedTicker("ENE", BANKRUPTCY, 2001, "Enron"),
    DelistedTicker("WCOM", BANKRUPTCY, 2002, "WorldCom"),
    DelistedTicker("LEH", BANKRUPTCY, 2008, "Lehman Brothers"),
    DelistedTicker("WAMUQ", BANKRUPTCY, 2008, "Washington Mutual"),
    DelistedTicker("GGP", BANKRUPTCY, 2009, "General Growth Properties"),
    DelistedTicker("MTLQQ", BANKRUPTCY, 2009, "General Motors (old)"),
    DelistedTicker("CIT", BANKRUPTCY, 2009, "CIT Group"),
    DelistedTicker("ABK", BANKRUPTCY, 2010, "Ambac Financial"),
    DelistedTicker("GTATQ", BANKRUPTCY, 2014, "GT Advanced Technologies"),
    DelistedTicker("SHLDQ", BANKRUPTCY, 2018, "Sears Holdings"),
    DelistedTicker("HTZGQ", BANKRUPTCY, 2020, "Hertz Global"),
    # --- acquisition / merger (left the universe, did not crash) ---
    DelistedTicker("WFM", ACQUISITION, 2017, "Whole Foods Market (Amazon)"),
    DelistedTicker("LLTC", ACQUISITION, 2017, "Linear Technology (Analog Devices)"),
    DelistedTicker("BRCD", ACQUISITION, 2017, "Brocade (Broadcom)"),
    DelistedTicker("TWX", ACQUISITION, 2018, "Time Warner (AT&T)"),
    DelistedTicker("COL", ACQUISITION, 2018, "Rockwell Collins (United Technologies)"),
    DelistedTicker("MON", ACQUISITION, 2018, "Monsanto (Bayer)"),
    DelistedTicker("PCLN", ACQUISITION, 2018, "Priceline (renamed Booking)"),
    DelistedTicker("CELG", ACQUISITION, 2019, "Celgene (Bristol-Myers Squibb)"),
    DelistedTicker("STI", ACQUISITION, 2019, "SunTrust Banks (Truist)"),
    DelistedTicker("RTN", ACQUISITION, 2020, "Raytheon (merged into RTX)"),
    DelistedTicker("AGN", ACQUISITION, 2020, "Allergan (AbbVie)"),
    DelistedTicker("ARNC", ACQUISITION, 2020, "Arconic (Apollo)"),
    DelistedTicker("TIF", ACQUISITION, 2021, "Tiffany & Co. (LVMH)"),
    DelistedTicker("XLNX", ACQUISITION, 2022, "Xilinx (AMD)"),
    DelistedTicker("ATVI", ACQUISITION, 2023, "Activision Blizzard (Microsoft)"),
)

#: Status labels for one symbol's coverage outcome.
REACHABLE = "reachable"   # source served >= min_bars usable rows
DROPPED = "dropped"       # source has no usable history (the survivorship drop)
ERROR = "error"           # unexpected failure (network, parse) — not a clean verdict


@dataclass(frozen=True)
class SymbolCoverage:
    """One symbol's audit outcome."""

    symbol: str
    status: str            # REACHABLE | DROPPED | ERROR
    bars: int = 0
    first: Optional[str] = None
    last: Optional[str] = None
    category: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class CoverageReport:
    """Result of :func:`audit_coverage` — the source's survivorship signature."""

    results: tuple[SymbolCoverage, ...]
    min_bars: int

    def _by(self, status: str) -> list[SymbolCoverage]:
        return [r for r in self.results if r.status == status]

    @property
    def reachable(self) -> list[SymbolCoverage]:
        return self._by(REACHABLE)

    @property
    def dropped(self) -> list[SymbolCoverage]:
        return self._by(DROPPED)

    @property
    def errored(self) -> list[SymbolCoverage]:
        return self._by(ERROR)

    @property
    def reachable_fraction(self) -> float:
        """Share of *non-errored* symbols the source can serve (a clean denominator).

        Errors (network/parse) are excluded so a flaky connection doesn't masquerade
        as good coverage; if everything errored the fraction is ``nan``.
        """
        clean = [r for r in self.results if r.status != ERROR]
        return len(self._by(REACHABLE)) / len(clean) if clean else float("nan")

    def fraction_by_category(self) -> dict[str, float]:
        """Reachable fraction split by ``category`` (e.g. bankruptcy vs acquisition)."""
        out: dict[str, float] = {}
        cats = {r.category for r in self.results if r.category}
        for cat in sorted(cats):
            clean = [r for r in self.results if r.category == cat and r.status != ERROR]
            if clean:
                out[cat] = sum(r.status == REACHABLE for r in clean) / len(clean)
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"symbol": r.symbol, "category": r.category, "status": r.status,
              "bars": r.bars, "first": r.first, "last": r.last, "detail": r.detail}
             for r in self.results]
        )

    def summary(self) -> str:
        n = len(self.results)
        frac = self.reachable_fraction
        lines = [
            f"coverage: {len(self.reachable)}/{n} reachable "
            f"({frac:.0%} of non-errored), {len(self.dropped)} dropped, "
            f"{len(self.errored)} errored  (min_bars={self.min_bars})"
        ]
        for cat, f in self.fraction_by_category().items():
            lines.append(f"  {cat:12s}: {f:.0%} reachable")
        return "\n".join(lines)


def audit_coverage(
    symbols: Iterable,
    fetch_fn: Callable[[str], pd.DataFrame],
    *,
    min_bars: int = 60,
    categories: Optional[Mapping[str, str]] = None,
) -> CoverageReport:
    """Classify how many *symbols* a source can serve, via an injected *fetch_fn*.

    Parameters
    ----------
    symbols:
        Either plain ticker strings, or :class:`DelistedTicker` records (in which
        case their ``category`` is carried into the report automatically).
    fetch_fn:
        ``fetch_fn(symbol) -> DataFrame``. Should raise
        :class:`~vpts.data.fetcher.DataFetchError` (or return an empty frame) when
        the source has no usable history for the symbol — that is the survivorship
        drop we are counting. Any *other* exception is recorded as ``ERROR`` so a
        flaky network is never mistaken for a clean coverage verdict.
    min_bars:
        Minimum usable rows for a symbol to count as ``REACHABLE``.
    categories:
        Optional ``{symbol: category}`` map (ignored for :class:`DelistedTicker`
        inputs, which already carry one).

    Returns a :class:`CoverageReport`.
    """
    cat_map = dict(categories or {})
    results: list[SymbolCoverage] = []
    for item in symbols:
        if isinstance(item, DelistedTicker):
            sym, cat = item.symbol, item.category
        else:
            sym, cat = str(item), cat_map.get(str(item))
        try:
            df = fetch_fn(sym)
            if df is None or len(df) < min_bars:
                results.append(SymbolCoverage(
                    sym, DROPPED, bars=0 if df is None else len(df), category=cat,
                    detail=f"{0 if df is None else len(df)} bar(s) < min_bars",
                ))
                continue
            idx = pd.to_datetime(df.index)
            results.append(SymbolCoverage(
                sym, REACHABLE, bars=len(df), category=cat,
                first=str(idx.min().date()), last=str(idx.max().date()),
            ))
        except DataFetchError as exc:
            results.append(SymbolCoverage(sym, DROPPED, category=cat, detail=str(exc)[:80]))
        except Exception as exc:  # network/parse — not a clean verdict
            results.append(SymbolCoverage(sym, ERROR, category=cat,
                                          detail=f"{type(exc).__name__}: {str(exc)[:60]}"))
    return CoverageReport(results=tuple(results), min_bars=int(min_bars))


def audit_known_delisted(
    fetch_fn: Callable[[str], pd.DataFrame],
    *,
    catalog: Sequence[DelistedTicker] = KNOWN_DELISTED,
    min_bars: int = 60,
) -> CoverageReport:
    """Convenience wrapper: run :func:`audit_coverage` over :data:`KNOWN_DELISTED`."""
    return audit_coverage(catalog, fetch_fn, min_bars=min_bars)
