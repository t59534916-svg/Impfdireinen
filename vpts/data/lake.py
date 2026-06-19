"""Materialize a delisted-inclusive parquet lake from any DataSource.

This is the **ingestion backbone** of the data-first track: pull a universe (survivors
*and* delisted names) from any :class:`~vpts.data.base.DataSource` and write it in the
Hive-partitioned layout (``{root}/{TICKER}/year=YYYY/data.parquet``) that
:class:`~vpts.data.sources.datalake_source.DataLakeSource` reads back. Once a real
delisted-capable source exists (a paid Polygon/FMP key, or a local export), this turns
the free-feed survivorship wall into a one-command, survivorship-free dataset the harness
can re-run on immediately.

It is source-agnostic and offline-testable: point it at a
:class:`~vpts.data.sources.synthetic_source.SyntheticSource` and it builds a real parquet
lake with no network. Writing parquet needs ``pyarrow`` (``pip install vpts[parquet]``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from vpts.data.base import DataSource
from vpts.data.fetcher import DataFetchError

logger = logging.getLogger(__name__)

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True)
class LakeBuildReport:
    """Outcome of a lake materialization — coverage, not just success/failure."""

    root: str
    written: tuple[str, ...]
    missing: tuple[str, ...]
    delisted_written: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_written(self) -> int:
        return len(self.written)

    @property
    def coverage_pct(self) -> float:
        total = len(self.written) + len(self.missing)
        return 100.0 * len(self.written) / total if total else 0.0

    def summary(self) -> str:
        lines = [
            f"Lake built at {self.root}",
            f"  wrote {self.n_written} / {self.n_written + len(self.missing)} "
            f"symbols ({self.coverage_pct:.0f}% reachable)",
        ]
        if self.delisted_written:
            lines.append(f"  delisted names captured: {len(self.delisted_written)} "
                         f"({', '.join(self.delisted_written[:8])}"
                         f"{' …' if len(self.delisted_written) > 8 else ''})")
        if self.missing:
            lines.append(f"  unreachable: {', '.join(self.missing[:12])}"
                         f"{' …' if len(self.missing) > 12 else ''}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"root": self.root, "written": list(self.written),
                "missing": list(self.missing), "delisted_written": list(self.delisted_written),
                "coverage_pct": self.coverage_pct}


def _write_symbol(root: Path, symbol: str, df: pd.DataFrame) -> None:
    """Write one symbol's frame as ``{root}/{symbol}/year=YYYY/data.parquet``."""
    cols = [c for c in _OHLCV if c in df.columns]
    for year, g in df[cols].groupby(df.index.year):
        d = root / symbol / f"year={int(year)}"
        d.mkdir(parents=True, exist_ok=True)
        out = g.copy()
        out.index.name = "Date"
        out.reset_index().to_parquet(d / "data.parquet", index=False)


def materialize_lake(
    root: str | Path,
    source: DataSource,
    symbols: Iterable[str],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    delist_caps: Optional[Mapping[str, str]] = None,
    delisted: Optional[Iterable[str]] = None,
) -> LakeBuildReport:
    """Pull *symbols* from *source* and write a parquet lake at *root*.

    Parameters
    ----------
    delist_caps:
        ``{symbol: 'YYYY-MM-DD'}`` — cap each name's fetch window at its delisting date
        so only genuine **pre-delisting** history is stored (and a reused ticker can't
        splice a different company's later life onto a dead identity).
    delisted:
        Optional set of symbols known to be delisted, recorded in the report so coverage
        of the *death leg* — the survivorship driver — is visible at a glance.

    A symbol that the source can't serve is recorded under ``missing`` and skipped; the
    build never aborts on one bad name. Returns a :class:`LakeBuildReport`.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    caps = dict(delist_caps or {})
    dead = {s.upper() for s in (delisted or [])}
    written: list[str] = []
    missing: list[str] = []
    dead_written: list[str] = []

    for sym in symbols:
        sym_end = min([end, caps[sym]], key=lambda d: pd.Timestamp(d)) if (end and sym in caps) \
            else (caps.get(sym) or end)
        try:
            df = source.get_bars(sym, period="max", interval="1d", start=start, end=sym_end)
        except DataFetchError as exc:
            logger.info("lake: %s unreachable (%s)", sym, exc)
            missing.append(sym)
            continue
        except Exception as exc:  # noqa: BLE001 - isolate a misbehaving source
            logger.warning("lake: %s raised (%s)", sym, exc)
            missing.append(sym)
            continue
        if df is None or df.empty:
            missing.append(sym)
            continue
        _write_symbol(root, sym, df)
        written.append(sym)
        if sym.upper() in dead:
            dead_written.append(sym)

    return LakeBuildReport(root=str(root), written=tuple(written),
                           missing=tuple(missing), delisted_written=tuple(dead_written))
