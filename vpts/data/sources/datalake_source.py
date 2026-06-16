"""Read a Hive-partitioned parquet **data lake** into the DataSource interface.

For a lake laid out as ``{root}/{TICKER}/year=YYYY/data.parquet`` (the structure of
the user's `data_lake/eod/{source}/daily/` tree), this turns the whole universe —
**including delisted names** — into a survivorship-aware source the existing harness
can consume. That is the one thing the free feeds couldn't give us.

Two things it adds beyond a plain reader:

* :meth:`available_symbols` enumerates the lake's universe from the directory tree.
* :meth:`build_universe` constructs a point-in-time :class:`~vpts.data.universe.Universe`
  with **inferred delist dates** — a name whose last bar predates the lake's global
  last date by more than ``active_gap_days`` is treated as delisted. That makes a
  survivorship-free backtest possible directly from the lake.

Reading parquet needs ``pyarrow`` (or ``fastparquet``) — install ``vpts[parquet]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from vpts.data.base import DataSource, DataSourceCapabilities, validate_ohlcv
from vpts.data.fetcher import DataFetchError
from vpts.data.universe import Membership, Universe

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataLakeSource(DataSource):
    """A parquet data lake (Hive-partitioned by ticker/year) as a DataSource.

    Parameters
    ----------
    root:
        Directory containing one subfolder per ticker (each with
        ``year=*/data.parquet``). For the user's lake this is e.g.
        ``data_lake/eod/yahoo_adjusted/daily`` or ``.../stooq/daily``.
    delisted:
        Optional explicit set of delisted symbols. If omitted, delisted status is
        *inferred* (see :meth:`build_universe` / :meth:`is_delisted`).
    active_gap_days:
        A symbol whose last bar is more than this many days before the lake's global
        last date is inferred delisted.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        delisted: Optional[Iterable[str]] = None,
        active_gap_days: int = 180,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise ValueError(f"data-lake root does not exist: {self.root}")
        self._explicit_delisted = {s.upper() for s in delisted} if delisted is not None else None
        self.active_gap_days = int(active_gap_days)
        self._last_date_cache: dict[str, pd.Timestamp] = {}
        self._global_last: Optional[pd.Timestamp] = None

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(
            name="datalake",
            is_free=True,
            requires_api_key=False,
            provides_delisted=True,          # the lake includes dead names — the whole point
            provides_intraday=False,
            provides_fundamentals=False,
            provides_point_in_time_universe=True,
        )

    # ------------------------------------------------------------------ #
    def _files(self, symbol: str) -> list[Path]:
        d = self.root / symbol
        files = sorted(d.glob("year=*/data.parquet"))
        if not files:                          # fall back to flatter layouts
            files = sorted(d.glob("*.parquet")) or sorted(self.root.glob(f"{symbol}.parquet"))
        return files

    @staticmethod
    def _read_one(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
        return df

    def get_bars(
        self,
        symbol: str,
        *,
        period: str = "max",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        files = self._files(symbol)
        if not files:
            raise DataFetchError(f"{symbol}: not found in data lake at {self.root}.")
        try:
            frames = [self._read_one(f) for f in files]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise DataFetchError(
                "Reading parquet needs pyarrow/fastparquet (pip install vpts[parquet])."
            ) from exc
        df = pd.concat(frames)
        df = df[[c for c in _OHLCV if c in df.columns]]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        df.attrs["symbol"] = symbol
        return validate_ohlcv(df, symbol=symbol, min_bars=1)

    # ------------------------------------------------------------------ #
    def available_symbols(self) -> list[str]:
        """Every ticker present in the lake (sorted)."""
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and self._files(p.name))

    def _last_date(self, symbol: str) -> Optional[pd.Timestamp]:
        if symbol in self._last_date_cache:
            return self._last_date_cache[symbol]
        files = self._files(symbol)
        if not files:
            return None
        last = self._read_one(files[-1]).index.max()
        self._last_date_cache[symbol] = last
        return last

    def _global_last_date(self) -> Optional[pd.Timestamp]:
        """Lake-wide latest bar date, computed and cached on demand.

        Decoupled from :meth:`build_universe` so :meth:`is_delisted` infers correctly
        regardless of call order (previously it returned ``None`` until a
        ``build_universe()`` call happened to populate ``_global_last``).
        """
        if self._global_last is None:
            lasts = [d for d in (self._last_date(s) for s in self.available_symbols())
                     if d is not None]
            self._global_last = max(lasts) if lasts else None
        return self._global_last

    def is_delisted(self, symbol: str) -> Optional[bool]:
        if self._explicit_delisted is not None:
            return symbol.upper() in self._explicit_delisted
        last = self._last_date(symbol)
        global_last = self._global_last_date()
        if last is None or global_last is None:
            return None
        return (global_last - last).days > self.active_gap_days

    def build_universe(self) -> Universe:
        """Point-in-time universe over the lake, with inferred delist dates.

        A symbol whose last bar predates the lake's global last date by more than
        ``active_gap_days`` is marked delisted (with ``delist`` = its last bar).
        Feed the result + ``get_bars`` straight into the survivorship-aware harness.
        """
        symbols = self.available_symbols()
        lasts = {s: self._last_date(s) for s in symbols}
        valid = {s: d for s, d in lasts.items() if d is not None}
        if not valid:
            raise DataFetchError("No readable symbols in the data lake.")
        self._global_last = max(valid.values())
        members = []
        for s, last in valid.items():
            dead = (self._explicit_delisted is not None and s.upper() in self._explicit_delisted) \
                or (self._explicit_delisted is None
                    and (self._global_last - last).days > self.active_gap_days)
            members.append(Membership(
                symbol=s,
                delist=pd.Timestamp(last) if dead else None,
                delisted=bool(dead),
                reason="inferred from last bar in data lake" if dead else "",
            ))
        return Universe(members)
