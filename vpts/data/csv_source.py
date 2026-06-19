"""Ingest OHLCV from CSV files — bring data you have the rights to into the pipeline.

The free feeds can't serve delisted names (the documented wall), but *you* can export
them: a broker download, a data-vendor CSV, or a site's own logged-in export. This turns
any such CSV into a :class:`~vpts.data.base.DataSource`, so it composes with the rest of
the backbone — :func:`~vpts.data.lake.materialize_lake` writes it into the parquet lake
that :class:`~vpts.data.sources.datalake_source.DataLakeSource` reads, and the harness
re-runs survivorship-free.

It is deliberately forgiving about real-world CSV mess:

* **delimiter** ``,`` or ``;`` (auto-detected from the header);
* **decimal comma** and **thousands dot** (German style, e.g. ``"1.234,56"``) or the
  US ``"1234.56"`` (auto-detected; override with *decimal*/*thousands*);
* **date formats** incl. ``DD.MM.YY`` (``dayfirst`` auto-on for ``;``-files) and ISO;
* **column names** in English or German via synonyms (``Datum/Schluss/Stücke`` …), or an
  explicit *column_map*.

⚠️ Two real-world gotchas it can't guess for you: a German "Volumen" column is often the
**€ turnover**, not share volume (share count is usually "Stücke" — auto-mapping prefers
it); and a foreign listing (e.g. a Stuttgart EUR line) is **not** the US USD series — keep
your universe on one venue/currency. Pass *column_map* when in doubt.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from vpts.data.base import DataSource, DataSourceCapabilities, validate_ohlcv
from vpts.data.fetcher import DataFetchError

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

#: Column-name synonyms (lower-cased), in priority order. Share *count* before € turnover.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Date":   ("date", "datum", "day", "tag", "zeit", "time"),
    "Open":   ("open", "erster", "eröffnung", "eroeffnung", "first", "open price"),
    "High":   ("high", "hoch", "höchst", "hoechst", "max"),
    "Low":    ("low", "tief", "tiefst", "niedrig", "min"),
    "Close":  ("close", "schluss", "schlusskurs", "closing", "last", "letzter", "adj close"),
    "Volume": ("volume", "stück", "stücke", "stueck", "stuecke", "stk", "units",
               "umsatz", "volumen", "vol"),
}


def _resolve_columns(columns, column_map: Optional[Mapping[str, str]]) -> dict[str, str]:
    """Map source columns → canonical OHLCV names (explicit *column_map* wins)."""
    lower = {str(c).strip().lower().strip('<>"'): c for c in columns}
    out: dict[str, str] = {}
    cmap = {k.title(): v for k, v in (column_map or {}).items()}
    for canon, syns in _SYNONYMS.items():
        if canon in cmap and cmap[canon] in columns:        # explicit override
            out[cmap[canon]] = canon
            continue
        for s in syns:
            if s in lower:
                out[lower[s]] = canon
                break
    return out


def read_ohlcv_csv(
    path: str | Path,
    *,
    symbol: Optional[str] = None,
    column_map: Optional[Mapping[str, str]] = None,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
    dayfirst: Optional[bool] = None,
    min_bars: int = 1,
) -> pd.DataFrame:
    """Read one OHLCV CSV → a clean, validated frame (DatetimeIndex, OHLCV columns).

    Delimiter, decimal/thousands and ``dayfirst`` are auto-detected for the common
    English (``,``/``.``) and German (``;``/``,``/``.``) layouts; pass them explicitly to
    override. Raises :class:`~vpts.data.fetcher.DataFetchError` on an unparseable or
    columns-missing file (so it fits the source-fallback contract).
    """
    name = symbol or Path(path).stem
    text = Path(path).read_text("utf-8", "replace")
    if not text.strip():
        raise DataFetchError(f"{name}: CSV is empty.")
    header = text.lstrip().splitlines()[0]
    german = header.count(";") >= header.count(",") and ";" in header
    sep = ";" if german else ","
    if decimal is None:
        decimal = "," if german else "."
    if thousands is None:
        thousands = "." if german else None
    if dayfirst is None:
        dayfirst = german

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal,
                         thousands=thousands, engine="python")
    except Exception as exc:  # noqa: BLE001 - unparseable as CSV
        raise DataFetchError(f"{name}: could not parse CSV ({exc}).") from exc

    ren = _resolve_columns(df.columns, column_map)
    df = df.rename(columns=ren)
    missing = [c for c in ("Date", *_OHLCV[:4]) if c not in df.columns]  # Date+OHLC required
    if missing:
        raise DataFetchError(
            f"{name}: CSV missing {missing} (saw columns {list(df.columns)}). "
            "Pass column_map={'Close': '<your column>', …} to map them explicitly."
        )
    if "Volume" not in df.columns:
        df["Volume"] = 1.0                                   # synthesize so validate passes

    df["Date"] = pd.to_datetime(df["Date"].astype(str), dayfirst=dayfirst,
                                format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df.index.name = None
    df = df[[c for c in _OHLCV if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    df.attrs["symbol"] = name
    return validate_ohlcv(df, symbol=name, min_bars=min_bars)


class CsvSource(DataSource):
    """Serve OHLCV from a directory of CSVs (``{root}/{SYMBOL}.csv`` / ``.txt``).

    Composes with the whole backbone: ``materialize_lake(lake, CsvSource(dir), symbols)``
    builds a parquet lake, or use it directly in a :class:`SourceRegistry`. Advertises
    ``provides_delisted=True`` because a CSV you supply can contain dead names — the
    whole point. Parsing keywords (``decimal``/``thousands``/``dayfirst``/``column_map``)
    are forwarded to :func:`read_ohlcv_csv`.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        delisted: Optional[Iterable[str]] = None,
        **parse_kwargs,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise ValueError(f"CsvSource root does not exist: {self.root}")
        self._explicit_delisted = {s.upper() for s in delisted} if delisted is not None else None
        self._parse_kwargs = parse_kwargs

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(
            name="csv",
            is_free=True,
            requires_api_key=False,
            provides_delisted=True,          # a user-supplied CSV can hold delisted names
            provides_intraday=False,
            provides_fundamentals=False,
            provides_point_in_time_universe=False,
        )

    def _file(self, symbol: str) -> Optional[Path]:
        for cand in (symbol, symbol.upper(), symbol.lower()):
            for ext in (".csv", ".txt"):
                p = self.root / f"{cand}{ext}"
                if p.exists():
                    return p
        hits = list(self.root.rglob(f"{symbol}.csv")) + list(self.root.rglob(f"{symbol}.txt"))
        return sorted(hits)[0] if hits else None

    def available_symbols(self) -> list[str]:
        """Every symbol with a CSV/TXT under the root (sorted, by file stem)."""
        return sorted({p.stem for p in self.root.rglob("*") if p.suffix.lower() in (".csv", ".txt")})

    def get_bars(
        self,
        symbol: str,
        *,
        period: str = "max",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        if interval.lower() not in ("1d", "d", "daily"):
            raise ValueError("CsvSource serves daily bars only (interval='1d').")
        path = self._file(symbol)
        if path is None:
            raise DataFetchError(f"{symbol}: no CSV/TXT found under {self.root}.")
        df = read_ohlcv_csv(path, symbol=symbol, **self._parse_kwargs)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return validate_ohlcv(df, symbol=symbol, min_bars=1)

    def is_delisted(self, symbol: str) -> Optional[bool]:
        if self._explicit_delisted is None:
            return None
        return symbol.upper() in self._explicit_delisted
