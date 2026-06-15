"""Stooq data source — free daily OHLCV that *retains delisted* US names.

Stooq is one of the few free feeds that keeps history for names that have since
delisted, so it advertises ``provides_delisted=True`` — the survivorship escape
hatch `RESEARCH.md` keeps asking for. Two access paths:

* **Live single-symbol CSV** — ``https://stooq.com/q/d/l/?s=<sym>&i=d``. Free and
  key-less, but Stooq now gates this endpoint behind a **JavaScript proof-of-work
  anti-bot wall** and serves an HTML challenge instead of CSV to non-browser clients
  (verified: it challenges even live tickers). When that happens this source raises a
  clear :class:`~vpts.data.fetcher.DataFetchError` rather than parsing the HTML as
  data — it does **not** attempt to defeat the wall.
* **Local bulk export** — point ``bulk_root`` at a Stooq database download
  (``d_us_txt`` and friends; the bulk archive is a Stooq Premium feature, HTTP 401
  otherwise). Files are parsed offline in either the simple ``Date,Open,…`` CSV or the
  bulk ``<TICKER>,<PER>,<DATE>,…`` layout — no network, no wall.

Testing
-------
The HTTP layer is **injectable** via *http_get* (``(url) -> str``), so CSV parsing,
wall detection and the bulk reader are unit-tested with no network — mirroring
:class:`~vpts.data.sources.polygon_source.PolygonSource`.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from vpts.data.base import DataSource, DataSourceCapabilities, validate_ohlcv
from vpts.data.fetcher import DataFetchError

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class StooqSource(DataSource):
    """Free daily OHLCV from Stooq (live CSV or a local bulk export).

    Parameters
    ----------
    http_get:
        Callable ``(url) -> str`` returning the response body. Defaults to a stdlib
        ``urllib`` GET (no new dependency). Inject a stub to test offline.
    bulk_root:
        Directory of a downloaded Stooq bulk database. When set, :meth:`get_bars`
        reads ``<root>/<symbol>.txt`` / ``.csv`` (recursively) instead of the network.
    suffix:
        Market suffix appended to bare symbols (Stooq uses ``.us`` for US equities).
    delisted:
        Optional explicit set of known-delisted symbols for :meth:`is_delisted`.
    """

    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(
        self,
        *,
        http_get: Optional[Callable[[str], str]] = None,
        bulk_root: Optional[str | Path] = None,
        suffix: str = ".us",
        delisted: Optional[Iterable[str]] = None,
        timeout: float = 30.0,
    ) -> None:
        self._http_get = http_get or self._urllib_get
        self.bulk_root = Path(bulk_root) if bulk_root else None
        if self.bulk_root is not None and not self.bulk_root.exists():
            raise ValueError(f"stooq bulk_root does not exist: {self.bulk_root}")
        self.suffix = suffix
        self._explicit_delisted = {s.upper() for s in delisted} if delisted is not None else None
        self.timeout = float(timeout)

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(
            name="stooq",
            is_free=True,
            requires_api_key=False,
            provides_delisted=True,          # Stooq retains delisted US names — the point
            provides_intraday=False,
            provides_fundamentals=False,
            provides_point_in_time_universe=False,
        )

    # ------------------------------------------------------------------ #
    def _urllib_get(self, url: str) -> str:  # pragma: no cover - needs network
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8", "replace")

    def stooq_symbol(self, symbol: str) -> str:
        """Map a bare ticker to Stooq's convention (lower-case, ``.us`` suffix)."""
        s = symbol.strip().lower()
        return s if "." in s else f"{s}{self.suffix}"

    # ------------------------------------------------------------------ #
    def _bulk_file(self, symbol: str) -> Optional[Path]:
        sym = self.stooq_symbol(symbol)
        stem = sym.split(".")[0]
        for cand in (sym, stem):
            for ext in (".txt", ".csv"):
                hits = list(self.bulk_root.rglob(f"{cand}{ext}"))
                if hits:
                    return sorted(hits)[0]
        return None

    @staticmethod
    def _parse(text: str, symbol: str) -> pd.DataFrame:
        """Parse either the simple ``Date,Open,…`` CSV or the bulk ``<TICKER>,…`` layout."""
        head = text.lstrip()[:400].lower()
        # An anti-bot/JS challenge or any HTML is NOT data — fail loudly. Detect HTML
        # specifically so Stooq's bulk "<TICKER>,<PER>,…" header is NOT misread as HTML.
        is_html = (head.startswith("<!") or head.startswith("<html") or "<!doctype" in head
                   or "<body" in head or "javascript" in head)
        if is_html:
            raise DataFetchError(
                f"{symbol}: Stooq returned an anti-bot/JS-challenge page, not CSV "
                "(the live endpoint is gated for non-browser clients; use a bulk_root export)."
            )
        if not text.strip() or "no data" in head or "brak danych" in head:
            raise DataFetchError(f"{symbol}: Stooq returned no data.")
        df = pd.read_csv(io.StringIO(text))
        cols = {c.lower().strip("<>"): c for c in df.columns}
        if "date" not in cols:
            raise DataFetchError(f"{symbol}: unrecognised Stooq columns {list(df.columns)}.")
        df = df.rename(columns={cols["date"]: "Date"})
        # Bulk format carries <OPEN>..<VOL>; simple format Open..Volume. Normalise.
        ren = {}
        for want, keys in {"Open": ("open",), "High": ("high",), "Low": ("low",),
                           "Close": ("close",), "Volume": ("vol", "volume")}.items():
            for k in keys:
                if k in cols:
                    ren[cols[k]] = want
                    break
        df = df.rename(columns=ren)
        missing = [c for c in _OHLCV if c not in df.columns]
        if missing:
            raise DataFetchError(f"{symbol}: Stooq data missing {missing}.")
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format="mixed", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        df = df[_OHLCV].apply(pd.to_numeric, errors="coerce")
        df.attrs["symbol"] = symbol
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
        if interval.lower() not in ("1d", "d", "daily"):
            raise ValueError("StooqSource serves daily bars only (interval='1d').")
        if self.bulk_root is not None:
            path = self._bulk_file(symbol)
            if path is None:
                raise DataFetchError(f"{symbol}: not found under stooq bulk_root {self.bulk_root}.")
            text = path.read_text("utf-8", "replace")
        else:
            url = f"{self.BASE_URL}?s={self.stooq_symbol(symbol)}&i=d"
            try:
                text = self._http_get(url)
            except DataFetchError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalise transport errors
                raise DataFetchError(f"{symbol}: Stooq request failed: {exc}") from exc
        df = self._parse(text, symbol)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return validate_ohlcv(df, symbol=symbol, min_bars=1)

    # ------------------------------------------------------------------ #
    def is_delisted(self, symbol: str) -> Optional[bool]:
        if self._explicit_delisted is None:
            return None
        return symbol.upper() in self._explicit_delisted
