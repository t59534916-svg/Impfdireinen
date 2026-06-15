"""Polygon.io data source — the delisted / point-in-time escape hatch.

Unlike yfinance, Polygon serves history for **delisted** tickers and exposes a
point-in-time reference universe (each ticker carries ``active`` / ``delisted_utc``).
That is exactly the capability `RESEARCH.md` names as the binding constraint, so
this adapter advertises ``provides_delisted=True``.

Activation & testing
--------------------
Requires a key (``POLYGON_API_KEY`` env var or the *api_key* argument; paid tiers
for full delisted history — verify your plan's coverage). The HTTP layer is
**injectable** via *http_get*, so the parsing/■shape logic is unit-tested against
canned Polygon JSON with no network and no key. Without a key, calls raise a clear
:class:`~vpts.data.fetcher.DataFetchError` rather than failing obscurely.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, Optional

import pandas as pd

from vpts.data.base import DataSource, DataSourceCapabilities, validate_ohlcv
from vpts.data.fetcher import DataFetchError

#: ``interval`` → Polygon ``(multiplier, timespan)``.
_TIMESPAN: dict[str, tuple[int, str]] = {
    "1m": (1, "minute"), "5m": (5, "minute"), "15m": (15, "minute"),
    "30m": (30, "minute"), "1h": (1, "hour"), "1d": (1, "day"),
    "1wk": (1, "week"), "1mo": (1, "month"),
}

#: ``period`` → approximate calendar days (for deriving a from/to window).
_PERIOD_DAYS: dict[str, int] = {
    "1mo": 31, "3mo": 93, "6mo": 186, "1y": 366, "2y": 731,
    "5y": 1827, "10y": 3653, "ytd": 366, "max": 36525,
}


class PolygonSource(DataSource):
    """OHLCV (and delisted-status) from Polygon.io.

    Parameters
    ----------
    api_key:
        Polygon key; falls back to ``POLYGON_API_KEY``.
    http_get:
        Callable ``(url) -> dict`` returning parsed JSON. Defaults to a stdlib
        ``urllib`` fetch (no new dependency). Inject a stub to test offline.
    adjusted:
        Request split/dividend-adjusted bars (default True).
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        http_get: Optional[Callable[[str], dict]] = None,
        adjusted: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        self._http_get = http_get or self._urllib_get
        self.adjusted = bool(adjusted)
        self.timeout = float(timeout)

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(
            name="polygon",
            is_free=False,
            requires_api_key=True,
            provides_delisted=True,          # the whole point: survivorship escape hatch
            provides_intraday=True,
            provides_fundamentals=True,
            provides_point_in_time_universe=True,
        )

    # ------------------------------------------------------------------ #
    def _urllib_get(self, url: str) -> dict:  # pragma: no cover - needs network
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _require_key(self) -> str:
        if not self.api_key:
            raise DataFetchError(
                "PolygonSource needs an API key (set POLYGON_API_KEY or pass api_key=). "
                "Delisted history requires a paid Polygon plan."
            )
        return self.api_key

    @staticmethod
    def _date_range(period: str, start: Optional[str], end: Optional[str]) -> tuple[str, str]:
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        if start:
            start_ts = pd.Timestamp(start)
        elif period.lower() == "ytd":
            start_ts = pd.Timestamp(year=end_ts.year, month=1, day=1)   # Jan-1 of end's year
        else:
            days = _PERIOD_DAYS.get(period.lower(), 186)
            start_ts = end_ts - pd.Timedelta(days=days)
        return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------ #
    def get_bars(
        self,
        symbol: str,
        *,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        key = self._require_key()
        iv = interval.lower()
        if iv not in _TIMESPAN:
            raise ValueError(f"Unsupported interval {interval!r}. Valid: {sorted(_TIMESPAN)}.")
        mult, span = _TIMESPAN[iv]
        frm, to = self._date_range(period, start, end)
        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{frm}/{to}"
            f"?adjusted={str(self.adjusted).lower()}&sort=asc&limit=50000&apiKey={key}"
        )
        try:
            payload = self._http_get(url)
        except DataFetchError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize transport errors
            raise DataFetchError(f"{symbol}: Polygon request failed: {exc}") from exc

        results = payload.get("results") or []
        if not results:
            raise DataFetchError(
                f"{symbol}: Polygon returned no bars "
                f"(status={payload.get('status')!r}, count={payload.get('resultsCount')})."
            )
        df = pd.DataFrame(results).rename(
            columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "ts"}
        )
        # Naive UTC index, consistent with YFinanceSource/SyntheticSource — a
        # tz-aware index here would crash any comparison against the tz-naive rest
        # of the system (e.g. Universe.members_asof).
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        df.attrs["symbol"] = symbol
        return validate_ohlcv(df, symbol=symbol, min_bars=1)

    # ------------------------------------------------------------------ #
    def is_delisted(self, symbol: str) -> Optional[bool]:
        """Point-in-time delisted status via the reference-ticker endpoint.

        Returns ``True``/``False`` from the ticker's ``active`` flag, or ``None`` if
        Polygon doesn't report it.
        """
        key = self._require_key()
        url = f"{self.BASE_URL}/v3/reference/tickers/{symbol}?apiKey={key}"
        try:
            res = (self._http_get(url) or {}).get("results") or {}
        except Exception:  # noqa: BLE001
            return None
        active = res.get("active")
        return (not bool(active)) if active is not None else None

    def list_delisted(self, *, market: str = "stocks", limit: int = 1000) -> list[dict]:
        """List delisted tickers (``active=false``) — the point-in-time universe seed.

        Returns the raw reference records (each carries ``ticker`` and
        ``delisted_utc``); pair with :meth:`get_bars` to build a survivorship-free
        backtest. Paginated by Polygon; this fetches the first page.
        """
        key = self._require_key()
        url = (
            f"{self.BASE_URL}/v3/reference/tickers"
            f"?market={market}&active=false&limit={int(limit)}&apiKey={key}"
        )
        return list((self._http_get(url) or {}).get("results") or [])
