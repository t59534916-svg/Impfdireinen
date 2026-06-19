"""Financial Modeling Prep (FMP) data source — survivor history now, delisted on a paid tier.

FMP serves daily EOD history for live names on the free/low tiers, and **delisted**
history on its higher plans (Starter/Premium and up). This adapter therefore advertises
``provides_delisted`` as a constructor flag (default ``False``): set
``FMPSource(delisted_capable=True)`` only when your plan actually includes dead names,
so :class:`~vpts.data.registry.SourceRegistry` routes survivorship-sensitive requests
correctly. (Verified against the live API: survivors like AAPL/MSFT return on the free
tier; delisted symbols like LEH/SIVB return "requires a higher plan" — the documented
data wall, re-confirmed.)

Activation & testing
--------------------
Requires a key (``FMP_API_KEY`` env var or the *api_key* argument). The HTTP layer is
**injectable** via *http_get*, so parsing and error-handling are unit-tested against
canned FMP JSON with no network and no key — mirroring
:class:`~vpts.data.sources.polygon_source.PolygonSource`.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, Optional

import pandas as pd

from vpts.data.base import DataSource, DataSourceCapabilities, validate_ohlcv
from vpts.data.fetcher import DataFetchError

#: ``period`` → approximate calendar days (for deriving a from/to window).
_PERIOD_DAYS: dict[str, int] = {
    "1mo": 31, "3mo": 93, "6mo": 186, "1y": 366, "2y": 731,
    "5y": 1827, "10y": 3653, "ytd": 366, "max": 36525,
}

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class FMPSource(DataSource):
    """Daily OHLCV from Financial Modeling Prep's stable EOD API.

    Parameters
    ----------
    api_key:
        FMP key; falls back to ``FMP_API_KEY``.
    http_get:
        Callable ``(url) -> dict | list`` returning parsed JSON. Defaults to a stdlib
        ``urllib`` fetch (no new dependency). Inject a stub to test offline.
    delisted_capable:
        Set ``True`` only if your FMP plan serves delisted history; this flips the
        advertised ``provides_delisted`` capability so the registry routes accordingly.
    adjusted:
        Use the dividend-adjusted EOD endpoint (default ``False`` → raw OHLCV, matching
        the rest of the stack which uses raw Close).
    """

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        http_get: Optional[Callable[[str], object]] = None,
        delisted_capable: bool = False,
        adjusted: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        self._http_get = http_get or self._urllib_get
        self.delisted_capable = bool(delisted_capable)
        self.adjusted = bool(adjusted)
        self.timeout = float(timeout)

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(
            name="fmp",
            is_free=False,
            requires_api_key=True,
            provides_delisted=self.delisted_capable,   # plan-dependent — opt in honestly
            provides_intraday=True,
            provides_fundamentals=True,
            provides_point_in_time_universe=False,
        )

    # ------------------------------------------------------------------ #
    def _urllib_get(self, url: str) -> object:  # pragma: no cover - needs network
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _require_key(self) -> str:
        if not self.api_key:
            raise DataFetchError(
                "FMPSource needs an API key (set FMP_API_KEY or pass api_key=). "
                "Delisted history requires a higher FMP plan."
            )
        return self.api_key

    @staticmethod
    def _date_range(period: str, start: Optional[str], end: Optional[str]) -> tuple[str, str]:
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        if start:
            start_ts = pd.Timestamp(start)
        elif period.lower() == "ytd":
            start_ts = pd.Timestamp(year=end_ts.year, month=1, day=1)
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
        if interval.lower() not in ("1d", "d", "daily"):
            raise ValueError("FMPSource serves daily bars only (interval='1d').")
        frm, to = self._date_range(period, start, end)
        kind = "historical-price-eod/dividend-adjusted" if self.adjusted else "historical-price-eod/full"
        url = f"{self.BASE_URL}/{kind}?symbol={symbol}&from={frm}&to={to}&apikey={key}"
        try:
            payload = self._http_get(url)
        except DataFetchError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize transport errors
            raise DataFetchError(f"{symbol}: FMP request failed: {exc}") from exc

        # FMP signals an error / plan gate via a dict with "Error Message" (or similar),
        # not a list of bars. Surface it instead of a generic "no bars" — e.g. delisted
        # history on a free plan returns a plan-upgrade message.
        if isinstance(payload, dict):
            msg = payload.get("Error Message") or payload.get("error") or payload.get("message")
            raise DataFetchError(f"{symbol}: FMP error: {msg or payload}")
        if not payload:
            raise DataFetchError(f"{symbol}: FMP returned no bars (empty result).")

        df = pd.DataFrame(payload)
        cols = {c.lower(): c for c in df.columns}
        need = {"date": "date", "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume"}
        missing = [k for k in need if k not in cols]
        if missing:
            raise DataFetchError(f"{symbol}: FMP payload missing {missing} (got {list(df.columns)}).")
        df = df.rename(columns={cols[k]: v for k, v in need.items()})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[_OHLCV].sort_index()
        df.index.name = None
        df.attrs["symbol"] = symbol
        return validate_ohlcv(df, symbol=symbol, min_bars=1)

    # ------------------------------------------------------------------ #
    def is_delisted(self, symbol: str) -> Optional[bool]:
        """FMP's stable API has no free point-in-time active flag; unknown here.

        Delisted *enumeration* (the screener / delisted-companies endpoint) is itself
        plan-gated, so this returns ``None``. Use a curated catalogue
        (:data:`~vpts.data.coverage.KNOWN_DELISTED`) to drive a survivorship audit.
        """
        return None
