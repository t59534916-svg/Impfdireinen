"""Fundamental data — point-in-time ingestion, ratios, and as-of alignment.

The fundamental half of Act V. Three layers:

1. **Sources** — a :class:`FundamentalsSource` contract mirroring
   :class:`~vpts.data.base.DataSource`, with an *injectable* HTTP transport so
   parsing is unit-tested with no network and no key
   (:class:`FMPFundamentalsSource`), plus a deterministic
   :class:`SyntheticFundamentalsSource` so the whole layer runs offline.
2. **Ratios** — :func:`compute_ratios` turns raw line items into the canonical
   :data:`~vpts.analysis.models.FUNDAMENTAL_FEATURES` row, including the
   Piotroski F-score and the Altman Z-score.
3. **Alignment** — :func:`align_fundamentals` merges a
   :class:`~vpts.analysis.models.FundamentalSeries` onto a price index
   **as-of the filing date**, never the period end.

Why the filing date is the whole ballgame
-----------------------------------------
A 10-K describing the fiscal year ending 2015-12-31 typically becomes public in
February or March 2016. Joining that row onto 2015-12-31 hands the model 45-90
days of hindsight on every observation — and because earnings surprises move
prices precisely in that window, the leak manufactures a large, extremely
persuasive "edge". This module makes the mistake structurally hard:
:class:`~vpts.analysis.models.FundamentalSnapshot` refuses to exist with
``available_at < period_end``, alignment is a backward as-of merge on
``available_at``, and :func:`audit_point_in_time` re-checks the result bar by
bar. When a feed gives no filing date, :data:`DEFAULT_REPORTING_LAG_DAYS` is
applied as an explicit, logged, deliberately conservative assumption rather than
silently defaulting to zero.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import zlib
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from vpts.analysis.models import (
    FUNDAMENTAL_FEATURES,
    LINE_ITEMS,
    FundamentalRatios,
    FundamentalSeries,
    FundamentalSnapshot,
)

logger = logging.getLogger(__name__)

_EPS = 1e-12
_NAN = float("nan")

#: Assumed gap between period end and public availability when a feed reports no
#: filing date. 75 days is past the SEC's 60/75-day large-accelerated 10-K
#: deadline — chosen to *over*-state the lag, because erring long costs a little
#: signal while erring short fabricates one.
DEFAULT_REPORTING_LAG_DAYS: int = 75


def _safe_div(a: float, b: float) -> float:
    """``a / b``, returning NaN for a non-finite or ~zero denominator."""
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)) or abs(b) <= _EPS:
        return _NAN
    return a / b


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _NAN
    return f if np.isfinite(f) else _NAN


# --------------------------------------------------------------------------- #
# Ratios & scores
# --------------------------------------------------------------------------- #
def piotroski_f_score(
    snap: FundamentalSnapshot, prior: Optional[FundamentalSnapshot]
) -> float:
    """Piotroski (2000) F-score, 0-9 — nine binary fundamental-health tests.

    Profitability (4): ROA > 0, operating cash flow > 0, ROA improving, and
    accruals healthy (OCF exceeds net income). Leverage/liquidity (3): leverage
    falling, current ratio rising, no share issuance. Efficiency (2): gross
    margin and asset turnover both improving.

    Returns ``NaN`` without a prior-year snapshot — five of the nine tests are
    year-over-year deltas, and scoring them as failures would bias the score
    down for every company's first observation.
    """
    if prior is None:
        return _NAN
    roa = _safe_div(snap.net_income, snap.total_assets)
    roa_prior = _safe_div(prior.net_income, prior.total_assets)
    lev = _safe_div(snap.total_debt, snap.total_assets)
    lev_prior = _safe_div(prior.total_debt, prior.total_assets)
    cur = _safe_div(snap.current_assets, snap.current_liabilities)
    cur_prior = _safe_div(prior.current_assets, prior.current_liabilities)
    gm = _safe_div(snap.gross_profit, snap.revenue)
    gm_prior = _safe_div(prior.gross_profit, prior.revenue)
    turn = _safe_div(snap.revenue, snap.total_assets)
    turn_prior = _safe_div(prior.revenue, prior.total_assets)

    tests = (
        roa > 0,
        snap.operating_cash_flow > 0,
        roa > roa_prior,
        snap.operating_cash_flow > snap.net_income,
        lev < lev_prior,
        cur > cur_prior,
        snap.shares_diluted <= prior.shares_diluted,
        gm > gm_prior,
        turn > turn_prior,
    )
    # A NaN comparison is False, which would silently read as "test failed".
    # Score only what is computable and refuse the whole score if too much is missing.
    inputs = (roa, roa_prior, lev, lev_prior, cur, cur_prior, gm, gm_prior, turn, turn_prior,
              snap.operating_cash_flow, snap.net_income, snap.shares_diluted, prior.shares_diluted)
    if sum(1 for v in inputs if not np.isfinite(v)) > 2:
        return _NAN
    return float(sum(1 for t in tests if bool(t)))


def altman_z_score(snap: FundamentalSnapshot, market_cap: float) -> float:
    """Altman (1968) Z-score for a public manufacturer — distress below ~1.8.

    ``Z = 1.2·WC/TA + 1.4·RE/TA + 3.3·EBIT/TA + 0.6·MVE/TL + 1.0·Sales/TA``.
    Included because it is one of the few fundamental constructs aimed squarely
    at the failure tail — the tail that survivor-only data deletes, and that
    ``RESEARCH.md`` identifies as this project's binding constraint.
    """
    ta = _num(snap.total_assets)
    if not np.isfinite(ta) or abs(ta) <= _EPS:
        return _NAN
    wc = _num(snap.current_assets) - _num(snap.current_liabilities)
    x1 = _safe_div(wc, ta)
    x2 = _safe_div(snap.retained_earnings, ta)
    x3 = _safe_div(snap.operating_income, ta)
    x4 = _safe_div(market_cap, snap.total_liabilities)
    x5 = _safe_div(snap.revenue, ta)
    parts = (x1, x2, x3, x4, x5)
    if not all(np.isfinite(p) for p in parts):
        return _NAN
    return float(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5)


def compute_ratios(
    snap: FundamentalSnapshot,
    *,
    prior: Optional[FundamentalSnapshot] = None,
    price: float = _NAN,
    shares: float = _NAN,
) -> FundamentalRatios:
    """Derive the :data:`~vpts.analysis.models.FUNDAMENTAL_FEATURES` row.

    ``price`` is the market price at the observation bar (so valuation ratios are
    as-of *now*, while the accounting numbers are as-of the last filing —
    the correct pairing). ``shares`` defaults to the snapshot's diluted count.
    Growth features and the F-score need *prior*; without it they are ``NaN``,
    which the dataset builder then drops rather than imputing.
    """
    shares = _num(shares)
    if not np.isfinite(shares):
        shares = _num(snap.shares_diluted)
    price = _num(price)
    mcap = price * shares if (np.isfinite(price) and np.isfinite(shares)) else _NAN

    fcf = snap.free_cash_flow
    growth = {"asset_growth": _NAN, "revenue_growth": _NAN, "earnings_growth": _NAN}
    if prior is not None:
        growth["asset_growth"] = _safe_div(
            _num(snap.total_assets) - _num(prior.total_assets), abs(_num(prior.total_assets)))
        growth["revenue_growth"] = _safe_div(
            _num(snap.revenue) - _num(prior.revenue), abs(_num(prior.revenue)))
        growth["earnings_growth"] = _safe_div(
            _num(snap.net_income) - _num(prior.net_income), abs(_num(prior.net_income)))

    return FundamentalRatios(
        earnings_yield=_safe_div(snap.net_income, mcap),
        book_to_price=_safe_div(snap.total_equity, mcap),
        sales_to_price=_safe_div(snap.revenue, mcap),
        fcf_yield=_safe_div(fcf, mcap),
        gross_margin=_safe_div(snap.gross_profit, snap.revenue),
        operating_margin=_safe_div(snap.operating_income, snap.revenue),
        roe=_safe_div(snap.net_income, snap.total_equity),
        roa=_safe_div(snap.net_income, snap.total_assets),
        debt_to_equity=_safe_div(snap.total_debt, snap.total_equity),
        current_ratio=_safe_div(snap.current_assets, snap.current_liabilities),
        interest_coverage=_safe_div(snap.operating_income, abs(_num(snap.interest_expense))),
        accruals=_safe_div(_num(snap.net_income) - _num(snap.operating_cash_flow),
                           snap.total_assets),
        piotroski_f=piotroski_f_score(snap, prior),
        altman_z=altman_z_score(snap, mcap),
        symbol=snap.symbol,
        period_end=snap.period_end,
        available_at=snap.available_at,
        price=price,
        market_cap=mcap,
        **growth,
    )


# --------------------------------------------------------------------------- #
# Point-in-time alignment
# --------------------------------------------------------------------------- #
def align_asof(table: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Backward as-of join of *table* (indexed by availability date) onto *index*.

    The single place point-in-time alignment happens, so every derived feature
    inherits the same guarantee: row *t* holds the most recent record public **on
    or before** *t*, and rows before the first record stay ``NaN`` — never
    back-filled, because nothing was knowable then.
    """
    index = pd.DatetimeIndex(index)
    if table is None or len(table) == 0:
        return pd.DataFrame(np.nan, index=index, columns=list(getattr(table, "columns", [])))
    right = table.sort_index().reset_index()
    av_col = right.columns[0]
    right = right.rename(columns={av_col: "_av"})
    right["available_at"] = right["_av"]
    left = pd.DataFrame({"_ts": index}).sort_values("_ts")
    merged = pd.merge_asof(
        left, right, left_on="_ts", right_on="_av",
        direction="backward", allow_exact_matches=True,
    )
    out = merged.set_index("_ts").drop(columns=["_av"])
    out.index = pd.DatetimeIndex(out.index, name=index.name)
    return out.reindex(index)


def align_fundamentals(
    series: FundamentalSeries,
    index: pd.DatetimeIndex,
    *,
    columns: Sequence[str] = LINE_ITEMS,
) -> pd.DataFrame:
    """As-of join of a series' **line items** onto *index*, keyed on ``available_at``.

    Thin wrapper over :func:`align_asof`. The returned frame carries
    ``period_end`` and ``available_at`` so the reporting lag stays auditable
    downstream; see :func:`audit_point_in_time`.
    """
    index = pd.DatetimeIndex(index)
    cols = ["period_end", "available_at", *columns]
    if len(series) == 0:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    table = pd.DataFrame(
        [{"available_at": s.available_at, "period_end": s.period_end,
          **{c: getattr(s, c) for c in columns}} for s in series.snapshots]
    ).set_index("available_at")
    return align_asof(table, index)[cols]


def audit_point_in_time(aligned: pd.DataFrame) -> dict:
    """Verify an aligned frame never uses a filing before it was public.

    Returns ``{"bars", "covered", "coverage_pct", "violations", "median_lag_days",
    "max_staleness_days"}``. ``violations`` must be 0 — a non-zero count means
    look-ahead reached the feature matrix, and the dataset builder raises on it
    rather than quietly producing a leaked (and flattering) result.
    """
    if "available_at" not in aligned.columns:
        raise ValueError("audit_point_in_time needs the 'available_at' column.")
    av = pd.to_datetime(aligned["available_at"])
    idx = pd.DatetimeIndex(aligned.index)
    mask = av.notna().to_numpy()
    n_cov = int(mask.sum())
    if not n_cov:
        return {"bars": int(len(aligned)), "covered": 0, "coverage_pct": 0.0,
                "violations": 0, "median_lag_days": None, "max_staleness_days": None}

    idx_cov = idx[mask]
    av_cov = pd.DatetimeIndex(av.to_numpy()[mask])
    violations = int((av_cov > idx_cov).sum())
    stale = (idx_cov - av_cov).days
    lag = None
    if "period_end" in aligned.columns:
        pe_cov = pd.DatetimeIndex(pd.to_datetime(aligned["period_end"]).to_numpy()[mask])
        lag = (av_cov - pe_cov).days
    return {
        "bars": int(len(aligned)),
        "covered": n_cov,
        "coverage_pct": round(100.0 * n_cov / max(len(aligned), 1), 2),
        "violations": violations,
        "median_lag_days": int(np.median(lag)) if lag is not None and len(lag) else None,
        "max_staleness_days": int(np.max(stale)) if len(stale) else None,
    }


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
class FundamentalsSource(ABC):
    """Provider-agnostic fundamentals contract — mirrors :class:`~vpts.data.base.DataSource`.

    Implementations return a :class:`~vpts.analysis.models.FundamentalSeries`
    whose snapshots carry a real ``available_at``. A source that cannot supply
    filing dates must say so via :attr:`provides_filing_dates` so callers know
    the point-in-time stamps are an assumption (:data:`DEFAULT_REPORTING_LAG_DAYS`)
    rather than a fact.
    """

    #: True when the feed supplies real filing/acceptance dates.
    provides_filing_dates: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Short source identifier."""

    @abstractmethod
    def get_fundamentals(self, symbol: str, *, limit: int = 40) -> FundamentalSeries:
        """Return up to *limit* most recent periods for *symbol*."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} filing_dates={self.provides_filing_dates}>"


def _pick(row: dict, *keys: str) -> float:
    """First present, numeric value among *keys* (feeds rename fields over time)."""
    for k in keys:
        if k in row and row[k] is not None:
            v = _num(row[k])
            if np.isfinite(v):
                return v
    return _NAN


def _pick_date(row: dict, *keys: str) -> Optional[pd.Timestamp]:
    for k in keys:
        v = row.get(k)
        if v:
            try:
                ts = pd.Timestamp(v)
            except (ValueError, TypeError):
                continue
            if pd.notna(ts):
                return ts.tz_localize(None) if ts.tzinfo else ts
    return None


class FMPFundamentalsSource(FundamentalsSource):
    """Financial Modeling Prep statements → point-in-time :class:`FundamentalSeries`.

    Pulls the income statement, balance sheet and cash-flow statement, joins them
    on the fiscal period, and takes ``available_at`` from FMP's ``filingDate`` /
    ``fillingDate`` / ``acceptedDate`` — real filing dates, which is what makes
    this feed usable for point-in-time work at all.

    The HTTP layer is injectable via *http_get* (matching
    :class:`~vpts.data.sources.fmp_source.FMPSource`), so the parser is tested
    against canned JSON offline. Statement depth beyond a few years is plan-gated
    on FMP; a 402/403 surfaces as a clear message rather than an empty series.
    """

    BASE_URL = "https://financialmodelingprep.com/stable"
    provides_filing_dates = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        http_get: Optional[Callable[[str], object]] = None,
        period: str = "annual",
        reporting_lag_days: int = DEFAULT_REPORTING_LAG_DAYS,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        self._http_get = http_get or self._urllib_get
        self.period = period
        self.reporting_lag_days = int(reporting_lag_days)
        self.timeout = float(timeout)

    @property
    def name(self) -> str:
        return "fmp-fundamentals"

    def _urllib_get(self, url: str) -> object:  # pragma: no cover - needs network
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _fetch(self, endpoint: str, symbol: str, limit: int) -> list[dict]:
        from vpts.data.fetcher import DataFetchError

        if not self.api_key:
            raise DataFetchError(
                "FMPFundamentalsSource needs an API key (set FMP_API_KEY or pass api_key=)."
            )
        url = (f"{self.BASE_URL}/{endpoint}?symbol={symbol}&period={self.period}"
               f"&limit={limit}&apikey={self.api_key}")
        try:
            payload = self._http_get(url)
        except DataFetchError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code in (402, 403):
                raise DataFetchError(
                    f"{symbol}: FMP {exc.code} — statement history requires a higher FMP plan."
                ) from exc
            raise DataFetchError(f"{symbol}: FMP statement request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - transport-agnostic seam
            raise DataFetchError(f"{symbol}: FMP statement request failed: {exc}") from exc
        if isinstance(payload, dict):
            if "Error Message" in payload:
                raise DataFetchError(f"{symbol}: FMP error — {payload['Error Message']}")
            payload = payload.get("data", [])
        return list(payload or [])

    def get_fundamentals(self, symbol: str, *, limit: int = 40) -> FundamentalSeries:
        income = self._fetch("income-statement", symbol, limit)
        balance = self._fetch("balance-sheet-statement", symbol, limit)
        cash = self._fetch("cash-flow-statement", symbol, limit)
        return self.parse(symbol, income, balance, cash,
                          reporting_lag_days=self.reporting_lag_days)

    @staticmethod
    def parse(
        symbol: str,
        income: Iterable[dict],
        balance: Iterable[dict],
        cash: Iterable[dict],
        *,
        reporting_lag_days: int = DEFAULT_REPORTING_LAG_DAYS,
    ) -> FundamentalSeries:
        """Join three FMP statement lists into a :class:`FundamentalSeries`.

        Rows are keyed on the fiscal ``date``; ``available_at`` is the earliest
        filing date across the three statements (all three must be public before
        the full row is), falling back to ``period_end + reporting_lag_days``
        with a warning when the feed omits filing dates entirely.
        """
        by_date: dict[str, dict[str, dict]] = {}
        for kind, rows in (("i", income), ("b", balance), ("c", cash)):
            for row in rows or []:
                d = row.get("date") or row.get("period_end")
                if not d:
                    continue
                by_date.setdefault(str(d)[:10], {})[kind] = row

        snaps: list[FundamentalSnapshot] = []
        assumed = 0
        for d, parts in by_date.items():
            i, b, c = parts.get("i", {}), parts.get("b", {}), parts.get("c", {})
            if not i and not b:
                continue
            try:
                period_end = pd.Timestamp(d)
            except (ValueError, TypeError):
                continue

            filed = [ts for ts in (
                _pick_date(i, "filingDate", "fillingDate", "acceptedDate"),
                _pick_date(b, "filingDate", "fillingDate", "acceptedDate"),
                _pick_date(c, "filingDate", "fillingDate", "acceptedDate"),
            ) if ts is not None]
            if filed:
                available_at = max(filed)          # the row is knowable once ALL parts are
            else:
                available_at = period_end + pd.Timedelta(days=reporting_lag_days)
                assumed += 1
            if available_at < period_end:          # a feed's bad date must not become look-ahead
                available_at = period_end + pd.Timedelta(days=reporting_lag_days)

            snaps.append(FundamentalSnapshot(
                symbol=symbol,
                period_end=period_end,
                available_at=available_at,
                revenue=_pick(i, "revenue"),
                gross_profit=_pick(i, "grossProfit"),
                operating_income=_pick(i, "operatingIncome"),
                net_income=_pick(i, "netIncome"),
                interest_expense=_pick(i, "interestExpense"),
                shares_diluted=_pick(i, "weightedAverageShsOutDil", "weightedAverageShsOut"),
                total_assets=_pick(b, "totalAssets"),
                total_liabilities=_pick(b, "totalLiabilities"),
                total_equity=_pick(b, "totalStockholdersEquity", "totalEquity"),
                total_debt=_pick(b, "totalDebt"),
                current_assets=_pick(b, "totalCurrentAssets"),
                current_liabilities=_pick(b, "totalCurrentLiabilities"),
                retained_earnings=_pick(b, "retainedEarnings"),
                cash=_pick(b, "cashAndCashEquivalents", "cashAndShortTermInvestments"),
                operating_cash_flow=_pick(c, "operatingCashFlow", "netCashProvidedByOperatingActivities"),
                capex=_pick(c, "capitalExpenditure"),
                period=str(i.get("period") or b.get("period") or "FY"),
                source="fmp",
            ))
        if assumed:
            logger.warning(
                "%s: %d/%d periods had no filing date — assumed period_end + %dd. "
                "These stamps are an assumption, not a fact.",
                symbol, assumed, len(snaps), reporting_lag_days,
            )
        return FundamentalSeries(symbol=symbol, snapshots=tuple(snaps))


class SyntheticFundamentalsSource(FundamentalsSource):
    """Deterministic offline fundamentals — for tests, demos and null checks.

    Generates plausible annual statements (a growing, profitable company with
    noise) with an explicit reporting lag, so the whole Act V pipeline runs with
    no network and no key. Pass *link* — a price frame — together with
    ``link_strength > 0`` to plant a **known** relationship between the earnings
    yield and the subsequent return; that is what the signal-detection test uses
    to prove the harness can find an edge that is really there. With
    ``link_strength = 0`` (the default) the fundamentals are pure noise, which is
    what the null-clearing test requires.
    """

    provides_filing_dates = True

    def __init__(
        self,
        *,
        seed: int = 0,
        n_periods: int = 12,
        start: str = "2010-12-31",
        freq: str = "A",
        reporting_lag_days: int = DEFAULT_REPORTING_LAG_DAYS,
        link: Optional[pd.DataFrame] = None,
        link_strength: float = 0.0,
        link_horizon: int = 20,
    ) -> None:
        if freq.upper() not in ("A", "Q"):
            raise ValueError("freq must be 'A' (annual) or 'Q' (quarterly).")
        self.seed = int(seed)
        self.n_periods = int(n_periods)
        self.start = pd.Timestamp(start)
        self.freq = freq.upper()
        self.reporting_lag_days = int(reporting_lag_days)
        self.link = link
        self.link_strength = float(link_strength)
        self.link_horizon = int(link_horizon)

    @property
    def name(self) -> str:
        return "synthetic-fundamentals"

    @staticmethod
    def stable_seed(seed: int, symbol: str) -> int:
        """Process-stable per-symbol seed.

        Uses CRC32 rather than :func:`hash` — Python randomises string hashing
        per process (``PYTHONHASHSEED``), so a ``hash``-derived seed would make
        the "deterministic" source silently irreproducible across runs.
        """
        return (int(seed) * 1_000_003 + zlib.crc32(symbol.encode("utf-8"))) % (2 ** 32)

    def get_fundamentals(self, symbol: str, *, limit: int = 40) -> FundamentalSeries:
        rng = np.random.default_rng(self.stable_seed(self.seed, symbol))
        n = min(self.n_periods, limit)
        assets = 1.0e10 * float(rng.uniform(0.5, 2.0))
        shares = 1.0e9 * float(rng.uniform(0.5, 2.0))
        margin = float(rng.uniform(0.06, 0.20))
        snaps: list[FundamentalSnapshot] = []

        quarterly = self.freq == "Q"
        growth = (1.00, 1.035) if quarterly else (1.00, 1.14)     # same annual pace either way
        scale = 0.25 if quarterly else 1.0                        # a quarter earns ~1/4 of a year

        for k in range(n):
            period_end = (self.start + pd.DateOffset(months=3 * k) if quarterly
                          else self.start + pd.DateOffset(years=k))
            available_at = period_end + pd.Timedelta(days=self.reporting_lag_days)
            assets *= float(rng.uniform(*growth))
            revenue = assets * float(rng.uniform(0.55, 0.95)) * scale
            margin = float(np.clip(margin + rng.normal(0, 0.02), 0.01, 0.35))
            net_income = revenue * margin

            # Optionally tie profitability to the FUTURE return after the filing,
            # so the earnings yield carries a real (planted) forward signal.
            if self.link is not None and self.link_strength > 0:
                fwd = self._forward_return(available_at)
                if np.isfinite(fwd):
                    net_income *= float(np.exp(self.link_strength * fwd * 10.0))

            equity = assets * float(rng.uniform(0.30, 0.55))
            liabilities = assets - equity
            snaps.append(FundamentalSnapshot(
                symbol=symbol,
                period_end=period_end,
                available_at=available_at,
                revenue=revenue,
                gross_profit=revenue * float(rng.uniform(0.30, 0.55)),
                operating_income=net_income * float(rng.uniform(1.15, 1.45)),
                net_income=net_income,
                interest_expense=liabilities * 0.03,
                total_assets=assets,
                total_liabilities=liabilities,
                total_equity=equity,
                total_debt=liabilities * float(rng.uniform(0.35, 0.65)),
                current_assets=assets * float(rng.uniform(0.20, 0.40)),
                current_liabilities=assets * float(rng.uniform(0.10, 0.25)),
                retained_earnings=equity * float(rng.uniform(0.30, 0.80)),
                cash=assets * float(rng.uniform(0.05, 0.20)),
                operating_cash_flow=net_income * float(rng.uniform(1.05, 1.45)),
                capex=revenue * float(rng.uniform(0.03, 0.09)),
                shares_diluted=shares * float(rng.uniform(0.98, 1.02)),
                period="FY",
                source="synthetic",
            ))
        return FundamentalSeries(symbol=symbol, snapshots=tuple(snaps))

    def _forward_return(self, available_at: pd.Timestamp) -> float:
        """Return over ``link_horizon`` bars starting at the first bar ≥ *available_at*."""
        idx = self.link.index
        pos = int(idx.searchsorted(available_at))
        if pos >= len(idx) - self.link_horizon:
            return _NAN
        close = self.link["Close"].to_numpy(float)
        return float(close[pos + self.link_horizon] / close[pos] - 1.0)


def fundamental_ratio_frame(
    series: FundamentalSeries,
    *,
    prices: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Every snapshot's ratios as a frame indexed by ``available_at``.

    The valuation ratios use the close on the filing date (from *prices*, as-of
    backward) when supplied, and are ``NaN`` otherwise — a book-to-price without
    a price is not a number worth guessing at.
    """
    rows = []
    for snap in series.snapshots:
        price = _NAN
        if prices is not None and len(prices):
            usable = prices.loc[prices.index <= snap.available_at]
            if len(usable):
                price = float(usable.iloc[-1])
        r = compute_ratios(snap, prior=series.prior_year_of(snap), price=price)
        row = {n: getattr(r, n) for n in FUNDAMENTAL_FEATURES}
        row["available_at"] = snap.available_at
        row["period_end"] = snap.period_end
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[*FUNDAMENTAL_FEATURES, "period_end"])
    out = pd.DataFrame(rows).set_index("available_at").sort_index()
    return out[[*FUNDAMENTAL_FEATURES, "period_end"]]
