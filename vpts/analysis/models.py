"""Immutable result objects for Act V — time-series & fundamental data analysis.

Three families live here:

* :class:`TimeSeriesReport` — the *descriptive* diagnostic of a price series
  (distribution, memory, volatility structure, drawdown, tails). It deliberately
  makes **no edge claim**: a fat tail or a rejected random walk is a property of
  the data, not a tradeable signal. Anything that wants to be called an edge has
  to go through :func:`vpts.harness.honest_backtest` like everything else.
* :class:`FundamentalSnapshot` / :class:`FundamentalSeries` — **point-in-time**
  fundamentals. Every snapshot carries the date it became *knowable*
  (``available_at``, i.e. the filing date) separately from the period it
  describes (``period_end``). The two are not interchangeable, and the class
  refuses to be constructed with ``available_at < period_end`` — using the
  period end as the observation date is the classic fundamental-data look-ahead.
* :class:`FundamentalRatios` — the derived ratio/score row, emitted in the
  canonical :data:`FUNDAMENTAL_FEATURES` order so it can be stacked straight
  into a :class:`~vpts.ml.models.FactorDataset`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

_NAN = float("nan")

#: The numeric fundamental feature matrix fed to the CPCV harness, in order.
#: Valuation ratios are expressed as *yields* (earnings/price, not price/earnings)
#: so they stay finite and monotone through zero earnings — a P/E explodes and
#: flips sign at ``E = 0``, which silently poisons a linear model.
FUNDAMENTAL_FEATURES: tuple[str, ...] = (
    "earnings_yield",       # trailing net income / market cap
    "book_to_price",        # total equity / market cap
    "sales_to_price",       # revenue / market cap
    "fcf_yield",            # (operating cash flow - capex) / market cap
    "gross_margin",         # gross profit / revenue
    "operating_margin",     # operating income / revenue
    "roe",                  # net income / total equity
    "roa",                  # net income / total assets
    "debt_to_equity",       # total debt / total equity
    "current_ratio",        # current assets / current liabilities
    "interest_coverage",    # operating income / interest expense
    "accruals",             # (net income - operating cash flow) / total assets
    "asset_growth",         # YoY growth in total assets (a known *negative* factor)
    "revenue_growth",       # YoY growth in revenue
    "earnings_growth",      # YoY growth in net income (scaled, sign-safe)
    "piotroski_f",          # 0..9 fundamental-quality score
    "altman_z",             # distress score (low = distressed)
)

#: Raw line items a :class:`FundamentalSnapshot` carries (the parser's target).
LINE_ITEMS: tuple[str, ...] = (
    "revenue", "gross_profit", "operating_income", "net_income", "interest_expense",
    "total_assets", "total_liabilities", "total_equity", "total_debt",
    "current_assets", "current_liabilities", "retained_earnings", "cash",
    "operating_cash_flow", "capex", "shares_diluted",
)


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeSeriesReport:
    """Descriptive diagnostics of one price series — *not* a signal.

    Read it as "what kind of process is this?", never as "is there money here?".
    The fields are grouped: returns/risk, distribution shape, tail risk, memory
    (autocorrelation, Ljung-Box, variance ratio, Hurst, ADF), volatility
    structure (clustering, ARCH-LM, range-based estimators) and drawdown.
    """

    symbol: Optional[str]
    n_bars: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    periods_per_year: float

    # --- returns / risk ---
    ann_return_pct: float
    ann_vol_pct: float
    sharpe: float

    # --- distribution ---
    skew: float
    excess_kurtosis: float
    jarque_bera_p: float

    # --- tails ---
    var_95_pct: float
    cvar_95_pct: float
    jump_frac: float                 # share of |z| > 4 return days

    # --- memory / efficiency ---
    autocorr_1: float
    ljung_box_p: float
    variance_ratio: float
    vr_z: float
    vr_p: float
    hurst: float
    adf_stat: float
    adf_stationary: bool             # vs the 5% asymptotic critical value

    # --- volatility structure ---
    abs_autocorr_1: float            # volatility clustering
    arch_lm_p: float
    parkinson_vol_pct: float
    garman_klass_vol_pct: float

    # --- drawdown ---
    max_drawdown_pct: float
    max_drawdown_bars: int
    ulcer_index: float

    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def random_walk_rejected(self) -> bool:
        """True iff the Lo-MacKinlay variance ratio rejects a random walk at 5%."""
        return bool(np.isfinite(self.vr_p) and self.vr_p < 0.05)

    @property
    def memory(self) -> str:
        """``"trending"`` / ``"mean-reverting"`` / ``"random-walk"`` (VR-based)."""
        if not self.random_walk_rejected:
            return "random-walk"
        return "trending" if self.variance_ratio > 1.0 else "mean-reverting"

    @property
    def fat_tailed(self) -> bool:
        return bool(np.isfinite(self.excess_kurtosis) and self.excess_kurtosis > 1.0)

    @property
    def vol_clustered(self) -> bool:
        return bool(np.isfinite(self.arch_lm_p) and self.arch_lm_p < 0.05)

    def as_dict(self) -> dict:
        out = {}
        for f in fields(self):
            if f.name == "extra":
                continue
            v = getattr(self, f.name)
            out[f.name] = (
                round(float(v), 4) if isinstance(v, float) else
                (str(v) if isinstance(v, pd.Timestamp) else v)
            )
        out["memory"] = self.memory
        return out

    def summary(self) -> str:
        sym = self.symbol or "series"
        span = ""
        if self.start is not None and self.end is not None:
            span = f" {self.start:%Y-%m-%d}→{self.end:%Y-%m-%d}"
        rw = "rejected" if self.random_walk_rejected else "not rejected"
        return "\n".join([
            f"Time-series diagnostic — {sym}{span}  ({self.n_bars} bars)",
            f"  return/risk   ann {self.ann_return_pct:+.1f}%  vol {self.ann_vol_pct:.1f}%  "
            f"Sharpe {self.sharpe:+.2f}",
            f"  distribution  skew {self.skew:+.2f}  excess-kurt {self.excess_kurtosis:+.2f}  "
            f"JB p {self.jarque_bera_p:.3f}  ({'fat-tailed' if self.fat_tailed else 'near-normal'})",
            f"  tails         VaR95 {self.var_95_pct:.2f}%  CVaR95 {self.cvar_95_pct:.2f}%  "
            f"jumps {self.jump_frac * 100:.2f}% of bars",
            f"  memory        AC(1) {self.autocorr_1:+.3f}  LB p {self.ljung_box_p:.3f}  "
            f"VR({self.extra.get('vr_q', '?')}) {self.variance_ratio:.3f} (z {self.vr_z:+.2f}, "
            f"p {self.vr_p:.3f}, RW {rw})  H {self.hurst:.3f}  → {self.memory}",
            f"  volatility    |r| AC(1) {self.abs_autocorr_1:+.3f}  ARCH-LM p {self.arch_lm_p:.3f}  "
            f"Parkinson {self.parkinson_vol_pct:.1f}%  Garman-Klass {self.garman_klass_vol_pct:.1f}%",
            f"  drawdown      max {self.max_drawdown_pct:.1f}% over {self.max_drawdown_bars} bars  "
            f"ulcer {self.ulcer_index:.2f}",
            "  NOTE: descriptive only — no edge is claimed or implied by these numbers.",
        ])


# --------------------------------------------------------------------------- #
# Fundamentals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FundamentalSnapshot:
    """One reporting period's fundamentals, with its point-in-time stamp.

    ``period_end`` is the fiscal period the numbers *describe*; ``available_at``
    is when they were first public (the filing/acceptance date). Only
    ``available_at`` may be used to decide what a model knew at bar *t* — keying
    on ``period_end`` leaks weeks-to-months of hindsight into every feature and
    is the single most common way a fundamental backtest fools itself.

    Missing line items are ``NaN`` rather than 0.0: a company that did not report
    a number is not a company that reported zero.
    """

    symbol: str
    period_end: pd.Timestamp
    available_at: pd.Timestamp

    revenue: float = _NAN
    gross_profit: float = _NAN
    operating_income: float = _NAN
    net_income: float = _NAN
    interest_expense: float = _NAN
    total_assets: float = _NAN
    total_liabilities: float = _NAN
    total_equity: float = _NAN
    total_debt: float = _NAN
    current_assets: float = _NAN
    current_liabilities: float = _NAN
    retained_earnings: float = _NAN
    cash: float = _NAN
    operating_cash_flow: float = _NAN
    capex: float = _NAN
    shares_diluted: float = _NAN

    period: str = "FY"               # "FY" | "Q1".."Q4" — cosmetic/grouping only
    source: Optional[str] = None

    line_items: ClassVar[tuple[str, ...]] = LINE_ITEMS

    def __post_init__(self) -> None:
        pe = pd.Timestamp(self.period_end)
        av = pd.Timestamp(self.available_at)
        object.__setattr__(self, "period_end", pe)
        object.__setattr__(self, "available_at", av)
        if av < pe:
            raise ValueError(
                f"{self.symbol}: available_at ({av:%Y-%m-%d}) precedes period_end "
                f"({pe:%Y-%m-%d}) — fundamentals cannot be known before the period "
                f"they describe. This is look-ahead."
            )

    @property
    def reporting_lag_days(self) -> int:
        """Days between the period end and the numbers becoming public."""
        return int((self.available_at - self.period_end).days)

    @property
    def free_cash_flow(self) -> float:
        """Operating cash flow less capital expenditure (capex sign-agnostic)."""
        return float(self.operating_cash_flow - abs(self.capex))

    def as_dict(self) -> dict:
        d = {
            "symbol": self.symbol,
            "period": self.period,
            "period_end": f"{self.period_end:%Y-%m-%d}",
            "available_at": f"{self.available_at:%Y-%m-%d}",
            "reporting_lag_days": self.reporting_lag_days,
        }
        d.update({n: getattr(self, n) for n in LINE_ITEMS})
        return d

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.period} ending {self.period_end:%Y-%m-%d} "
            f"(public {self.available_at:%Y-%m-%d}, +{self.reporting_lag_days}d): "
            f"revenue {self.revenue:,.0f}  net income {self.net_income:,.0f}  "
            f"assets {self.total_assets:,.0f}"
        )


@dataclass(frozen=True)
class FundamentalSeries:
    """A symbol's snapshots, ordered by ``available_at`` — queryable as-of a date."""

    symbol: str
    snapshots: tuple[FundamentalSnapshot, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.snapshots, key=lambda s: (s.available_at, s.period_end)))
        object.__setattr__(self, "snapshots", ordered)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self):
        return iter(self.snapshots)

    def as_of(self, ts) -> Optional[FundamentalSnapshot]:
        """Latest snapshot **public on or before** *ts* — ``None`` before the first filing.

        This is the only sanctioned way to ask "what did the market know then?".
        """
        ts = pd.Timestamp(ts)
        found = None
        for s in self.snapshots:          # ordered; small N (quarterly over years)
            if s.available_at <= ts:
                found = s
            else:
                break
        return found

    def prior_year_of(self, snap: FundamentalSnapshot) -> Optional[FundamentalSnapshot]:
        """The snapshot ~4 quarters before *snap* (for YoY growth / Piotroski deltas)."""
        target = snap.period_end - pd.Timedelta(days=365)
        best, best_gap = None, pd.Timedelta(days=120)
        for s in self.snapshots:
            if s.period_end >= snap.period_end:
                continue
            gap = abs(s.period_end - target)
            if gap <= best_gap:
                best, best_gap = s, gap
        return best

    def to_frame(self) -> pd.DataFrame:
        """All snapshots as a frame indexed by ``available_at`` (the PIT index)."""
        if not self.snapshots:
            return pd.DataFrame(columns=["period_end", *LINE_ITEMS])
        rows = [s.as_dict() for s in self.snapshots]
        out = pd.DataFrame(rows)
        out["available_at"] = pd.to_datetime(out["available_at"])
        out["period_end"] = pd.to_datetime(out["period_end"])
        return out.set_index("available_at").sort_index()

    def summary(self) -> str:
        if not self.snapshots:
            return f"{self.symbol}: no fundamental snapshots."
        lags = [s.reporting_lag_days for s in self.snapshots]
        return (
            f"{self.symbol}: {len(self.snapshots)} snapshots "
            f"{self.snapshots[0].period_end:%Y-%m-%d}→{self.snapshots[-1].period_end:%Y-%m-%d}, "
            f"reporting lag median {int(np.median(lags))}d "
            f"(min {min(lags)}d, max {max(lags)}d)"
        )


@dataclass(frozen=True)
class FundamentalRatios:
    """Derived ratios/scores for one snapshot, in :data:`FUNDAMENTAL_FEATURES` order."""

    earnings_yield: float = _NAN
    book_to_price: float = _NAN
    sales_to_price: float = _NAN
    fcf_yield: float = _NAN
    gross_margin: float = _NAN
    operating_margin: float = _NAN
    roe: float = _NAN
    roa: float = _NAN
    debt_to_equity: float = _NAN
    current_ratio: float = _NAN
    interest_coverage: float = _NAN
    accruals: float = _NAN
    asset_growth: float = _NAN
    revenue_growth: float = _NAN
    earnings_growth: float = _NAN
    piotroski_f: float = _NAN
    altman_z: float = _NAN

    # --- metadata (not part of the feature vector) ---
    symbol: Optional[str] = None
    period_end: Optional[pd.Timestamp] = None
    available_at: Optional[pd.Timestamp] = None
    price: float = _NAN
    market_cap: float = _NAN

    feature_names: ClassVar[tuple[str, ...]] = FUNDAMENTAL_FEATURES

    def to_vector(self) -> np.ndarray:
        """Return the row in :data:`FUNDAMENTAL_FEATURES` order."""
        return np.array([getattr(self, n) for n in FUNDAMENTAL_FEATURES], dtype=float)

    @property
    def is_complete(self) -> bool:
        return bool(np.all(np.isfinite(self.to_vector())))

    def as_dict(self) -> dict:
        d = {n: round(float(getattr(self, n)), 4) for n in FUNDAMENTAL_FEATURES}
        d["symbol"] = self.symbol
        if self.period_end is not None:
            d["period_end"] = f"{self.period_end:%Y-%m-%d}"
        if self.available_at is not None:
            d["available_at"] = f"{self.available_at:%Y-%m-%d}"
        return d

    def summary(self) -> str:
        head = f"{self.symbol or 'ratios'}"
        if self.period_end is not None:
            head += f" @ {self.period_end:%Y-%m-%d}"
        return "\n".join([
            f"Fundamental ratios — {head}",
            f"  value      E/P {self.earnings_yield:+.4f}  B/P {self.book_to_price:.4f}  "
            f"S/P {self.sales_to_price:.4f}  FCF/P {self.fcf_yield:+.4f}",
            f"  quality    gross {self.gross_margin:.3f}  oper {self.operating_margin:+.3f}  "
            f"ROE {self.roe:+.3f}  ROA {self.roa:+.3f}  accruals {self.accruals:+.4f}",
            f"  balance    D/E {self.debt_to_equity:.2f}  current {self.current_ratio:.2f}  "
            f"int-cover {self.interest_coverage:.1f}  Altman-Z {self.altman_z:.2f}",
            f"  growth     revenue {self.revenue_growth:+.3f}  earnings {self.earnings_growth:+.3f}  "
            f"assets {self.asset_growth:+.3f}  Piotroski F {self.piotroski_f:.0f}/9",
        ])
