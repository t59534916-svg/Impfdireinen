"""Tests for vpts.analysis — Act V time-series & fundamental data analysis.

Every estimator is checked against an input whose answer is known in advance,
and the two evaluators carry the repo's mandatory pair:

* a **signal** test — a planted, real relationship must be found;
* a **null-clearing** test — pure noise must report nothing.

Fully offline and deterministic: prices come from the synthetic generator,
fundamentals from :class:`SyntheticFundamentalsSource`, and the FMP parser is
exercised against canned JSON through its injectable transport.

    python tests/test_analysis.py
    pytest tests/test_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.analysis import (  # noqa: E402
    FUNDAMENTAL_FEATURES,
    DEFAULT_REPORTING_LAG_DAYS,
    FMPFundamentalsSource,
    FundamentalSeries,
    FundamentalSnapshot,
    SyntheticFundamentalsSource,
    TimeSeriesReport,
    adf_test,
    align_fundamentals,
    altman_z_score,
    analyze_timeseries,
    arch_lm_test,
    audit_point_in_time,
    autocorrelation,
    build_combined_dataset,
    build_fundamental_dataset,
    build_fundamental_panel,
    compute_ratios,
    distribution_stats,
    drawdown_stats,
    fundamental_feature_frame,
    fundamental_ratio_frame,
    garman_klass_vol,
    hurst_exponent,
    jump_fraction,
    ljung_box,
    log_returns,
    parkinson_vol,
    piotroski_f_score,
    realized_vol,
    tail_risk,
    variance_ratio,
)
from vpts.data.synthetic import synthetic_survivor_ohlcv  # noqa: E402
from vpts.ml.models import CrossSectionalPanel, FactorDataset  # noqa: E402
from vpts import permutation_test_cross_sectional  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ar1(phi: float, n: int = 4000, seed: int = 1, sigma: float = 0.01) -> np.ndarray:
    """AR(1) series with known autocorrelation *phi*."""
    e = np.random.default_rng(seed).normal(0.0, sigma, n)
    a = np.zeros(n)
    for i in range(1, n):
        a[i] = phi * a[i - 1] + e[i]
    return a


def _snap(symbol="T", period_end="2015-12-31", available_at=None, **kw) -> FundamentalSnapshot:
    base = dict(
        revenue=1000.0, gross_profit=400.0, operating_income=200.0, net_income=100.0,
        interest_expense=20.0, total_assets=2000.0, total_liabilities=1200.0,
        total_equity=800.0, total_debt=600.0, current_assets=500.0,
        current_liabilities=250.0, retained_earnings=300.0, cash=100.0,
        operating_cash_flow=150.0, capex=50.0, shares_diluted=100.0,
    )
    base.update(kw)
    pe = pd.Timestamp(period_end)
    av = pd.Timestamp(available_at) if available_at is not None else pe + pd.Timedelta(days=60)
    return FundamentalSnapshot(symbol=symbol, period_end=pe, available_at=av, **base)


def _universe(n_names=10, n_bars=2600, seed0=2000, link_strength=0.0, trial=0):
    """Price frames + matching quarterly fundamentals (optionally signal-linked)."""
    frames, smap = {}, {}
    for i in range(n_names):
        s = f"N{i}"
        f = synthetic_survivor_ohlcv(n_bars, seed=seed0 + trial * 30 + i, start_date="2010-01-04")
        frames[s] = f
        smap[s] = SyntheticFundamentalsSource(
            seed=trial * 77 + i, n_periods=44, start="2009-09-30", freq="Q",
            link=f if link_strength else None, link_strength=link_strength,
        ).get_fundamentals(s)
    return frames, smap


# =========================================================================== #
# Time series — volatility & distribution
# =========================================================================== #
def test_log_returns_and_realized_vol_known():
    close = np.array([100.0, 110.0, 121.0, 133.1])
    r = log_returns(close)
    assert r.size == 3
    assert np.allclose(r, np.log(1.1), atol=1e-12)      # constant 10% growth
    assert realized_vol(r) == pytest.approx(0.0, abs=1e-9)   # no dispersion → zero vol
    assert log_returns([100.0]).size == 0
    assert log_returns([]).size == 0


def test_range_volatility_estimators_known():
    # Constant bar: H/L ratio fixed, C == O → Parkinson from the range, GK slightly lower.
    n = 500
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    open_ = np.full(n, 100.0)
    close = np.full(n, 100.0)
    expected_pk = np.sqrt((np.log(101.0 / 99.0) ** 2) / (4 * np.log(2)) * 252)
    assert parkinson_vol(high, low) == pytest.approx(expected_pk, rel=1e-12)
    # C == O kills the second GK term, leaving 0.5·(ln H/L)² annualised.
    expected_gk = np.sqrt(0.5 * np.log(101.0 / 99.0) ** 2 * 252)
    assert garman_klass_vol(open_, high, low, close) == pytest.approx(expected_gk, rel=1e-12)
    assert np.isnan(parkinson_vol([1.0], [1.0]))


def test_distribution_stats_normal_vs_fat_tailed():
    normal = np.random.default_rng(0).normal(0, 1, 6000)
    sk, ek, p = distribution_stats(normal)
    assert abs(sk) < 0.15 and abs(ek) < 0.25
    assert p > 0.01                                    # normality NOT rejected

    fat = np.random.default_rng(0).standard_t(3, 6000)
    _, ek_fat, p_fat = distribution_stats(fat)
    assert ek_fat > 1.0                                # heavy tails detected
    assert p_fat < 0.01                                # normality rejected


def test_tail_risk_and_jump_fraction():
    r = np.random.default_rng(0).normal(0.0, 0.01, 20000)
    var, cvar = tail_risk(r, level=0.95)
    assert var == pytest.approx(-1.645 * 0.01, rel=0.08)   # normal 5% quantile
    assert cvar < var                                       # shortfall is worse than VaR
    assert jump_fraction(r, k=4.0) < 0.002                  # ~0 jumps in a Gaussian

    spiked = r.copy()
    spiked[:200] = 0.20                                     # 1% of bars are 20σ moves
    assert jump_fraction(spiked, k=4.0) == pytest.approx(0.01, abs=0.002)


# =========================================================================== #
# Time series — memory (signal + null for each estimator)
# =========================================================================== #
def test_variance_ratio_clears_random_walk_null():
    """i.i.d. increments → VR ≈ 1 and a non-significant z, across seeds."""
    for seed in range(6):
        r = np.random.default_rng(seed).normal(0.0, 0.01, 4000)
        vr, z, p = variance_ratio(r, 5)
        assert vr == pytest.approx(1.0, abs=0.10)
        assert abs(z) < 2.6                     # would be ~63x larger with a bad scaling
        assert p > 0.01


def test_variance_ratio_detects_both_signs_of_memory():
    vr_up, z_up, p_up = variance_ratio(_ar1(0.3), 5)
    assert vr_up > 1.2 and z_up > 3.0 and p_up < 0.01        # trending
    vr_dn, z_dn, p_dn = variance_ratio(_ar1(-0.3), 5)
    assert vr_dn < 0.8 and z_dn < -3.0 and p_dn < 0.01       # mean-reverting
    assert np.isnan(variance_ratio(np.zeros(5), 5)[0])       # degenerate input


def test_hurst_clears_random_walk_null():
    """The Anis-Lloyd correction must recentre i.i.d. data on 0.5, not ~0.57."""
    for seed in range(5):
        r = np.random.default_rng(seed).normal(0.0, 0.01, 4000)
        assert hurst_exponent(r) == pytest.approx(0.5, abs=0.05)


def test_hurst_orders_persistent_above_antipersistent():
    # R/S measures long-range dependence, so a short-memory AR(1) moves it only
    # a little — the ordering is the honest assertion, not a magnitude.
    assert hurst_exponent(_ar1(0.3)) > hurst_exponent(_ar1(-0.3))
    assert np.isnan(hurst_exponent(np.zeros(10)))


def test_ljung_box_null_and_signal():
    white = np.random.default_rng(3).normal(0, 1, 3000)
    _, p_white = ljung_box(white, 10)
    assert p_white > 0.05                       # white noise is not rejected
    _, p_ar = ljung_box(_ar1(0.3), 10)
    assert p_ar < 1e-6                          # AR(1) is


def test_autocorrelation_recovers_ar1_coefficient():
    ac = autocorrelation(_ar1(0.5, n=8000), 3)
    assert ac[0] == pytest.approx(0.5, abs=0.05)
    assert ac[1] == pytest.approx(0.25, abs=0.06)   # phi^2
    assert np.all(np.isnan(autocorrelation(np.arange(3.0), 10)))


def test_arch_lm_null_and_signal():
    homo = np.random.default_rng(4).normal(0, 0.01, 4000)
    _, p_homo = arch_lm_test(homo, lags=5)
    assert p_homo > 0.05                        # constant variance → no ARCH

    rng = np.random.default_rng(5)              # volatility that clusters
    vol = np.zeros(4000)
    vol[0] = 0.01
    for i in range(1, 4000):
        vol[i] = np.sqrt(1e-6 + 0.85 * vol[i - 1] ** 2 + 0.1 * (0.01 * rng.normal()) ** 2)
    hetero = vol * rng.normal(0, 1, 4000)
    _, p_het = arch_lm_test(hetero, lags=5)
    assert p_het < 0.01


def test_adf_separates_unit_root_from_stationary():
    rng = np.random.default_rng(6)
    walk = np.cumsum(rng.normal(0, 1, 3000))        # unit root → NOT stationary
    stat_walk, ok_walk = adf_test(walk)
    assert not ok_walk and stat_walk > -3.0

    stat_ret, ok_ret = adf_test(np.diff(walk))      # its differences are stationary
    assert ok_ret and stat_ret < -3.43


# =========================================================================== #
# Time series — drawdown & the one-call report
# =========================================================================== #
def test_drawdown_stats_known_path():
    close = np.array([100.0, 120.0, 60.0, 90.0, 130.0])   # peak 120 → trough 60 = -50%
    max_dd, longest, ulcer = drawdown_stats(close)
    assert max_dd == pytest.approx(-0.5)
    assert longest == 2                                    # bars 2 and 3 are underwater
    assert ulcer > 0.0
    assert np.isnan(drawdown_stats([])[0])


def test_log_returns_drops_session_crossing_moves():
    """An overnight gap is not a one-bar return, and must not be counted as one."""
    close = np.array([100.0, 101.0, 102.0, 200.0, 202.0, 204.0])   # 102→200 is a gap
    sess = np.array([0, 0, 0, 1, 1, 1])
    naive = log_returns(close)
    split = log_returns(close, sessions=sess)
    assert naive.size == 5 and split.size == 4                     # the gap is dropped
    assert np.max(np.abs(naive)) > 0.6                             # the gap dominates
    assert np.max(np.abs(split)) < 0.02                            # …and is gone
    with pytest.raises(ValueError, match="sessions has length"):
        log_returns(close, sessions=np.array([0, 0]))


def test_analyze_timeseries_session_split_changes_the_verdict():
    """Reproduces the real defect: 2 gap bars flip the volatility-clustering call.

    Built to mirror what real 1-minute data does — a small number of overnight
    jumps among many intraday bars, large enough to swamp the squared-return
    regression that ARCH-LM depends on.
    """
    rng = np.random.default_rng(0)
    seg, n_days = [], 3
    level = 100.0
    for d in range(n_days):                       # clustered intraday vol per session
        vol = 0.001 * (1.0 + 4.0 * (np.arange(400) < 200))
        path = level * np.exp(np.cumsum(rng.normal(0, 1, 400) * vol))
        seg.append(path)
        level = path[-1] * 3.0                    # a violent overnight gap
    close = np.concatenate(seg)
    idx = pd.DatetimeIndex([
        pd.Timestamp("2023-03-01") + pd.Timedelta(days=d, minutes=i)
        for d in range(n_days) for i in range(400)])
    frame = pd.DataFrame({"Close": close}, index=idx)

    naive = analyze_timeseries(frame, symbol="X")
    split = analyze_timeseries(frame, symbol="X", sessions="date")
    assert split.extra["session_split"] and not naive.extra["session_split"]
    assert split.extra["n_returns"] == naive.extra["n_returns"] - (n_days - 1)
    # The gaps blow up the tails and hide the clustering that is really there.
    assert naive.excess_kurtosis > 10 * split.excess_kurtosis
    assert naive.arch_lm_p > 0.05 and split.arch_lm_p < 0.05
    with pytest.raises(ValueError, match="DatetimeIndex"):
        analyze_timeseries(pd.DataFrame({"Close": close}), sessions="date")


def test_analyze_timeseries_report_contract():
    df = synthetic_survivor_ohlcv(2500, seed=1, start_date="2010-01-04")
    rep = analyze_timeseries(df, symbol="SYN")
    assert isinstance(rep, TimeSeriesReport)
    assert rep.n_bars == 2500
    assert rep.memory in ("trending", "mean-reverting", "random-walk")
    assert rep.ann_vol_pct > 0 and rep.max_drawdown_pct <= 0
    assert "descriptive only" in rep.summary()             # the honesty disclaimer stays
    d = rep.as_dict()
    assert d["n_bars"] == 2500 and "memory" in d
    with pytest.raises(ValueError):
        analyze_timeseries(pd.DataFrame({"Volume": [1, 2, 3]}))


def test_analyze_timeseries_calls_synthetic_survivor_a_random_walk():
    """The generator makes i.i.d. returns; the diagnostic must not claim memory."""
    rejections = 0
    for seed in range(8):
        df = synthetic_survivor_ohlcv(2500, seed=seed, start_date="2010-01-04")
        if analyze_timeseries(df).random_walk_rejected:
            rejections += 1
    assert rejections <= 2                                  # ~nominal at 5%, not systematic


# =========================================================================== #
# Fundamentals — the point-in-time contract
# =========================================================================== #
def test_snapshot_refuses_lookahead_construction():
    with pytest.raises(ValueError, match="look-ahead"):
        FundamentalSnapshot(symbol="T", period_end=pd.Timestamp("2015-12-31"),
                            available_at=pd.Timestamp("2015-06-30"))
    ok = _snap()                                            # available_at after period_end
    assert ok.reporting_lag_days == 60
    assert ok.free_cash_flow == pytest.approx(100.0)


def test_series_as_of_is_strictly_point_in_time():
    a = _snap(period_end="2014-12-31", available_at="2015-03-01", net_income=10.0)
    b = _snap(period_end="2015-12-31", available_at="2016-03-01", net_income=20.0)
    ser = FundamentalSeries(symbol="T", snapshots=(b, a))    # deliberately out of order
    assert ser.snapshots[0] is a                            # sorted by availability

    assert ser.as_of("2015-01-01") is None                  # before ANY filing
    assert ser.as_of("2015-02-28") is None                  # day before the first filing
    assert ser.as_of("2015-03-01") is a                     # exact filing date counts
    assert ser.as_of("2016-02-29") is a                     # still the old one
    assert ser.as_of("2016-03-01") is b


def test_prior_year_lookup():
    snaps = tuple(_snap(period_end=f"{y}-12-31", available_at=f"{y + 1}-03-01")
                  for y in (2013, 2014, 2015))
    ser = FundamentalSeries(symbol="T", snapshots=snaps)
    assert ser.prior_year_of(snaps[2]).period_end == pd.Timestamp("2014-12-31")
    assert ser.prior_year_of(snaps[0]) is None              # nothing before the first


def test_align_fundamentals_never_leaks():
    idx = pd.date_range("2015-01-01", periods=400, freq="B")
    ser = FundamentalSeries(symbol="T", snapshots=(
        _snap(period_end="2014-12-31", available_at="2015-03-02", revenue=111.0),
        _snap(period_end="2015-06-30", available_at="2015-08-14", revenue=222.0),
    ))
    al = align_fundamentals(ser, idx)
    report = audit_point_in_time(al)
    assert report["violations"] == 0
    assert report["bars"] == 400

    # Explicit bar-by-bar: nothing before the filing, exact value after.
    # (All dates chosen are business days, so they exist in the index.)
    assert np.isnan(al.loc[pd.Timestamp("2015-02-27"), "revenue"])   # Friday before
    assert al.loc[pd.Timestamp("2015-03-02"), "revenue"] == 111.0
    assert al.loc[pd.Timestamp("2015-08-13"), "revenue"] == 111.0     # not yet filed
    assert al.loc[pd.Timestamp("2015-08-14"), "revenue"] == 222.0


def test_audit_point_in_time_catches_a_planted_violation():
    """The auditor must fail loudly when a filing date is later than its bar."""
    idx = pd.date_range("2015-01-05", periods=5, freq="B")
    bad = pd.DataFrame({
        "period_end": [pd.Timestamp("2014-12-31")] * 5,
        "available_at": [pd.Timestamp("2015-12-31")] * 5,     # public AFTER every bar
        "revenue": [1.0] * 5,
    }, index=idx)
    assert audit_point_in_time(bad)["violations"] == 5


# =========================================================================== #
# Fundamentals — ratios & scores against hand-computed answers
# =========================================================================== #
def test_compute_ratios_exact_values():
    r = compute_ratios(_snap(), price=10.0)               # 100 shares × 10 → mcap 1000
    assert r.earnings_yield == pytest.approx(0.10)
    assert r.book_to_price == pytest.approx(0.80)
    assert r.sales_to_price == pytest.approx(1.00)
    assert r.fcf_yield == pytest.approx(0.10)
    assert r.gross_margin == pytest.approx(0.40)
    assert r.operating_margin == pytest.approx(0.20)
    assert r.roe == pytest.approx(0.125)
    assert r.roa == pytest.approx(0.05)
    assert r.debt_to_equity == pytest.approx(0.75)
    assert r.current_ratio == pytest.approx(2.0)
    assert r.interest_coverage == pytest.approx(10.0)
    assert r.accruals == pytest.approx(-0.025)            # OCF exceeds earnings → healthy
    assert np.isnan(r.piotroski_f)                        # no prior → refuse, don't guess
    assert r.to_vector().shape == (len(FUNDAMENTAL_FEATURES),)


def test_ratios_are_nan_not_zero_when_inputs_missing():
    r = compute_ratios(_snap(revenue=float("nan"), total_equity=0.0), price=10.0)
    assert np.isnan(r.gross_margin) and np.isnan(r.sales_to_price)
    assert np.isnan(r.roe)                                # zero denominator → NaN, not inf
    assert not r.is_complete
    assert np.isnan(compute_ratios(_snap()).earnings_yield)   # no price → no valuation ratio


def test_altman_z_exact_value():
    # 1.2(.125) + 1.4(.15) + 3.3(.1) + 0.6(1000/1200) + 1.0(.5)
    assert altman_z_score(_snap(), market_cap=1000.0) == pytest.approx(1.69, abs=1e-9)
    assert np.isnan(altman_z_score(_snap(total_assets=0.0), market_cap=1000.0))


def test_piotroski_scores_a_perfect_and_a_failing_year():
    good = _snap(period_end="2015-12-31")
    prior = _snap(period_end="2014-12-31", net_income=60.0, operating_cash_flow=80.0,
                  total_debt=800.0, current_assets=400.0, revenue=800.0, gross_profit=280.0)
    assert piotroski_f_score(good, prior) == 9.0          # all nine signals improve

    # Mirror image: everything deteriorates.
    bad = _snap(period_end="2015-12-31", net_income=-50.0, operating_cash_flow=-70.0,
                total_debt=900.0, current_assets=300.0, revenue=600.0,
                gross_profit=150.0, shares_diluted=140.0)
    assert piotroski_f_score(bad, _snap(period_end="2014-12-31")) == 0.0
    assert np.isnan(piotroski_f_score(good, None))


# =========================================================================== #
# Fundamentals — sources (offline, injected transport)
# =========================================================================== #
_INCOME = [{"date": "2015-12-31", "period": "FY", "filingDate": "2016-02-24",
            "revenue": 1000, "grossProfit": 400, "operatingIncome": 200,
            "netIncome": 100, "interestExpense": 20, "weightedAverageShsOutDil": 100}]
_BALANCE = [{"date": "2015-12-31", "filingDate": "2016-02-24", "totalAssets": 2000,
             "totalLiabilities": 1200, "totalStockholdersEquity": 800, "totalDebt": 600,
             "totalCurrentAssets": 500, "totalCurrentLiabilities": 250,
             "retainedEarnings": 300, "cashAndCashEquivalents": 100}]
_CASH = [{"date": "2015-12-31", "filingDate": "2016-02-24",
          "operatingCashFlow": 150, "capitalExpenditure": -50}]


def test_fmp_parser_uses_the_real_filing_date():
    ser = FMPFundamentalsSource.parse("T", _INCOME, _BALANCE, _CASH)
    assert len(ser) == 1
    s = ser.snapshots[0]
    assert s.period_end == pd.Timestamp("2015-12-31")
    assert s.available_at == pd.Timestamp("2016-02-24")      # from the feed, not assumed
    assert s.revenue == 1000 and s.total_assets == 2000
    assert s.capex == -50 and s.free_cash_flow == pytest.approx(100.0)   # sign-agnostic
    r = compute_ratios(s, price=10.0)
    assert r.earnings_yield == pytest.approx(0.10)


def test_fmp_parser_assumes_a_lag_when_no_filing_date_is_given():
    income = [{**_INCOME[0]}]
    income[0].pop("filingDate")
    balance = [{**_BALANCE[0]}]
    balance[0].pop("filingDate")
    cash = [{**_CASH[0]}]
    cash[0].pop("filingDate")
    s = FMPFundamentalsSource.parse("T", income, balance, cash).snapshots[0]
    assert s.available_at == pd.Timestamp("2015-12-31") + pd.Timedelta(
        days=DEFAULT_REPORTING_LAG_DAYS)


def test_fmp_parser_refuses_a_filing_date_before_the_period_end():
    """A feed's bad date must degrade to the assumed lag, never become look-ahead."""
    income = [{**_INCOME[0], "filingDate": "2015-01-01"}]     # impossible
    s = FMPFundamentalsSource.parse("T", income, _BALANCE, _CASH).snapshots[0]
    assert s.available_at >= s.period_end


def test_fmp_fundamentals_source_offline_via_injected_transport():
    calls: list[str] = []

    def fake_get(url: str):
        calls.append(url)
        if "income-statement" in url:
            return _INCOME
        if "balance-sheet" in url:
            return _BALANCE
        return _CASH

    src = FMPFundamentalsSource(api_key="k", http_get=fake_get)
    ser = src.get_fundamentals("T")
    assert len(calls) == 3 and all("apikey=k" in u for u in calls)
    assert src.provides_filing_dates and src.name == "fmp-fundamentals"
    assert ser.snapshots[0].available_at == pd.Timestamp("2016-02-24")


def test_fmp_fundamentals_source_reports_the_plan_gate():
    from vpts.data.fetcher import DataFetchError

    def boom(url: str):
        import urllib.error
        raise urllib.error.HTTPError(url, 402, "Payment Required", {}, None)

    with pytest.raises(DataFetchError, match="higher FMP plan"):
        FMPFundamentalsSource(api_key="k", http_get=boom).get_fundamentals("T")
    with pytest.raises(DataFetchError, match="API key"):
        FMPFundamentalsSource(api_key=None, http_get=boom).get_fundamentals("T")


def test_synthetic_source_seed_is_stable_across_processes():
    """Seeds must not come from Python's per-process randomised string hash."""
    assert SyntheticFundamentalsSource.stable_seed(3, "AAPL") == (
        3 * 1_000_003 + __import__("zlib").crc32(b"AAPL")) % (2 ** 32)
    # Same symbol+seed → same stream; different symbol → different stream.
    s = SyntheticFundamentalsSource.stable_seed
    assert s(3, "AAPL") == s(3, "AAPL") and s(3, "AAPL") != s(3, "MSFT")


def test_synthetic_source_is_deterministic_and_lagged():
    a = SyntheticFundamentalsSource(seed=3, n_periods=8).get_fundamentals("X")
    b = SyntheticFundamentalsSource(seed=3, n_periods=8).get_fundamentals("X")
    assert [s.as_dict() for s in a] == [s.as_dict() for s in b]
    assert len(a) == 8
    assert all(s.reporting_lag_days == DEFAULT_REPORTING_LAG_DAYS for s in a)
    q = SyntheticFundamentalsSource(seed=3, n_periods=8, freq="Q").get_fundamentals("X")
    assert (q.snapshots[1].period_end - q.snapshots[0].period_end).days < 100   # quarterly
    with pytest.raises(ValueError):
        SyntheticFundamentalsSource(freq="W")


def test_fundamental_ratio_frame_indexed_by_availability():
    ser = SyntheticFundamentalsSource(seed=1, n_periods=6).get_fundamentals("X")
    fr = fundamental_ratio_frame(ser)
    assert list(fr.index) == [s.available_at for s in ser.snapshots]
    assert all(c in fr.columns for c in FUNDAMENTAL_FEATURES)


# =========================================================================== #
# Datasets — the evaluation contract
# =========================================================================== #
def test_feature_frame_is_point_in_time_and_complete():
    df = synthetic_survivor_ohlcv(2600, seed=3, start_date="2010-01-04")
    ser = SyntheticFundamentalsSource(seed=7, n_periods=11, start="2009-12-31").get_fundamentals("S")
    ff = fundamental_feature_frame(df, ser)
    assert list(ff.columns)[:len(FUNDAMENTAL_FEATURES)] == list(FUNDAMENTAL_FEATURES)
    assert audit_point_in_time(ff)["violations"] == 0
    # Nothing is knowable before the first filing.
    first = ser.snapshots[0].available_at
    before = ff.loc[ff.index < first, list(FUNDAMENTAL_FEATURES)]
    assert before.notna().to_numpy().sum() == 0


def test_feature_frame_ignores_the_future_entirely():
    """Truncating future filings must not change a single earlier row.

    The strongest available no-look-ahead check: rebuild the features from a
    series with the later filings deleted and demand bit-identical history.
    """
    df = synthetic_survivor_ohlcv(2600, seed=4, start_date="2010-01-04")
    full = SyntheticFundamentalsSource(seed=9, n_periods=11, start="2009-12-31").get_fundamentals("S")
    cut = FundamentalSeries(symbol="S", snapshots=full.snapshots[:6])
    boundary = full.snapshots[6].available_at

    a = fundamental_feature_frame(df, full).loc[df.index < boundary, list(FUNDAMENTAL_FEATURES)]
    b = fundamental_feature_frame(df, cut).loc[df.index < boundary, list(FUNDAMENTAL_FEATURES)]
    pd.testing.assert_frame_equal(a, b)


def test_fundamental_dataset_samples_once_per_filing():
    df = synthetic_survivor_ohlcv(2600, seed=5, start_date="2010-01-04")
    ser = SyntheticFundamentalsSource(seed=11, n_periods=44, start="2009-09-30",
                                      freq="Q").get_fundamentals("S")
    ds = build_fundamental_dataset(df, ser, horizon=20, symbol="S", min_samples=25)
    assert isinstance(ds, FactorDataset)
    assert ds.X.shape[1] == len(FUNDAMENTAL_FEATURES)
    assert len(ds) <= len(ser)                       # never more rows than filings
    assert ds.stride > 40                            # ~a quarter apart, not a few bars
    assert ds.purge_samples == 1                     # so the derived purge is honest
    assert np.all(np.isfinite(ds.X)) and np.all(np.isfinite(ds.y))

    # Every sampled bar postdates the filing it used.
    ff = fundamental_feature_frame(df, ser)
    av = pd.to_datetime(ff.loc[ds.timestamps, "available_at"])
    assert bool((av.to_numpy() <= ds.timestamps.to_numpy()).all())

    denser = build_fundamental_dataset(df, ser, horizon=20, symbol="S",
                                       rows_per_filing=3, min_samples=25)
    assert len(denser) > len(ds)                     # opt-in only, and it shrinks the stride
    assert denser.stride < ds.stride


def test_fundamental_dataset_raises_when_history_is_too_short():
    df = synthetic_survivor_ohlcv(600, seed=6, start_date="2015-01-02")
    ser = SyntheticFundamentalsSource(seed=2, n_periods=3, start="2014-12-31").get_fundamentals("S")
    with pytest.raises(ValueError, match="usable fundamental samples"):
        build_fundamental_dataset(df, ser, horizon=20, symbol="S")


def test_combined_dataset_stacks_structural_and_fundamental():
    from vpts.structure.models import STRUCTURAL_FEATURES

    df = synthetic_survivor_ohlcv(2600, seed=7, start_date="2010-01-04")
    ser = SyntheticFundamentalsSource(seed=13, n_periods=44, start="2009-09-30",
                                      freq="Q").get_fundamentals("S")
    ds = build_combined_dataset(df, ser, horizon=20, stride=10, symbol="S")
    assert ds.X.shape[1] == len(STRUCTURAL_FEATURES) + len(FUNDAMENTAL_FEATURES)
    assert ds.feature_names[:len(STRUCTURAL_FEATURES)] == STRUCTURAL_FEATURES
    assert np.all(np.isfinite(ds.X))
    assert len(ds.timestamps) == len(ds)


def test_fundamental_panel_is_ranked_and_market_neutral():
    frames, smap = _universe(n_names=8, trial=1)
    panel = build_fundamental_panel(frames, smap, horizon=20, rebalance=63)
    assert isinstance(panel, CrossSectionalPanel)
    assert panel.feature_names == FUNDAMENTAL_FEATURES
    assert panel.n_names == 8 and panel.n_dates > 5
    assert np.all(np.abs(panel.X) <= 0.5 + 1e-9)          # centred ranks
    for d in range(panel.n_dates):                        # each date sums to ~0 per feature
        rows = panel.X[panel.date_id == d]
        assert np.allclose(rows.sum(axis=0), 0.0, atol=1e-9)
    with pytest.raises(ValueError, match="need >="):
        build_fundamental_panel({k: frames[k] for k in list(frames)[:2]}, smap)


# --------------------------------------------------------------------------- #
# The mandatory pair: finds a planted edge, reports nothing on noise
# --------------------------------------------------------------------------- #
def test_fundamental_panel_finds_a_planted_edge():
    frames, smap = _universe(n_names=14, link_strength=6.0, trial=3)
    panel = build_fundamental_panel(frames, smap, horizon=20, rebalance=63)
    res = permutation_test_cross_sectional(panel, n_permutations=120, seed=0)
    assert res.real_ic > 0.05
    assert res.p_value < 0.05                              # the edge is really there


def test_fundamental_panel_clears_the_null():
    """Fundamentals generated independently of price must report nothing."""
    frames, smap = _universe(n_names=14, link_strength=0.0, trial=4)
    panel = build_fundamental_panel(frames, smap, horizon=20, rebalance=63)
    res = permutation_test_cross_sectional(panel, n_permutations=120, seed=0)
    assert res.p_value > 0.05
    assert abs(res.null_ic_mean) < 0.05                    # the null is centred on zero


def _run_all() -> int:  # pragma: no cover - manual runner
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} analysis tests …\n")
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
