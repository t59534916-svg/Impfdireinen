"""Financial time-series analysis — distribution, memory, volatility, drawdown.

This is the *diagnostic* half of Act V. It answers "what kind of process is this
series?" — is it fat-tailed, does volatility cluster, does it deviate from a
random walk, how deep and how long are its drawdowns — using only
``numpy``/``pandas``/``scipy``, so it stays inside the dependency-light core.

Deliberately **not** a signal layer
-----------------------------------
None of these statistics is an edge. A rejected random walk says the increments
are not independent; it does not say the dependence is exploitable net of cost,
nor that it survives out-of-sample or the removal of survivorship conditioning.
The repo has an apparatus for that question (:func:`vpts.harness.honest_backtest`)
and this module's job is to *describe*, then hand any candidate to it. Every
statistic here is computed on the whole sample and is therefore **in-sample by
construction** — which is exactly why it is presented as description, not proof.

The tests (``tests/test_analysis.py``) check each estimator against inputs with
a known answer — a pure random walk must come back with VR ≈ 1, Hurst ≈ 0.5 and
a non-rejecting Ljung-Box; a planted AR(1) or GARCH-like series must be caught.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from vpts.analysis.models import TimeSeriesReport

_EPS = 1e-12

#: Asymptotic Dickey-Fuller critical values, constant-no-trend case (Fuller 1976).
#: Deliberately the *asymptotic* table rather than an interpolated response
#: surface: the honest statement is "beyond/short of the 5% critical value",
#: not a spuriously precise p-value.
ADF_CRITICAL: dict[int, float] = {1: -3.43, 5: -2.86, 10: -2.57}


def _clean(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


# --------------------------------------------------------------------------- #
# Returns & volatility
# --------------------------------------------------------------------------- #
def log_returns(
    close: Sequence[float], *, sessions: Optional[Sequence] = None
) -> np.ndarray:
    """Log returns of a price series (non-finite bars dropped).

    ``sessions`` is a per-bar label (e.g. the calendar date of each intraday
    bar). When supplied, returns that **span a session boundary** are dropped
    instead of being reported as one bar's move.

    This matters far more than it looks. On daily bars every row *is* a session,
    so the default is correct. On 1-minute bars, an overnight gap is a ~17.5-hour
    move wearing a 1-minute label: on one month of AAPL minute bars, 19 such
    returns out of 7,389 (**0.26% of the data**) inflated excess kurtosis from
    6.6 to 240.4, flipped skew from -0.10 to +6.71, and turned the ARCH-LM test
    from p = 1.4e-74 (overwhelming volatility clustering) to p = 0.988 (none) —
    a completely inverted conclusion from a quarter of a percent of the rows.
    """
    c = np.asarray(close, dtype=float).ravel()
    ok = np.isfinite(c) & (c > 0)
    if sessions is None:
        c = c[ok]
        if c.size < 2:
            return np.empty(0, dtype=float)
        return np.diff(np.log(c))

    s = np.asarray(sessions).ravel()
    if s.size != c.size:
        raise ValueError(f"sessions has length {s.size}, expected {c.size}.")
    c, s = c[ok], s[ok]
    if c.size < 2:
        return np.empty(0, dtype=float)
    r = np.diff(np.log(c))
    return r[s[1:] == s[:-1]]


def realized_vol(returns: Sequence[float], *, periods_per_year: float = 252.0) -> float:
    """Annualised close-to-close volatility (sample std × √ppy)."""
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def parkinson_vol(
    high: Sequence[float], low: Sequence[float], *, periods_per_year: float = 252.0
) -> float:
    """Parkinson (1980) high-low range volatility, annualised.

    Uses the bar's *range* rather than its close, so it is ~5× more efficient
    than close-to-close for the same sample — at the cost of assuming no drift
    and no overnight gap.
    """
    h, l = np.asarray(high, float).ravel(), np.asarray(low, float).ravel()
    n = min(h.size, l.size)
    h, l = h[:n], l[:n]
    m = np.isfinite(h) & np.isfinite(l) & (h > 0) & (l > 0)
    if m.sum() < 2:
        return float("nan")
    hl = np.log(h[m] / l[m]) ** 2
    var = hl.mean() / (4.0 * np.log(2.0))
    return float(np.sqrt(max(var, 0.0) * periods_per_year))


def garman_klass_vol(
    open_: Sequence[float], high: Sequence[float], low: Sequence[float],
    close: Sequence[float], *, periods_per_year: float = 252.0,
) -> float:
    """Garman-Klass (1980) OHLC volatility estimator, annualised.

    Combines the range and the open-close move; more efficient still than
    Parkinson, and like it assumes a driftless, gap-free process.
    """
    o, h = np.asarray(open_, float).ravel(), np.asarray(high, float).ravel()
    l, c = np.asarray(low, float).ravel(), np.asarray(close, float).ravel()
    n = min(o.size, h.size, l.size, c.size)
    o, h, l, c = o[:n], h[:n], l[:n], c[:n]
    m = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    m &= (o > 0) & (h > 0) & (l > 0) & (c > 0)
    if m.sum() < 2:
        return float("nan")
    hl = np.log(h[m] / l[m]) ** 2
    co = np.log(c[m] / o[m]) ** 2
    var = (0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co).mean()
    return float(np.sqrt(max(var, 0.0) * periods_per_year))


# --------------------------------------------------------------------------- #
# Distribution & tails
# --------------------------------------------------------------------------- #
def distribution_stats(returns: Sequence[float]) -> tuple[float, float, float]:
    """Return ``(skew, excess_kurtosis, jarque_bera_p)``.

    Excess kurtosis is Fisher's (normal → 0.0). The Jarque-Bera p-value is the
    probability of seeing this much skew+kurtosis under normality — for daily
    equity returns it is essentially always ~0, which is the point.
    """
    r = _clean(returns)
    if r.size < 8:
        return float("nan"), float("nan"), float("nan")
    sk = float(stats.skew(r, bias=False))
    ek = float(stats.kurtosis(r, fisher=True, bias=False))
    try:
        _, p = stats.jarque_bera(r)
    except Exception:  # noqa: BLE001 - scipy raises on degenerate input
        p = float("nan")
    return sk, ek, float(p)


def tail_risk(returns: Sequence[float], *, level: float = 0.95) -> tuple[float, float]:
    """Historical ``(VaR, CVaR)`` at *level*, as **negative** return fractions.

    VaR is the ``1-level`` empirical quantile; CVaR (expected shortfall) is the
    mean of the returns at or below it — the number that actually describes what
    a bad day costs, since VaR is silent about the shape beyond the threshold.
    """
    r = _clean(returns)
    if r.size < 20:
        return float("nan"), float("nan")
    var = float(np.quantile(r, 1.0 - level))
    tail = r[r <= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar


def jump_fraction(returns: Sequence[float], *, k: float = 4.0) -> float:
    """Share of bars whose return is more than *k* robust sigmas from the median.

    Uses the median and the MAD-implied sigma so that the jumps themselves do
    not inflate the yardstick used to detect them.
    """
    r = _clean(returns)
    if r.size < 20:
        return float("nan")
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    sigma = mad * 1.4826
    if sigma <= _EPS:
        return float("nan")
    return float(np.mean(np.abs(r - med) > k * sigma))


# --------------------------------------------------------------------------- #
# Memory / efficiency
# --------------------------------------------------------------------------- #
def autocorrelation(x: Sequence[float], lags: int = 10) -> np.ndarray:
    """Sample autocorrelations for lags ``1..lags`` (biased 1/n estimator)."""
    a = _clean(x)
    n = a.size
    if n < lags + 2:
        return np.full(lags, np.nan)
    a = a - a.mean()
    denom = float(np.sum(a * a))
    if denom <= _EPS:
        return np.full(lags, np.nan)
    return np.array([float(np.sum(a[k:] * a[:-k]) / denom) for k in range(1, lags + 1)])


def ljung_box(x: Sequence[float], lags: int = 10) -> tuple[float, float]:
    """Ljung-Box portmanteau test for joint autocorrelation → ``(Q, p)``.

    Small p ⇒ the first *lags* autocorrelations are jointly non-zero, i.e. the
    series is not white noise.
    """
    a = _clean(x)
    n = a.size
    if n < lags + 5:
        return float("nan"), float("nan")
    rho = autocorrelation(a, lags)
    if not np.all(np.isfinite(rho)):
        return float("nan"), float("nan")
    k = np.arange(1, lags + 1)
    q = float(n * (n + 2) * np.sum(rho ** 2 / (n - k)))
    return q, float(stats.chi2.sf(q, lags))


def variance_ratio(returns: Sequence[float], q: int = 5) -> tuple[float, float, float]:
    """Lo-MacKinlay (1988) variance ratio → ``(VR, z, p)``, heteroskedasticity-robust.

    Under a random walk the variance of *q*-period returns is *q* times the
    one-period variance, so ``VR = 1``. ``VR > 1`` means positively autocorrelated
    increments (trending / momentum); ``VR < 1`` means mean reversion. The
    robust z-statistic is used because equity returns are conditionally
    heteroskedastic, and the homoskedastic version rejects a random walk purely
    on volatility clustering.
    """
    r = _clean(returns)
    n = r.size
    if q < 2 or n < 2 * q + 4:
        return float("nan"), float("nan"), float("nan")
    mu = float(r.mean())
    dev = r - mu
    var_1 = float(np.sum(dev ** 2) / (n - 1))
    if var_1 <= _EPS:
        return float("nan"), float("nan"), float("nan")

    # Overlapping q-period sums, with the Lo-MacKinlay unbiasing constant m.
    csum = np.concatenate(([0.0], np.cumsum(r)))
    q_sums = csum[q:] - csum[:-q]                     # length n - q + 1
    m = q * (n - q + 1) * (1.0 - q / n)
    if m <= _EPS:
        return float("nan"), float("nan"), float("nan")
    var_q = float(np.sum((q_sums - q * mu) ** 2) / m)
    vr = var_q / var_1

    # Heteroskedasticity-robust variance of VR (Lo-MacKinlay theorem 2).
    d2 = dev ** 2
    denom = float(np.sum(d2)) ** 2
    if denom <= _EPS:
        return float(vr), float("nan"), float("nan")
    theta = 0.0
    for j in range(1, q):
        delta_j = float(np.sum(d2[j:] * d2[:-j])) / denom
        theta += (2.0 * (q - j) / q) ** 2 * delta_j
    if theta <= _EPS:
        return float(vr), float("nan"), float("nan")
    # delta_j already carries the 1/n (its denominator is the *square* of the sum
    # of squared deviations), so theta ~ O(1/n) and z needs no further scaling.
    z = float((vr - 1.0) / np.sqrt(theta))
    return float(vr), z, float(2.0 * stats.norm.sf(abs(z)))


def _expected_rs(n: int) -> float:
    """Anis-Lloyd expected ``R/S`` for i.i.d. data of length *n* (Peters' form)."""
    i = np.arange(1, n)
    tail = float(np.sum(np.sqrt((n - i) / i)))
    if n <= 340:
        factor = float(np.exp(gammaln((n - 1) / 2.0) - gammaln(n / 2.0)) / np.sqrt(np.pi))
    else:
        factor = float(1.0 / np.sqrt(n * np.pi / 2.0))
    return (n - 0.5) / n * factor * tail


def hurst_exponent(x: Sequence[float], *, min_window: int = 16) -> float:
    """Rescaled-range Hurst exponent of a *return* series, Anis-Lloyd corrected.

    ``H ≈ 0.5`` random walk, ``H > 0.5`` persistent/trending, ``H < 0.5``
    anti-persistent. The raw R/S statistic is badly biased upward on the window
    sizes a few thousand bars allow — an i.i.d. series comes back near 0.57 —
    so the observed R/S is divided by its Anis-Lloyd expectation under
    independence before the log-log regression, which recentres i.i.d. data on
    0.5. Even corrected, treat roughly ``0.45-0.55`` as indistinguishable from a
    random walk rather than as weak evidence of memory.
    """
    a = _clean(x)
    n = a.size
    if n < 4 * min_window:
        return float("nan")
    windows, ratios = [], []
    w = min_window
    while w <= n // 2:
        rs_vals = []
        for i in range(n // w):
            seg = a[i * w:(i + 1) * w]
            sd = seg.std(ddof=1)
            if sd <= _EPS:
                continue
            dev = np.cumsum(seg - seg.mean())
            rng = float(dev.max() - dev.min())
            if rng > _EPS:
                rs_vals.append(rng / sd)
        if rs_vals:
            exp_rs = _expected_rs(w)
            if exp_rs > _EPS:
                windows.append(w)
                ratios.append(float(np.mean(rs_vals)) / exp_rs)
        w *= 2
    if len(windows) < 3:
        return float("nan")
    # log(R/S observed) - log(E[R/S]) has slope H - 0.5 against log(window).
    slope = float(np.polyfit(np.log(windows), np.log(ratios), 1)[0])
    return 0.5 + slope


def _ols_tstat(y: np.ndarray, X: np.ndarray, idx: int) -> float:
    """t-statistic of coefficient *idx* in an OLS fit of ``y`` on ``X``."""
    n, k = X.shape
    if n <= k:
        return float("nan")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    s2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:  # pragma: no cover - pinv is very forgiving
        return float("nan")
    se = float(np.sqrt(max(s2 * xtx_inv[idx, idx], 0.0)))
    if se <= _EPS:
        return float("nan")
    return float(beta[idx] / se)


def adf_test(x: Sequence[float], *, lags: int = 1) -> tuple[float, bool]:
    """Augmented Dickey-Fuller test (constant, no trend) → ``(t-stat, stationary@5%)``.

    Regresses ``Δy_t`` on ``y_{t-1}``, a constant and *lags* lagged differences;
    a sufficiently negative t-statistic on ``y_{t-1}`` rejects a unit root.
    Compared against the **asymptotic** critical values in :data:`ADF_CRITICAL`,
    so near-boundary results should be read as "inconclusive", not decided.
    """
    a = _clean(x)
    n = a.size
    if n < 4 * (lags + 2):
        return float("nan"), False
    dy = np.diff(a)
    y_lag = a[:-1]
    rows = n - 1 - lags
    if rows <= lags + 3:
        return float("nan"), False
    Y = dy[lags:]
    cols = [y_lag[lags:], np.ones(rows)]
    for i in range(1, lags + 1):
        cols.append(dy[lags - i: len(dy) - i])
    X = np.column_stack(cols)
    t = _ols_tstat(Y, X, 0)
    return t, bool(np.isfinite(t) and t < ADF_CRITICAL[5])


def arch_lm_test(returns: Sequence[float], *, lags: int = 5) -> tuple[float, float]:
    """Engle's ARCH-LM test for conditional heteroskedasticity → ``(LM, p)``.

    Regresses squared returns on their own lags; ``LM = n·R²`` is χ²(lags) under
    "no ARCH effects". Small p ⇒ volatility clusters, the defining stylised fact
    of financial returns (and the reason a homoskedastic variance ratio misleads).
    """
    r = _clean(returns)
    n = r.size
    if n < 5 * lags + 10:
        return float("nan"), float("nan")
    r2 = (r - r.mean()) ** 2
    rows = n - lags
    Y = r2[lags:]
    X = np.column_stack([np.ones(rows)] + [r2[lags - i: n - i] for i in range(1, lags + 1)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    if ss_tot <= _EPS:
        return float("nan"), float("nan")
    r_sq = 1.0 - float(resid @ resid) / ss_tot
    lm = float(rows * max(r_sq, 0.0))
    return lm, float(stats.chi2.sf(lm, lags))


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
def drawdown_curve(close: Sequence[float]) -> np.ndarray:
    """Fractional drawdown from the running peak (≤ 0 at every point)."""
    c = _clean(close)
    if c.size == 0:
        return np.empty(0, dtype=float)
    peak = np.maximum.accumulate(c)
    return c / peak - 1.0


def drawdown_stats(close: Sequence[float]) -> tuple[float, int, float]:
    """Return ``(max_drawdown_frac, longest_underwater_bars, ulcer_index)``.

    The ulcer index — RMS of the drawdown curve — is reported alongside the max
    because a single deep spike and a long shallow grind have very different
    lived risk but can share a maximum.
    """
    dd = drawdown_curve(close)
    if dd.size == 0:
        return float("nan"), 0, float("nan")
    max_dd = float(dd.min())
    ulcer = float(np.sqrt(np.mean((dd * 100.0) ** 2)))
    longest = run = 0
    for v in dd:
        run = run + 1 if v < -_EPS else 0
        longest = max(longest, run)
    return max_dd, int(longest), ulcer


def rolling_correlation(a: Sequence[float], b: Sequence[float], window: int = 60) -> pd.Series:
    """Rolling Pearson correlation of two aligned return series."""
    sa, sb = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    return sa.rolling(window).corr(sb)


# --------------------------------------------------------------------------- #
# The one-call diagnostic
# --------------------------------------------------------------------------- #
def analyze_timeseries(
    frame: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    periods_per_year: float = 252.0,
    vr_q: int = 5,
    lb_lags: int = 10,
    arch_lags: int = 5,
    sessions: Optional[Sequence] | str = None,
) -> TimeSeriesReport:
    """Run the full descriptive battery over an OHLCV frame → :class:`TimeSeriesReport`.

    Only ``Close`` is required; ``Open/High/Low`` unlock the range-based
    volatility estimators (they come back ``NaN`` otherwise). Nothing here is
    predictive — see the module docstring.

    ``sessions``
        Per-bar session label, so return-based statistics never span a session
        boundary. Pass ``"date"`` to derive it from a ``DatetimeIndex`` — the
        right choice for **intraday** bars, where an untreated overnight gap is a
        multi-hour move mislabelled as one bar (see :func:`log_returns` for how
        badly that distorts the result). Leave ``None`` for daily-or-slower bars,
        where every row is already its own session.

        Drawdown is deliberately *not* session-split: the equity path is real
        across the gap even though the one-bar return is not.
    """
    if "Close" not in frame.columns:
        raise ValueError("analyze_timeseries needs at least a 'Close' column.")
    if isinstance(sessions, str):
        if sessions != "date":
            raise ValueError("sessions must be an array-like, 'date', or None.")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("sessions='date' needs a DatetimeIndex.")
        sessions = frame.index.normalize().to_numpy()
    close = frame["Close"].to_numpy(float)
    r = log_returns(close, sessions=sessions)
    if r.size < 30:
        raise ValueError(f"{symbol or '?'}: need ≥30 usable returns, got {r.size}.")

    n_years = r.size / periods_per_year
    total = float(np.exp(np.sum(r)))
    ann_ret = (total ** (1.0 / n_years) - 1.0) if n_years > 0 else float("nan")
    ann_vol = realized_vol(r, periods_per_year=periods_per_year)
    sharpe = float(ann_ret / ann_vol) if (np.isfinite(ann_vol) and ann_vol > _EPS) else float("nan")

    sk, ek, jb_p = distribution_stats(r)
    var, cvar = tail_risk(r)
    ac = autocorrelation(r, 1)
    _, lb_p = ljung_box(r, lb_lags)
    vr, vr_z, vr_p = variance_ratio(r, vr_q)
    hurst = hurst_exponent(r)
    adf_stat, adf_stat_ok = adf_test(r, lags=1)
    abs_ac = autocorrelation(np.abs(r), 1)
    _, arch_p = arch_lm_test(r, lags=arch_lags)

    has_ohlc = all(c in frame.columns for c in ("Open", "High", "Low"))
    pk = parkinson_vol(frame["High"], frame["Low"], periods_per_year=periods_per_year) \
        if all(c in frame.columns for c in ("High", "Low")) else float("nan")
    gk = garman_klass_vol(frame["Open"], frame["High"], frame["Low"], frame["Close"],
                          periods_per_year=periods_per_year) if has_ohlc else float("nan")

    max_dd, dd_bars, ulcer = drawdown_stats(close)
    is_dt = isinstance(frame.index, pd.DatetimeIndex) and len(frame.index) > 0

    return TimeSeriesReport(
        symbol=symbol,
        n_bars=int(len(frame)),
        start=frame.index[0] if is_dt else None,
        end=frame.index[-1] if is_dt else None,
        periods_per_year=float(periods_per_year),
        ann_return_pct=float(ann_ret * 100.0),
        ann_vol_pct=float(ann_vol * 100.0),
        sharpe=sharpe,
        skew=sk,
        excess_kurtosis=ek,
        jarque_bera_p=jb_p,
        var_95_pct=float(var * 100.0),
        cvar_95_pct=float(cvar * 100.0),
        jump_frac=jump_fraction(r),
        autocorr_1=float(ac[0]),
        ljung_box_p=lb_p,
        variance_ratio=vr,
        vr_z=vr_z,
        vr_p=vr_p,
        hurst=hurst,
        adf_stat=adf_stat,
        adf_stationary=adf_stat_ok,
        abs_autocorr_1=float(abs_ac[0]),
        arch_lm_p=arch_p,
        parkinson_vol_pct=float(pk * 100.0),
        garman_klass_vol_pct=float(gk * 100.0),
        max_drawdown_pct=float(max_dd * 100.0),
        max_drawdown_bars=dd_bars,
        ulcer_index=ulcer,
        extra={"vr_q": vr_q, "lb_lags": lb_lags, "arch_lags": arch_lags,
               "n_returns": int(r.size), "session_split": sessions is not None,
               # With sessions applied, every return-based figure — including the
               # annualised return — is INTRADAY-ONLY: overnight moves are excluded
               # by construction, so this is not a buy-and-hold return.
               "returns_exclude_overnight": sessions is not None},
    )
