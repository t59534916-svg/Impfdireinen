"""Behavioral-dynamics feature primitives (causal, single-instrument OHLCV).

Each function turns OHLCV into a per-bar series that proxies a *human/institutional
behavior* rather than a generic indicator. The behavioral reading for each is in
its docstring. Everything here is **causal**: a feature at bar *t* uses only bars
``<= t`` (rolling/shift only), so the no-look-ahead invariant holds by construction
(and is unit-tested via truncation invariance).

These are *hypotheses*, not validated edges. The project's `RESEARCH.md` already
found single-timeframe feature richness did not beat the wall; the point here is to
express the behavioral ideas precisely and feed them through the same CPCV +
permutation + survivorship harness, then report what survives — including nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def _ret(close: pd.Series) -> pd.Series:
    return close.pct_change()


def _zscore(s: pd.Series, window: int) -> pd.Series:
    """Trailing z-score over *window* (causal; min_periods=window)."""
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std()
    return (s - mean) / (std + _EPS)


def close_location_value(df: pd.DataFrame) -> pd.Series:
    """Per-bar CLV ∈ [−1, +1]: where the bar closed within its range.

    **Behavior:** closing near the high (CLV→+1) is buyers winning the day's
    auction (accumulation); near the low (CLV→−1) is sellers in control
    (distribution). The raw building block for order-flow pressure proxies.
    """
    rng = (df["High"] - df["Low"]).replace(0.0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    return clv.fillna(0.0)


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """RVOL − 1: current volume vs its trailing average.

    **Behavior:** a participation surge. Crowds act together — earnings, news,
    capitulation, breakouts all show up as volume well above the recent norm.
    Centred at 0 (0 = average participation, +1 = double).
    """
    avg = df["Volume"].rolling(window, min_periods=window).mean()
    return (df["Volume"] / (avg + _EPS)) - 1.0


def absorption(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Effort-minus-result: z(volume) − z(|return|) over *window*.

    **Behavior:** high effort (volume) with little result (price move) is
    *absorption* — a large passive participant soaking up market orders at a price
    (icebergs at support/resistance). Positive ⇒ effort without result (absorption);
    negative ⇒ result without effort (thin, easily-moved tape).
    """
    vol_z = _zscore(df["Volume"], window)
    move_z = _zscore(_ret(df["Close"]).abs(), window)
    return (vol_z - move_z).fillna(0.0)


def clv_pressure(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling mean CLV: sustained accumulation vs distribution pressure.

    **Behavior:** a run of closes near the highs (positive) is persistent demand
    absorbing supply — accumulation; near the lows (negative) is distribution.
    Smooths single-bar CLV into a conviction-direction read.
    """
    return close_location_value(df).rolling(window, min_periods=window).mean().fillna(0.0)


def accumulation_slope(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Normalized slope of the CLV×volume accumulation line over *window*.

    **Behavior:** the Williams Accumulation/Distribution idea — signed volume
    (CLV×V) cumulated. A rising line on flat price is stealth accumulation
    (smart money building before the move); a falling line on flat price is
    distribution into strength. Normalized by trailing dollar-ish volume so it is
    comparable across names.
    """
    ad = (close_location_value(df) * df["Volume"]).cumsum()
    delta = ad - ad.shift(window)
    norm = df["Volume"].rolling(window, min_periods=window).mean() * window
    return (delta / (norm + _EPS)).fillna(0.0)


def liquidity_grab(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Signed stop-run / liquidity-grab strength against the prior *lookback* range.

    **Behavior:** price pokes **below** a recent low (triggering resting stops /
    inducing sellers) then **closes back above** it — a bullish liquidity grab
    (+); the mirror above a recent high that closes back below is bearish (−).
    This is the "stop hunt then reverse" footprint of a large player sourcing
    liquidity. Strength = sweep depth as a fraction of the bar range.
    """
    prior_low = df["Low"].rolling(lookback, min_periods=lookback).min().shift(1)
    prior_high = df["High"].rolling(lookback, min_periods=lookback).max().shift(1)
    rng = (df["High"] - df["Low"]).replace(0.0, np.nan)
    bull = ((df["Low"] < prior_low) & (df["Close"] > prior_low)) * (prior_low - df["Low"]) / rng
    bear = ((df["High"] > prior_high) & (df["Close"] < prior_high)) * (df["High"] - prior_high) / rng
    return (bull.fillna(0.0) - bear.fillna(0.0)).fillna(0.0)


def wick_asymmetry(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling mean of (upper wick − lower wick) / range.

    **Behavior:** upper wicks are rejection of higher prices (supply stepping in /
    distribution); lower wicks are rejection of lower prices (demand / absorption).
    Persistent upper-wick dominance (positive) is sellers defending; lower-wick
    dominance (negative) is buyers defending.
    """
    body_hi = df[["Open", "Close"]].max(axis=1)
    body_lo = df[["Open", "Close"]].min(axis=1)
    rng = (df["High"] - df["Low"]).replace(0.0, np.nan)
    asym = ((df["High"] - body_hi) - (body_lo - df["Low"])) / rng
    return asym.fillna(0.0).rolling(window, min_periods=window).mean().fillna(0.0)


def price_extension(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Trailing z-score of close vs its moving average — how stretched price is.

    **Behavior:** the further price is from its recent anchor, the more the move is
    driven by late, emotional participants (greed on the upside, fear on the down).
    Extreme positive extension is the FOMO zone; extreme negative is capitulation.
    """
    return _zscore(df["Close"], window).fillna(0.0)


def fomo_thrust(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Price extension gated by a volume surge — the chase signal.

    **Behavior:** extension *with* a participation surge is the classic FOMO
    blow-off (everyone piling in at once, late); extension on quiet volume is far
    less emotional. ``price_extension × max(RVOL, 0)``, signed by the extension.
    """
    ext = price_extension(df, window)
    thrust = relative_volume(df, window).clip(lower=0.0)
    return (ext * thrust).fillna(0.0)


def volatility_expansion(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.Series:
    """Short- vs long-horizon realized-volatility ratio − 1.

    **Behavior:** markets coil (volatility contracts) then expand. A ratio rising
    through 0 is the *coil-to-expansion* transition — the moment a quiet,
    balanced auction breaks into a trend as one side gains conviction.
    """
    r = _ret(df["Close"])
    vs = r.rolling(short, min_periods=short).std()
    vl = r.rolling(long, min_periods=long).std()
    return ((vs / (vl + _EPS)) - 1.0).fillna(0.0)


def trend_emergence(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volatility expansion signed by momentum direction.

    **Behavior:** combines *that* a move is starting (expansion) with *which way*
    (momentum sign) — a directional read on trend formation out of a coil. Positive
    ⇒ expansion to the upside; negative ⇒ expansion to the downside.
    """
    mom = (df["Close"] / df["Close"].shift(window) - 1.0)
    return (volatility_expansion(df, max(2, window // 4), window) * np.sign(mom)).fillna(0.0)


def conviction_shift(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Change in the accumulation slope — acceleration of order-flow conviction.

    **Behavior:** not the level of buying pressure but its *change*. A positive
    shift is conviction building (accumulation accelerating); negative is conviction
    fading (the move losing its sponsor) — an early read on exhaustion/regime change.
    """
    acc = accumulation_slope(df, window)
    return (acc - acc.shift(window)).fillna(0.0)
