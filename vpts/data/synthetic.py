"""Deterministic synthetic OHLCV generators (survivor & delisted paths).

These are promoted, library-grade versions of the generator that lived in
``examples/survivorship_stress.py``. Two roles:

1. **Offline testing / demos** — let the whole pipeline run with zero network.
2. **Survivorship stress** — the *delisted* generator produces the "name that
   died" path (normal, then a negative-drift, elevated-volatility decline to a
   penny floor with capitulation volume) that `RESEARCH.md` injects to show the
   structural edge is a survivorship mirage. Keeping one canonical implementation
   means every experiment stresses survivorship the same, documented way.

The processes are **strategy-agnostic** (not tuned to any model) and fully
seeded, so results are reproducible.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _ohlc_from_close(close: np.ndarray, bar_vol: np.ndarray, rng: np.random.Generator):
    """Build self-consistent O/H/L from a close path and per-bar volatility."""
    high = close * (1.0 + np.abs(rng.normal(0, bar_vol)))
    low = close * (1.0 - np.abs(rng.normal(0, bar_vol)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, close, open_])
    low = np.minimum.reduce([low, close, open_])
    return open_, high, low


def synthetic_survivor_ohlcv(
    n: int = 504, *, seed: int = 0, start_date: str = "2015-01-02"
) -> pd.DataFrame:
    """A 'name that lived': a near-random-walk with mild positive drift.

    Deliberately **near-efficient** — returns are i.i.d. with a small drift, so the
    forward return is *not* predictable from past bars. This matters: a survivor
    built from a smooth deterministic cycle would let any feature "predict" it and
    manufacture a fake edge in demos/tests. A geometric random walk is the honest
    null — features should score ≈0 IC on it. Volume carries a benign cycle (so
    RVOL-type features have something to measure) that is uncorrelated with returns.
    Deterministic given *seed*.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    start = rng.uniform(30.0, 150.0)
    mu = rng.uniform(0.0001, 0.0004)          # mild positive drift
    sigma = rng.uniform(0.010, 0.018)         # daily volatility
    rets = rng.normal(mu, sigma, n)           # i.i.d. returns ⇒ no free predictability
    close = np.maximum(start * np.exp(np.cumsum(rets)), 1.0)
    bar_vol = sigma * (1.0 + 0.3 * np.abs(np.sin(t / 26.0)))
    open_, high, low = _ohlc_from_close(close, bar_vol, rng)
    volume = (3e6 + 1e6 * np.cos(t / 26.0) + rng.integers(-3e5, 3e5, n)).astype(float)
    volume = np.maximum(volume, 1e5)
    idx = pd.date_range(start_date, periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


#: Expected bear-rally bursts per bar of the decline, by `rally` mode.
_RALLY_RATE = {"off": 0.0, "mild": 0.02, "strong": 0.05}


def synthetic_delisted_ohlcv(
    n: int = 504,
    *,
    seed: int = 0,
    start_date: str = "2015-01-02",
    terminal_frac: Optional[float] = None,
    rally: str = "off",
) -> pd.DataFrame:
    """A 'name that died': ~normal, then a sustained decline to pennies.

    Canonical survivorship-injection path (mirrors the `RESEARCH.md` harness): a
    flat-ish first phase, then a negative-drift, elevated-volatility decline
    floored near zero, with capitulation volume spikes on big down days.

    Parameters
    ----------
    terminal_frac:
        If given, the decline drift is set deterministically so the name ends at
        ≈ ``terminal_frac × start`` (e.g. ``0.1`` = a 90% terminal loss, calibrated
        to performance-related delisting returns). ``None`` uses a random
        −0.2%…−0.5%/day drift. Realistic delistings are *slow* (months/quarters),
        which this preserves — the decline spans the post-``cut`` portion, not days.
    rally:
        Counter-trend **bear-rally** structure layered on the decline:
        ``"off"`` (monotone, the original), ``"mild"``, or ``"strong"`` (or a float
        rate). Poisson-timed +15–40% bounces over 5–20 bars. When ``terminal_frac``
        is set the base drift is **rescaled so the terminal loss is unchanged** — so
        this tests whether a result depends on the decline being structureless, not
        on how far it falls. (The original "inversion" was found with ``rally="off"``;
        dip-buying signals can win on a bounce, so rallies are the adversarial case.)
    """
    rng = np.random.default_rng(seed)
    start = rng.uniform(40.0, 160.0)
    cut = int(rng.uniform(0.2, 0.5) * n)               # when the decline begins
    vol_n, vol_d = rng.uniform(0.012, 0.022), rng.uniform(0.03, 0.06)
    m = max(n - cut, 1)

    # Bear-rally bursts over the decline phase (positive log-return runs).
    rate = _RALLY_RATE[rally] if isinstance(rally, str) and rally in _RALLY_RATE else None
    if rate is None:
        try:
            rate = float(rally)
        except (TypeError, ValueError):
            raise ValueError(f"rally must be one of {sorted(_RALLY_RATE)} or a float.")
    rally_rets = np.zeros(m)
    for _ in range(int(rng.poisson(rate * m))):
        w = int(rng.integers(5, 21))
        s0 = int(rng.integers(0, max(1, m - w)))
        rally_rets[s0:s0 + w] += np.log1p(rng.uniform(0.15, 0.40)) / w   # +15–40% over w bars
    rally_total = float(rally_rets.sum())

    if terminal_frac is not None:
        if not 0.0 < terminal_frac < 1.0:
            raise ValueError("terminal_frac must be in (0, 1).")
        # rescale base drift so terminal ≈ start*terminal_frac DESPITE the rallies
        drift_d = float((np.log(terminal_frac) - rally_total) / m)
    else:
        drift_d = -rng.uniform(0.002, 0.005)           # -0.2%..-0.5% / day
    rets = np.empty(n)
    rets[:cut] = rng.normal(0.0, vol_n, cut)
    rets[cut:] = rng.normal(drift_d, vol_d, m) + rally_rets
    close = np.maximum(start * np.exp(np.cumsum(rets)), 0.20)   # penny floor
    bar_vol = np.where(np.arange(n) < cut, vol_n, vol_d)
    open_, high, low = _ohlc_from_close(close, bar_vol, rng)
    base = rng.uniform(2e6, 8e6)
    volume = base * (1.0 + 3.0 * np.maximum(0.0, -rets) / (vol_d + 1e-9))
    idx = pd.date_range(start_date, periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
