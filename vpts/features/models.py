"""Configuration and the canonical behavioral feature spec.

:func:`feature_spec` is the single source of truth: it maps a
:class:`BehavioralConfig` to an ordered list of ``(name, fn, kwargs)`` so the
feature *names* and the feature *values* can never drift apart. The dataset
builder and the exported :data:`BEHAVIORAL_FEATURES` both derive from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vpts.features import behavioral as bh


@dataclass(frozen=True)
class BehavioralConfig:
    """Window choices for the multi-horizon behavioral feature set.

    The horizons act as *timeframes*: ``short`` ≈ a week, ``med`` ≈ a month,
    ``long`` ≈ a quarter of daily bars. Computing the same behavior at several
    horizons is the (strictly causal) multi-timeframe construction — no calendar
    resampling, so no alignment look-ahead risk in the core path.
    """

    short: int = 5
    med: int = 21
    long: int = 63
    grab_lookback: int = 20


#: One feature = (column name, primitive function, kwargs). Order is the X-column order.
FeatureSpec = list[tuple[str, Callable, dict]]


def feature_spec(config: BehavioralConfig = BehavioralConfig()) -> FeatureSpec:
    """Return the ordered behavioral feature spec for *config*."""
    s, m, l, g = config.short, config.med, config.long, config.grab_lookback
    return [
        (f"rvol_{s}", bh.relative_volume, {"window": s}),
        (f"rvol_{m}", bh.relative_volume, {"window": m}),
        (f"absorption_{m}", bh.absorption, {"window": m}),
        (f"clv_pressure_{s}", bh.clv_pressure, {"window": s}),
        (f"clv_pressure_{m}", bh.clv_pressure, {"window": m}),
        (f"accum_slope_{m}", bh.accumulation_slope, {"window": m}),
        (f"accum_slope_{l}", bh.accumulation_slope, {"window": l}),
        (f"liquidity_grab_{g}", bh.liquidity_grab, {"lookback": g}),
        (f"wick_asym_{m}", bh.wick_asymmetry, {"window": m}),
        (f"price_ext_{m}", bh.price_extension, {"window": m}),
        (f"price_ext_{l}", bh.price_extension, {"window": l}),
        (f"vol_expansion_{s}_{m}", bh.volatility_expansion, {"short": s, "long": m}),
        (f"trend_emergence_{m}", bh.trend_emergence, {"window": m}),
        (f"conviction_shift_{m}", bh.conviction_shift, {"window": m}),
        (f"fomo_thrust_{m}", bh.fomo_thrust, {"window": m}),
    ]


def behavioral_feature_names(config: BehavioralConfig = BehavioralConfig()) -> tuple[str, ...]:
    return tuple(name for name, _, _ in feature_spec(config))


#: Default feature-name list (15 multi-horizon behavioral features).
BEHAVIORAL_FEATURES: tuple[str, ...] = behavioral_feature_names()
