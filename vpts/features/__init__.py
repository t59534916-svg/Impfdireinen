"""``vpts.features`` — multi-timeframe behavioral-dynamics features.

Behavioral proxies for what humans and institutions actually *do* in the tape —
participation surges (RVOL), absorption (effort without result), liquidity grabs
(stop-runs), accumulation/distribution pressure and its acceleration (conviction
shift), FOMO extension, wick rejection, and coil-to-expansion trend formation —
each computed causally and at multiple horizons.

Honest scope
------------
These are **hypotheses**, not validated edges. `RESEARCH.md` already found that
single-timeframe feature richness did not beat the survivorship wall. The value
here is precise, causal behavioral definitions wired straight into the CPCV +
permutation + survivorship harness (:func:`build_behavioral_dataset` →
``FactorDataset``), so each idea is judged honestly and negatives are reported.
"""
from __future__ import annotations

from vpts.features import behavioral
from vpts.features.dataset import behavioral_feature_frame, build_behavioral_dataset
from vpts.features.indicators import build_indicator_dataset
from vpts.features.models import (
    BEHAVIORAL_FEATURES,
    BehavioralConfig,
    behavioral_feature_names,
    feature_spec,
)
from vpts.features.multitimeframe import (
    align_to_base,
    multi_timeframe_feature,
    resample_ohlcv,
)

__all__ = [
    "behavioral",
    "BehavioralConfig",
    "BEHAVIORAL_FEATURES",
    "behavioral_feature_names",
    "feature_spec",
    "behavioral_feature_frame",
    "build_behavioral_dataset",
    "build_indicator_dataset",
    "resample_ohlcv",
    "align_to_base",
    "multi_timeframe_feature",
]
