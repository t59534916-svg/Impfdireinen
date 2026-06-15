"""Merge alternative signals into a behavioral feature frame — only what exists.

:func:`merge_alt_features` appends an alt provider's options-flow / sentiment series
to a :func:`~vpts.features.behavioral_feature_frame`-style frame, **skipping columns
that are entirely missing** so the merged frame never introduces all-NaN columns
that would break the harness's finite-row requirement. The provider is responsible
for causality; this helper does not shift or look ahead.
"""
from __future__ import annotations

import pandas as pd

from vpts.altdata.base import AltSignalSource


def merge_alt_features(
    frame: pd.DataFrame,
    source: AltSignalSource,
    symbol: str,
    *,
    prefix: str = "alt_",
) -> pd.DataFrame:
    """Return *frame* plus any non-empty alt-signal columns from *source*.

    Columns are named ``{prefix}options_flow`` / ``{prefix}sentiment``. A signal the
    provider can't supply for *symbol* (all-NaN) is omitted, so with a
    :class:`~vpts.altdata.sources.NullAltSource` the frame is returned unchanged.
    """
    out = frame.copy()
    caps = source.capabilities
    candidates = []
    if caps.has_options_flow:
        candidates.append((f"{prefix}options_flow", source.options_flow(symbol, frame.index)))
    if caps.has_sentiment:
        candidates.append((f"{prefix}sentiment", source.sentiment(symbol, frame.index)))

    for col, series in candidates:
        series = series.reindex(frame.index)
        if series.notna().any():            # only add columns that carry data
            out[col] = series
    return out
