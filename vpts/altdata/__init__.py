"""``vpts.altdata`` — integration points for alternative behavioral signals.

Interfaces only (no live data). Subclass :class:`AltSignalSource` to wire a real
**options-flow** (dealer gamma/skew positioning) or **sentiment** (news/social/retail)
feed; both are causal, per-bar signals that extend the behavioral feature set with
*direct* measurements of institutional positioning and crowd psychology — the things
the free OHLCV layer can only proxy.

>>> from vpts.altdata import NullAltSource, merge_alt_features
>>> # with no provider configured, the frame passes through unchanged:
>>> # merge_alt_features(behavioral_frame, NullAltSource(), "AAPL").equals(behavioral_frame)

Plug a provider into :func:`merge_alt_features` to append its columns to a behavioral
feature frame, then run the combined frame through the usual CPCV + permutation +
survivorship harness — alt signals are hypotheses too, judged on the same bars.
"""
from __future__ import annotations

from vpts.altdata.base import AltDataCapabilities, AltSignalSource
from vpts.altdata.integrate import merge_alt_features
from vpts.altdata.sources import NullAltSource, StaticAltSource

__all__ = [
    "AltSignalSource",
    "AltDataCapabilities",
    "NullAltSource",
    "StaticAltSource",
    "merge_alt_features",
]
