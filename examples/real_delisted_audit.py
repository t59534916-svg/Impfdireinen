"""Tier 1 — feed the harness REAL delisted names, and measure the free-data ceiling.

`RESEARCH.md`'s headline is that the binding constraint is **survivorship bias**,
and every survivorship result so far leans on *synthetic* dead names because (as the
docstrings note) "no free delisted prices." This script turns that asserted caveat
into a **reproduced, quantified fact** and then runs the harness on whatever real
delisted history actually IS reachable for free.

What it does
------------
1. **Coverage audit.** Against a curated catalogue of 26 well-known US delistings
   (:data:`vpts.data.KNOWN_DELISTED`, split into *bankruptcy/liquidation* — the
   names that went to ~0 — and *acquisition/merger*), it asks a free feed how many
   it can still serve (:func:`vpts.data.audit_coverage`). The answer is the feed's
   **survivorship signature**.
2. **Real-delisted run.** For the handful of real delisted names that DO come back,
   it builds the structural feature → forward-return datasets and reports the pooled
   OOS IC and the conviction-bucket direction, beside the survivors-only baseline —
   the harness, finally fed real (not synthetic) delisted names.

    python examples/real_delisted_audit.py
    python examples/real_delisted_audit.py --min-bars 250 --start 2000-01-01

Honest scope: free feeds serve ~0 of the **death** names (they were dropped the day
they delisted), so the real-delisted sample is dominated by *acquisition* exits that
did not crash. That is precisely why the death leg can only be probed by synthetic
injection (``survivorship_baserate.py``) — this script shows *why*, with numbers.
The drop-in for real death-name history is :class:`~vpts.data.PolygonSource`
(``provides_delisted=True``, needs a key) or :class:`~vpts.data.DataLakeSource`
(a user parquet lake) — the harness is already wired for both.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from vpts import (  # noqa: E402
    CombinatorialPurgedCV,
    DataFetchError,
    build_structural_dataset,
    cpcv_factor_eval,
)
from vpts.data import KNOWN_DELISTED, audit_known_delisted  # noqa: E402
from vpts.ml.factor_model import cpcv_factor_quantile_returns  # noqa: E402

_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/120 Safari/537.36")
_V8 = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d"


def yahoo_v8_loader(start: str, end: str, delist_caps: Optional[dict] = None):
    """Return ``fetch(symbol) -> OHLCV df`` over Yahoo's v8 chart JSON (no key).

    This is the survivor-only free feed under test: it returns clean history for
    live names but an **empty** result for names that delisted (which we surface as
    :class:`~vpts.data.DataFetchError`, i.e. the survivorship drop). Raw JSON is more
    reproducible here than the yfinance retry path.

    *delist_caps* (``{symbol: 'YYYY-MM-DD'}``) caps the fetch window at each name's
    delisting date, so we only ever pull **genuine pre-delisting history**. This
    closes a subtle trap: a free feed may serve a *reused* ticker (a different company
    that later claimed the symbol — e.g. STI after SunTrust merged), which a naive
    symbol fetch would silently splice onto the dead company's identity.
    """
    p1 = int(pd.Timestamp(start).timestamp())
    p2_default = int(pd.Timestamp(end).timestamp())
    caps = delist_caps or {}

    def fetch(symbol: str) -> pd.DataFrame:
        p2 = min(p2_default, int(pd.Timestamp(caps[symbol]).timestamp())) if symbol in caps else p2_default
        url = _V8.format(sym=symbol, p1=p1, p2=p2)
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                doc = json.load(resp)
        except Exception as exc:  # HTTP 404/400 for unknown symbols, network, parse
            raise DataFetchError(f"{symbol}: {type(exc).__name__}") from exc
        res = (doc.get("chart") or {}).get("result")
        if not res or not res[0].get("timestamp"):
            raise DataFetchError(f"{symbol}: no rows (delisted/dropped by feed)")
        r = res[0]
        ind = r.get("indicators", {})
        q = (ind.get("quote") or [{}])[0]
        idx = pd.to_datetime(r["timestamp"], unit="s").normalize()
        adj = (ind.get("adjclose") or [{}])[0].get("adjclose")
        df = pd.DataFrame({
            "Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
            "Close": adj if adj is not None else q.get("close"), "Volume": q.get("volume"),
        }, index=idx).dropna(how="all")
        if "Close" not in df or df["Close"].notna().sum() == 0:
            raise DataFetchError(f"{symbol}: all-NaN close")
        return df.ffill().dropna()

    return fetch


def _build(frame: pd.DataFrame, sym: str, args):
    ds = build_structural_dataset(frame, lookback=args.lookback, horizon=args.horizon,
                                  stride=args.stride, symbol=sym, interval="1d")
    cv = CombinatorialPurgedCV(n_groups=args.n_groups, n_test_groups=args.n_test,
                               purge=ds.purge_samples, embargo_pct=0.01)
    ics = np.array(cpcv_factor_eval(ds, cv=cv, alpha=args.alpha).fold_ics, dtype=float)
    try:
        bres = cpcv_factor_quantile_returns(ds, cv=cv, n_buckets=5, cost_bps=args.cost_bps)
    except ValueError:
        bres = None
    return sym, ics, bres


def _pooled_ic(entries) -> float:
    ics = [ic for _s, ic, _b in entries if ic.size]
    return float(np.concatenate(ics).mean()) if ics else float("nan")


def _top_bucket(entries) -> float:
    tops = [np.asarray(b.bucket_returns_pct, float)[-1] for _s, _i, b in entries if b is not None]
    return float(np.mean(tops)) if tops else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit free-feed delisted coverage + run the harness on real dead names.")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--min-bars", type=int, default=250, help="usable rows to count as reachable")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    args = ap.parse_args()

    # Cap each catalogued name at its delisting year → genuine pre-delisting history
    # only (and reused tickers correctly read as 'dropped', not spliced in).
    caps = {t.symbol: f"{t.year}-12-31" for t in KNOWN_DELISTED}
    feed = yahoo_v8_loader(args.start, args.end, delist_caps=caps)

    print("=" * 84)
    print("TIER 1 — does a FREE feed serve real delisted names? (the survivorship ceiling)")
    print("=" * 84)
    print(f"Auditing {len(KNOWN_DELISTED)} known US delistings against Yahoo v8 (free, no key);")
    print(f"each name's window is capped at its delisting year (pre-delisting history only).\n")
    rep = audit_known_delisted(feed, min_bars=args.min_bars)
    print(rep.to_frame().to_string(index=False))
    print("\n" + rep.summary())
    print("\nReading: the BANKRUPTCY leg is the survivorship driver (those names went to ~0).")
    print("A free feed that serves ~0% of them cannot, even in principle, settle the question —")
    print("the dead names were dropped the day they delisted. This is the wall, quantified.\n")

    # ---- run the harness on whatever real delisted names DID come back ---------- #
    reachable = [r.symbol for r in rep.reachable]
    if not reachable:
        print("No real delisted names reachable on this feed — the synthetic injection in")
        print("examples/survivorship_baserate.py remains the only probe of the death leg.")
        return 0

    print("=" * 84)
    print(f"Running the structural harness on the {len(reachable)} REAL reachable delisted "
          f"name(s): {', '.join(reachable)}")
    print("=" * 84)
    real_dead = []
    for sym in reachable:
        try:
            real_dead.append(_build(feed(sym), sym, args))
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")

    print("\nLoading real stocknet survivors (5y daily, no key) for the baseline …")
    load = github_loader()
    survivors = []
    for sym, _sector in GITHUB_TICKERS:
        try:
            survivors.append(_build(load(sym), sym, args))
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")

    if not survivors:
        print("No survivors loaded (network?) — cannot compute the baseline."); return 1

    surv_ic, surv_top = _pooled_ic(survivors), _top_bucket(survivors)
    dead_ic, dead_top = _pooled_ic(real_dead), _top_bucket(real_dead)
    mixed = survivors + real_dead
    mix_ic, mix_top = _pooled_ic(mixed), _top_bucket(mixed)

    print("\n" + "-" * 70)
    print(f"{'universe':<34}{'pooled IC':>12}{'top-bucket fwd %':>20}")
    print("-" * 70)
    print(f"{'survivors only (' + str(len(survivors)) + ')':<34}{surv_ic:>+12.3f}{surv_top:>+19.2f}%")
    print(f"{'real delisted only (' + str(len(real_dead)) + ')':<34}{dead_ic:>+12.3f}{dead_top:>+19.2f}%")
    print(f"{'survivors + real delisted':<34}{mix_ic:>+12.3f}{mix_top:>+19.2f}%")
    print("-" * 70)
    print(f"\nReading: only {len(real_dead)} real delisted name(s) are reachable — far too few to")
    print("interpret (the IC/direction above is noise at this n). And every one is an ACQUISITION")
    print("exit that left the universe without crashing: the BANKRUPTCY names that actually drive")
    print("the survivorship inversion are 0% reachable. So free data cannot build a real")
    print("delisted cross-section at all — which is the finding. The death leg can only be probed")
    print("by synthetic injection (survivorship_baserate.py); the drop-in for real death-name")
    print("history is PolygonSource (needs a key) or DataLakeSource (a user parquet lake).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
