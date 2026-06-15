"""Tier 2b — does the structural signal behave differently OUT of its regime?

The whole `RESEARCH.md` arc lives in one regime: US large-cap equities, daily bars,
2012–2017, survivor-only. A fair question is whether the structural microstructure
edge is specific to that regime or shows up elsewhere. Crypto is a useful contrast:
a different asset class and era, 24/7 daily bars, and — usefully — **no delisting /
survivorship confound** for the majors (they were continuously traded over the
window), so the survivorship caveat that dogs the equity work doesn't apply here.

This reuses the harness **unchanged** (`build_structural_dataset` →
`cpcv_factor_eval` + block-permutation null) on a basket of large-cap crypto pairs
from a free, no-key feed (Binance.US klines), and prints the pooled OOS IC and
block-permutation p beside the equity baseline (the same 20 stocknet survivors).

    python examples/structural_out_of_regime.py
    python examples/structural_out_of_regime.py --perms 500 --symbols BTCUSDT ETHUSDT LTCUSDT

Reading: this is a generalization probe, not a tradeable result (crypto costs are
ignored). If the structural IC is materially different out-of-regime, that bounds
how much the equity finding can be read as a general microstructure law.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

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
from vpts.ml.models import FactorDataset  # noqa: E402
from vpts.stats import block_shuffle_indices, recommend_block_size  # noqa: E402

_UA = "Mozilla/5.0"
_KLINES = "https://api.binance.us/api/v3/klines?symbol={sym}&interval={iv}&limit={lim}"
DEFAULT_CRYPTO = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "BCHUSDT", "ADAUSDT",
                  "SOLUSDT", "DOGEUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT"]


def binance_loader(limit: int = 1000, interval: str = "1d"):
    """Return ``load(symbol) -> OHLCV df`` from free Binance.US klines (no key)."""

    def load(symbol: str) -> pd.DataFrame:
        url = _KLINES.format(sym=symbol, iv=interval, lim=limit)
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                rows = json.load(resp)
        except Exception as exc:
            raise DataFetchError(f"{symbol}: {type(exc).__name__}") from exc
        if not rows:
            raise DataFetchError(f"{symbol}: no klines")
        idx = pd.to_datetime([r[0] for r in rows], unit="ms")
        df = pd.DataFrame({
            "Open": [float(r[1]) for r in rows], "High": [float(r[2]) for r in rows],
            "Low": [float(r[3]) for r in rows], "Close": [float(r[4]) for r in rows],
            "Volume": [float(r[5]) for r in rows],
        }, index=idx)
        return df

    return load


def _eval_ics(ds, cv, alpha):
    try:
        return np.array(cpcv_factor_eval(ds, cv=cv, alpha=alpha).fold_ics, dtype=float)
    except ValueError:
        return None


def _build(frame, sym, args):
    ds = build_structural_dataset(frame, lookback=args.lookback, horizon=args.horizon,
                                  stride=args.stride, symbol=sym, interval="1d")
    cv = CombinatorialPurgedCV(n_groups=args.n_groups, n_test_groups=args.n_test,
                               purge=ds.purge_samples, embargo_pct=0.01)
    return ds, cv


def _pooled(entries, alpha) -> tuple[float, int]:
    ics, n = [], 0
    for ds, cv in entries:
        a = _eval_ics(ds, cv, alpha)
        if a is not None and a.size:
            ics.append(a); n += len(ds)
    return (float(np.concatenate(ics).mean()) if ics else float("nan")), n


def _block_perm_p(entries, real_ic, *, alpha, perms, seed=0) -> float:
    rng = np.random.default_rng(seed)
    null = np.empty(perms, dtype=float)
    for p in range(perms):
        folds = []
        for ds, cv in entries:
            bs = recommend_block_size(ds.horizon, ds.stride)
            perm = block_shuffle_indices(len(ds), bs, rng=rng)
            shuf = FactorDataset(X=ds.X, y=ds.y[perm], baseline=ds.baseline,
                                 feature_names=ds.feature_names, horizon=ds.horizon,
                                 stride=ds.stride, timestamps=ds.timestamps, symbol=ds.symbol)
            ic = _eval_ics(shuf, cv, alpha)
            if ic is not None:
                folds.append(ic)
        null[p] = float(np.concatenate(folds).mean()) if folds else np.nan
    null = null[np.isfinite(null)]
    return float((np.sum(null >= real_ic) + 1) / (null.size + 1))


def _load_basket(loader, names, args, label):
    entries = []
    for sym in names:
        try:
            entries.append(_build(loader(sym), sym, args))
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
    print(f"  {label}: built {len(entries)} names")
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description="Out-of-regime test: structural IC on crypto vs equities.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_CRYPTO, help="crypto pairs (Binance.US)")
    ap.add_argument("--limit", type=int, default=1000, help="daily klines per symbol (max 1000)")
    args = ap.parse_args()

    print("=" * 78)
    print("TIER 2b — structural IC OUT of regime (crypto, 24/7 daily, no delist confound)")
    print("=" * 78)

    print(f"\nLoading crypto basket from Binance.US (free, no key), {args.limit} daily bars …")
    crypto = _load_basket(binance_loader(limit=args.limit), args.symbols, args, "crypto")
    print("Loading equity baseline (stocknet survivors, 2012–2017) …")
    equity = _load_basket(github_loader(), [t for t, _ in GITHUB_TICKERS], args, "equity")

    if not crypto:
        print("\nNo crypto names built (network?)."); return 1

    eq_ic, eq_n = _pooled(equity, args.alpha)
    cr_ic, cr_n = _pooled(crypto, args.alpha)
    eq_p = _block_perm_p(equity, eq_ic, alpha=args.alpha, perms=args.perms) if equity else float("nan")
    cr_p = _block_perm_p(crypto, cr_ic, alpha=args.alpha, perms=args.perms)

    print("\n" + "-" * 66)
    print(f"{'regime':<34}{'pooled OOS IC':>14}{'p(block)':>10}{'samples':>8}")
    print("-" * 66)
    print(f"{'equity (US large-cap 2012-2017)':<34}{eq_ic:>+14.3f}{eq_p:>10.3f}{eq_n:>8}")
    print(f"{'crypto (majors, ~3y daily)':<34}{cr_ic:>+14.3f}{cr_p:>10.3f}{cr_n:>8}")
    print("-" * 66)

    sig = np.isfinite(cr_p) and cr_p < 0.05
    same_sign = np.isfinite(eq_ic) and np.sign(cr_ic) == np.sign(eq_ic)
    print()
    if sig and same_sign:
        print("Reading: the structural IC carries the SAME sign and is significant out-of-regime →")
        print("evidence the microstructure effect is not purely an equity-2012-2017 artifact.")
    elif sig and not same_sign:
        print("Reading: the structural IC is significant but FLIPS SIGN out-of-regime → the effect")
        print("is regime-dependent; the equity finding does not generalize as a directional law.")
    else:
        print("Reading: the structural IC is NOT significant out-of-regime (crypto) → the equity")
        print("finding does not obviously generalize; it may be specific to that asset class/era.")
    print("(Generalization probe only — crypto costs ignored; not a tradeable result.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
