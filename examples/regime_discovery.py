"""Unsupervised regime / pattern discovery — does any discovered regime predict?

Builds the multi-timeframe behavioral feature matrix per name, assigns each bar a
regime **walk-forward** (clusters fit only on prior bars), and asks the honest
question: do the discovered regimes *separate the forward return* out-of-sample, or
is the separation within the label-shuffle null? Reported per name, then aggregated
against the ~5%-by-chance rate (so we don't fool ourselves with multiple testing).

    python examples/regime_discovery.py

Non-overlapping labels (stride = horizon) keep the per-name permutation null honest.
Honest scope: survivorship-biased survivors; a discovered cluster is a hypothesis
until it clears its null. No edge is assumed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from vpts import DataFetchError, build_behavioral_dataset  # noqa: E402
from vpts.ml.regime import regime_forward_stats, walk_forward_regimes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward regime / pattern discovery.")
    ap.add_argument("--horizon", type=int, default=5)      # 1-week forward return
    ap.add_argument("--stride", type=int, default=5)       # = horizon → non-overlapping
    ap.add_argument("--n-regimes", type=int, default=3)
    ap.add_argument("--min-train", type=int, default=100)
    ap.add_argument("--refit-every", type=int, default=25)
    ap.add_argument("--perms", type=int, default=400)
    ap.add_argument("--tickers", nargs="*", default=[t for t, _ in GITHUB_TICKERS])
    args = ap.parse_args()

    load = github_loader()
    print(f"Walk-forward regime discovery — {args.n_regimes} regimes on "
          f"{len(args.tickers)} survivors, horizon {args.horizon} (non-overlapping)\n")
    print(f"  {'name':6} {'samples':>8} {'best−worst spread':>18} {'p(shuffle)':>11}  separates?")
    print("  " + "-" * 56)

    sig, total, spreads = 0, 0, []
    for sym in args.tickers:
        try:
            ds = build_behavioral_dataset(load(sym), horizon=args.horizon,
                                          stride=args.stride, symbol=sym)
            labels = walk_forward_regimes(ds.X, n_regimes=args.n_regimes,
                                          min_train=args.min_train, refit_every=args.refit_every)
            res = regime_forward_stats(labels, ds.y, n_permutations=args.perms)
        except (DataFetchError, ValueError):
            continue
        total += 1
        spreads.append(res.spread)
        is_sig = res.significant
        sig += int(is_sig)
        print(f"  {sym:6} {res.n_samples:>8} {res.spread * 100:>16.2f}% {res.p_value:>11.3f}"
              f"  {'YES' if is_sig else 'no'}")

    if not total:
        print("\nNo usable names (network?)."); return 1
    expected = 0.05 * total
    print("  " + "-" * 56)
    print(f"\n{sig}/{total} names show significant regime→return separation "
          f"(p<0.05); ~{expected:.1f} expected by chance.")
    verdict = ("MORE than chance — worth a deeper, multiple-testing-corrected look"
               if sig > expected + 2 * np.sqrt(max(expected, 1e-9))
               else "≈ chance — the discovered regimes do NOT predict forward returns out of sample")
    print(f"Verdict: {verdict}.")
    print("\n(Survivorship-biased survivors; a discovered cluster is a hypothesis until it clears")
    print(" its null. Pattern discovery is a feature factory, not an edge — judged by the harness.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
