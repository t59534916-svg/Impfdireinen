"""How does the structural 'edge' hold up as the population gets loser-heavy?

`RESEARCH.md` showed the structural microstructure features have a real OOS
correlation **on survivors** that *inverts* once delisted names are injected. This
script makes that quantitative as a function of the **loser:winner ratio** —
because the real equity population is loser-heavy (Bessembinder 2018: most stocks
underperform T-bills over their lifetime; a tiny fraction create all the wealth),
yet a survivor-only backtest universe throws those losers away.

It loads the real, free `stocknet` survivors (split-adjusted 5y daily, no key),
builds the structural feature → forward-return datasets, and sweeps the injected
**delisted fraction** from 0 (survivors only) up past 0.5 (more losers than
winners). For each fraction it reports the pooled out-of-sample IC, decomposed into
the survivor group and the (synthetic, calibrated) dead group, plus a block-permutation
p-value at the endpoints. The dead names decline *slowly* to a calibrated terminal
loss (``--terminal-frac``), so the only thing changing across the sweep is the base
rate of losers — exactly the survivorship lever.

    python examples/survivorship_baserate.py
    python examples/survivorship_baserate.py --fractions 0 0.2 0.33 0.5 0.67 --terminal-frac 0.1

Honest scope: the dead names are *synthetic* (no free delisted prices — verified
across Polygon/FMP free tiers), so this is a calibrated stress, not ground truth.
The survivors are real; the result shows the *mechanism and magnitude* of the bias.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from vpts import CombinatorialPurgedCV, DataFetchError, build_structural_dataset, cpcv_factor_eval  # noqa: E402
from vpts.data import synthetic_delisted_ohlcv  # noqa: E402
from vpts.ml.models import FactorDataset  # noqa: E402
from vpts.stats import block_shuffle_indices, recommend_block_size  # noqa: E402


def _eval_fold_ics(ds: FactorDataset, cv: CombinatorialPurgedCV, alpha: float) -> np.ndarray:
    return np.array(cpcv_factor_eval(ds, cv=cv, alpha=alpha).fold_ics, dtype=float)


def _pooled_ic(entries) -> float:
    """Mean OOS IC pooled over every fold of every (name, fold_ics) entry."""
    ics = [f for _name, f, _ds, _cv in entries]
    return float(np.concatenate(ics).mean()) if ics else float("nan")


def _block_perm_p(entries, real_ic: float, *, alpha: float, perms: int, seed: int = 0) -> float:
    """Block-permutation p-value for the pooled IC (preserves label autocorrelation)."""
    rng = np.random.default_rng(seed)
    null = np.empty(perms, dtype=float)
    for p in range(perms):
        folds = []
        for _name, _f, ds, cv in entries:
            bs = recommend_block_size(ds.horizon, ds.stride)
            perm = block_shuffle_indices(len(ds), bs, rng=rng)
            shuf = FactorDataset(X=ds.X, y=ds.y[perm], baseline=ds.baseline,
                                 feature_names=ds.feature_names, horizon=ds.horizon,
                                 stride=ds.stride, symbol=ds.symbol)
            try:
                folds.append(_eval_fold_ics(shuf, cv, alpha))
            except ValueError:
                continue
        null[p] = float(np.concatenate(folds).mean()) if folds else np.nan
    null = null[np.isfinite(null)]
    return float((np.sum(null >= real_ic) + 1) / (null.size + 1))


def _build(frame, sym, args):
    ds = build_structural_dataset(frame, lookback=args.lookback, horizon=args.horizon,
                                  stride=args.stride, symbol=sym, interval="1d")
    cv = CombinatorialPurgedCV(n_groups=args.n_groups, n_test_groups=args.n_test,
                               purge=ds.purge_samples, embargo_pct=0.01)
    return sym, _eval_fold_ics(ds, cv, args.alpha), ds, cv


def main() -> int:
    ap = argparse.ArgumentParser(description="Survivorship bias vs the loser:winner ratio.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--terminal-frac", type=float, default=0.1, help="dead-name terminal loss level")
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.0, 0.2, 0.33, 0.5, 0.67])
    ap.add_argument("--perms", type=int, default=60, help="block-perm shuffles at the endpoints")
    ap.add_argument("--tickers", nargs="*", default=[t for t, _ in GITHUB_TICKERS])
    args = ap.parse_args()

    load = github_loader()
    print(f"Loading {len(args.tickers)} real stocknet survivors (5y daily) …")
    survivors = []
    for sym in args.tickers:
        try:
            survivors.append(_build(load(sym), sym, args))
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
    if not survivors:
        print("No survivors loaded (network?)."); return 1
    n_surv = len(survivors)
    surv_ic = _pooled_ic(survivors)
    print(f"  {n_surv} survivors built · structural OOS IC (survivors only) = {surv_ic:+.3f}\n")

    # Pre-build a pool of synthetic dead names (calibrated slow decline), once.
    max_frac = max(args.fractions)
    n_dead_max = int(round(max_frac / (1.0 - max_frac) * n_surv)) if max_frac < 1 else 0
    cal_bars, start_date = 1258, "2012-09-04"           # the stocknet calendar (5y daily)
    print(f"Pre-building {n_dead_max} synthetic dead names (terminal≈{args.terminal_frac:g}×, slow decline) …")
    dead_pool = []
    for k in range(n_dead_max):
        df = synthetic_delisted_ohlcv(cal_bars, seed=500 + k, start_date=start_date,
                                      terminal_frac=args.terminal_frac)
        try:
            dead_pool.append(_build(df, f"DEAD{k}", args))
        except ValueError:
            continue
    dead_ic = _pooled_ic(dead_pool)
    print(f"  dead-group structural OOS IC = {dead_ic:+.3f}  "
          f"(dip-buying features anti-predict a name on its way to zero)\n")

    print("=" * 74)
    print(f"{'delisted':>9} {'dead:alive':>11} {'pooledIC':>9} {'survIC':>8} {'deadIC':>8} {'p(block)':>9}")
    print("-" * 74)
    rows = []
    for f in sorted(args.fractions):
        k = int(round(f / (1.0 - f) * n_surv)) if f < 1 else len(dead_pool)
        k = min(k, len(dead_pool))
        entries = survivors + dead_pool[:k]
        pooled = _pooled_ic(entries)
        frac_actual = k / (n_surv + k)
        # block-perm p only at the endpoints (it is the expensive part)
        endpoint = f in (min(args.fractions), max(args.fractions))
        p = _block_perm_p(entries, pooled, alpha=args.alpha, perms=args.perms) if endpoint else None
        ps = f"{p:.3f}" if p is not None else "  —"
        print(f"{frac_actual:>8.0%} {k:>5}:{n_surv:<5} {pooled:>+9.3f} {surv_ic:>+8.3f} "
              f"{dead_ic:>+8.3f} {ps:>9}")
        rows.append((frac_actual, pooled, p))

    print("=" * 74)
    base = rows[0][1]
    worst = rows[-1][1]
    print(f"\nReading: the survivor-only IC ({base:+.3f}) is the number a survivor-biased")
    print(f"backtest would report. As the population goes loser-heavy it moves to {worst:+.3f}")
    verdict = ("INVERTS sign" if base > 0 and worst < 0 else
               "erodes toward zero" if base > 0 else "no survivor edge to erode")
    print(f"→ the apparent edge {verdict}. The bias is a base-rate/universe effect, not a")
    print("  per-trade one: nothing 'dies' inside a ~1-month hold — the losers were simply")
    print("  never in the survivor universe. (Dead names synthetic; survivors real.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
