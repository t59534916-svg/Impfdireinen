"""Tier 2a — is the 'structural' edge just short-horizon reversal in disguise?

`RESEARCH.md`'s one real signal is the structural microstructure block (carried by
``cost_basis_migration`` and ``delta_net`` — dip-buying / accumulation features). A
fair skeptic asks: is that just a generic **k-day reversal** (recent losers bounce)
or **12-1 momentum** wearing Volume-Profile clothing? If so, the "structural" IC
should vanish once we regress those two classic factors out.

Method (disconfirm-by-default)
------------------------------
For each real survivor, at the *same* decision bars the structural dataset uses:
  1. build a plain **k-day reversal** factor (past k-day return) and a **12-1
     momentum** factor (return from t-252 to t-21);
  2. **orthogonalize** every structural feature against them (per-name OLS on
     ``[1, reversal, momentum]``, keep the residual) — removing all linear
     reversal/-momentum content;
  3. re-run the purged-CPCV IC on the **residualized** structural features.
Then compare pooled IC: raw structural vs reversal-only vs momentum-only vs
residualized structural, with a block-permutation p-value on the residual.

    python examples/structural_reversal.py
    python examples/structural_reversal.py --k 10 --perms 500

Reading: if the residual IC collapses to ≈0, the "structural" edge is generic
reversal and should be reported as such; if it survives orthogonalization, that is
the first evidence of a genuinely **profile-specific** component. Either way it goes
in `RESEARCH.md` — survivor-only, so still a validity check, not a tradeable result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from vpts import (  # noqa: E402
    CombinatorialPurgedCV,
    DataFetchError,
    build_structural_dataset,
    cpcv_factor_eval,
)
from vpts.ml.models import FactorDataset  # noqa: E402
from vpts.stats import block_shuffle_indices, recommend_block_size  # noqa: E402


def _residualize(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Column-wise OLS residual of X on ``[1, Z]`` — removes all linear Z content."""
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, X, rcond=None)
    return X - Z1 @ beta


def _subset(ds: FactorDataset, mask: np.ndarray) -> FactorDataset:
    ts = ds.timestamps[mask] if ds.timestamps is not None else None
    return FactorDataset(X=ds.X[mask], y=ds.y[mask], baseline=ds.baseline[mask],
                         feature_names=ds.feature_names, horizon=ds.horizon,
                         stride=ds.stride, timestamps=ts, symbol=ds.symbol)


def _eval_ics(ds: FactorDataset, cv: CombinatorialPurgedCV, alpha: float) -> np.ndarray:
    return np.array(cpcv_factor_eval(ds, cv=cv, alpha=alpha).fold_ics, dtype=float)


def _pooled(ic_lists) -> float:
    ics = [a for a in ic_lists if a is not None and a.size]
    return float(np.concatenate(ics).mean()) if ics else float("nan")


def _factor_ds(values: np.ndarray, ref: FactorDataset, name: str) -> FactorDataset:
    """A one-column FactorDataset for a single factor, aligned to *ref*'s labels."""
    return FactorDataset(X=values.reshape(-1, 1), y=ref.y, baseline=values,
                         feature_names=(name,), horizon=ref.horizon, stride=ref.stride,
                         timestamps=ref.timestamps, symbol=ref.symbol)


def _build_entry(frame, sym, args):
    """Return per-name datasets (raw structural, residualized, reversal, momentum) + cv."""
    ds = build_structural_dataset(frame, lookback=args.lookback, horizon=args.horizon,
                                  stride=args.stride, symbol=sym, interval="1d")
    if ds.timestamps is None:
        raise ValueError("no timestamps (need a DatetimeIndex)")
    close = frame["Close"].to_numpy(float)
    pos = frame.index.get_indexer(ds.timestamps)               # price position per sample
    rev = np.full(pos.shape, np.nan)
    mom = np.full(pos.shape, np.nan)
    ok_rev = pos - args.k >= 0
    ok_mom = pos - args.mom_long >= 0
    rev[ok_rev] = close[pos[ok_rev]] / close[pos[ok_rev] - args.k] - 1.0          # k-day past return
    mp = pos[ok_mom]
    mom[ok_mom] = close[mp - args.mom_skip] / close[mp - args.mom_long] - 1.0     # 12-1 momentum
    mask = np.isfinite(rev) & np.isfinite(mom) & (pos >= 0)
    if mask.sum() < args.min_samples:
        raise ValueError(f"only {int(mask.sum())} aligned samples")

    raw = _subset(ds, mask)
    Z = np.column_stack([rev[mask], mom[mask]])
    resid = FactorDataset(X=_residualize(raw.X, Z), y=raw.y, baseline=raw.baseline,
                          feature_names=raw.feature_names, horizon=raw.horizon,
                          stride=raw.stride, timestamps=raw.timestamps, symbol=sym)
    cv = CombinatorialPurgedCV(n_groups=args.n_groups, n_test_groups=args.n_test,
                               purge=raw.purge_samples, embargo_pct=0.01)
    return {
        "sym": sym, "cv": cv, "raw": raw, "resid": resid,
        "rev": _factor_ds(rev[mask], raw, "reversal_k"),
        "mom": _factor_ds(mom[mask], raw, "momentum_12_1"),
    }


def _safe_ics(ds, cv, alpha):
    try:
        return _eval_ics(ds, cv, alpha)
    except ValueError:
        return None


def _block_perm_p(entries, key, real_ic, *, alpha, perms, seed=0) -> float:
    """Block-permutation p for the pooled IC of dataset *key* (preserves autocorrelation)."""
    rng = np.random.default_rng(seed)
    null = np.empty(perms, dtype=float)
    for p in range(perms):
        folds = []
        for e in entries:
            ds = e[key]
            bs = recommend_block_size(ds.horizon, ds.stride)
            perm = block_shuffle_indices(len(ds), bs, rng=rng)
            shuf = FactorDataset(X=ds.X, y=ds.y[perm], baseline=ds.baseline,
                                 feature_names=ds.feature_names, horizon=ds.horizon,
                                 stride=ds.stride, timestamps=ds.timestamps, symbol=ds.symbol)
            ic = _safe_ics(shuf, e["cv"], alpha)
            if ic is not None:
                folds.append(ic)
        null[p] = float(np.concatenate(folds).mean()) if folds else np.nan
    null = null[np.isfinite(null)]
    return float((np.sum(null >= real_ic) + 1) / (null.size + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description="Is the structural edge just reversal/momentum?")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=21, help="reversal lookback (days)")
    ap.add_argument("--mom-skip", type=int, default=21, help="momentum skip (days, the '1' in 12-1)")
    ap.add_argument("--mom-long", type=int, default=252, help="momentum lookback (days, the '12')")
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--min-samples", type=int, default=30)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--tickers", nargs="*", default=[t for t, _ in GITHUB_TICKERS])
    args = ap.parse_args()

    print("=" * 80)
    print("TIER 2a — orthogonalize the structural features against reversal + 12-1 momentum")
    print("=" * 80)
    print(f"reversal=k{args.k}d  momentum={args.mom_long}->{args.mom_skip}d  "
          f"horizon={args.horizon}  stride={args.stride}\n")

    load = github_loader()
    entries = []
    for sym in args.tickers:
        try:
            entries.append(_build_entry(load(sym), sym, args))
        except (DataFetchError, ValueError) as exc:
            print(f"  ! {sym}: skipped ({exc})")
    if not entries:
        print("No names built (network?)."); return 1

    raw_ic = _pooled([_safe_ics(e["raw"], e["cv"], args.alpha) for e in entries])
    rev_ic = _pooled([_safe_ics(e["rev"], e["cv"], args.alpha) for e in entries])
    mom_ic = _pooled([_safe_ics(e["mom"], e["cv"], args.alpha) for e in entries])
    res_ic = _pooled([_safe_ics(e["resid"], e["cv"], args.alpha) for e in entries])

    p_raw = _block_perm_p(entries, "raw", raw_ic, alpha=args.alpha, perms=args.perms)
    p_res = _block_perm_p(entries, "resid", res_ic, alpha=args.alpha, perms=args.perms)

    n_samples = sum(len(e["raw"]) for e in entries)
    print(f"{len(entries)} names · {n_samples} aligned samples (block-perm null, {args.perms} perms)\n")
    print("-" * 64)
    print(f"{'factor':<40}{'pooled OOS IC':>14}{'p(block)':>10}")
    print("-" * 64)
    print(f"{'structural (raw, same samples)':<40}{raw_ic:>+14.3f}{p_raw:>10.3f}")
    print(f"{'reversal only (k=' + str(args.k) + 'd)':<40}{rev_ic:>+14.3f}{'':>10}")
    print(f"{'momentum only (12-1)':<40}{mom_ic:>+14.3f}{'':>10}")
    print(f"{'structural ⟂ reversal+momentum (residual)':<40}{res_ic:>+14.3f}{p_res:>10.3f}")
    print("-" * 64)

    retained = (res_ic / raw_ic * 100.0) if np.isfinite(raw_ic) and abs(raw_ic) > 1e-9 else float("nan")
    print(f"\nResidual retains {retained:.0f}% of the raw structural IC "
          f"(p {p_raw:.3f} → {p_res:.3f}).")
    # Verdict is driven by IC RETENTION (does the signal survive removing reversal/
    # momentum?), with the permutation p reported as supporting context.
    if res_ic > 0 and retained >= 70:
        print("Verdict: the structural IC SURVIVES orthogonalization (residual IC ≈ or > raw) →")
        print("it is NOT generic reversal/momentum; a profile-specific component remains. The")
        print('"it\'s just reversal" hypothesis is not supported (still survivor-only).')
    elif res_ic <= 0 or retained <= 30:
        print("Verdict: the structural IC COLLAPSES once reversal/momentum are removed → it is")
        print("substantially generic reversal/momentum dressed in Volume-Profile language.")
    else:
        print("Verdict: PARTIAL — some profile-specific signal remains but reversal/momentum")
        print("explain a meaningful share. Report the residual IC and p beside the raw.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
