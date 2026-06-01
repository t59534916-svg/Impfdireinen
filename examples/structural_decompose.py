"""Decompose the structural signal: which features survive survivorship — and does it survive costs?

The structural matrix shows a small OOS IC that decays gracefully under delisted
injection. This asks the two questions that decide whether any of it is *genuine*
and *tradeable*:

  A. **Per-feature robustness** — each feature's single-factor OOS IC on survivors
     vs survivors+delisted. Features that stay positive under injection are
     survivorship-resistant; ones that collapse/flip were survivorship.
  B. **Subgroup ablation** — a "regime/shape" sub-model (vacr_z, skew, POC slope,
     footprints, shapes) vs a "dip-buying" sub-model (delta, POC location,
     cost-basis migration), each pushed through the survivorship sweep. Which
     subgroup keeps its significance?
  C. **Cost-aware** — the full model's gross long/short return per bet, net of
     realistic round-trip costs. A +0.035 IC is small; does it clear costs?

    python examples/structural_decompose.py --n-delisted 9 --perms 200

Honest scope: survivorship-biased survivors + *synthetic* delisted; gross-of-cost
IC; a research decomposition, not a tradeable result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from github_data_scan import GITHUB_TICKERS, github_loader  # noqa: E402
from structural_survivorship import _build, Built  # noqa: E402
from survivorship_stress import synthetic_delisted_ohlcv  # noqa: E402
from vpts import STRUCTURAL_FEATURES, DataFetchError, cpcv_factor_eval  # noqa: E402
from vpts.ml.models import FactorDataset  # noqa: E402

DIP = ("delta_net", "delta_poc", "poc_loc", "cost_basis_migration")
REGIME = ("vacr_z", "poc_slope", "skew", "kurtosis", "n_ledges", "poor_high",
          "is_P", "is_b", "is_B")


def _subset(ds: FactorDataset, names: tuple[str, ...]) -> FactorDataset:
    idx = [ds.feature_names.index(n) for n in names]
    return FactorDataset(X=ds.X[:, idx], y=ds.y, baseline=ds.baseline,
                         feature_names=tuple(names), horizon=ds.horizon,
                         stride=ds.stride, symbol=ds.symbol)


def _pooled(built: Built, names: tuple[str, ...] | None, alpha: float = 1.0) -> tuple[float, float]:
    """Pooled OOS IC and L/S return (%) for an optional feature subset."""
    ics, lss = [], []
    for ds, cv in built:
        d = _subset(ds, names) if names else ds
        try:
            r = cpcv_factor_eval(d, cv=cv, alpha=alpha)
        except ValueError:
            continue
        ics.extend(r.fold_ics)
        lss.append(r.oos_ls_return_pct)
    return (float(np.mean(ics)) if ics else float("nan"),
            float(np.mean(lss)) if lss else float("nan"))


def _perm_p(built: Built, names: tuple[str, ...] | None, n_perms: int,
            alpha: float = 1.0, seed: int = 0) -> float:
    real, _ = _pooled(built, names, alpha)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perms):
        ics = []
        for ds, cv in built:
            d = _subset(ds, names) if names else ds
            perm = rng.permutation(len(d))
            sh = FactorDataset(X=d.X, y=d.y[perm], baseline=d.baseline,
                               feature_names=d.feature_names, horizon=d.horizon,
                               stride=d.stride, symbol=d.symbol)
            try:
                ics.extend(cpcv_factor_eval(sh, cv=cv, alpha=alpha).fold_ics)
            except ValueError:
                continue
        if ics:
            null.append(float(np.mean(ics)))
    arr = np.array(null, float)
    return float((np.sum(arr >= real) + 1) / (arr.size + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description="Structural feature decomposition + cost test.")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--n-delisted", type=int, default=9)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--survivors", nargs="*", default=[t for t, _ in GITHUB_TICKERS][:20])
    args = ap.parse_args()

    load = github_loader()
    kw = dict(lookback=args.lookback, horizon=args.horizon, stride=args.stride)
    surv: Built = []
    for sym in args.survivors:
        try:
            surv.append(_build(load(sym), sym, kw))
        except (DataFetchError, ValueError):
            pass
    dead: Built = [_build(synthetic_delisted_ohlcv(seed=100 + k), f"DEAD{k}", kw)
                   for k in range(args.n_delisted)]
    both = surv + dead
    print(f"Built {len(surv)} survivors + {len(dead)} delisted\n")

    # ---- A: per-feature single-factor OOS IC, survivors vs +delisted ---- #
    print("A) Per-feature OOS IC (single factor)         survivors   +delisted     Δ")
    rows = []
    for f in STRUCTURAL_FEATURES:
        ic_s, _ = _pooled(surv, (f,))
        ic_b, _ = _pooled(both, (f,))
        rows.append((f, ic_s, ic_b, ic_b - ic_s))
    for f, s, b, d in sorted(rows, key=lambda r: -r[2]):
        mark = "  <- robust" if (b > 0.01 and s > 0.01) else ("  <- collapses" if s > 0.01 and b <= 0 else "")
        print(f"   {f:22s} {s:+10.3f} {b:+11.3f} {d:+8.3f}{mark}")

    # ---- B: subgroup ablation under the survivorship sweep ---- #
    print("\nB) Subgroup pooled IC / permutation-p under injection")
    print(f"   {'subgroup':>10} {'survivors':>20} {'+all delisted':>22}")
    for tag, names in (("ALL", None), ("REGIME", REGIME), ("DIP", DIP)):
        ic_s, _ = _pooled(surv, names); p_s = _perm_p(surv, names, args.perms)
        ic_b, _ = _pooled(both, names); p_b = _perm_p(both, names, args.perms)
        print(f"   {tag:>10}   IC {ic_s:+.3f} (p={p_s:.3f})     IC {ic_b:+.3f} (p={p_b:.3f})")

    # ---- C: cost-aware net long/short return per bet (survivor universe) ---- #
    print("\nC) Cost-aware full-model L/S return per "
          f"{args.horizon}-bar bet (survivors)")
    _, gross = _pooled(surv, None)
    bets_yr = 252.0 / args.horizon
    print(f"   gross L/S return/bet : {gross:+.3f}%   (~{gross * bets_yr:+.1f}%/yr, non-overlap approx)")
    for bps in (5, 10, 20):
        net = gross - bps / 100.0           # round-trip cost in %
        print(f"   net @ {bps:>2}bps round-trip: {net:+.3f}%/bet  (~{net * bets_yr:+.1f}%/yr)")

    print("\nRead: REGIME features surviving injection => a small genuine (non-survivorship) "
          "signal; only DIP surviving => mostly survivorship. Costs eating the gross L/S => "
          "not tradeable even if genuine. (Synthetic delisted; gross IC; research decomposition.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
