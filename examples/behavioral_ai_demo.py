"""End-to-end Stock-Behavior AI pipeline — offline, honest, all four Act III modules.

This is the "How to use" capstone. It wires the new modules into one flow and keeps
**research mode** (does this behavior carry survivorship-robust OOS signal?) strictly
separate from **production mode** (read the current behavioral state of one name —
and refuse to call it an edge unless the research verdict earned it).

It runs with **zero network and zero API keys**: data comes from the deterministic
:class:`~vpts.data.sources.SyntheticSource`, and the insight layer falls back to its
deterministic template when no LLM client is supplied. Pass ``--llm`` to narrate the
verdict with Claude instead (requires ``pip install vpts[llm]`` + credentials).

    python examples/behavioral_ai_demo.py
    python examples/behavioral_ai_demo.py --names 8 --perms 100 --llm

Honest scope: synthetic data, so there is no real edge to find — the point is to
demonstrate that the pipeline *reports the absence of one correctly*, and that the
same machinery would judge real data the same way. The verdict is computed by the
harness, never asserted by the narration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vpts import (  # noqa: E402
    CombinatorialPurgedCV,
    Evidence,
    InsightGenerator,
    build_behavioral_dataset,
    cpcv_factor_eval,
    probability_of_backtest_overfitting,
)
from vpts.data import SurvivorshipInjector, SyntheticSource  # noqa: E402
from vpts.features import BEHAVIORAL_FEATURES, behavioral_feature_frame  # noqa: E402
from vpts.ml.models import FactorDataset  # noqa: E402


# --------------------------------------------------------------------------- #
def _pooled_ic(frames: dict, *, horizon: int, stride: int, alpha: float,
               n_groups: int, n_test: int):
    """Build behavioral datasets, evaluate pooled OOS IC, return (built, pooled_ic)."""
    built: list[tuple[str, FactorDataset, CombinatorialPurgedCV]] = []
    fold_ics: list[np.ndarray] = []
    for sym, df in frames.items():
        try:
            ds = build_behavioral_dataset(df, horizon=horizon, stride=stride, symbol=sym)
            cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test,
                                       purge=ds.purge_samples, embargo_pct=0.01)
            res = cpcv_factor_eval(ds, cv=cv, alpha=alpha)
        except ValueError:
            continue
        built.append((sym, ds, cv))
        fold_ics.append(np.array(res.fold_ics, dtype=float))
    pooled = float(np.concatenate(fold_ics).mean()) if fold_ics else float("nan")
    return built, pooled, fold_ics


def _permutation_p(built, real_ic: float, *, alpha: float, perms: int) -> float:
    """Label-shuffle permutation p-value for the pooled IC (the decisive test)."""
    rng = np.random.default_rng(0)
    null = np.empty(perms, dtype=float)
    for p in range(perms):
        folds = []
        for _sym, ds, cv in built:
            perm = rng.permutation(len(ds))
            shuf = FactorDataset(X=ds.X, y=ds.y[perm], baseline=ds.baseline,
                                 feature_names=ds.feature_names, horizon=ds.horizon,
                                 stride=ds.stride, symbol=ds.symbol)
            try:
                folds.append(np.array(cpcv_factor_eval(shuf, cv=cv, alpha=alpha).fold_ics, float))
            except ValueError:
                continue
        null[p] = float(np.concatenate(folds).mean()) if folds else np.nan
    null = null[np.isfinite(null)]
    return float((np.sum(null >= real_ic) + 1) / (null.size + 1))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Behavioral-AI end-to-end demo (offline).")
    ap.add_argument("--names", type=int, default=6, help="number of synthetic survivors")
    ap.add_argument("--horizon", type=int, default=20)
    # stride >= horizon ⇒ non-overlapping labels. Overlapping labels (stride < horizon)
    # share most of their forward window, and the per-row permutation null does not
    # preserve that autocorrelation — which makes the test anti-conservative (it will
    # flag spurious significance). Non-overlapping sampling keeps the null honest.
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--perms", type=int, default=60)
    ap.add_argument("--n-delisted", type=int, default=3)
    ap.add_argument("--llm", action="store_true", help="narrate the verdict with Claude")
    args = ap.parse_args()

    src = SyntheticSource()
    survivors = {f"SURV{i}": src.get_bars(f"SURV{i}", period="5y") for i in range(args.names)}

    print("=" * 70)
    print("RESEARCH MODE — does the behavioral feature set carry OOS signal?")
    print("=" * 70)
    print(f"  {len(survivors)} synthetic survivors · {len(BEHAVIORAL_FEATURES)} behavioral "
          f"features · horizon {args.horizon} · stride {args.stride}\n")

    built, real_ic, fold_ics = _pooled_ic(
        survivors, horizon=args.horizon, stride=args.stride, alpha=args.alpha,
        n_groups=args.n_groups, n_test=args.n_test)
    if not built:
        print("No usable datasets (try --names higher or --horizon lower)."); return 1
    p_surv = _permutation_p(built, real_ic, alpha=args.alpha, perms=args.perms)
    print(f"  Survivors  : pooled OOS IC {real_ic:+.3f}  ·  permutation p = {p_surv:.3f}")

    # --- selection-overfitting check across names (PBO) -------------------- #
    # Treat each name as a candidate "config" and its per-fold ICs as a period
    # series: would picking the best-IS name generalize OOS? (illustrative).
    width = min(len(f) for f in fold_ics)
    if width >= 8 and len(fold_ics) >= 2:
        mat = np.column_stack([f[:width] for f in fold_ics])
        n_splits = max(4, min(10, (width // 2) * 2))
        try:
            pbo = probability_of_backtest_overfitting(mat, n_splits=n_splits, metric="mean").pbo
            print(f"  PBO (name-selection across {mat.shape[1]} names): {pbo:.0%}")
        except ValueError:
            pbo = None
    else:
        pbo = None

    # --- the decisive test: inject synthetic delisted names, re-run -------- #
    print(f"\n  Injecting {args.n_delisted} synthetic delisted names and re-running …")
    injected = SurvivorshipInjector(n_delisted=args.n_delisted, seed=321).inject(survivors)
    built_inj, real_ic_inj, _ = _pooled_ic(
        injected.frames, horizon=args.horizon, stride=args.stride, alpha=args.alpha,
        n_groups=args.n_groups, n_test=args.n_test)
    p_inj = _permutation_p(built_inj, real_ic_inj, alpha=args.alpha, perms=args.perms) \
        if built_inj else float("nan")
    survives = bool(np.isfinite(p_inj) and p_inj < 0.05)
    inverts = bool(np.isfinite(real_ic_inj) and real_ic > 0.0 and real_ic_inj < 0.0)
    print(f"  Injected   : pooled OOS IC {real_ic_inj:+.3f}  ·  permutation p = {p_inj:.3f}"
          f"  ·  {'survives' if survives else 'fails'}"
          + ("  ·  INVERTS" if inverts else ""))

    # --- assemble validated evidence and get the honest verdict ----------- #
    evidence = Evidence(
        name="multi-timeframe behavioral feature set",
        oos_ic=real_ic, p_value=p_surv, pbo=pbo,
        survives_injection=survives, inverts_under_injection=inverts,
        n_samples=int(sum(len(ds) for _, ds, _ in built)), horizon=args.horizon,
        notes="synthetic data; this demonstrates the validation flow only.",
    )
    client = None
    if args.llm:
        from vpts.insight import AnthropicClient
        client = AnthropicClient()
    insight = InsightGenerator(client=client).explain(evidence)

    print("\n" + "-" * 70)
    print(f"  VERDICT (computed in code): {insight.verdict.value.upper()}")
    print("-" * 70)
    print("  " + insight.text.replace("\n", "\n  "))
    if insight.warnings:
        print(f"\n  [guardrail warnings: {insight.warnings}]")

    # ---------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("PRODUCTION MODE — read the current behavioral state of one name")
    print("=" * 70)
    name = next(iter(survivors))
    frame = behavioral_feature_frame(survivors[name]).dropna()
    latest = frame.iloc[-1]
    top = latest.reindex(latest.abs().sort_values(ascending=False).index)[:5]
    print(f"  {name} — most pronounced behavioral readings right now:")
    for feat, val in top.items():
        print(f"    {feat:24s} {val:+.3f}")
    actable = insight.verdict.permits_edge_claim
    print(f"\n  Act on these as a signal? {'YES' if actable else 'NO'} — the research "
          f"verdict is '{insight.verdict.value}'.")
    if not actable:
        print("  The behavioral read is DESCRIPTIVE only: the feature set is not a "
              "survivorship-robust, validated edge on this data, so production must "
              "not trade it. This gate is the whole point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
