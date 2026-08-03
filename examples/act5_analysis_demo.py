#!/usr/bin/env python3
"""Act V — data analysis of financial time series *and* fundamental data.

Runs the whole of :mod:`vpts.analysis` end to end, offline and deterministic:

1. **Time-series diagnostics** — distribution, tails, memory (Ljung-Box,
   Lo-MacKinlay variance ratio, Hurst, ADF), volatility structure and drawdown
   for each name, plus a cross-name summary.
2. **Fundamentals** — point-in-time statements → ratios, Piotroski F, Altman Z,
   with a filing-lag audit proving no bar used a filing before it was public.
3. **The trial** — the same fundamentals put through the project's one
   evaluation contract: a per-name ``FactorDataset`` and a cross-sectional panel,
   each with a label-shuffle permutation null, on data where the answer is known
   (fundamentals generated independently of price → there is *no* edge to find).

Usage
-----
    python examples/act5_analysis_demo.py                 # offline synthetic (default)
    python examples/act5_analysis_demo.py --names 20      # wider synthetic universe
    python examples/act5_analysis_demo.py --live AAPL MSFT --fmp-key $FMP_API_KEY

``--live`` pulls real prices via ``MarketDataFetcher`` and, with an FMP key, real
point-in-time statements. Without a key it runs prices-live / fundamentals-
synthetic and says so, rather than pretending.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.analysis import (  # noqa: E402
    FMPFundamentalsSource,
    SyntheticFundamentalsSource,
    align_fundamentals,
    analyze_timeseries,
    audit_point_in_time,
    build_fundamental_dataset,
    build_fundamental_panel,
    compute_ratios,
)
from vpts.data.synthetic import synthetic_survivor_ohlcv  # noqa: E402
from vpts.harness import MIN_SAMPLES_FOR_PERM, honest_backtest  # noqa: E402
from vpts.ml.cross_sectional import permutation_test_cross_sectional  # noqa: E402

HR = "=" * 78


def _rule(title: str) -> None:
    print(f"\n{HR}\n{title}\n{HR}")


# --------------------------------------------------------------------------- #
def load_universe(args) -> tuple[dict, dict, str]:
    """Return ``(price_frames, fundamental_series, provenance)``."""
    if args.live:
        from vpts.data.fetcher import MarketDataFetcher

        fetcher = MarketDataFetcher()
        frames = {}
        for sym in args.live:
            try:
                frames[sym] = fetcher.fetch(sym, period="10y", interval="1d")
            except Exception as exc:  # noqa: BLE001 - a dead symbol shouldn't kill the demo
                print(f"  ! {sym}: {type(exc).__name__}: {exc}")
        if not frames:
            raise SystemExit("no live prices fetched — check connectivity.")
        if args.fmp_key:
            src = FMPFundamentalsSource(api_key=args.fmp_key, period="quarter")
            series = {}
            for sym, f in frames.items():
                try:
                    series[sym] = src.get_fundamentals(sym, limit=44)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {sym} fundamentals: {exc}")
            if series:
                return frames, series, "prices: live · fundamentals: FMP (real filing dates)"
            print("  ! no FMP statements returned — falling back to synthetic fundamentals.")
        series = {s: SyntheticFundamentalsSource(seed=i, n_periods=44, freq="Q",
                                                 start="2009-09-30").get_fundamentals(s)
                  for i, s in enumerate(frames)}
        return frames, series, "prices: live · fundamentals: SYNTHETIC (no FMP key)"

    frames, series = {}, {}
    for i in range(args.names):
        sym = f"SYN{i:02d}"
        frames[sym] = synthetic_survivor_ohlcv(2600, seed=2000 + i, start_date="2010-01-04")
        series[sym] = SyntheticFundamentalsSource(
            seed=100 + i, n_periods=44, start="2009-09-30", freq="Q",
            link=frames[sym] if args.plant_edge else None,
            link_strength=6.0 if args.plant_edge else 0.0,
        ).get_fundamentals(sym)
    tag = "PLANTED EDGE (a positive result here means the harness works)" if args.plant_edge \
        else "pure noise (fundamentals independent of price — there is NO edge to find)"
    return frames, series, f"synthetic survivors · {tag}"


# --------------------------------------------------------------------------- #
def part_one_timeseries(frames: dict) -> None:
    _rule("PART 1 — financial time-series analysis (descriptive)")
    reports = {}
    for sym, df in frames.items():
        try:
            reports[sym] = analyze_timeseries(df, symbol=sym)
        except ValueError as exc:
            print(f"  ! {sym}: {exc}")
    first = next(iter(reports.values()))
    print(first.summary())

    rows = [{
        "name": s, "ann_ret_%": r.ann_return_pct, "ann_vol_%": r.ann_vol_pct,
        "skew": r.skew, "exc_kurt": r.excess_kurtosis, "CVaR95_%": r.cvar_95_pct,
        "AC(1)": r.autocorr_1, "VR(5)": r.variance_ratio, "VR_p": r.vr_p,
        "Hurst": r.hurst, "ARCH_p": r.arch_lm_p, "maxDD_%": r.max_drawdown_pct,
        "memory": r.memory,
    } for s, r in reports.items()]
    tbl = pd.DataFrame(rows).set_index("name")
    print(f"\nAll {len(tbl)} names:\n")
    print(tbl.round(3).to_string())

    n_rej = int(sum(r.random_walk_rejected for r in reports.values()))
    n_fat = int(sum(r.fat_tailed for r in reports.values()))
    n_clust = int(sum(r.vol_clustered for r in reports.values()))
    print(f"\n  random walk rejected (VR, 5%): {n_rej}/{len(reports)}"
          f"   fat-tailed: {n_fat}/{len(reports)}   vol-clustered: {n_clust}/{len(reports)}")
    print("  Reminder: these are in-sample descriptive statistics. None of them is an edge —\n"
          "  they say what the process looks like, not that anything is predictable net of cost.")


# --------------------------------------------------------------------------- #
def part_two_fundamentals(frames: dict, series: dict) -> None:
    _rule("PART 2 — fundamental data (point-in-time)")
    sym = next(iter(series))
    ser = series[sym]
    print(ser.summary())
    print("\nMost recent filing:")
    print(" ", ser.snapshots[-1].summary())
    print()
    price = float(frames[sym]["Close"].iloc[-1])
    print(compute_ratios(ser.snapshots[-1], prior=ser.prior_year_of(ser.snapshots[-1]),
                         price=price).summary())

    _rule("PART 2b — the look-ahead audit (the whole reason fundamentals are hard)")
    print("A filing describing a period is public only weeks-to-months later. Joining it on the\n"
          "period end hands the model that gap as hindsight — and manufactures an edge.\n")
    audits = []
    for s, f in frames.items():
        audits.append({"name": s, **audit_point_in_time(align_fundamentals(series[s], f.index))})
    at = pd.DataFrame(audits).set_index("name")
    print(at.to_string())
    total_viol = int(at["violations"].sum())
    print(f"\n  point-in-time violations across all names: {total_viol}"
          f"  ({'PASS — no bar used a filing before it was public' if total_viol == 0 else 'FAIL'})")
    print(f"  median reporting lag: {int(at['median_lag_days'].median())} days — "
          "that is exactly how much hindsight a period-end join would have leaked.")


# --------------------------------------------------------------------------- #
def part_three_trial(frames: dict, series: dict, perms: int) -> None:
    _rule("PART 3 — the same fundamentals, put on trial")
    print("A feature family only counts here once it clears the project's evaluation contract:\n"
          "purged CPCV out-of-sample IC + a label-shuffle permutation null.\n")

    print("--- 3a. Per-name time series (one row per filing) ---")
    datasets = []
    for s, f in frames.items():
        try:
            datasets.append(build_fundamental_dataset(f, series[s], horizon=20,
                                                      symbol=s, min_samples=20))
        except ValueError as exc:
            print(f"  ! {s}: {exc}")
    if datasets:
        med_n = float(np.median([len(d) for d in datasets]))
        print(f"  {len(datasets)} datasets, {med_n:.0f} rows each (one per filing), "
              f"stride ~{datasets[0].stride} bars")
        rep = honest_backtest(datasets, perms=perms, seed=0, n_trials=1, n_groups=5, n_test=2)
        print(f"  OOS IC {rep.oos_ic_mean:+.4f}   block-perm p {rep.block_perm_p:.4f}   "
              f"DSR {rep.deflated_sharpe:.3f}   PBO {rep.pbo:.2f}")
        print(f"  verdict: {rep.verdict}")
        if med_n < MIN_SAMPLES_FOR_PERM:
            print(f"  ⚠ {med_n:.0f} rows/dataset is below the harness's "
                  f"MIN_SAMPLES_FOR_PERM={MIN_SAMPLES_FOR_PERM}. Fundamentals land here BY\n"
                  "    CONSTRUCTION — one row per filing is the honest sampling, and honest\n"
                  "    sampling is small. Read this p-value as unreliable; use 3b instead.")

    print("\n--- 3b. Cross-section (the frame fundamentals are actually used in) ---")
    panel = build_fundamental_panel(frames, series, horizon=20, rebalance=63)
    print(f"  panel {panel.X.shape[0]} rows = {panel.n_dates} dates × ~{panel.n_names} names, "
          f"{len(panel.feature_names)} ranked factors")
    res = permutation_test_cross_sectional(panel, n_permutations=perms, seed=0)
    print(f"  combined OOS rank IC {res.real_ic:+.4f}   null mean {res.null_ic_mean:+.4f}   "
          f"p {res.p_value:.4f}  ({res.n_permutations} shuffles)")
    verdict = ("SIGNIFICANT at 5%" if res.p_value < 0.05
               else "NOT significant — nothing to report")
    print(f"  verdict: {verdict}")
    print("\n  Why 3b and not 3a: pooling across names by date gives ~20× the effective sample\n"
          "  and a within-date shuffle that respects the panel structure. Measured on pure\n"
          "  noise, 3b false-positives at ~the nominal 5%; 3a, at ~37 rows/name, does not.")


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", type=int, default=12, help="synthetic universe size (default 12)")
    p.add_argument("--live", nargs="*", metavar="SYM", help="fetch these symbols live instead")
    p.add_argument("--fmp-key", default=None, help="FMP API key for real point-in-time statements")
    p.add_argument("--perms", type=int, default=200, help="permutations (default 200)")
    p.add_argument("--plant-edge", action="store_true",
                   help="plant a REAL fundamental→return link, to prove the harness finds one")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    frames, series, provenance = load_universe(args)

    print(HR)
    print("Act V — data analysis: financial time series & fundamental data")
    print(HR)
    print(f"universe: {len(frames)} names · {provenance}")

    part_one_timeseries(frames)
    part_two_fundamentals(frames, series)
    part_three_trial(frames, series, args.perms)

    _rule("Bottom line")
    print("The time-series half describes; it never claims. The fundamental half is\n"
          "point-in-time by construction and is judged by the same harness as every other\n"
          "feature family in this repo. On survivor-only data a fundamental result would\n"
          "still owe the survivorship-injection sweep — Altman-Z exists precisely to score\n"
          "the failure tail that survivor-only data deletes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
