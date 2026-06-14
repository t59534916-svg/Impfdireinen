"""End-to-end demo of the report's defensible recipe — on SYNTHETIC data.

Gradient-boosted trees for cross-sectional ranking → purged + embargoed CPCV →
directional accuracy vs the **up-rate baseline** → a cost-aware tails-only
long/short → a deflated-Sharpe significance gate. Runs offline (no network): it
fabricates a date×name panel with a *known* signal strength so you can watch the
harness correctly find signal when present and clear the null when not.

    python examples/gbrt_significance_demo.py
    python examples/gbrt_significance_demo.py --null   # signal switched off
"""
from __future__ import annotations

import argparse

import numpy as np

from vpts.ml.gbrt import gbrt_cross_sectional_eval
from vpts.ml.models import CrossSectionalPanel
from vpts.ml.significance import deflated_sharpe_ratio


def make_panel(n_dates: int, n_names: int, beta: float, seed: int) -> CrossSectionalPanel:
    """date×name panel; feature 0 predicts within-date returns with strength ``beta``."""
    rng = np.random.default_rng(seed)
    feats = ("f0_signal", "f1_noise", "f2_noise", "f3_noise")
    X, y, d = [], [], []
    for date in range(n_dates):
        raw = rng.standard_normal((n_names, len(feats)))
        ranks = np.empty_like(raw)
        for j in range(len(feats)):
            order = np.argsort(raw[:, j])
            r = np.empty(n_names)
            r[order] = np.arange(n_names)
            ranks[:, j] = r / (n_names - 1) - 0.5         # centred cross-sectional rank
        ret = beta * ranks[:, 0] + 0.4 * rng.standard_normal(n_names)
        for i in range(n_names):
            X.append(ranks[i].tolist())
            y.append(float(ret[i]))
            d.append(date)
    return CrossSectionalPanel(
        X=np.array(X, float), y=np.array(y, float), date_id=np.array(d, int),
        feature_names=feats, horizon=5, rebalance=5, n_dates=n_dates,
        symbols=tuple(f"N{i}" for i in range(n_names)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true", help="switch the signal off (beta=0)")
    ap.add_argument("--dates", type=int, default=120)
    ap.add_argument("--names", type=int, default=40)
    args = ap.parse_args()

    beta = 0.0 if args.null else 0.6
    panel = make_panel(args.dates, args.names, beta=beta, seed=0)
    print(f"\nPanel: {panel.n_names} names × {panel.n_dates} dates "
          f"({len(panel)} rows), signal beta={beta}\n")

    res = gbrt_cross_sectional_eval(
        panel, n_estimators=80, learning_rate=0.05, max_depth=3,
        min_samples_leaf=20, subsample=0.8)
    print(res.summary())

    # Significance gate on the OUT-OF-SAMPLE per-date IC stream (res.oos_ics),
    # treated as the return stream of a daily-rebalanced rank book, and declaring
    # a realistic search size of 50 trials. Using the OOS stream (not an in-sample
    # refit) is the whole point — it is what makes the null correctly collapse.
    print("\nSignificance gate (OOS per-date rank-IC as the 'return' stream, 50 trials):")
    sig = deflated_sharpe_ratio(np.array(res.oos_ics, dtype=float), n_trials=50, sr_std=0.05)
    print(sig.summary())

    print("\nReading: with a real signal the OOS IC is clearly positive, directional "
          "accuracy beats the up-rate, and the deflated Sharpe survives the 50-trial "
          "correction. Run with --null to watch all three collapse to ~no-edge — note "
          "the gate runs on OUT-OF-SAMPLE ICs, so the null does not falsely pass.\n")


if __name__ == "__main__":
    main()
