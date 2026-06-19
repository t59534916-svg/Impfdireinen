"""Put the classic swing-trading toolkit on trial: RSI · MACD · MA-cross · momentum · Fibonacci.

The newsletter-staple indicators, run through the *same* honest harness as everything else —
purged-CPCV OOS IC, a block-permutation p-value, the Deflated Sharpe / PBO, and a
survivorship-injection sweep (survivors-only vs survivors+synthetic-delisted). The point is
discipline, not a sales pitch: this is the same class of input that experiments 2–6 already
showed does not clear the survivorship wall.

    python examples/indicator_swing_eval.py
    python examples/indicator_swing_eval.py --names 10 --perms 300

NOTE: runs on SYNTHETIC survivors (near-random-walks) — the real stocknet feed is offline (404).
So a "NO EDGE" here mainly proves the machinery + null discipline; the genuine test needs real,
delisted-inclusive data (see examples/v2_survivorship_free.py --lake/--source).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import honest_backtest                              # noqa: E402
from vpts.data import synthetic_survivor_ohlcv               # noqa: E402
from vpts.features import build_indicator_dataset            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate classic swing indicators through the harness.")
    ap.add_argument("--names", type=int, default=8)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    frames = {f"S{i}": synthetic_survivor_ohlcv(820, seed=i) for i in range(args.names)}

    def feats(frame, sym):
        return build_indicator_dataset(frame, lookback=120, horizon=args.horizon,
                                       stride=args.stride, symbol=sym, interval="1d")

    datasets = [feats(f, s) for s, f in frames.items()]

    print("=" * 84)
    print("Classic swing indicators (RSI · MACD · MA-cross · momentum · Fibonacci) on trial")
    print("=" * 84)
    rep = honest_backtest(datasets, perms=args.perms, n_trials=11,
                          frames=frames, feature_builder=feats,
                          injection_fractions=(0.0, 0.5), terminal_frac=0.08, seed=args.seed)
    print(rep.summary())
    print("\nReading: classic indicators are the same FAMILY of input experiments 2–6 tested.")
    print("On synthetic survivors they clear nothing; the real test needs delisted-inclusive data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
