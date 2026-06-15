"""Judge any idea in <20 lines — the honest backtest harness, end to end.

This is the capstone of what `RESEARCH.md` calls the durable asset: not a signal, but
the harness that refuses to flatter one. It runs fully **offline** on a deterministic
synthetic universe so it doubles as a smoke test; swap the synthetic frames for your
own ``{symbol: OHLCV}`` and feature builder to judge a real idea.

    python examples/honest_harness_demo.py

The single :func:`~vpts.honest_backtest` call returns the whole skeptic's checklist:
purged-CPCV OOS IC, a block-permutation p-value, the conviction-bucket curve, the
Deflated Sharpe (selection-adjusted) and PBO, and a survivorship-injection sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import build_structural_dataset, honest_backtest          # noqa: E402
from vpts.data import synthetic_survivor_ohlcv                       # noqa: E402


def main() -> int:
    # ── the <20-line core: your data + a feature builder → one verdict ────────── #
    frames = {f"SYM{i}": synthetic_survivor_ohlcv(900, seed=i) for i in range(6)}

    def features(frame, symbol):
        return build_structural_dataset(frame, lookback=120, horizon=20, stride=8,
                                         symbol=symbol, interval="1d")

    datasets = [features(f, s) for s, f in frames.items()]
    report = honest_backtest(
        datasets,
        n_trials=11,                      # be honest: variants you tried (DSR adjustment)
        perms=200,
        frames=frames, feature_builder=features,   # → also run the survivorship sweep
        injection_fractions=(0.0, 0.2, 0.5),
    )
    print(report.summary())
    # ──────────────────────────────────────────────────────────────────────────── #

    print("\n(Synthetic survivors with no planted edge → the harness should report NO EDGE;")
    print(" point `frames`/`features` at real data to judge a real idea.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
