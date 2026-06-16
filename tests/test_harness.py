"""Tests for the one-call honest backtest harness (vpts.harness).

Offline and deterministic. The load-bearing checks are the **null-clearing** ones:
on data with no planted signal the harness must report a non-significant IC and a
"NO EDGE" verdict; on data with a planted signal it must detect it. This is the same
disconfirm-by-default discipline every evaluator in the repo is held to.

    python tests/test_harness.py
    pytest tests/test_harness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import HonestReport, honest_backtest                       # noqa: E402
from vpts.ml.models import FactorDataset                             # noqa: E402


def _ds(n: int, seed: int, *, signal: float = 0.0) -> FactorDataset:
    """A FactorDataset with `signal`·X[:,0] in the label (signal=0 → pure null)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = signal * X[:, 0] + rng.standard_normal(n) * 0.5
    return FactorDataset(X=X, y=y, baseline=X[:, 0].copy(),
                         feature_names=("a", "b", "c"), horizon=20, stride=8)


# --------------------------------------------------------------------------- #
def test_null_data_clears_no_edge() -> None:
    # Pool several independent null datasets → pooled IC ≈ 0, p ≈ 0.5 (robustly n.s.).
    datasets = [_ds(500, seed=s, signal=0.0) for s in range(8)]
    rep = honest_backtest(datasets, n_trials=5, perms=120, seed=0)
    assert isinstance(rep, HonestReport)
    assert rep.block_perm_p > 0.05 and not rep.ic_significant
    assert rep.verdict.startswith("NO EDGE")


def test_planted_signal_is_detected() -> None:
    datasets = [_ds(500, seed=s, signal=0.6) for s in range(4)]
    rep = honest_backtest(datasets, n_trials=1, perms=120, seed=0)
    assert rep.oos_ic_mean > 0.1 and rep.ic_significant and rep.block_perm_p < 0.05


def test_verdict_overfit_gate() -> None:
    # A significant IC that is also overfit (PBO ≥ 0.5) must NOT pass the checklist.
    from vpts.harness import _verdict

    assert _verdict(sig=True, overfit=True, inverts=False, dsr_ok=True, pbo=0.7
                    ).startswith("OVERFIT")
    # …and a clean significant result still passes.
    assert _verdict(sig=True, overfit=False, inverts=False, dsr_ok=True, pbo=0.1
                    ).startswith("PASSES")
    # Order: not-significant dominates; inversion beats DSR; overfit beats inversion.
    assert _verdict(sig=False, overfit=True, inverts=True, dsr_ok=False, pbo=0.9
                    ).startswith("NO EDGE")
    assert _verdict(sig=True, overfit=True, inverts=True, dsr_ok=False, pbo=0.9
                    ).startswith("OVERFIT")
    assert _verdict(sig=True, overfit=False, inverts=True, dsr_ok=False, pbo=0.1
                    ).startswith("SURVIVORSHIP MIRAGE")


def test_single_dataset_accepted() -> None:
    rep = honest_backtest(_ds(600, seed=1, signal=0.0), perms=60, seed=0)
    assert rep.n_datasets == 1 and rep.n_samples == 600


def test_report_to_dict_and_summary_shape() -> None:
    rep = honest_backtest([_ds(500, seed=s) for s in range(3)], perms=60, seed=0)
    d = rep.to_dict()
    for k in ("oos_ic_mean", "block_perm_p", "deflated_sharpe", "pbo", "verdict",
              "bucket_curve", "ic_significant"):
        assert k in d
    assert "block-perm p" in rep.summary()
    assert "injection_sweep" not in d                # no sweep when frames omitted


def test_empty_datasets_raises() -> None:
    try:
        honest_backtest([])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty datasets")


def test_injection_sweep_runs_with_frames() -> None:
    # End-to-end sweep on a tiny synthetic survivor universe (fast).
    from vpts import build_structural_dataset
    from vpts.data import synthetic_survivor_ohlcv

    frames = {f"S{i}": synthetic_survivor_ohlcv(620, seed=i) for i in range(3)}

    def feats(frame, sym):
        return build_structural_dataset(frame, lookback=120, horizon=20, stride=8, symbol=sym)

    datasets = [feats(f, s) for s, f in frames.items()]
    rep = honest_backtest(datasets, perms=40, frames=frames, feature_builder=feats,
                          injection_fractions=(0.0, 0.5), terminal_frac=0.08, seed=7)
    assert rep.injection_sweep is not None and len(rep.injection_sweep) == 2
    # fractions are ordered and the first rung is survivors-only (0%).
    assert rep.injection_sweep[0].delisted_fraction == 0.0
    assert rep.injection_sweep[-1].delisted_fraction > 0.0
    assert "injection_sweep" in rep.to_dict()


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} harness tests …\n")
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
