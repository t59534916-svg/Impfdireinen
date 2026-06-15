"""Tests for the block-permutation null (vpts.stats.block_permutation).

The load-bearing test demonstrates *why it exists*: on autocorrelated targets the
block permutation yields a more conservative p-value than the per-row shuffle that
the rest of the harness uses by default.

    python tests/test_block_permutation.py
    pytest tests/test_block_permutation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.stats import (  # noqa: E402
    block_permutation_test,
    block_shuffle_indices,
    recommend_block_size,
)


def test_recommend_block_size() -> None:
    assert recommend_block_size(20, 5) == 5          # ceil(20/5)+1
    assert recommend_block_size(20, 20) == 2         # non-overlapping ⇒ tiny block
    for bad in ((0, 1), (1, 0), (-5, 1)):
        try:
            recommend_block_size(*bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {bad}")


def test_block_shuffle_is_a_permutation_preserving_blocks() -> None:
    rng = np.random.default_rng(0)
    idx = block_shuffle_indices(20, 5, rng=rng)
    assert sorted(idx.tolist()) == list(range(20))   # a true permutation
    # Each original contiguous block of 5 survives intact somewhere in the output.
    s = idx.tolist()
    for b in range(0, 20, 5):
        block = list(range(b, b + 5))
        assert any(s[i:i + 5] == block for i in range(0, 20 - 4))


def test_pvalue_uniformish_under_true_null() -> None:
    # Feature and target independent ⇒ p should not be tiny.
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    y = rng.normal(size=400)
    res = block_permutation_test(
        y, lambda yy: float(np.corrcoef(x, yy)[0, 1]),
        block_size=10, n_permutations=300, seed=2)
    assert res.p_value > 0.05                          # no spurious significance


def test_block_is_more_conservative_than_per_row_on_autocorrelated_target() -> None:
    # The spurious-regression trap: two INDEPENDENT random walks correlate by chance.
    # Per-row shuffling destroys the integration and falsely declares significance;
    # block permutation preserves it and correctly does not.
    rng = np.random.default_rng(2)
    n = 500
    x = np.cumsum(rng.normal(size=n))
    y = np.cumsum(rng.normal(size=n))                 # independent of x

    def stat(yy: np.ndarray) -> float:
        return float(np.corrcoef(x, yy)[0, 1])

    real = stat(y)

    rng2 = np.random.default_rng(99)
    per_row_null = np.array([stat(y[rng2.permutation(n)]) for _ in range(300)])
    p_per_row = (np.sum(np.abs(per_row_null) >= abs(real)) + 1) / (per_row_null.size + 1)

    block = block_permutation_test(y, stat, block_size=50, n_permutations=300,
                                   alternative="two-sided", seed=99)

    assert abs(real) > 0.3                              # a sizeable spurious correlation
    assert p_per_row < 0.05                             # per-row: FALSE POSITIVE
    assert block.null_std > per_row_null.std()          # block null is wider (honest)
    assert block.p_value > p_per_row and block.p_value > 0.05  # block: correctly not sig.
    assert "block" in block.summary().lower()


def test_alternatives_and_validation() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(size=100)
    for alt in ("greater", "less", "two-sided"):
        r = block_permutation_test(y, lambda yy: float(yy.mean()), block_size=10,
                                   n_permutations=50, alternative=alt, seed=1)
        assert 0.0 < r.p_value <= 1.0
    try:
        block_permutation_test(y, lambda yy: 0.0, block_size=5, alternative="bogus")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for bad alternative")


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} block-permutation tests …\n")
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
