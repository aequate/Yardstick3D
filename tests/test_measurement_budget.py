"""Deterministic unit tests for measurement-budget subsampling helpers.

Pure index arithmetic only: no GT, no file I/O, no randomness outside
seeded helpers, no DA3 inference.
"""

from __future__ import annotations

import numpy as np

from yardstick3d.constraints.base import ConstraintSet
from yardstick3d.constraints.speed import PathDisplacementConstraint
from yardstick3d.evaluation.measurement_budget_helpers import (
    BUDGETS,
    STRATEGIES,
    budget_to_k,
    build_constraint_subset,
    subset_indices_largest_predicted_displacement,
    subset_indices_longest_baseline,
    subset_indices_random,
    subset_indices_uniform,
)


def _toy_cs(n: int) -> ConstraintSet:
    cs = ConstraintSet()
    for i in range(n):
        cs.add(
            PathDisplacementConstraint(
                t_i=float(i), t_j=float(i + 1), length_m=float(i + 1), sigma=2.0
            )
        )
    return cs


def test_budget_to_k_at_n7():
    assert budget_to_k("100pct", 7) == 7
    assert budget_to_k("50pct", 7) == 4
    assert budget_to_k("25pct", 7) == 2
    assert budget_to_k("10pct", 7) == 1
    assert budget_to_k("5pct", 7) == 1
    assert budget_to_k("k2", 7) == 2
    assert budget_to_k("k1_long", 7) == 1


def test_budget_to_k_collapse_documented():
    # 10pct and 5pct collapse at n=7; k2 duplicates 25pct by construction.
    assert budget_to_k("10pct", 7) == budget_to_k("5pct", 7) == 1
    assert budget_to_k("k2", 7) == budget_to_k("25pct", 7) == 2
    # Absolute budgets cap correctly on tiny windows.
    assert budget_to_k("k2", 1) == 1
    assert budget_to_k("k1_long", 1) == 1
    assert budget_to_k("50pct", 1) == 1
    assert budget_to_k("100pct", 0) == 0


def test_random_is_seeded_sorted_and_sized():
    a = subset_indices_random(7, 4, seed=0)
    b = subset_indices_random(7, 4, seed=0)
    assert a == b  # deterministic
    assert len(a) == 4
    assert a == sorted(a)
    assert len(set(a)) == 4
    assert all(0 <= i < 7 for i in a)
    # Different seeds may differ (not asserted strictly), but shape holds.
    c = subset_indices_random(7, 4, seed=1)
    assert len(c) == 4 and c == sorted(c)
    # Full budget returns all indices sorted regardless of seed permutation.
    assert subset_indices_random(7, 7, seed=0) == list(range(7))
    assert subset_indices_random(7, 1, seed=2) == subset_indices_random(7, 1, seed=2)


def test_uniform_even_spacing():
    assert subset_indices_uniform(7, 7) == list(range(7))
    assert subset_indices_uniform(7, 4) == [0, 2, 4, 6]
    assert subset_indices_uniform(7, 2) == [0, 6]
    assert subset_indices_uniform(7, 1) == [0]
    assert subset_indices_uniform(1, 1) == [0]
    assert subset_indices_uniform(0, 3) == []


def test_longest_baseline_picks_largest_earliest_tiebreak_sorted():
    lengths = [3.9, 3.1, 3.6, 2.9, 3.3, 3.6, 3.7]
    # Top-1 is index 0 (3.9).
    assert subset_indices_longest_baseline(lengths, 1) == [0]
    # Top-2: 0 (3.9) and 6 (3.7).
    assert subset_indices_longest_baseline(lengths, 2) == [0, 6]
    # Ties: two 3.6 at idx 2 and 5; earliest wins for k=3? top3 = 0,6,2.
    assert subset_indices_longest_baseline(lengths, 3) == [0, 2, 6]
    # Output sorted ascending even though selection is by value.
    assert subset_indices_longest_baseline([1.0, 5.0, 2.0], 2) == [1, 2]


def test_largest_predicted_displacement_same_ordering_rule():
    norms = [0.5, 2.0, 1.0, 2.0, 0.1]
    # Tie at 2.0 (idx 1,3): earliest wins for k=1.
    assert subset_indices_largest_predicted_displacement(norms, 1) == [1]
    assert subset_indices_largest_predicted_displacement(norms, 2) == [1, 3]
    assert subset_indices_largest_predicted_displacement(norms, 5) == [0, 1, 2, 3, 4]


def test_build_subset_preserves_sorted_order_and_length():
    cs = _toy_cs(7)
    sub = build_constraint_subset(cs, [6, 0, 2])
    assert len(sub) == 3
    # Sorted ascending: t_i should be 0, 2, 6.
    assert [c.t_i for c in sub.constraints] == [0.0, 2.0, 6.0]
    # k=1 edge case: single interval preserved (median-of-one defined by solver).
    one = build_constraint_subset(cs, [3])
    assert len(one) == 1
    assert one.constraints[0].length_m == 4.0


def test_helpers_are_pure_no_gt_io():
    # No NaN leakage: NaN values sort last deterministically.
    idx = subset_indices_longest_baseline([1.0, float("nan"), 2.0], 1)
    assert idx == [2]
    idx2 = subset_indices_largest_predicted_displacement(
        [float("nan"), 0.3, 0.2], 1
    )
    assert idx2 == [1]
    # Budget labels cover the frozen protocol grid.
    assert set(BUDGETS) == {"100pct", "50pct", "25pct", "10pct", "5pct", "k2", "k1_long"}
    assert set(STRATEGIES) == {
        "random",
        "uniform",
        "longest_baseline",
        "largest_predicted_displacement",
    }
    # Uniform with numpy int input still returns python ints sorted.
    out = subset_indices_uniform(np.int64(7), np.int64(2))
    assert out == [0, 6] and all(isinstance(v, int) for v in out)
