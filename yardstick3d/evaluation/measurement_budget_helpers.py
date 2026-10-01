"""Deterministic measurement-budget subsampling helpers (GT-free, no I/O).

Pure index arithmetic for choosing which k cue measurements the solver
sees (random, uniform, longest baseline or largest predicted step). No
ground truth, no file I/O, no DA3 inference. All strategies return
constraint indices sorted ascending for solver input.
"""

from __future__ import annotations

import numpy as np


BUDGETS = ("100pct", "50pct", "25pct", "10pct", "5pct", "k2", "k1_long")
STRATEGIES = ("random", "uniform", "longest_baseline", "largest_predicted_displacement")
RANDOM_SEEDS = (0, 1, 2, 3, 4)


def budget_to_k(budget: str, n: int) -> int:
    """Map budget label to integer k per protocol Sec.3.

    k=max(1, round(frac*n)) for fractions; absolute budgets override.
    Uses Python round (bankers) to match scripts/measurement_budget.py
    ``int(round(frac * n))`` exactly.
    """
    n = int(n)
    if n <= 0:
        return 0
    if budget == "100pct":
        return n
    if budget == "50pct":
        return max(1, int(round(0.50 * n)))
    if budget == "25pct":
        return max(1, int(round(0.25 * n)))
    if budget == "10pct":
        return max(1, int(round(0.10 * n)))
    if budget == "5pct":
        return max(1, int(round(0.05 * n)))
    if budget == "k2":
        return min(2, n)
    if budget == "k1_long":
        return min(1, n)
    raise ValueError(f"unknown budget {budget!r}")


def subset_indices_random(n: int, k: int, seed: int) -> list[int]:
    """Seeded choice without replacement, sorted ascending (protocol Sec.4)."""
    n, k = int(n), int(k)
    if n <= 0 or k <= 0:
        return []
    kk = min(k, n)
    rng = np.random.default_rng(int(seed))
    take = rng.choice(n, size=kk, replace=False)
    return sorted(int(j) for j in take)


def subset_indices_uniform(n: int, k: int) -> list[int]:
    """Evenly spaced coverage: linspace(0,n-1,k).round(), unique, sorted."""
    n, k = int(n), int(k)
    if n <= 0 or k <= 0:
        return []
    kk = min(k, n)
    raw = np.linspace(0, n - 1, kk).round().astype(int).tolist()
    # Unique preserving order of appearance (raw is already ascending).
    seen: list[int] = []
    for v in raw:
        iv = int(v)
        iv = max(0, min(n - 1, iv))
        if iv not in seen:
            seen.append(iv)
    # If rounding duplicates collapsed entries (only possible when kk>n,
    # already capped, but handle defensively), fill smallest missing first.
    if len(seen) < kk:
        for cand in range(n):
            if cand not in seen:
                seen.append(cand)
            if len(seen) >= kk:
                break
    return sorted(seen[:kk])


def _topk_by_value(values: list[float] | np.ndarray, k: int) -> list[int]:
    """Top-k indices by descending value, earliest-index tie-break, sorted asc."""
    n = len(values)
    k = int(k)
    if n <= 0 or k <= 0:
        return []
    kk = min(k, n)
    vals = [float(v) for v in list(values)]
    # NaN sorts last (treated as -inf).
    order = sorted(
        range(n),
        key=lambda i: (-vals[i] if np.isfinite(vals[i]) else float("inf"), i),
    )
    take = order[:kk]
    return sorted(take)


def subset_indices_longest_baseline(lengths: list[float] | np.ndarray, k: int) -> list[int]:
    """k constraints with largest cue length_m, earliest tie-break, sorted asc."""
    return _topk_by_value(list(lengths), k)


def subset_indices_largest_predicted_displacement(
    disp_norms: list[float] | np.ndarray, k: int
) -> list[int]:
    """k constraints with largest predicted visual displacement norm, sorted asc."""
    return _topk_by_value(list(disp_norms), k)


def build_constraint_subset(cs, indices: list[int] | np.ndarray):
    """Build a new ConstraintSet with constraints at sorted indices."""
    from yardstick3d.constraints.base import ConstraintSet

    idx = sorted(int(i) for i in list(indices))
    out = ConstraintSet()
    for i in idx:
        out.add(cs.constraints[i])
    return out


def predicted_displacement_norms(pred, cs) -> list[float]:
    """GT-free visual displacement norms per constraint (for largest-disp strategy).

    For each constraint, ||c_j - c_i|| from cached pred.centers() at the
    constraint endpoint timestamps. Non-matching constraints yield NaN
    (sorted last by the selector). Never uses GT.
    """
    import numpy as np

    from yardstick3d.geometry.trajectory import find_frame_index

    centers = np.asarray(pred.centers(), dtype=np.float64)
    ts = np.asarray(pred.timestamps, dtype=np.float64)
    out: list[float] = []
    for c in cs.constraints:
        try:
            max_dt = float(getattr(c, "max_dt_match", 0.05))
            t_i = float(getattr(c, "t_i"))
            t_j = float(getattr(c, "t_j"))
        except (AttributeError, TypeError, ValueError):
            out.append(float("nan"))
            continue
        i = find_frame_index(ts, t_i, max_dt)
        j = find_frame_index(ts, t_j, max_dt)
        if i is None or j is None:
            out.append(float("nan"))
            continue
        try:
            out.append(float(np.linalg.norm(centers[j] - centers[i])))
        except Exception:
            out.append(float("nan"))
    return out
