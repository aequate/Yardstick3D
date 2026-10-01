from __future__ import annotations

from yardstick3d.constraints.base import ConstraintSet

FORBIDDEN_KEYS = (
    "gt_pose",
    "gt_scale",
    "ground_truth",
    "state_groundtruth_estimate",
    "poses_gt",
)


def assert_no_gt_in_constraints(constraints: ConstraintSet) -> None:
    for c in constraints.constraints:
        meta = c.metadata()
        for k in meta:
            if any(f in str(k).lower() for f in FORBIDDEN_KEYS):
                raise RuntimeError(f"Metric leakage: constraint metadata contains {k}")
