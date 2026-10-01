"""Build Yardstick3D constraints from ADVIO *inference* channels only."""

from __future__ import annotations

from yardstick3d.constraints.base import ConstraintSet
from yardstick3d.constraints.speed import PathDisplacementConstraint
from yardstick3d.datasets.advio import InferenceChannels, corelocation_interval_distance
from yardstick3d.types import FloatArray


FORBIDDEN_INFERENCE_ATTRS = (
    "centers",
    "quat_wxyz",
    "evaluation",
    "pose.csv",
    "arkit",
    "arcore",
    "accelerometer",
    "gyro",
)


def assert_inference_only(inf: InferenceChannels) -> None:
    inf.assert_imu_denied()
    blob = inf.notes
    for k in blob:
        lk = str(k).lower()
        if any(f in lk for f in FORBIDDEN_INFERENCE_ATTRS):
            raise RuntimeError(f"leakage in inference notes: {k}")


def constraints_from_corelocation_displacement(
    inf: InferenceChannels,
    frame_t: FloatArray,
    min_dt: float = 0.3,
    sigma_floor_m: float = 2.0,
) -> ConstraintSet:
    assert_inference_only(inf)
    cs = ConstraintSet()
    t = list(map(float, frame_t))
    for i in range(len(t) - 1):
        if t[i + 1] - t[i] < min_dt:
            continue
        d = corelocation_interval_distance(inf, t[i], t[i + 1])
        if d is None or d < 1e-3:
            continue
        # uncertainty from reported horizontal accuracy (mean in window)
        m = (inf.location_t >= t[i]) & (inf.location_t <= t[i + 1])
        h = float(inf.hacc_m[m].mean()) if m.any() else 10.0
        sigma = max(sigma_floor_m, 0.5 * h)
        cs.add(
            PathDisplacementConstraint(
                t_i=t[i],
                t_j=t[i + 1],
                length_m=float(d),
                sigma=sigma,
                max_dt_match=0.05,
            )
        )
    return cs
