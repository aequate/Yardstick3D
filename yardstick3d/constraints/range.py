from __future__ import annotations

import numpy as np

from yardstick3d.constraints.speed import _metric_centers
from yardstick3d.geometry.sim3 import Sim3, apply_sim3_points
from yardstick3d.types import (
    ConstraintClass,
    ConstraintValidity,
    FloatArray,
    MetricConstraint,
    MetricState,
    PredictionBundle,
)


class SparseRangeConstraint(MetricConstraint):
    """||X_j - p_i|| ≈ d, or camera-frame Z ≈ d for a pixel.

    If `point_world` is in the 3DFM frame, scale applies to both camera and point.
    Range is invariant to the global translation/rotation of a consistent Sim(3)
    only after both are transformed; the residual then scales as s * d_hat - d.
    """

    name = "sparse_range"
    cue_class = ConstraintClass.SPARSE_DISTANCE

    def __init__(
        self,
        frame_index: int,
        range_m: float,
        point_world: FloatArray | None = None,
        depth_z: float | None = None,
        sigma: float = 0.05,
    ) -> None:
        self.frame_index = int(frame_index)
        self.range_m = float(range_m)
        self.point_world = None if point_world is None else np.asarray(point_world, dtype=np.float64)
        self.depth_z = None if depth_z is None else float(depth_z)
        self.sigma = float(sigma)
        if self.point_world is None and self.depth_z is None:
            raise ValueError("SparseRangeConstraint requires point_world or depth_z")

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        if self.frame_index < 0 or self.frame_index >= prediction.n_frames():
            return ConstraintValidity.OUT_OF_WINDOW
        return ConstraintValidity.VALID

    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        if self.depth_z is not None:
            pred = state.scale * state.depth_scale * self.depth_z
            return np.array([pred - self.range_m], dtype=np.float64)
        C = _metric_centers(prediction, state)
        sim = Sim3(scale=state.scale, R=state.rotation, t=state.translation)
        X = apply_sim3_points(self.point_world, sim)
        return np.array([float(np.linalg.norm(X - C[self.frame_index]) - self.range_m)], dtype=np.float64)

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        if self.depth_z is not None:
            return np.array([state.depth_scale * self.depth_z], dtype=np.float64)
        C = prediction.centers()
        a = float(np.linalg.norm(self.point_world - C[self.frame_index]))
        return np.array([a], dtype=np.float64)

    def covariance(self) -> FloatArray:
        return np.array([[self.sigma**2]], dtype=np.float64)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        if self.depth_z is not None:
            a = float(self.depth_z)
            if abs(a) < 1e-15:
                return None
            return a, self.range_m, self.sigma**2
        C = prediction.centers()
        a = float(np.linalg.norm(self.point_world - C[self.frame_index]))
        if a < 1e-15:
            return None
        return a, self.range_m, self.sigma**2
