from __future__ import annotations

import numpy as np

from yardstick3d.geometry.sim3 import Sim3, apply_sim3_points
from yardstick3d.types import (
    ConstraintClass,
    ConstraintValidity,
    FloatArray,
    MetricConstraint,
    MetricState,
    PredictionBundle,
)


class ObjectSizeConstraint(MetricConstraint):
    """||X_a - X_b|| ≈ L. Weak semantic prior — default large sigma."""

    name = "object_size"
    cue_class = ConstraintClass.SEMANTIC_PRIOR

    def __init__(
        self,
        point_a: FloatArray,
        point_b: FloatArray,
        length_m: float,
        sigma: float = 0.4,
    ) -> None:
        self.point_a = np.asarray(point_a, dtype=np.float64).reshape(3)
        self.point_b = np.asarray(point_b, dtype=np.float64).reshape(3)
        self.length_m = float(length_m)
        self.sigma = float(sigma)

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        if np.linalg.norm(self.point_a - self.point_b) < 1e-15:
            return ConstraintValidity.DEGENERATE
        return ConstraintValidity.VALID

    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        sim = Sim3(scale=state.scale, R=state.rotation, t=state.translation)
        a = apply_sim3_points(self.point_a, sim)
        b = apply_sim3_points(self.point_b, sim)
        return np.array([float(np.linalg.norm(a - b) - self.length_m)], dtype=np.float64)

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        a = float(np.linalg.norm(self.point_a - self.point_b))
        return np.array([a], dtype=np.float64)

    def covariance(self) -> FloatArray:
        return np.array([[self.sigma**2]], dtype=np.float64)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        a = float(np.linalg.norm(self.point_a - self.point_b))
        if a < 1e-15:
            return None
        return a, self.length_m, self.sigma**2
