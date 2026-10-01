from __future__ import annotations

import numpy as np

from yardstick3d.types import (
    ConstraintClass,
    ConstraintValidity,
    FloatArray,
    MetricConstraint,
    MetricState,
    PredictionBundle,
)


class GravityConstraint(MetricConstraint):
    """Align 3DFM gravity direction with measured gravity. Does not observe scale."""

    name = "gravity"
    cue_class = ConstraintClass.DIRECTIONAL

    def __init__(self, gravity_world: FloatArray, sigma_rad: float = 0.05) -> None:
        g = np.asarray(gravity_world, dtype=np.float64).reshape(3)
        self.gravity_world = g / (np.linalg.norm(g) + 1e-15)
        self.sigma_rad = float(sigma_rad)

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        return ConstraintValidity.VALID

    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        # Predicted down in metric world is R * e_down_fm. Default e_down = -z of first camera.
        T = np.asarray(prediction.T_w2c[0], dtype=np.float64)
        R_w2c = T[:3, :3]
        down_fm = R_w2c.T @ np.array([0.0, 1.0, 0.0])  # OpenCV +Y down
        down_m = state.rotation @ down_fm
        down_m = down_m / (np.linalg.norm(down_m) + 1e-15)
        return (down_m - self.gravity_world).astype(np.float64)

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        return np.zeros(3, dtype=np.float64)

    def covariance(self) -> FloatArray:
        return (self.sigma_rad**2) * np.eye(3)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        return None
