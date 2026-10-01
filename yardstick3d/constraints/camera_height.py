from __future__ import annotations

import numpy as np

from yardstick3d.constraints.speed import _metric_centers
from yardstick3d.types import (
    ConstraintClass,
    ConstraintValidity,
    FloatArray,
    MetricConstraint,
    MetricState,
    PredictionBundle,
)


class CameraHeightConstraint(MetricConstraint):
    """dist(p_i, ground plane) ≈ h.

    Plane is n^T x + d = 0 in the 3DFM frame (n unit). Weak prior: large sigma.
    Scale-only observation uses unsigned distance: s * |n^T C + d| ≈ h
    when the plane coefficients are expressed in the unscaled 3DFM frame.
    """

    name = "camera_height"
    cue_class = ConstraintClass.STRUCTURAL_PRIOR

    def __init__(
        self,
        frame_index: int,
        height_m: float,
        plane_n: FloatArray,
        plane_d: float,
        sigma: float = 0.15,
    ) -> None:
        self.frame_index = int(frame_index)
        self.height_m = float(height_m)
        n = np.asarray(plane_n, dtype=np.float64).reshape(3)
        self.plane_n = n / (np.linalg.norm(n) + 1e-15)
        self.plane_d = float(plane_d)
        self.sigma = float(sigma)

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        if self.frame_index < 0 or self.frame_index >= prediction.n_frames():
            return ConstraintValidity.OUT_OF_WINDOW
        return ConstraintValidity.VALID

    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        C = _metric_centers(prediction, state)
        # Plane was defined in unscaled frame: n^T X + d = 0.
        # After X' = s X, plane becomes n^T X'/s + d = 0 so dist' = |n^T C' + s d|.
        dist = abs(float(self.plane_n @ C[self.frame_index] + state.scale * self.plane_d))
        return np.array([dist - self.height_m], dtype=np.float64)

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        C = prediction.centers()
        signed = float(self.plane_n @ C[self.frame_index] + self.plane_d)
        return np.array([abs(signed)], dtype=np.float64)

    def covariance(self) -> FloatArray:
        return np.array([[self.sigma**2]], dtype=np.float64)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        C = prediction.centers()
        a = abs(float(self.plane_n @ C[self.frame_index] + self.plane_d))
        if a < 1e-15:
            return None
        return a, self.height_m, self.sigma**2
