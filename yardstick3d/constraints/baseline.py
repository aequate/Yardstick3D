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


class BaselineConstraint(MetricConstraint):
    """Known inter-camera distance ||p_i - p_j|| ≈ b (stereo / rig)."""

    name = "baseline"
    cue_class = ConstraintClass.STRUCTURAL_PRIOR

    def __init__(self, i: int, j: int, baseline_m: float, sigma: float = 0.01) -> None:
        self.i = int(i)
        self.j = int(j)
        self.baseline_m = float(baseline_m)
        self.sigma = float(sigma)

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        n = prediction.n_frames()
        if self.i < 0 or self.j < 0 or self.i >= n or self.j >= n or self.i == self.j:
            return ConstraintValidity.OUT_OF_WINDOW
        return ConstraintValidity.VALID

    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        C = _metric_centers(prediction, state)
        return np.array(
            [float(np.linalg.norm(C[self.j] - C[self.i]) - self.baseline_m)], dtype=np.float64
        )

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        C = prediction.centers()
        return np.array([float(np.linalg.norm(C[self.j] - C[self.i]))], dtype=np.float64)

    def covariance(self) -> FloatArray:
        return np.array([[self.sigma**2]], dtype=np.float64)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        C = prediction.centers()
        a = float(np.linalg.norm(C[self.j] - C[self.i]))
        if a < 1e-15:
            return None
        return a, self.baseline_m, self.sigma**2
