from __future__ import annotations

import numpy as np

from yardstick3d.geometry.cameras import camera_centers_from_w2c
from yardstick3d.geometry.sim3 import Sim3, apply_sim3_centers
from yardstick3d.geometry.trajectory import find_frame_index
from yardstick3d.types import (
    ConstraintClass,
    ConstraintValidity,
    FloatArray,
    MetricConstraint,
    MetricState,
    PredictionBundle,
)


def _metric_centers(prediction: PredictionBundle, state: MetricState) -> FloatArray:
    C = prediction.centers()
    sim = Sim3(scale=state.scale, R=state.rotation, t=state.translation)
    return apply_sim3_centers(C, sim)


class SpeedConstraint(MetricConstraint):
    """Scalar speed over an interval: ||p_j - p_i|| / Δt ≈ v.

    Observes scale only. Does not constrain orientation or translation gauge.
    """

    name = "speed"
    cue_class = ConstraintClass.DIRECT_DISPLACEMENT

    def __init__(
        self,
        t_i: float,
        t_j: float,
        speed_mps: float,
        sigma: float = 0.1,
        max_dt_match: float = 0.05,
    ) -> None:
        self.t_i = float(t_i)
        self.t_j = float(t_j)
        self.speed_mps = float(speed_mps)
        self.sigma = float(sigma)
        self.max_dt_match = float(max_dt_match)

    def _indices(self, prediction: PredictionBundle) -> tuple[int, int] | None:
        i = find_frame_index(prediction.timestamps, self.t_i, self.max_dt_match)
        j = find_frame_index(prediction.timestamps, self.t_j, self.max_dt_match)
        if i is None or j is None or i == j:
            return None
        return i, j

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        idx = self._indices(prediction)
        if idx is None:
            return ConstraintValidity.OUT_OF_WINDOW
        i, j = idx
        C = camera_centers_from_w2c(prediction.T_w2c)
        if np.linalg.norm(C[j] - C[i]) < 1e-12 and abs(self.speed_mps) < 1e-12:
            return ConstraintValidity.DEGENERATE
        return ConstraintValidity.VALID

    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        idx = self._indices(prediction)
        if idx is None:
            return np.array([np.nan])
        i, j = idx
        C = _metric_centers(prediction, state)
        dt = abs(self.t_j - self.t_i)
        pred = float(np.linalg.norm(C[j] - C[i]) / max(dt, 1e-12))
        return np.array([pred - self.speed_mps], dtype=np.float64)

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        idx = self._indices(prediction)
        if idx is None:
            return np.array([0.0])
        i, j = idx
        C = prediction.centers()
        dt = abs(self.t_j - self.t_i)
        a = float(np.linalg.norm(C[j] - C[i]) / max(dt, 1e-12))
        return np.array([a], dtype=np.float64)

    def covariance(self) -> FloatArray:
        return np.array([[self.sigma**2]], dtype=np.float64)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        idx = self._indices(prediction)
        if idx is None:
            return None
        i, j = idx
        C = prediction.centers()
        dt = abs(self.t_j - self.t_i)
        a = float(np.linalg.norm(C[j] - C[i]) / max(dt, 1e-12))
        if a < 1e-15:
            return None
        return a, self.speed_mps, self.sigma**2


class PathDisplacementConstraint(MetricConstraint):
    """Integrated path length: ||p_j - p_i|| ≈ L (straight-line displacement)."""

    name = "path_displacement"
    cue_class = ConstraintClass.DIRECT_DISPLACEMENT

    def __init__(
        self,
        t_i: float,
        t_j: float,
        length_m: float,
        sigma: float = 0.1,
        max_dt_match: float = 0.05,
    ) -> None:
        self.t_i = float(t_i)
        self.t_j = float(t_j)
        self.length_m = float(length_m)
        self.sigma = float(sigma)
        self.max_dt_match = float(max_dt_match)

    def _indices(self, prediction: PredictionBundle) -> tuple[int, int] | None:
        i = find_frame_index(prediction.timestamps, self.t_i, self.max_dt_match)
        j = find_frame_index(prediction.timestamps, self.t_j, self.max_dt_match)
        if i is None or j is None or i == j:
            return None
        return i, j

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        return (
            ConstraintValidity.VALID
            if self._indices(prediction) is not None
            else ConstraintValidity.OUT_OF_WINDOW
        )

    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        idx = self._indices(prediction)
        if idx is None:
            return np.array([np.nan])
        i, j = idx
        C = _metric_centers(prediction, state)
        return np.array([float(np.linalg.norm(C[j] - C[i]) - self.length_m)], dtype=np.float64)

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        idx = self._indices(prediction)
        if idx is None:
            return np.array([0.0])
        i, j = idx
        C = prediction.centers()
        return np.array([float(np.linalg.norm(C[j] - C[i]))], dtype=np.float64)

    def covariance(self) -> FloatArray:
        return np.array([[self.sigma**2]], dtype=np.float64)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        idx = self._indices(prediction)
        if idx is None:
            return None
        i, j = idx
        C = prediction.centers()
        a = float(np.linalg.norm(C[j] - C[i]))
        if a < 1e-15:
            return None
        return a, self.length_m, self.sigma**2
