from __future__ import annotations

import numpy as np

from yardstick3d.constraints.speed import _metric_centers
from yardstick3d.geometry.trajectory import find_frame_index
from yardstick3d.types import (
    ConstraintClass,
    ConstraintValidity,
    FloatArray,
    MetricConstraint,
    MetricState,
    PredictionBundle,
)


class AltitudeConstraint(MetricConstraint):
    """Barometric / GNSS height change: e_z^T (p_j - p_i) ≈ Δh.

    Requires a gravity-aligned z axis (or Mode B roll/pitch). If the 3DFM
    frame is arbitrarily rotated, this cue is degenerate for scale-only Mode A
    unless `use_magnitude_fallback` is set and Δh dominates the motion.
    """

    name = "altitude"
    cue_class = ConstraintClass.DIRECT_DISPLACEMENT

    def __init__(
        self,
        t_i: float,
        t_j: float,
        delta_h_m: float,
        sigma: float = 0.5,
        axis: FloatArray | None = None,
        max_dt_match: float = 0.05,
        gravity_aligned: bool = True,
    ) -> None:
        self.t_i = float(t_i)
        self.t_j = float(t_j)
        self.delta_h_m = float(delta_h_m)
        self.sigma = float(sigma)
        self.axis = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, dtype=np.float64)
        self.axis = self.axis / (np.linalg.norm(self.axis) + 1e-15)
        self.max_dt_match = float(max_dt_match)
        self.gravity_aligned = bool(gravity_aligned)

    def _indices(self, prediction: PredictionBundle) -> tuple[int, int] | None:
        i = find_frame_index(prediction.timestamps, self.t_i, self.max_dt_match)
        j = find_frame_index(prediction.timestamps, self.t_j, self.max_dt_match)
        if i is None or j is None or i == j:
            return None
        return i, j

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        if self._indices(prediction) is None:
            return ConstraintValidity.OUT_OF_WINDOW
        if not self.gravity_aligned:
            return ConstraintValidity.DEGENERATE
        return ConstraintValidity.VALID

    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        idx = self._indices(prediction)
        if idx is None:
            return np.array([np.nan])
        i, j = idx
        C = _metric_centers(prediction, state)
        pred = float(self.axis @ (C[j] - C[i]))
        return np.array([pred - self.delta_h_m], dtype=np.float64)

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        idx = self._indices(prediction)
        if idx is None:
            return np.array([0.0])
        i, j = idx
        C = prediction.centers()
        a = float(self.axis @ (state.rotation @ (C[j] - C[i])))
        return np.array([a], dtype=np.float64)

    def covariance(self) -> FloatArray:
        return np.array([[self.sigma**2]], dtype=np.float64)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        if not self.gravity_aligned:
            return None
        idx = self._indices(prediction)
        if idx is None:
            return None
        i, j = idx
        C = prediction.centers()
        a = float(self.axis @ (C[j] - C[i]))
        if abs(a) < 1e-15:
            return None
        return a, self.delta_h_m, self.sigma**2
