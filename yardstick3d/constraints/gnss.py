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


class GNSSDisplacementConstraint(MetricConstraint):
    """Vector displacement: (p_j - p_i) ≈ Δg_ij in a world frame.

    If the 3DFM frame is not aligned with GNSS, Mode A uses the magnitude
    ||p_j - p_i|| ≈ ||Δg|| which constrains scale only.
    Set `use_vector=True` only when frames are believed aligned (Mode B).
    """

    name = "gnss_displacement"
    cue_class = ConstraintClass.DIRECT_DISPLACEMENT

    def __init__(
        self,
        t_i: float,
        t_j: float,
        delta_xyz_m: FloatArray,
        sigma: float | FloatArray = 1.0,
        max_dt_match: float = 0.05,
        use_vector: bool = False,
    ) -> None:
        self.t_i = float(t_i)
        self.t_j = float(t_j)
        self.delta = np.asarray(delta_xyz_m, dtype=np.float64).reshape(3)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.max_dt_match = float(max_dt_match)
        self.use_vector = bool(use_vector)

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
        dp = C[j] - C[i]
        if self.use_vector:
            return (dp - self.delta).astype(np.float64)
        return np.array([float(np.linalg.norm(dp) - np.linalg.norm(self.delta))], dtype=np.float64)

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        idx = self._indices(prediction)
        if idx is None:
            return np.array([0.0])
        i, j = idx
        C = prediction.centers()
        dhat = C[j] - C[i]
        if self.use_vector:
            return dhat.astype(np.float64)
        n = float(np.linalg.norm(dhat))
        return np.array([n], dtype=np.float64)

    def covariance(self) -> FloatArray:
        if self.use_vector:
            if self.sigma.ndim == 0:
                return (float(self.sigma) ** 2) * np.eye(3)
            if self.sigma.ndim == 1:
                return np.diag(self.sigma**2)
            return np.asarray(self.sigma, dtype=np.float64)
        s = float(np.mean(self.sigma)) if self.sigma.ndim else float(self.sigma)
        return np.array([[s**2]], dtype=np.float64)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        idx = self._indices(prediction)
        if idx is None:
            return None
        i, j = idx
        C = prediction.centers()
        a = float(np.linalg.norm(C[j] - C[i]))
        b = float(np.linalg.norm(self.delta))
        if a < 1e-15:
            return None
        s = float(np.mean(self.sigma)) if self.sigma.ndim else float(self.sigma)
        return a, b, s**2


class GNSSPositionConstraint(MetricConstraint):
    """Absolute position p_i ≈ g_i, only well-posed with translation gauge free (Mode B)."""

    name = "gnss_position"
    cue_class = ConstraintClass.ABSOLUTE_POSITION

    def __init__(
        self,
        t: float,
        xyz_m: FloatArray,
        sigma: float | FloatArray = 2.0,
        max_dt_match: float = 0.05,
    ) -> None:
        self.t = float(t)
        self.xyz = np.asarray(xyz_m, dtype=np.float64).reshape(3)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.max_dt_match = float(max_dt_match)

    def _index(self, prediction: PredictionBundle) -> int | None:
        return find_frame_index(prediction.timestamps, self.t, self.max_dt_match)

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        return (
            ConstraintValidity.VALID
            if self._index(prediction) is not None
            else ConstraintValidity.OUT_OF_WINDOW
        )

    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        k = self._index(prediction)
        if k is None:
            return np.full(3, np.nan)
        C = _metric_centers(prediction, state)
        return (C[k] - self.xyz).astype(np.float64)

    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        k = self._index(prediction)
        if k is None:
            return np.zeros(3)
        C = prediction.centers()
        return (state.rotation @ C[k]).astype(np.float64)

    def covariance(self) -> FloatArray:
        if self.sigma.ndim == 0:
            return (float(self.sigma) ** 2) * np.eye(3)
        if self.sigma.ndim == 1:
            return np.diag(self.sigma**2)
        return np.asarray(self.sigma, dtype=np.float64)

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        return None
