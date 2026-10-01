from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


class ConstraintClass(str, Enum):
    DIRECT_DISPLACEMENT = "direct_displacement"
    SPARSE_DISTANCE = "sparse_distance"
    STRUCTURAL_PRIOR = "structural_prior"
    DIRECTIONAL = "directional"
    ABSOLUTE_POSITION = "absolute_position"
    SEMANTIC_PRIOR = "semantic_prior"


class ConstraintValidity(str, Enum):
    VALID = "valid"
    DEGENERATE = "degenerate"
    OUT_OF_WINDOW = "out_of_window"
    REJECTED = "rejected"
    MISSING = "missing"


@dataclass
class MetricState:
    """Gauge variables applied to a frozen 3DFM prediction.

    Mode A uses scale only (R=I, t=0).
    Mode B additionally estimates Sim(3) alignment.
    Camera-frame depth is scaled by `scale` (and optionally `depth_scale`).
    """

    scale: float = 1.0
    rotation: FloatArray = field(default_factory=lambda: np.eye(3))
    translation: FloatArray = field(default_factory=lambda: np.zeros(3))
    depth_scale: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> MetricState:
        return MetricState(
            scale=float(self.scale),
            rotation=np.array(self.rotation, dtype=np.float64, copy=True),
            translation=np.array(self.translation, dtype=np.float64, copy=True),
            depth_scale=float(self.depth_scale),
            extra=dict(self.extra),
        )


@dataclass
class PredictionBundle:
    """Canonical 3DFM output. Poses are OpenCV, world-to-camera.

    Camera center in the 3DFM world frame: C = -R^T t for T_w2c = [R|t].
    Depth is pinhole Z in the camera frame.
    Units of the 3DFM world are unknown unless `is_metric` is True.
    """

    timestamps: FloatArray
    T_w2c: FloatArray
    K: FloatArray | None = None
    depth_z: FloatArray | None = None
    points_world: FloatArray | None = None
    camera_centers: FloatArray | None = None
    depth_conf: FloatArray | None = None
    point_conf: FloatArray | None = None
    is_metric: bool = False
    model_id: str = "unknown"
    aux: dict[str, Any] = field(default_factory=dict)

    def n_frames(self) -> int:
        return int(self.T_w2c.shape[0])

    def centers(self) -> FloatArray:
        if self.camera_centers is not None:
            return np.asarray(self.camera_centers, dtype=np.float64)
        from yardstick3d.geometry.cameras import camera_centers_from_w2c

        return camera_centers_from_w2c(self.T_w2c)


@dataclass
class SolverDiagnostics:
    n_iterations: int = 0
    converged: bool = False
    cost: float = np.nan
    cost_history: list[float] = field(default_factory=list)
    hessian_scale: float = np.nan
    condition_number: float = np.nan
    n_accepted: int = 0
    n_rejected: int = 0
    residual_breakdown: dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class GroundingResult:
    state: MetricState
    trajectory_metric: FloatArray
    depth_metric: FloatArray | None
    points_metric: FloatArray | None
    scale: float
    scale_std: float
    observability_score: float
    observable: bool
    diagnostics: SolverDiagnostics
    accepted_constraints: int
    rejected_constraints: int
    residual_breakdown: dict[str, float]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "scale_std": float(self.scale_std),
            "observable": bool(self.observable),
            "observability_score": float(self.observability_score),
            "accepted_constraints": int(self.accepted_constraints),
            "rejected_constraints": int(self.rejected_constraints),
            "residual_breakdown": {k: float(v) for k, v in self.residual_breakdown.items()},
            "diagnostics": {
                "n_iterations": self.diagnostics.n_iterations,
                "converged": self.diagnostics.converged,
                "cost": float(self.diagnostics.cost),
                "hessian_scale": float(self.diagnostics.hessian_scale),
                "condition_number": float(self.diagnostics.condition_number),
                "message": self.diagnostics.message,
            },
            "provenance": self.provenance,
        }


class MetricConstraint(ABC):
    name: str = "constraint"
    cue_class: ConstraintClass = ConstraintClass.DIRECT_DISPLACEMENT

    @abstractmethod
    def residual(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        """Residual vector in measurement space (metric units)."""

    @abstractmethod
    def jacobian_scale(self, prediction: PredictionBundle, state: MetricState) -> FloatArray:
        """dr/ds, same length as residual."""

    @abstractmethod
    def covariance(self) -> FloatArray:
        """Covariance of the residual (scalar, vector, or diagonal)."""

    def validity(self, prediction: PredictionBundle, state: MetricState) -> ConstraintValidity:
        return ConstraintValidity.VALID

    def scale_observation(self, prediction: PredictionBundle) -> tuple[float, float, float] | None:
        """Optional linear observation s * a ≈ b with variance σ².

        Returns (a, b, variance) or None if this cue is not a pure scale factor.
        """
        return None

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "class": self.cue_class.value}
