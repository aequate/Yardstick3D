"""Yardstick3D: metrically ground frozen 3D foundation models with weak cues."""

from yardstick3d.types import (
    ConstraintValidity,
    MetricConstraint,
    MetricState,
    PredictionBundle,
    SolverDiagnostics,
    GroundingResult,
)
from yardstick3d.optimization.grounder import MetricGrounder

__version__ = "0.1.0"
__all__ = [
    "ConstraintValidity",
    "MetricConstraint",
    "MetricState",
    "PredictionBundle",
    "SolverDiagnostics",
    "GroundingResult",
    "MetricGrounder",
    "__version__",
]
