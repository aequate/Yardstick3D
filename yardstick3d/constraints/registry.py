from __future__ import annotations

from typing import Any, Callable

from yardstick3d.constraints.altitude import AltitudeConstraint
from yardstick3d.constraints.baseline import BaselineConstraint
from yardstick3d.constraints.camera_height import CameraHeightConstraint
from yardstick3d.constraints.gnss import GNSSDisplacementConstraint, GNSSPositionConstraint
from yardstick3d.constraints.gravity import GravityConstraint
from yardstick3d.constraints.object_size import ObjectSizeConstraint
from yardstick3d.constraints.range import SparseRangeConstraint
from yardstick3d.constraints.speed import PathDisplacementConstraint, SpeedConstraint
from yardstick3d.types import MetricConstraint

CONSTRAINT_REGISTRY: dict[str, type[MetricConstraint]] = {
    "speed": SpeedConstraint,
    "path_displacement": PathDisplacementConstraint,
    "gnss_displacement": GNSSDisplacementConstraint,
    "gnss_position": GNSSPositionConstraint,
    "altitude": AltitudeConstraint,
    "baseline": BaselineConstraint,
    "sparse_range": SparseRangeConstraint,
    "camera_height": CameraHeightConstraint,
    "object_size": ObjectSizeConstraint,
    "gravity": GravityConstraint,
}


def create_constraint(name: str, **kwargs: Any) -> MetricConstraint:
    if name not in CONSTRAINT_REGISTRY:
        raise KeyError(f"Unknown constraint '{name}'. Known: {sorted(CONSTRAINT_REGISTRY)}")
    cls: Callable[..., MetricConstraint] = CONSTRAINT_REGISTRY[name]
    return cls(**kwargs)
