from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from yardstick3d.types import PredictionBundle

PoseKind = Literal["w2c", "c2w"]


@dataclass
class AdapterCapabilities:
    name: str
    min_vram_gb: float
    max_views_on_6gb: int | None
    poses: bool
    depth: bool
    metric: bool
    pose_kind: PoseKind
    license: str
    local: bool
    notes: str = ""


@dataclass
class PredictRequest:
    images: Sequence[np.ndarray]
    timestamps: np.ndarray | None = None
    K: np.ndarray | None = None
    aux: dict[str, Any] = field(default_factory=dict)


class FoundationModelAdapter(ABC):
    caps: AdapterCapabilities

    @abstractmethod
    def load(self, device: str = "cpu") -> None: ...

    @abstractmethod
    def predict(self, req: PredictRequest) -> PredictionBundle: ...

    def unload(self) -> None:
        return None
