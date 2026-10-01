from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yardstick3d.types import FloatArray


@dataclass
class Sim3:
    """X' = s R X + t."""

    scale: float
    R: FloatArray
    t: FloatArray

    @staticmethod
    def identity() -> Sim3:
        return Sim3(scale=1.0, R=np.eye(3), t=np.zeros(3))

    @staticmethod
    def scale_only(s: float) -> Sim3:
        return Sim3(scale=float(s), R=np.eye(3), t=np.zeros(3))

    def inverse(self) -> Sim3:
        s = float(self.scale)
        Rt = np.asarray(self.R, dtype=np.float64).T
        t = np.asarray(self.t, dtype=np.float64).reshape(3)
        return Sim3(scale=1.0 / s, R=Rt, t=-(1.0 / s) * Rt @ t)

    def compose(self, other: Sim3) -> Sim3:
        s = float(self.scale) * float(other.scale)
        R = np.asarray(self.R) @ np.asarray(other.R)
        t = float(self.scale) * (np.asarray(self.R) @ np.asarray(other.t).reshape(3)) + np.asarray(
            self.t
        ).reshape(3)
        return Sim3(scale=s, R=R, t=t)

    def matrix(self) -> FloatArray:
        T = np.eye(4)
        T[:3, :3] = float(self.scale) * np.asarray(self.R)
        T[:3, 3] = np.asarray(self.t, dtype=np.float64).reshape(3)
        return T


def apply_sim3_points(X: FloatArray, sim: Sim3) -> FloatArray:
    X = np.asarray(X, dtype=np.float64)
    shape = X.shape
    flat = X.reshape(-1, 3)
    out = float(sim.scale) * (np.asarray(sim.R) @ flat.T).T + np.asarray(sim.t).reshape(3)
    return out.reshape(shape)


def apply_sim3_centers(C: FloatArray, sim: Sim3) -> FloatArray:
    return apply_sim3_points(C, sim)
