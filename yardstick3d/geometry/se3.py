from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yardstick3d.types import FloatArray


@dataclass
class SE3:
    R: FloatArray
    t: FloatArray

    @staticmethod
    def identity() -> SE3:
        return SE3(R=np.eye(3), t=np.zeros(3))

    @staticmethod
    def from_matrix(T: FloatArray) -> SE3:
        T = np.asarray(T, dtype=np.float64)
        return SE3(R=T[:3, :3].copy(), t=T[:3, 3].copy())

    def matrix(self) -> FloatArray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = np.asarray(self.t, dtype=np.float64).reshape(3)
        return T

    def inverse(self) -> SE3:
        Rt = np.asarray(self.R, dtype=np.float64).T
        t = np.asarray(self.t, dtype=np.float64).reshape(3)
        return SE3(R=Rt, t=-Rt @ t)

    def compose(self, other: SE3) -> SE3:
        R = np.asarray(self.R) @ np.asarray(other.R)
        t = np.asarray(self.R) @ np.asarray(other.t).reshape(3) + np.asarray(self.t).reshape(3)
        return SE3(R=R, t=t)

    def transform_points(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=np.float64)
        return (np.asarray(self.R) @ X.reshape(-1, 3).T).T + np.asarray(self.t).reshape(3)


def se3_inverse(T: FloatArray) -> FloatArray:
    return SE3.from_matrix(T).inverse().matrix()


def se3_compose(A: FloatArray, B: FloatArray) -> FloatArray:
    return SE3.from_matrix(A).compose(SE3.from_matrix(B)).matrix()
