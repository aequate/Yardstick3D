from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from yardstick3d.types import FloatArray


def so3_exp(omega: FloatArray) -> FloatArray:
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    return Rotation.from_rotvec(omega).as_matrix()


def so3_log(R: FloatArray) -> FloatArray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return Rotation.from_matrix(R).as_rotvec()


def so3_from_quat(q_xyzw: FloatArray) -> FloatArray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    return Rotation.from_quat(q).as_matrix()


def so3_to_quat(R: FloatArray) -> FloatArray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return Rotation.from_matrix(R).as_quat()


def random_so3(rng: np.random.Generator) -> FloatArray:
    return Rotation.random(random_state=int(rng.integers(0, 2**31 - 1))).as_matrix()
