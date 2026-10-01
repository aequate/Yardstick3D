from __future__ import annotations

import numpy as np

from yardstick3d.geometry.cameras import (
    camera_centers_from_w2c,
    invert_w2c,
)


def test_invert_w2c_known_answer():
    # Rot_z(90): X_world=(1,0,0) maps to camera (0,1,0); center C=(0,1,0).
    R = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    T = np.zeros((1, 3, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        camera_centers_from_w2c(T), [[0.0, 1.0, 0.0]], rtol=0, atol=1e-12
    )
    c2w = invert_w2c(T)
    np.testing.assert_allclose(c2w[0, :3, :3], R[0].T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(c2w[0, :3, 3], [0.0, 1.0, 0.0], rtol=0, atol=1e-12)
