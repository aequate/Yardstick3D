from __future__ import annotations

import numpy as np

from yardstick3d.geometry.cameras import apply_sim3_w2c, camera_centers_from_w2c, w2c_from_centers
from yardstick3d.geometry.rotations import so3_exp, so3_log, so3_from_quat, so3_to_quat
from yardstick3d.geometry.se3 import SE3, se3_compose, se3_inverse
from yardstick3d.geometry.sim3 import Sim3, apply_sim3_points


def test_so3_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    w = rng.normal(size=3) * 0.4
    R = so3_exp(w)
    w2 = so3_log(R)
    np.testing.assert_allclose(so3_exp(w2), R, atol=1e-10)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert abs(np.linalg.det(R) - 1) < 1e-10


def test_quat_roundtrip():
    R = so3_exp(np.array([0.2, -0.1, 0.4]))
    q = so3_to_quat(R)
    R2 = so3_from_quat(q)
    np.testing.assert_allclose(R2, R, atol=1e-10)


def test_se3_inverse_compose():
    a = SE3(R=so3_exp(np.array([0.1, 0.2, 0.0])), t=np.array([1.0, 2.0, 3.0]))
    b = SE3(R=so3_exp(np.array([0.0, 0.0, 0.3])), t=np.array([-1.0, 0.5, 0.0]))
    I = a.compose(a.inverse()).matrix()
    np.testing.assert_allclose(I, np.eye(4), atol=1e-10)
    c = se3_compose(a.matrix(), b.matrix())
    np.testing.assert_allclose(se3_inverse(se3_inverse(c)), c, atol=1e-10)
    X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
    np.testing.assert_allclose(a.inverse().transform_points(a.transform_points(X)), X, atol=1e-10)


def test_sim3_inverse_compose():
    s = Sim3(scale=2.5, R=so3_exp(np.array([0.2, 0.0, 0.1])), t=np.array([1.0, -2.0, 0.5]))
    X = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    Y = apply_sim3_points(X, s)
    X2 = apply_sim3_points(Y, s.inverse())
    np.testing.assert_allclose(X2, X, atol=1e-10)
    ident = s.compose(s.inverse())
    np.testing.assert_allclose(ident.scale, 1.0, atol=1e-10)
    np.testing.assert_allclose(ident.R, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(ident.t, 0.0, atol=1e-10)


def test_camera_centers_w2c_convention():
    R = so3_exp(np.array([0.1, 0.2, -0.3]))
    C = np.array([4.0, -1.0, 2.0])
    T = w2c_from_centers(R, C)
    C2 = camera_centers_from_w2c(T)
    np.testing.assert_allclose(C2[0], C, atol=1e-10)
    t = T[0, :3, 3]
    np.testing.assert_allclose(t, -R @ C, atol=1e-10)
    np.testing.assert_allclose(C2[0], -R.T @ t, atol=1e-10)


def test_sim3_on_cameras_preserves_relative_geometry():
    rng = np.random.default_rng(3)
    n = 5
    C = rng.normal(size=(n, 3))
    R = np.stack([so3_exp(rng.normal(size=3) * 0.2) for _ in range(n)])
    T = w2c_from_centers(R, C)
    sim = Sim3(scale=3.0, R=so3_exp(np.array([0.0, 0.4, 0.0])), t=np.array([10.0, 0.0, -2.0]))
    T2 = apply_sim3_w2c(T, sim)
    C2 = camera_centers_from_w2c(T2)
    # Distances scale by s
    d0 = np.linalg.norm(C[1] - C[0])
    d1 = np.linalg.norm(C2[1] - C2[0])
    np.testing.assert_allclose(d1, 3.0 * d0, atol=1e-10)
