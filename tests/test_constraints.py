from __future__ import annotations

import numpy as np
import pytest

from yardstick3d.constraints.altitude import AltitudeConstraint
from yardstick3d.constraints.baseline import BaselineConstraint
from yardstick3d.constraints.gnss import GNSSDisplacementConstraint
from yardstick3d.constraints.range import SparseRangeConstraint
from yardstick3d.constraints.speed import SpeedConstraint
from yardstick3d.datasets.synthetic import generate_scene
from yardstick3d.types import MetricState


def test_speed_zero_residual_at_truth():
    scene = generate_scene(n_frames=10, regime="line", true_scale=2.0, seed=0)
    dt = 0.1
    v = float(np.linalg.norm(scene.centers_true[1] - scene.centers_true[0]) / dt)
    c = SpeedConstraint(0.0, 0.1, v, sigma=0.05)
    state = MetricState(scale=scene.true_scale, rotation=scene.sim_unknown.R, translation=scene.sim_unknown.t)
    r = c.residual(scene.prediction, state)
    np.testing.assert_allclose(r, 0.0, atol=1e-8)
    r_bad = c.residual(scene.prediction, MetricState(scale=1.0))
    assert abs(r_bad[0]) > 0.1


def test_gnss_magnitude_zero_at_truth():
    scene = generate_scene(n_frames=8, regime="line", true_scale=3.0, seed=1)
    d = scene.centers_true[2] - scene.centers_true[0]
    c = GNSSDisplacementConstraint(0.0, 0.2, d, sigma=0.1, use_vector=False)
    state = MetricState(scale=scene.true_scale, rotation=scene.sim_unknown.R, translation=scene.sim_unknown.t)
    r = c.residual(scene.prediction, state)
    np.testing.assert_allclose(r, 0.0, atol=1e-8)


def test_altitude_zero_when_aligned():
    scene = generate_scene(n_frames=12, regime="vertical", true_scale=1.8, seed=2)
    # Reconstruct with identity rotation by using scale-only unknown
    scene = generate_scene(n_frames=12, regime="vertical", true_scale=1.8, seed=2)
    # Force identity gauge except scale: rebuild prediction
    from yardstick3d.geometry.cameras import w2c_from_centers
    from yardstick3d.types import PredictionBundle

    s = 1.8
    C_fm = scene.centers_true / s
    T = w2c_from_centers(scene.R_w2c_true, C_fm)
    pred = PredictionBundle(timestamps=scene.timestamps, T_w2c=T, camera_centers=C_fm, model_id="x")
    dh = float(scene.centers_true[5, 2] - scene.centers_true[0, 2])
    c = AltitudeConstraint(scene.timestamps[0], scene.timestamps[5], dh, sigma=0.1)
    r = c.residual(pred, MetricState(scale=s))
    np.testing.assert_allclose(r, 0.0, atol=1e-8)


def test_baseline_and_range():
    scene = generate_scene(n_frames=6, regime="line", true_scale=2.2, seed=4)
    from yardstick3d.geometry.cameras import w2c_from_centers
    from yardstick3d.types import PredictionBundle

    s = 2.2
    C_fm = scene.centers_true / s
    pts_fm = scene.points_true / s
    T = w2c_from_centers(scene.R_w2c_true, C_fm)
    pred = PredictionBundle(
        timestamps=scene.timestamps, T_w2c=T, camera_centers=C_fm, points_world=pts_fm, model_id="x"
    )
    b = float(np.linalg.norm(scene.centers_true[1] - scene.centers_true[0]))
    c = BaselineConstraint(0, 1, b, sigma=0.01)
    np.testing.assert_allclose(c.residual(pred, MetricState(scale=s)), 0.0, atol=1e-8)
    d = float(np.linalg.norm(scene.points_true[0] - scene.centers_true[0]))
    r = SparseRangeConstraint(0, d, point_world=pts_fm[0], sigma=0.01)
    np.testing.assert_allclose(r.residual(pred, MetricState(scale=s)), 0.0, atol=1e-8)


def test_covariance_positive():
    c = SpeedConstraint(0.0, 0.1, 1.0, sigma=0.2)
    cov = c.covariance()
    assert cov[0, 0] == pytest.approx(0.04)
