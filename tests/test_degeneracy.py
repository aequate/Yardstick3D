from __future__ import annotations

import numpy as np

from yardstick3d.datasets.synthetic import generate_scene, make_constraints
from yardstick3d.observability.metrics import (
    interval_scales,
    metric_observability_score,
    pairwise_scale_cv,
    robust_cv,
)
from yardstick3d.optimization.grounder import MetricGrounder


def test_stationary_not_observable():
    scene = generate_scene(n_frames=10, regime="stationary", true_scale=2.0, seed=0)
    cs = make_constraints(scene, "speed", sparsity=1.0, seed=0)
    rep = metric_observability_score(scene.prediction, cs)
    assert rep.score < 0.45
    r = MetricGrounder().solve(scene.prediction, cs, solver="robust")
    assert (not r.observable) or (not np.isfinite(r.scale))


def test_pure_rotation_not_observable():
    scene = generate_scene(n_frames=10, regime="pure_rotation", true_scale=2.0, seed=1)
    cs = make_constraints(scene, "speed", sparsity=1.0, seed=1)
    r = MetricGrounder().solve(scene.prediction, cs, solver="robust")
    assert not r.observable or not np.isfinite(r.scale)


def test_translating_is_observable():
    scene = generate_scene(n_frames=16, regime="line", true_scale=2.0, seed=2)
    cs = make_constraints(scene, "speed", sparsity=1.0, seed=2)
    r = MetricGrounder().solve(scene.prediction, cs, solver="robust")
    assert r.observable
    assert r.observability_score > 0.45


def test_altitude_degenerate_without_vertical_motion():
    scene = generate_scene(n_frames=12, regime="line", true_scale=2.0, seed=3)
    from yardstick3d.geometry.cameras import w2c_from_centers
    from yardstick3d.types import PredictionBundle
    from yardstick3d.constraints.altitude import AltitudeConstraint
    from yardstick3d.constraints.base import ConstraintSet

    C_fm = scene.centers_true / 2.0
    pred = PredictionBundle(
        timestamps=scene.timestamps,
        T_w2c=w2c_from_centers(scene.R_w2c_true, C_fm),
        camera_centers=C_fm,
        model_id="x",
    )
    cs = ConstraintSet()
    for i in range(len(scene.timestamps) - 1):
        dh = float(scene.centers_true[i + 1, 2] - scene.centers_true[i, 2])
        cs.add(
            AltitudeConstraint(
                scene.timestamps[i], scene.timestamps[i + 1], dh, sigma=0.1, gravity_aligned=True
            )
        )
    so = [c.scale_observation(pred) for c in cs.constraints]
    # Planar drive with constant z: a ≈ 0
    assert all(x is None or abs(x[0]) < 1e-8 for x in so)


def test_pairwise_scale_cv_low_when_consistent():
    scene = generate_scene(n_frames=12, regime="line", true_scale=2.7, seed=4)
    cs = make_constraints(scene, "speed", sparsity=1.0, seed=4)
    cv = pairwise_scale_cv(scene.prediction, cs)
    assert cv < 0.05


def test_pairwise_scale_cv_high_with_one_bad_interval():
    scene = generate_scene(n_frames=12, regime="line", true_scale=2.7, seed=5)
    cs = make_constraints(scene, "speed", sparsity=1.0, seed=5)
    cs.constraints[0].speed_mps *= 8.0
    cv = pairwise_scale_cv(scene.prediction, cs)
    assert cv > 0.2


def test_robust_cv_low_when_consistent():
    scene = generate_scene(n_frames=12, regime="line", true_scale=2.7, seed=4)
    cs = make_constraints(scene, "speed", sparsity=1.0, seed=4)
    s, _sig = interval_scales(scene.prediction, cs)
    assert robust_cv(s) < 0.05


def test_robust_cv_resists_one_outlier():
    s = np.ones(11, dtype=np.float64)
    s[0] = 8.0
    assert robust_cv(s) < 0.05
    assert float(np.std(s, ddof=0) / np.median(s)) > 0.2


def test_robust_cv_high_when_dispersed():
    s = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    assert robust_cv(s) > 0.5
