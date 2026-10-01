from __future__ import annotations

import numpy as np

from yardstick3d.evaluation.alignment import apply_se3, apply_sim3, ate_rmse, umeyama
from yardstick3d.io.prediction_cache import load_prediction, prediction_hash, save_prediction
from yardstick3d.types import PredictionBundle


def test_se3_alignment_preserves_scale():
    rng = np.random.default_rng(0)
    gt = np.cumsum(rng.normal(size=(12, 3)) * 0.4, axis=0)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([5.0, -2.0, 1.0])
    pred = (R @ gt.T).T + t
    s, R_hat, t_hat = umeyama(pred, gt, with_scale=False)
    assert abs(s - 1.0) < 1e-12
    aligned = apply_se3(pred, R_hat, t_hat)
    assert ate_rmse(aligned, gt) < 1e-8
    # path length unchanged by SE3
    from yardstick3d.geometry.trajectory import path_length

    assert abs(path_length(pred) - path_length(gt)) < 1e-8


def test_sim3_alignment_removes_scale():
    rng = np.random.default_rng(1)
    gt = np.cumsum(rng.normal(size=(12, 3)) * 0.5, axis=0)
    pred = 2.7 * gt + np.array([3.0, 0.0, -1.0])
    s, R, t = umeyama(pred, gt, with_scale=True)
    assert abs(s - 1 / 2.7) < 1e-6
    assert ate_rmse(apply_sim3(pred, s, R, t), gt) < 1e-6


def test_oracle_not_in_solver_namespace():
    import yardstick3d.optimization.grounder as g
    import inspect

    src = inspect.getsource(g)
    assert "oracle_scale" not in src
    assert "interpolate_gt" not in src


def test_prediction_cache_roundtrip(tmp_path):
    T = np.zeros((4, 3, 4))
    T[:, :3, :3] = np.eye(3)
    T[:, 0, 3] = np.arange(4)
    b = PredictionBundle(
        timestamps=np.arange(4, dtype=np.float64),
        T_w2c=T,
        is_metric=False,
        model_id="da3-small/test",
    )
    p = tmp_path / "p.npz"
    meta = save_prediction(b, p, meta={"contains_ground_truth": False})
    assert meta["contains_ground_truth"] is False
    b2 = load_prediction(p)
    np.testing.assert_allclose(b2.T_w2c, T)
    assert b2.is_metric is False
    assert prediction_hash(b) == prediction_hash(b2)


def test_config_hash_stable():
    from yardstick3d.experiments.advio_real import config_hash

    a = config_hash({"x": 1, "y": [1, 2]})
    b = config_hash({"y": [1, 2], "x": 1})
    assert a == b
