from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from yardstick3d.baselines.chained_vo import preprocess_like_vggt_crop, run_vo


def _synthetic_scene(n_pts=60, seed=0):
    rng = np.random.default_rng(seed)
    K = np.array([[400.0, 0, 160.0], [0, 400.0, 120.0], [0, 0, 1]])
    pts = rng.uniform([-3, -2, 4], [3, 2, 10], size=(n_pts, 3))
    # unique random patch per point: blocky binary texture, stable across views
    patches = (rng.random((n_pts, 4, 4)) < 0.5).astype(np.uint8) * 255
    patches = np.kron(patches, np.ones((3, 3), np.uint8))  # 12x12
    centers = np.array([[i * 0.6, 0.1 * i, 0.05 * i] for i in range(8)])
    angles = [0.02 * i for i in range(8)]
    Rs = [np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) for a in angles]
    imgs = []
    for R, C in zip(Rs, centers):
        t = -R @ C
        uv = (K @ (pts @ R.T + t).T).T
        uv = uv[:, :2] / uv[:, 2:3]
        img = np.zeros((240, 320), np.uint8)
        for (u, v), pat in zip(uv, patches):
            u0, v0 = int(round(u)) - 6, int(round(v)) - 6
            if u0 >= 0 and v0 >= 0 and u0 + 12 < 320 and v0 + 12 < 240:
                img[v0:v0 + 12, u0:u0 + 12] = np.maximum(img[v0:v0 + 12, u0:u0 + 12], pat)
        imgs.append(img)
    return imgs, K, np.asarray(centers)


def test_vo_recovers_synthetic_trajectory_up_to_sim3():
    from yardstick3d.evaluation.alignment import apply_sim3, ate_rmse, umeyama

    imgs, K, centers_gt = _synthetic_scene()
    res = run_vo(imgs, K)
    assert res.ok, res.error
    assert res.T_w2c.shape == (8, 3, 4)
    s, R, t = umeyama(res.centers, centers_gt, with_scale=True)
    assert ate_rmse(apply_sim3(res.centers, s, R, t), centers_gt) < 0.15
    assert s > 0


def test_vo_reports_failure_openly_not_nan():
    K = np.eye(3) * 400
    K[0, 2] = K[1, 2] = 100
    blank = [np.zeros((240, 320), np.uint8) for _ in range(3)]
    res = run_vo(blank, K)
    assert res.ok is False and res.error and res.centers is None


def test_preprocess_shape_matches_vggt_grid(tmp_path):
    import cv2 as _cv2

    p = tmp_path / "f.png"
    _cv2.imwrite(str(p), np.zeros((720, 1280, 3), np.uint8))
    g = preprocess_like_vggt_crop(p)
    assert g.shape == (294, 518), g.shape  # pinned mode=crop recipe for 1280x720
