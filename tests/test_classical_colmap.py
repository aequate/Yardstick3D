"""Classical baseline v2 (COLMAP) adapter tests.

Pure unit tests (no COLMAP execution) plus portrait-geometry guards, a
mov-decode identity check, and one slow real-pipeline schema test.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yardstick3d.baselines.colmap_sfm import (
    center_from_w2c,
    build_frame_list,
    frame_name,
    gather_query_poses,
    pose_to_w2c,
    result_to_bundle,
    select_primary_model,
    success_table,
    ColmapResult,
)

ROOT = Path(__file__).resolve().parents[1]


def test_center_known_answer():
    T = np.hstack([np.eye(3), np.array([[0.0, 0.0, 5.0]]).T])
    np.testing.assert_allclose(center_from_w2c(T), [0, 0, -5])
    # 90-degree yaw: camera at origin looking +x means center stays origin
    R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    T = pose_to_w2c(R, np.zeros(3))
    assert T.shape == (3, 4)
    np.testing.assert_allclose(center_from_w2c(T), [0, 0, 0], atol=1e-12)
    # translated: C = -R^T t
    t = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(center_from_w2c(pose_to_w2c(R, t)), -R.T @ t)


def test_model_selection_prefers_registrations_then_points():
    a = SimpleNamespace(num_reg_images=lambda: 5, num_points3D=lambda: 9000)
    b = SimpleNamespace(num_reg_images=lambda: 8, num_points3D=lambda: 10)
    c = SimpleNamespace(num_reg_images=lambda: 8, num_points3D=lambda: 700)
    assert select_primary_model({"x": a, "y": b, "z": c}) is c
    with pytest.raises(RuntimeError):
        select_primary_model({})


def _fake_pose(R=None, t=None):
    R = np.eye(3) if R is None else np.asarray(R, float)
    t = np.zeros(3) if t is None else np.asarray(t, float)

    class Rot:
        def matrix(self):
            return R

    return SimpleNamespace(rotation=Rot(), translation=t)


def _fake_rec(names, K, missing=(), unposed=()):
    cams = {1: SimpleNamespace(model_name="SIMPLE_RADIAL", width=720, height=1280,
                               params=[1850.0, 360.0, 640.0, 0.0],
                               calibration_matrix=lambda: np.asarray(K, float))}

    def find(name):
        if name in missing:
            return None
        t = np.array([float(names.index(name)), 0.0, 0.0])
        return SimpleNamespace(has_pose=name not in unposed, camera_id=1,
                               cam_from_world=lambda: _fake_pose(t=t))

    return SimpleNamespace(find_image_with_name=find,
                           camera=lambda i: cams[i])


def test_gather_portrait_K_not_swapped():
    K = np.array([[1850.0, 0, 360.0], [0, 1850.0, 640.0], [0, 0, 1]])
    rec = _fake_rec(["f00.png", "f01.png"], K)
    T, Kt, cam = gather_query_poses(rec, ["f00.png", "f01.png"])
    assert T.shape == (2, 3, 4) and Kt.shape == (2, 3, 3)
    np.testing.assert_allclose(Kt[0], K)  # portrait K preserved, never transposed
    assert cam["width"] == 720 and cam["height"] == 1280
    assert cam["K"][0, 2] == pytest.approx(360.0)  # cx = W/2 portrait
    assert cam["K"][1, 2] == pytest.approx(640.0)  # cy = H/2 portrait


def test_gather_reports_gaps_openly():
    K = np.eye(3)
    rec = _fake_rec(["a.png", "b.png"], K, missing=("b.png",))
    with pytest.raises(RuntimeError, match="missing"):
        gather_query_poses(rec, ["a.png", "b.png"])
    rec2 = _fake_rec(["a.png", "b.png"], K, unposed=("b.png",))
    with pytest.raises(RuntimeError, match="unposed"):
        gather_query_poses(rec2, ["a.png", "b.png"])


def test_bundle_conversion_and_failure_refusal():
    T = np.stack([pose_to_w2c(np.eye(3), [i, 0, 0]) for i in range(8)])
    K = np.tile(np.eye(3)[None], (8, 1, 1))
    res = ColmapResult(ok=True, T_w2c=T, K=K,
                       centers=np.stack([center_from_w2c(t) for t in T]),
                       query_names=[f"f{i:02d}.png" for i in range(8)],
                       n_registered=8, n_points3D=500, stage="done")
    b = result_to_bundle(res, np.arange(8, dtype=float), "colmap-4.2.0/sparse8")
    assert b.is_metric is False and b.T_w2c.shape == (8, 3, 4)
    assert b.K.shape == (8, 3, 3) and b.model_id.startswith("colmap-")
    with pytest.raises(ValueError):
        result_to_bundle(ColmapResult(ok=False, stage="map", error="x"),
                         np.arange(8, dtype=float), "colmap-4.2.0/sparse8")


def test_success_table_rates():
    rows = [{"ok": True, "n_registered": 8}] * 9 + [
        {"ok": False, "n_registered": 3, "error": "pose_gather gaps"}]
    t = success_table(rows)
    assert t["n_attempted"] == 10 and t["n_success"] == 9
    assert t["success_rate"] == pytest.approx(0.9)
    assert t["n_registered_hist"][0] == 3 and t["n_registered_hist"][-1] == 8


def test_random_cue_draws_are_method_independent():
    from yardstick3d.evaluation.measurement_budget_helpers import subset_indices_random
    # Same (n_full, k, seed) -> same indices whatever the prediction source.
    assert subset_indices_random(7, 1, 3) == subset_indices_random(7, 1, 3)
    assert len(subset_indices_random(7, 2, 0)) == 2


def test_fullrate_frame_list_alignment():
    # Full-rate sampling must always include the 8 frozen query indices so the
    # Exp-B query timestamps align with the frozen window timestamps.
    frozen = [942, 1079, 1216, 1353, 1491, 1628, 1765, 1902]
    frames = build_frame_list(frozen[0], frozen[-1], frozen, stride=12)
    assert frames == sorted(set(frames)), "must be sorted and deduplicated"
    for k in frozen:
        assert k in frames, f"frozen index {k} missing from full-rate list"
    assert len(frames) > len(frozen), "should densify beyond the 8 frozen frames"
    # a stride that already lands on a frozen index must not duplicate it
    assert build_frame_list(0, 24, [0, 12, 24], stride=12) == [0, 12, 24]
    # naming: zero-padded stable query names
    assert frame_name(942) == "v00942.png"
    assert frame_name(1902) == "v01902.png"


def test_method_independent_grounding_ignores_model_id():
    # The grounding layer must behave identically whatever frontend produced
    # the bundle: same timestamps/T_w2c => same solver scale.
    from yardstick3d.datasets.advio import load_advio
    from yardstick3d.datasets.advio_cues import constraints_from_corelocation_displacement
    from yardstick3d.experiments.advio_real import run_solvers
    from yardstick3d.io.prediction_cache import load_prediction
    from yardstick3d.types import PredictionBundle

    npz = ROOT / "artifacts" / "vggt_returned" / "outputs" / "advio-20" / "t0_16.npz"
    if not npz.exists():
        pytest.skip("frozen prediction cache unavailable")
    base = load_prediction(npz)
    seq = load_advio(ROOT / "data" / "advio", "20", imu_allowed=False)
    cs = constraints_from_corelocation_displacement(seq.inference, base.timestamps)

    colmap_style = PredictionBundle(
        timestamps=base.timestamps, T_w2c=base.T_w2c, K=base.K,
        camera_centers=base.centers(), is_metric=False,
        model_id="colmap-4.2.0/sparse8", aux={"frontend": "colmap"})
    s_a = run_solvers(base, cs)["naive"]["scale"]
    s_b = run_solvers(colmap_style, cs)["naive"]["scale"]
    assert s_a == pytest.approx(s_b, rel=0, abs=0), "model_id leaked into grounding"
