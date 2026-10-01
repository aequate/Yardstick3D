from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from yardstick3d.constraints.altitude import AltitudeConstraint
from yardstick3d.constraints.base import ConstraintSet
from yardstick3d.constraints.baseline import BaselineConstraint
from yardstick3d.constraints.gnss import GNSSDisplacementConstraint
from yardstick3d.constraints.range import SparseRangeConstraint
from yardstick3d.constraints.speed import SpeedConstraint
from yardstick3d.geometry.cameras import w2c_from_centers
from yardstick3d.geometry.rotations import random_so3, so3_exp
from yardstick3d.geometry.sim3 import Sim3, apply_sim3_points
from yardstick3d.types import FloatArray, PredictionBundle

MotionRegime = Literal[
    "line",
    "circle",
    "planar_drive",
    "vertical",
    "stationary",
    "pure_rotation",
    "dynamic",
    "intermittent",
]


@dataclass
class SyntheticScene:
    timestamps: FloatArray
    centers_true: FloatArray
    R_w2c_true: FloatArray
    T_w2c_true: FloatArray
    points_true: FloatArray
    sim_unknown: Sim3
    prediction: PredictionBundle
    true_scale: float
    speeds: FloatArray
    regime: str
    rng_seed: int
    metadata: dict = field(default_factory=dict)


def _trajectory(regime: MotionRegime, n: int, dt: float, rng: np.random.Generator) -> FloatArray:
    t = np.arange(n) * dt
    C = np.zeros((n, 3))
    if regime == "stationary":
        C[:] = rng.normal(scale=1e-6, size=(n, 3))
    elif regime == "pure_rotation":
        C[:] = np.array([0.0, 0.0, 0.0])
    elif regime == "line":
        v = 5.0
        C[:, 0] = v * t
        C[:, 2] = 1.2
    elif regime == "circle":
        r = 8.0
        w = 0.4
        C[:, 0] = r * np.cos(w * t)
        C[:, 1] = r * np.sin(w * t)
        C[:, 2] = 1.2
    elif regime == "planar_drive":
        C[:, 0] = 8.0 * t + 0.3 * np.sin(0.7 * t)
        C[:, 1] = 0.4 * np.sin(0.3 * t)
        C[:, 2] = 1.5
    elif regime == "vertical":
        C[:, 2] = 1.0 + 3.0 * t
        C[:, 0] = 0.2 * t
    elif regime == "dynamic":
        C[:, 0] = 4.0 * t + 0.5 * np.sin(2.0 * t)
        C[:, 1] = 1.5 * np.sin(1.3 * t)
        C[:, 2] = 1.2 + 0.8 * np.sin(0.9 * t)
    elif regime == "intermittent":
        speed = np.zeros(n)
        speed[(t % 4.0) < 2.0] = 6.0
        C[:, 0] = np.cumsum(speed) * dt
        C[:, 2] = 1.2
    else:
        raise ValueError(regime)
    return C


def generate_scene(
    n_frames: int = 30,
    dt: float = 0.1,
    regime: MotionRegime = "line",
    true_scale: float = 2.5,
    seed: int = 0,
    fm_noise_std: float = 0.0,
    n_points: int = 40,
    unknown_rotation: bool = True,
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    ts = np.arange(n_frames, dtype=np.float64) * dt
    C = _trajectory(regime, n_frames, dt, rng)
    # Camera looks forward along +X of motion when possible; OpenCV: +Z forward, +Y down.
    R = np.repeat(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])[None, ...], n_frames, axis=0)
    if regime == "pure_rotation":
        for i in range(n_frames):
            R[i] = so3_exp(np.array([0.0, 0.15 * i * dt, 0.0]))
    T = w2c_from_centers(R, C)
    pts = rng.normal(scale=3.0, size=(n_points, 3))
    pts[:, 2] += 8.0

    R_unk = (
        random_so3(rng)
        if unknown_rotation and regime not in ("stationary",)
        else np.eye(3)
    )
    # Unknown gauge applied to create the 3DFM output: X_fm = (1/s) R^T (X - t)
    # Equivalent: 3DFM is true geometry transformed by inverse of unknown Sim3.
    t_unk = rng.normal(scale=0.5, size=3)
    sim_true_from_fm = Sim3(scale=true_scale, R=R_unk, t=t_unk)
    sim_fm_from_true = sim_true_from_fm.inverse()
    C_fm = apply_sim3_points(C, sim_fm_from_true)
    if fm_noise_std > 0:
        C_fm = C_fm + rng.normal(scale=fm_noise_std, size=C_fm.shape)
    R_fm = np.einsum("ij,njk->nik", R_unk.T, R)
    T_fm = w2c_from_centers(R_fm, C_fm)
    pts_fm = apply_sim3_points(pts, sim_fm_from_true)

    speeds = np.linalg.norm(C[1:] - C[:-1], axis=1) / dt
    pred = PredictionBundle(
        timestamps=ts,
        T_w2c=T_fm,
        camera_centers=C_fm,
        points_world=pts_fm,
        depth_z=None,
        is_metric=False,
        model_id="synthetic_fm",
        aux={"regime": regime},
    )
    return SyntheticScene(
        timestamps=ts,
        centers_true=C,
        R_w2c_true=R,
        T_w2c_true=T,
        points_true=pts,
        sim_unknown=sim_true_from_fm,
        prediction=pred,
        true_scale=float(true_scale),
        speeds=speeds,
        regime=regime,
        rng_seed=seed,
        metadata={"dt": dt, "n_frames": n_frames, "unknown_rotation": unknown_rotation},
    )


def make_constraints(
    scene: SyntheticScene,
    cue: str,
    sparsity: float = 1.0,
    noise_std: float = 0.0,
    outlier_frac: float = 0.0,
    time_offset_s: float = 0.0,
    seed: int = 1,
    gravity_aligned: bool = True,
) -> ConstraintSet:
    """Build metric cues from GT, then hide GT. `sparsity` keeps that fraction of intervals."""
    rng = np.random.default_rng(seed)
    ts = scene.timestamps
    C = scene.centers_true
    dt = float(np.mean(np.diff(ts))) if len(ts) > 1 else 0.1
    n = len(ts)
    cs = ConstraintSet()
    n_int = max(n - 1, 1)
    keep = rng.random(n_int) < sparsity
    if sparsity >= 1.0:
        keep[:] = True

    def _outlier_idx(n_slots: int) -> set[int]:
        slots = [i for i in range(n_slots) if i < len(keep) and keep[i]]
        n_out = int(round(float(outlier_frac) * len(slots)))
        if n_out <= 0 or not slots:
            return set()
        chosen = rng.choice(np.array(slots), size=min(n_out, len(slots)), replace=False)
        return set(int(x) for x in np.atleast_1d(chosen))

    def maybe_outlier(val: float, scale: float) -> float:
        return val + scale * 10.0 * float(rng.choice([-1.0, 1.0]))

    speed_out = _outlier_idx(n - 1) if cue in ("speed", "all") else set()

    if cue in ("speed", "all"):
        for i in range(n - 1):
            if not keep[i]:
                continue
            v = float(np.linalg.norm(C[i + 1] - C[i]) / dt)
            if i in speed_out:
                v = maybe_outlier(v, max(abs(v), 1.0))
            elif noise_std > 0:
                v = v + rng.normal(scale=noise_std)
            cs.add(
                SpeedConstraint(
                    t_i=float(ts[i] + time_offset_s),
                    t_j=float(ts[i + 1] + time_offset_s),
                    speed_mps=v,
                    sigma=max(noise_std, 0.05),
                    max_dt_match=max(0.06, abs(time_offset_s) + 0.02),
                )
            )
    gnss_out = _outlier_idx(n - 1) if cue in ("gnss", "all") else set()
    if cue in ("gnss", "all"):
        for i in range(n - 1):
            if not keep[i]:
                continue
            d = C[i + 1] - C[i]
            if i in gnss_out:
                d = d + rng.normal(scale=5.0, size=3)
            elif noise_std > 0:
                d = d + rng.normal(scale=noise_std, size=3)
            cs.add(
                GNSSDisplacementConstraint(
                    t_i=float(ts[i] + time_offset_s),
                    t_j=float(ts[i + 1] + time_offset_s),
                    delta_xyz_m=d,
                    sigma=max(noise_std, 0.2),
                    max_dt_match=max(0.06, abs(time_offset_s) + 0.02),
                    use_vector=False,
                )
            )
    alt_out = _outlier_idx(n - 1) if (
        cue == "altitude" or (cue == "all" and gravity_aligned and not scene.metadata.get("unknown_rotation", True))
    ) else set()
    if cue == "altitude" or (cue == "all" and gravity_aligned and not scene.metadata.get("unknown_rotation", True)):
        for i in range(n - 1):
            if not keep[i]:
                continue
            dh = float(C[i + 1, 2] - C[i, 2])
            if i in alt_out:
                dh = maybe_outlier(dh, 2.0)
            elif noise_std > 0:
                dh = dh + rng.normal(scale=noise_std)
            cs.add(
                AltitudeConstraint(
                    t_i=float(ts[i] + time_offset_s),
                    t_j=float(ts[i + 1] + time_offset_s),
                    delta_h_m=dh,
                    sigma=max(noise_std, 0.3),
                    max_dt_match=max(0.06, abs(time_offset_s) + 0.02),
                    gravity_aligned=gravity_aligned,
                )
            )
    if cue in ("baseline", "all") and n >= 2:
        b = float(np.linalg.norm(C[1] - C[0]))
        if b > 1e-8:
            if noise_std > 0:
                b = b + rng.normal(scale=noise_std)
            cs.add(BaselineConstraint(i=0, j=1, baseline_m=b, sigma=max(noise_std, 0.02)))
    if cue in ("range", "all"):
        n_keep = max(1, int(round(sparsity * min(8, len(scene.points_true)))))
        idx = rng.choice(len(scene.points_true), size=min(n_keep, len(scene.points_true)), replace=False)
        for k, pi in enumerate(idx):
            fi = int(k % n)
            d = float(np.linalg.norm(scene.points_true[pi] - C[fi]))
            if rng.random() < outlier_frac:
                d = maybe_outlier(d, 3.0)
            elif noise_std > 0:
                d = d + rng.normal(scale=noise_std)
            # Range constraint uses 3DFM-frame points so scale maps correctly.
            cs.add(
                SparseRangeConstraint(
                    frame_index=fi,
                    range_m=d,
                    point_world=scene.prediction.points_world[pi],
                    sigma=max(noise_std, 0.05),
                )
            )
    return cs
