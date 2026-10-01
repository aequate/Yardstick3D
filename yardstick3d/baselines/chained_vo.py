"""Minimal classical monocular VO baseline (evaluation comparison only).

Chained two-view geometry over 8 frames: Essential-matrix pose for the first
pair, triangulation, then PnP against the triangulated map for subsequent
frames (the textbook fix for per-pair scale ambiguity). Arbitrary monocular
scale, OpenCV w2c output, no metric input, no learned weights.

Deterministic: callers must set cv2.setRNGSeed; all RANSAC thresholds are
fixed parameters of run_vo, never tuned per window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VOResult:
    ok: bool
    T_w2c: np.ndarray | None = None
    centers: np.ndarray | None = None
    n_inliers_pair01: int = 0
    n_map_points: int = 0
    per_frame_inliers: list = field(default_factory=list)
    error: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


def preprocess_like_vggt_crop(png_path: str | Path, width: int = 1280, height: int = 720) -> np.ndarray:
    """Reproduce pinned VGGT mode=crop pixels for the frozen 1280x720 PNGs.

    Recipe at source revision a288dd0f: new_width=518,
    new_height=round(h*518/w/14)*14 = 294 for 1280x720 (no center crop since
    294 < 518), PIL BICUBIC. Returns HxWx uint8 grayscale for feature work.
    """
    import cv2

    img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(png_path)
    new_width, new_height = 518, 294
    small = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _detect(g: np.ndarray):
    import cv2

    sift = cv2.SIFT_create()
    k, d = sift.detectAndCompute(g, None)
    if d is None:
        return [], np.zeros((0, 128), np.float32)
    return k, d


def _match(d0: np.ndarray, d1: np.ndarray, ratio: float = 0.75):
    import cv2

    if len(d0) < 2 or len(d1) < 2:
        raise RuntimeError("too few descriptors")
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
    good = [a for a, b in m if a.distance < ratio * b.distance]
    if len(good) < 8:
        raise RuntimeError(f"too few matches after ratio test: {len(good)}")
    return good


def _cheiral(pts3: np.ndarray, Rs: list, ts: list, idx: tuple[int, ...]) -> np.ndarray:
    """Boolean mask: triangulated world points in front of all listed cameras."""
    ok = np.ones(len(pts3), bool)
    for i in idx:
        z = (pts3 @ np.asarray(Rs[i]).T + np.asarray(ts[i]))[:, 2]
        ok &= np.isfinite(z) & (z > 0)
    return ok


def _euclid(pts4: np.ndarray) -> np.ndarray:
    """Homogeneous -> Euclidean, preserving sign (never clamp negative w).

    Points with |w| ~ 0 become non-finite and are rejected by _cheiral.
    """
    w = pts4[3:]
    return (pts4[:3] / np.where(np.abs(w) > 1e-12, w, np.nan)).T


def run_vo(
    grays: list[np.ndarray],
    K: np.ndarray,
    *,
    e_thresh: float = 1.0,
    pnp_reproj: float = 4.0,
    min_inliers: int = 15,
) -> VOResult:
    """Chain poses over pre-loaded grayscale frames with fixed intrinsics K."""
    import cv2

    cv2.setRNGSeed(0)
    K = np.asarray(K, dtype=np.float64)
    n = len(grays)
    if n < 2:
        return VOResult(ok=False, error="need >=2 frames")
    try:
        k0, d0 = _detect(grays[0])
        k1, d1 = _detect(grays[1])
        good = _match(d0, d1)
        corr = np.stack(
            [
                np.float64([k0[g.queryIdx].pt for g in good]),
                np.float64([k1[g.trainIdx].pt for g in good]),
            ],
            axis=0,
        )
        E, mask_e = cv2.findEssentialMat(corr[0], corr[1], K, method=cv2.RANSAC, threshold=e_thresh)
        if E is None or mask_e is None or int(mask_e.sum()) < min_inliers:
            return VOResult(ok=False, error="essential-matrix RANSAC failed")
        _, R01, t01, mask_p = cv2.recoverPose(E, corr[0], corr[1], K, mask=mask_e)
        inl = (mask_p.ravel() > 0) & (mask_e.ravel() > 0)
        if int(inl.sum()) < min_inliers:
            return VOResult(ok=False, error="recoverPose cheirality failed")
        p0, p1 = corr[0][inl], corr[1][inl]
        good_inl = [g for g, f in zip(good, inl) if f]
        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K @ np.hstack([R01, t01])
        pts4 = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
        pts3 = _euclid(pts4)
        # First camera at origin: C0=(0,0,0); second center C1 = -R01^T t01.
        Rs = [np.eye(3), R01]
        ts = [np.zeros(3), t01.ravel()]
        keep = _cheiral(pts3, Rs, ts, (0, 1))
        # map: track_id -> {"xyz": 3d point, "desc": last descriptor}
        tracks: dict[int, dict] = {}
        for j in np.flatnonzero(keep):
            tracks[j] = {"xyz": pts3[j], "desc": d1[good_inl[j].trainIdx].copy(),
                         "uv": {0: p0[j], 1: p1[j]}}
        if len(tracks) < min_inliers:
            return VOResult(ok=False, error="too few cheiral seed points")
        per_frame = [int(inl.sum())]
        for i in range(2, n):
            ki, di = _detect(grays[i])
            if len(di) < 2:
                return VOResult(ok=False, error=f"no features at frame {i}")
            ids = list(tracks)
            td = np.asarray([tracks[j]["desc"] for j in ids]).astype(np.float32)
            matches = _match(td, di)
            obj, img_pts, seen = [], [], set()
            for g in matches:
                j = ids[g.queryIdx]
                if j in seen:
                    continue
                seen.add(j)
                obj.append(tracks[j]["xyz"])
                img_pts.append(ki[g.trainIdx].pt)
                tracks[j]["desc"] = di[g.trainIdx].copy()
                tracks[j]["uv"][i] = np.asarray(ki[g.trainIdx].pt)
            if len(obj) < 6:
                return VOResult(ok=False, error=f"PnP: only {len(obj)} tracked points at frame {i}")
            ok_p, rvec, tvec, inl_p = cv2.solvePnPRansac(
                np.asarray(obj), np.asarray(img_pts), K, None,
                reprojectionError=pnp_reproj, confidence=0.99, flags=cv2.SOLVEPNP_AP3P,
            )
            if not ok_p or inl_p is None or len(inl_p) < min_inliers:
                return VOResult(ok=False, error=f"PnP RANSAC failed at frame {i}")
            R_i, _ = cv2.Rodrigues(rvec)
            Rs.append(R_i)
            ts.append(tvec.ravel())
            per_frame.append(int(len(inl_p)))
            # triangulate fresh points from pair (i-1, i) for future frames
            k_prev, d_prev = _detect(grays[i - 1])
            try:
                fresh = _match(d_prev, di)
            except RuntimeError:
                fresh = []
            Pi_1 = K @ np.hstack([Rs[i - 1], ts[i - 1][:, None]])
            Pi = K @ np.hstack([R_i, ts[i][:, None]])
            if len(fresh):
                q_prev = np.float64([k_prev[g.queryIdx].pt for g in fresh])
                q_cur = np.float64([ki[g.trainIdx].pt for g in fresh])
                p4 = cv2.triangulatePoints(Pi_1, Pi, q_prev.T, q_cur.T)
                p3 = _euclid(p4)
                keep3 = _cheiral(p3, Rs, ts, (i - 1, i))
                nxt = max(tracks) + 1
                for m in np.flatnonzero(keep3):
                    tracks[nxt] = {"xyz": p3[m], "desc": di[fresh[m].trainIdx].copy(),
                                   "uv": {i - 1: q_prev[m], i: q_cur[m]}}
                    nxt += 1
        T = np.zeros((n, 3, 4))
        for i in range(n):
            T[i, :3, :3] = Rs[i]
            T[i, :3, 3] = ts[i]
        from yardstick3d.geometry.cameras import camera_centers_from_w2c

        return VOResult(
            ok=True, T_w2c=T, centers=camera_centers_from_w2c(T),
            n_inliers_pair01=int(inl.sum()), n_map_points=len(tracks),
            per_frame_inliers=per_frame,
            provenance={"method": "chained-E-PnP", "K": np.asarray(K).tolist()},
        )
    except RuntimeError as e:
        return VOResult(ok=False, error=str(e)[:300])
