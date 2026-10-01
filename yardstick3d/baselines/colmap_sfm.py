"""COLMAP incremental SfM classical baseline (v2, frozen protocol).

Single strong conventional monocular-geometry comparator. Pure vision
(SIFT + RANSAC + bundle adjustment), arbitrary-scale output, self-calibrated
shared camera, no metric/IMU/GNSS/learned input. Output is canonicalized to
:class:`PredictionBundle` so the *unchanged* Yardstick3D grounding layer
consumes it through the same interface as the 3DFM predictions.

Frozen configuration lives in ``configs/classical_baseline_v2.yaml``; the
defaults below mirror it exactly. Do not tune per window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROTOCOL_ID = "classical_baseline_v2"
PYCOLMAP_VERSION_PIN = "4.2.0"
COLMAP_VERSION_PIN = "COLMAP 4.2.0"
MODEL_ID_SPARSE = "colmap-4.2.0/sparse8"
MODEL_ID_FULLRATE = "colmap-4.2.0/fullrate-seq"
MAPPER_SEED = 0
NUM_THREADS = 1
SEQ_OVERLAP = 10
FULLRATE_STRIDE = 12


def build_frame_list(v0: int, v1: int, frozen: Sequence[int],
                     stride: int = FULLRATE_STRIDE) -> list[int]:
    """Full-rate sampling: stride grid over [v0,v1] UNION the frozen indices.

    Guarantees the 8 frozen frames are always present (so Exp-B query poses
    align with the frozen timestamps), deduplicated and sorted.
    """
    grid = list(range(int(v0), int(v1) + 1, int(stride)))
    return sorted(set(grid) | {int(x) for x in frozen})


def frame_name(idx: int) -> str:
    return f"v{int(idx):05d}.png"


@dataclass
class ColmapResult:
    ok: bool
    T_w2c: np.ndarray | None = None          # (n_query, 3, 4) OpenCV w2c
    K: np.ndarray | None = None              # (n_query, 3, 3) estimated shared cam
    centers: np.ndarray | None = None        # (n_query, 3), C = -R^T t
    query_names: list[str] = field(default_factory=list)
    n_registered: int = 0                    # registered images in chosen model
    n_points3D: int = 0
    n_models: int = 0
    estimated_camera: dict[str, Any] = field(default_factory=dict)
    stage: str = ""
    error: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


def colmap_versions() -> dict[str, str]:
    import pycolmap

    v = getattr(pycolmap, "__version__", "unknown")
    cv = pycolmap.COLMAP_version
    cv = cv() if callable(cv) else cv
    return {"pycolmap": str(v), "colmap": str(cv)}


def check_pinned_versions() -> dict[str, str]:
    vers = colmap_versions()
    if vers["pycolmap"] != PYCOLMAP_VERSION_PIN:
        raise RuntimeError(f"pycolmap {vers['pycolmap']} != pin {PYCOLMAP_VERSION_PIN}")
    if COLMAP_VERSION_PIN not in vers["colmap"]:
        raise RuntimeError(f"colmap {vers['colmap']} != pin {COLMAP_VERSION_PIN}")
    return vers


def _extraction_options():
    import pycolmap

    o = pycolmap.FeatureExtractionOptions()
    o.use_gpu = False
    o.num_threads = NUM_THREADS
    return o


def _matching_options():
    import pycolmap

    o = pycolmap.FeatureMatchingOptions()
    o.use_gpu = False
    o.num_threads = NUM_THREADS
    return o


def _mapper_options(seed: int = MAPPER_SEED):
    import pycolmap

    o = pycolmap.IncrementalPipelineOptions()
    o.random_seed = int(seed)
    o.num_threads = NUM_THREADS
    return o


def run_extraction(db_path: str | Path, image_dir: str | Path) -> None:
    import pycolmap

    pycolmap.extract_features(
        str(db_path), str(image_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=_extraction_options(),
    )


def run_matching_exhaustive(db_path: str | Path) -> None:
    import pycolmap

    pycolmap.match_exhaustive(str(db_path), matching_options=_matching_options())


def run_matching_sequential(db_path: str | Path, overlap: int = SEQ_OVERLAP) -> None:
    import pycolmap

    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = int(overlap)
    pycolmap.match_sequential(
        str(db_path), matching_options=_matching_options(), pairing_options=pairing
    )


def run_mapper(
    db_path: str | Path, image_dir: str | Path, out_dir: str | Path,
    seed: int = MAPPER_SEED,
):
    import pycolmap

    return pycolmap.incremental_mapping(
        str(db_path), str(image_dir), str(out_dir), options=_mapper_options(seed)
    )


def select_primary_model(reconstructions: dict) -> Any:
    """Largest model wins: max registered images, tie-break max 3D points."""
    if not reconstructions:
        raise RuntimeError("no reconstructions produced")
    return max(
        reconstructions.values(),
        key=lambda r: (r.num_reg_images(), r.num_points3D()),
    )


def pose_to_w2c(rotation_matrix: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return np.hstack([R, t[:, None]])


def center_from_w2c(T: np.ndarray) -> np.ndarray:
    R, t = T[:, :3], T[:, 3]
    return -R.T @ t


def gather_query_poses(reconstruction: Any, query_names: Sequence[str]):
    """Poses for query images in order. Raises RuntimeError listing gaps."""
    T_list, missing, unposed, nonfinite = [], [], [], []
    cam_info: dict[str, Any] = {}
    for name in query_names:
        im = reconstruction.find_image_with_name(name)
        if im is None:
            missing.append(name)
            continue
        if not im.has_pose:
            unposed.append(name)
            continue
        pose = im.cam_from_world()
        T = pose_to_w2c(pose.rotation.matrix(), np.asarray(pose.translation))
        if not np.all(np.isfinite(T)):
            nonfinite.append(name)
            continue
        T_list.append(T)
        cam = reconstruction.camera(im.camera_id)
        cam_info = {
            "model": str(cam.model_name() if callable(getattr(cam, "model_name", None)) else cam.model_name),
            "width": int(cam.width),
            "height": int(cam.height),
            "params": [float(v) for v in cam.params],
            "K": np.asarray(cam.calibration_matrix(), dtype=np.float64),
        }
    problems = {"missing": missing, "unposed": unposed, "nonfinite": nonfinite}
    if any(problems.values()):
        raise RuntimeError(f"query pose gaps: {problems}")
    T_w2c = np.stack(T_list)
    n = len(query_names)
    K = np.tile(cam_info["K"][None], (n, 1, 1))
    return T_w2c, K, cam_info


def run_sparse_window(
    png_paths: Sequence[str | Path],
    query_names: Sequence[str],
    workdir: str | Path,
    seed: int = MAPPER_SEED,
) -> ColmapResult:
    """Experiment A: incremental SfM over matched sparse frames (exhaustive)."""
    import shutil

    vers = check_pinned_versions()
    workdir = Path(workdir)
    imgdir = workdir / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    for src, name in zip(png_paths, query_names):
        shutil.copy(src, imgdir / name)
    db = workdir / "database.db"
    out = workdir / "sparse"
    prov = {
        "protocol": PROTOCOL_ID, "experiment": "A", "seed": int(seed),
        "threads": NUM_THREADS, **vers,
    }
    try:
        run_extraction(db, imgdir)
    except Exception as e:  # noqa: BLE001
        return ColmapResult(ok=False, query_names=list(query_names), stage="extract",
                            error=str(e)[-300:], provenance=prov)
    try:
        run_matching_exhaustive(db)
    except Exception as e:  # noqa: BLE001
        return ColmapResult(ok=False, query_names=list(query_names), stage="match",
                            error=str(e)[-300:], provenance=prov)
    try:
        recs = run_mapper(db, imgdir, out, seed=seed)
    except Exception as e:  # noqa: BLE001
        return ColmapResult(ok=False, query_names=list(query_names), stage="map",
                            error=str(e)[-300:], provenance=prov)
    if not recs:
        return ColmapResult(ok=False, query_names=list(query_names), stage="map",
                            error="empty reconstruction set", n_models=0, provenance=prov)
    best = select_primary_model(recs)
    n_reg, n_pts = best.num_reg_images(), best.num_points3D()
    try:
        T_w2c, K, cam_info = gather_query_poses(best, list(query_names))
    except RuntimeError as e:
        return ColmapResult(ok=False, query_names=list(query_names), stage="pose_gather",
                            error=str(e)[-300:], n_registered=n_reg, n_points3D=n_pts,
                            n_models=len(recs), provenance=prov)
    centers = np.stack([center_from_w2c(T) for T in T_w2c])
    return ColmapResult(
        ok=True, T_w2c=T_w2c, K=K, centers=centers,
        query_names=list(query_names), n_registered=n_reg, n_points3D=n_pts,
        n_models=len(recs), estimated_camera=cam_info, stage="done", provenance=prov,
    )


def decode_mov_frames(
    video_path: str | Path, frame_indices: Sequence[int], outdir: str | Path,
    name_fn=None,
) -> list[Path]:
    """Decode exact mov frames by index (raw portrait pixels, no enhancement)."""
    import cv2

    video_path, outdir = Path(video_path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video_path}")
    paths = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"decode failed at mov frame {idx}")
        name = name_fn(int(idx)) if name_fn else f"v{int(idx):05d}.png"
        p = outdir / name
        if not cv2.imwrite(str(p), frame):
            cap.release()
            raise RuntimeError(f"write failed for {p}")
        paths.append(p)
    cap.release()
    return paths


def run_fullrate_window(
    image_paths: Sequence[str | Path],
    query_names: Sequence[str],
    workdir: str | Path,
    seed: int = MAPPER_SEED,
    overlap: int = SEQ_OVERLAP,
) -> ColmapResult:
    """Experiment B: same pipeline with sequential matching over dense frames."""
    import shutil

    vers = check_pinned_versions()
    workdir = Path(workdir)
    imgdir = workdir / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    names = [Path(p).name for p in image_paths]
    for src, name in zip(image_paths, names):
        shutil.copy(src, imgdir / name)
    db = workdir / "database.db"
    out = workdir / "sparse"
    prov = {
        "protocol": PROTOCOL_ID, "experiment": "B", "seed": int(seed),
        "threads": NUM_THREADS, "seq_overlap": int(overlap),
        "n_images": len(image_paths), **vers,
    }
    try:
        run_extraction(db, imgdir)
    except Exception as e:  # noqa: BLE001
        return ColmapResult(ok=False, query_names=list(query_names), stage="extract",
                            error=str(e)[-300:], provenance=prov)
    try:
        run_matching_sequential(db, overlap=overlap)
    except Exception as e:  # noqa: BLE001
        return ColmapResult(ok=False, query_names=list(query_names), stage="match",
                            error=str(e)[-300:], provenance=prov)
    try:
        recs = run_mapper(db, imgdir, out, seed=seed)
    except Exception as e:  # noqa: BLE001
        return ColmapResult(ok=False, query_names=list(query_names), stage="map",
                            error=str(e)[-300:], provenance=prov)
    if not recs:
        return ColmapResult(ok=False, query_names=list(query_names), stage="map",
                            error="empty reconstruction set", n_models=0, provenance=prov)
    best = select_primary_model(recs)
    n_reg, n_pts = best.num_reg_images(), best.num_points3D()
    try:
        T_w2c, K, cam_info = gather_query_poses(best, list(query_names))
    except RuntimeError as e:
        return ColmapResult(ok=False, query_names=list(query_names), stage="pose_gather",
                            error=str(e)[-300:], n_registered=n_reg, n_points3D=n_pts,
                            n_models=len(recs), provenance=prov)
    centers = np.stack([center_from_w2c(T) for T in T_w2c])
    return ColmapResult(
        ok=True, T_w2c=T_w2c, K=K, centers=centers,
        query_names=list(query_names), n_registered=n_reg, n_points3D=n_pts,
        n_models=len(recs), estimated_camera=cam_info, stage="done", provenance=prov,
    )


def result_to_bundle(
    res: ColmapResult, timestamps: np.ndarray, model_id: str
):
    """Canonicalize a successful result to PredictionBundle (depth optional/absent)."""
    from yardstick3d.types import PredictionBundle

    if not res.ok or res.T_w2c is None or res.K is None:
        raise ValueError("cannot canonicalize failed COLMAP result")
    ts = np.asarray(timestamps, dtype=np.float64)
    return PredictionBundle(
        timestamps=ts,
        T_w2c=np.asarray(res.T_w2c, dtype=np.float64),
        K=np.asarray(res.K, dtype=np.float64),
        camera_centers=np.asarray(res.centers, dtype=np.float64),
        is_metric=False,
        model_id=model_id,
        aux={
            "protocol": PROTOCOL_ID,
            "n_registered": res.n_registered,
            "n_points3D": res.n_points3D,
            "estimated_camera": res.estimated_camera,
            **res.provenance,
        },
    )


def success_table(rows: Sequence[dict]) -> dict:
    """Pure helper: availability + conditional-rate summary over window rows."""
    ok = [r for r in rows if r.get("ok")]
    return {
        "n_attempted": len(rows),
        "n_success": len(ok),
        "success_rate": (len(ok) / len(rows)) if rows else float("nan"),
        "n_registered_hist": sorted(r.get("n_registered", 0) for r in rows),
    }
