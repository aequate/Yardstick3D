from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from yardstick3d.adapters.base import AdapterCapabilities, FoundationModelAdapter, PredictRequest
from yardstick3d.geometry.cameras import camera_centers_from_w2c
from yardstick3d.types import PredictionBundle

VGGT_EXPECTED_MODEL_ID = "vggt-1b/facebook/VGGT-1B"
VGGT_EXPECTED_SOURCE_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_EXPECTED_MODEL_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
VGGT_REQUIRED_FIELDS = frozenset(
    {"timestamps", "T_w2c", "K", "depth_z", "depth_conf", "is_metric", "model_id"}
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_vggt_bundle(
    npz_path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    expected_timestamps: np.ndarray | None = None,
    check_provenance: bool = True,
) -> PredictionBundle:
    """Map a frozen remote VGGT output bundle to the canonical PredictionBundle.

    Enforces the frozen-pack contract: exact field set, (8,3,4) OpenCV w2c
    poses, finite values, positive depth, unknown-scale flag, and pinned
    model identity. Camera centers are derived as C = -R^T t, the same
    convention used by the DA3 adapter. The grounding solver is untouched;
    this only changes the prediction source.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        if set(z.files) != set(VGGT_REQUIRED_FIELDS):
            raise ValueError(
                f"VGGT bundle field mismatch in {npz_path.name}: {sorted(z.files)}"
            )
        timestamps = np.asarray(z["timestamps"], dtype=np.float64)
        T_w2c = np.asarray(z["T_w2c"], dtype=np.float64)
        K = np.asarray(z["K"], dtype=np.float64)
        depth_z = np.asarray(z["depth_z"], dtype=np.float32)
        depth_conf = np.asarray(z["depth_conf"], dtype=np.float32)
        is_metric = bool(np.asarray(z["is_metric"]).reshape(()))
        model_id = str(z["model_id"].item())
    if timestamps.shape != (8,) or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"bad timestamps in {npz_path.name}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"non-increasing timestamps in {npz_path.name}")
    if T_w2c.shape != (8, 3, 4) or not np.all(np.isfinite(T_w2c)):
        raise ValueError(f"bad T_w2c in {npz_path.name}")
    if K.shape != (8, 3, 3) or not np.all(np.isfinite(K)):
        raise ValueError(f"bad K in {npz_path.name}")
    if depth_z.ndim != 3 or depth_z.shape[0] != 8 or not np.all(np.isfinite(depth_z)):
        raise ValueError(f"bad depth_z in {npz_path.name}")
    if np.any(depth_z <= 0):
        raise ValueError(f"non-positive depth in {npz_path.name}")
    if depth_conf.shape != depth_z.shape or not np.all(np.isfinite(depth_conf)):
        raise ValueError(f"bad depth_conf in {npz_path.name}")
    if np.any(depth_conf < 0):
        raise ValueError(f"negative depth_conf in {npz_path.name}")
    if is_metric:
        raise ValueError(f"VGGT bundle {npz_path.name} must be unknown-scale (is_metric=false)")
    if check_provenance and model_id != VGGT_EXPECTED_MODEL_ID:
        raise ValueError(f"foreign model_id {model_id!r} in {npz_path.name}")
    if expected_timestamps is not None:
        np.testing.assert_array_equal(
            timestamps,
            np.asarray(expected_timestamps, dtype=np.float64),
            err_msg=f"timestamp mismatch vs frozen input for {npz_path.name}",
        )
    aux: dict[str, Any] = {
        "source_npz": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "is_metric": False,
        "pose_convention": "OpenCV world-to-camera 3x4; camera center = -R.T @ t",
        "depth_convention": "camera Z, unknown global scale",
    }
    if sidecar_path is None:
        candidate = npz_path.with_suffix(".json")
        sidecar_path = candidate if candidate.exists() else None
    if sidecar_path is not None:
        meta = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        if check_provenance:
            if meta.get("source_revision") != VGGT_EXPECTED_SOURCE_REVISION:
                raise ValueError(f"foreign source_revision in {sidecar_path}")
            if meta.get("model_revision") != VGGT_EXPECTED_MODEL_REVISION:
                raise ValueError(f"foreign model_revision in {sidecar_path}")
            if meta.get("model_id") != VGGT_EXPECTED_MODEL_ID:
                raise ValueError(f"foreign model_id in {sidecar_path}")
            if meta.get("is_metric") is not False:
                raise ValueError(f"sidecar must declare is_metric=false: {sidecar_path}")
            if meta.get("contains_ground_truth") is not False:
                raise ValueError(f"sidecar must declare no GT: {sidecar_path}")
        if meta.get("npz_sha256") != aux["npz_sha256"]:
            raise ValueError(f"npz sha mismatch vs sidecar: {npz_path.name}")
        aux["sidecar"] = meta
    return PredictionBundle(
        timestamps=timestamps,
        T_w2c=T_w2c,
        K=K,
        depth_z=depth_z,
        camera_centers=camera_centers_from_w2c(T_w2c),
        depth_conf=depth_conf,
        is_metric=False,
        model_id=model_id,
        aux=aux,
    )


class VGGTCachedAdapter(FoundationModelAdapter):
    """Serve frozen remote VGGT bundles through the adapter interface.

    Live VGGT inference stays blocked (see VGGTAdapter); this class only
    reads pinned off-box outputs from ``root`` (``<seq>/<folder>.npz``).
    CPU-only, no weights, no GT channel.
    """

    caps = AdapterCapabilities(
        name="vggt-1b-cached",
        min_vram_gb=0.0,
        max_views_on_6gb=8,
        poses=True,
        depth=True,
        metric=False,
        pose_kind="w2c",
        license="weights CC-BY-NC-4.0 (commercial gated)",
        local=True,
        notes="Frozen off-box VGGT-1B outputs mapped to PredictionBundle. No inference.",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, device: str = "cpu") -> None:
        if not self.root.is_dir():
            raise RuntimeError(f"VGGT cached root missing: {self.root}")

    def bundle_for(self, entry: str) -> PredictionBundle:
        return load_vggt_bundle(self.root / f"{entry}.npz")

    def predict(self, req: PredictRequest) -> PredictionBundle:
        npz = (req.aux or {}).get("npz")
        if npz is None:
            raise RuntimeError("VGGTCachedAdapter.predict requires req.aux['npz']")
        bundle = load_vggt_bundle(npz, expected_timestamps=req.timestamps)
        return bundle


VGGT_BLOCKER = (
    "VGGT-1B is not run in-process: it needs a CUDA GPU with 12 GB or more (weights are ~5 GB fp32). "
    "Run vggt_runner/ on a GPU host and load the returned npz as a cached PredictionBundle, "
    "or use DummyFoundationModel for solver tests. Canonical pose: OpenCV w2c 3x4, depth is Z."
)


class VGGTAdapter(FoundationModelAdapter):
    caps = AdapterCapabilities(
        name="vggt-1b",
        min_vram_gb=12.0,
        max_views_on_6gb=None,
        poses=True,
        depth=True,
        metric=False,
        pose_kind="w2c",
        license="weights CC-BY-NC-4.0 (commercial gated)",
        local=False,
        notes=VGGT_BLOCKER,
    )

    def load(self, device: str = "cuda") -> None:
        raise RuntimeError(VGGT_BLOCKER)

    def predict(self, req: PredictRequest) -> PredictionBundle:
        raise RuntimeError(VGGT_BLOCKER)
