from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from yardstick3d.types import PredictionBundle


def prediction_hash(bundle: PredictionBundle, extra: dict[str, Any] | None = None) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(bundle.T_w2c).tobytes())
    h.update(np.ascontiguousarray(bundle.timestamps).tobytes())
    h.update(bundle.model_id.encode())
    if extra:
        h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def save_prediction(bundle: PredictionBundle, npz_path: str | Path, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamps": np.asarray(bundle.timestamps),
        "T_w2c": np.asarray(bundle.T_w2c),
        "is_metric": np.array(bool(bundle.is_metric)),
        "model_id": np.array(bundle.model_id),
    }
    if bundle.K is not None:
        payload["K"] = np.asarray(bundle.K)
    if bundle.depth_z is not None:
        payload["depth_z"] = np.asarray(bundle.depth_z)
    if bundle.camera_centers is not None:
        payload["camera_centers"] = np.asarray(bundle.camera_centers)
    if bundle.depth_conf is not None:
        payload["depth_conf"] = np.asarray(bundle.depth_conf)
    np.savez_compressed(npz_path, **payload)
    digest = prediction_hash(bundle, meta)
    meta_out = {
        "npz": str(npz_path),
        "hash": digest,
        "model_id": bundle.model_id,
        "n_frames": bundle.n_frames(),
        "is_metric": bool(bundle.is_metric),
        "contains_ground_truth": False,
        "T_w2c_shape": list(bundle.T_w2c.shape),
        **(meta or {}),
    }
    json_path = npz_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    return meta_out


def load_prediction(npz_path: str | Path) -> PredictionBundle:
    z = np.load(npz_path, allow_pickle=True)
    mid = z["model_id"]
    model_id = str(mid.item() if hasattr(mid, "item") else mid)
    is_metric = bool(np.array(z["is_metric"]).reshape(()))
    return PredictionBundle(
        timestamps=np.asarray(z["timestamps"], dtype=np.float64),
        T_w2c=np.asarray(z["T_w2c"], dtype=np.float64),
        K=np.asarray(z["K"], dtype=np.float64) if "K" in z.files else None,
        depth_z=np.asarray(z["depth_z"], dtype=np.float32) if "depth_z" in z.files else None,
        camera_centers=np.asarray(z["camera_centers"], dtype=np.float64) if "camera_centers" in z.files else None,
        depth_conf=np.asarray(z["depth_conf"], dtype=np.float32) if "depth_conf" in z.files else None,
        is_metric=is_metric,
        model_id=model_id,
        aux={"source_npz": str(npz_path)},
    )
