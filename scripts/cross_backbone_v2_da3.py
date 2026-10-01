"""Cross-backbone v2 - DA3-BASE local inference on the frozen Stage-1 windows (v5 pooled).

Contract: configs/prospective_cross_backbone_v2.yaml (FROZEN, sha in v2_scoring.CONTRACT_SHA).
Reads ONLY Stage-1 sensor artifacts (images, UTC-converted ublox cue, windows.json). No reference/GT is
opened. Writes immutable DA3 PredictionBundles, da3_manifest.json, and the GT-free rho(a,b) diagnostic.

Run under the DA3 environment:  .venv-da3/Scripts/python.exe scripts/cross_backbone_v2_da3.py infer
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yardstick3d.datasets import v2_scoring as v2  # noqa: E402

OUT = ROOT / "artifacts/cross_backbone_v5"  # v5 pooled (A3)
CONTRACT = v2.CONTRACT_PATH
CONTRACT_SHA = v2.CONTRACT_SHA


def load_cue(path: Path):
    """(t_utc, llh[n,3]) from the Stage-1 cue csv (already UTC-converted)."""
    t, llh = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_utc"]))
            llh.append((float(row["lat_deg"]), float(row["lon_deg"]), float(row["alt_m"])))
    return np.asarray(t), np.asarray(llh)


def cue_for(s1: Path, w: dict, cache: dict):
    """(t_utc, llh) of the window's own sequence (v5: cue_<SEQ>.csv; v2-v4: cue_receiver_lla.csv)."""
    key = w.get("sequence")
    if key not in cache:
        cache[key] = load_cue(s1 / v2.cue_csv_name(key))
    return cache[key]


def window_cue_enu(cue_t, cue_llh, t_start):
    """Window-local ENU about the window's first cue fix (shared origin, contract enu_origin)."""
    from yardstick3d.datasets import mars_lvig as ml

    lat0, lon0, alt0 = v2.window_origin(cue_t, cue_llh, t_start)
    alt = np.where(np.isfinite(cue_llh[:, 2]), cue_llh[:, 2], 0.0)
    return ml.cue_enu_from_llh(cue_llh[:, 0], cue_llh[:, 1], alt, lat0, lon0, alt0)


def check_stage1(s1: Path) -> dict:
    """Load windows.json / manifest, verify contract sha and frame hashes. GT-free."""
    manifest = json.loads((s1 / "stage1_manifest.json").read_text())
    if manifest.get("contract_sha256") != CONTRACT_SHA:
        raise RuntimeError("stage1 manifest contract sha != frozen v2 contract sha")
    if manifest.get("reference_values_read") is not False:
        raise RuntimeError("stage1 manifest does not attest reference_values_read == False")
    wj = json.loads((s1 / "windows.json").read_text())
    for name, digest in manifest["frame_sha256"].items():
        if v2.sha256_file(s1 / "frames" / name) != digest:
            raise RuntimeError(f"frame hash mismatch: {name}")
    return wj


def infer() -> int:
    import types as _types

    for _name in ("pycolmap", "gsplat", "open3d"):
        sys.modules.setdefault(_name, _types.ModuleType(_name))
    import cv2

    from yardstick3d.adapters.base import PredictRequest
    from yardstick3d.adapters.da3 import DA3Adapter
    from yardstick3d.datasets import mars_lvig as ml
    from yardstick3d.io.prediction_cache import save_prediction
    from yardstick3d.optimization.grounder import MetricGrounder

    v2.check_contract(CONTRACT, CONTRACT_SHA)
    s1 = OUT / "stage1"
    wins = check_stage1(s1)["windows"]
    pred_dir = OUT / "predictions_da3"
    if pred_dir.exists() and any(pred_dir.glob("*.npz")):
        raise RuntimeError("DA3 predictions exist; refusing silent rerun")
    cue_cache: dict = {}

    adapter = DA3Adapter(model_name="da3-base", process_res=378)
    adapter.load("cuda")
    if not adapter.from_pretrained_ok:
        raise RuntimeError("DA3-BASE not from_pretrained")
    print("DA3 hub:", adapter.hub_id)

    g = MetricGrounder()
    rows = []
    try:
        for i, w in enumerate(wins):
            cue_t, cue_llh = cue_for(s1, w, cue_cache)
            imgs = [cv2.cvtColor(cv2.imread(str(s1 / "frames" / v2.frame_relpath(w, k))), cv2.COLOR_BGR2RGB)
                    for k in w["frame_indices_bag"]]
            ts = np.asarray(w["frame_times_utc"], dtype=np.float64)
            pred = adapter.predict(PredictRequest(images=imgs, timestamps=ts))
            dest = pred_dir / f"{i:04d}.npz"
            save_prediction(pred, dest, meta={
                "contract_sha256": CONTRACT_SHA, "window": i, "role": w["role"], "sequence": v2.seq_of(w),
                "contains_ground_truth": False, "gt_used": False,
                "checkpoint": adapter.hub_id, "from_pretrained": True, "backbone": "da3-base",
            })
            enu = window_cue_enu(cue_t, cue_llh, float(ts[0]))
            cs, cinfo = ml.constraints_from_ublox_displacement(ts, cue_t, enu, max_dt_match=v2.MAX_DT_CUE_S)
            scales = {}
            for name in ("none", "naive", "ls", "robust"):
                r = g.solve(pred, cs, solver=name)
                scales[name] = {
                    "scale": float(r.scale) if np.isfinite(r.scale) else None,
                    "observable": bool(r.observable),
                    "observability_score": float(r.observability_score),
                    "accepted": int(r.accepted_constraints),
                }
            rho = v2.rho_diagnostic(pred.centers(), ts, cue_t, enu)
            rows.append({
                "window": i, "sequence": v2.seq_of(w), "role": w["role"], "cls": w.get("cls"), "t0": w["t0"],
                "prediction": str(dest.relative_to(ROOT)),
                "prediction_sha256": v2.sha256_file(dest),
                "prediction_hash": json.loads(dest.with_suffix(".json").read_text())["hash"],
                "solver": scales, "cue": cinfo, "n_constraints": len(cs), "rho_diagnostic": rho,
            })
            v2.dump_json(OUT / "da3_progress.json", {"predictions": rows, "gt_loaded": False})
            print(f"window {i} {v2.seq_of(w)} ({w['role']}) scale_naive={scales['naive']['scale']} cue_kept={cinfo['n_kept']}")
    finally:
        adapter.unload()

    v2.dump_json(OUT / "da3_manifest.json", {
        "contract_sha256": CONTRACT_SHA,
        "sequences": sorted({v2.seq_of(w) for w in wins}, key=lambda s: (s not in v2.POOL, v2.POOL.index(s) if s in v2.POOL else 0)),
        "backbone": "da3-base",
        "model_id": "da3-base/depth-anything/DA3-BASE",
        "hub_id": "depth-anything/DA3-BASE",
        "from_pretrained": True,
        "process_res": 378,
        "process_res_method": "upper_bound_resize",
        "device": "cuda",
        "contains_ground_truth": False,
        "reference_values_read": False,
        "windows": rows,
    })
    print("DA3 inference complete")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["infer"])
    ap.parse_args()
    return infer()


if __name__ == "__main__":
    raise SystemExit(main())
