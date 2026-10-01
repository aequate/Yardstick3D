"""Cross-backbone v2 - build the deterministic off-box VGGT pack (contract v2, FROZEN).

Copies the Stage-1 v2 window RGB frames + UTC timestamps into a self-contained pack. The runner files
(bootstrap.py, run_vggt.py, verify.py, requirements.txt) are copied byte-for-byte from the v1 pack so the
VGGT semantics are identical; model_lock.json carries the SAME pinned VGGT source/checkpoint with a new
pack_id, the v2 contract sha and the v2 sequence. The pack contains no GT, no reference, no cue value.
Roles (test/validation/cc) are NOT written into the pack (verify.py's window.json allow-list has no role).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yardstick3d.datasets import v2_scoring as v2  # noqa: E402

SRC = ROOT / "artifacts/cross_backbone_v5/stage1"  # v5 pooled (A3)
PACK = ROOT / "artifacts/vggt_pack_cross_backbone_v5"
TEMPLATE_PACK = ROOT / "vggt_runner"
TEMPLATE_FILES = ("bootstrap.py", "run_vggt.py", "verify.py", "requirements.txt")
# VGGT pins that MUST appear verbatim in the frozen v2 contract (backbones.vggt-1b)
PIN_KEYS = ("source_url", "source_revision", "model_id", "model_revision", "checkpoint", "checkpoint_sha256")


def window_folder(t0: float) -> str:
    return f"t{int(round(t0))}_{int(round(t0 + v2.WINDOW_DURATION_S))}"


def build_model_lock(template_lock: dict, contract_text: str, contract_sha: str, sequence: str = v2.SEQUENCE) -> dict:
    """v1 model_lock with a new pack_id/sequence/contract sha; pins asserted against the v2 contract text."""
    for k in PIN_KEYS:
        if str(template_lock[k]) not in contract_text:
            raise RuntimeError(f"model_lock pin {k}={template_lock[k]!r} not found in the v2 contract")
    if str(template_lock["checkpoint_bytes"]) not in contract_text:
        raise RuntimeError("checkpoint_bytes not found in the v2 contract")
    lock = dict(template_lock)
    lock.update({
        "pack_id": v2.PACK_ID,
        "sequence": sequence,
        "contract": "configs/prospective_cross_backbone_v5.yaml",
        "contract_sha256": contract_sha,
        "contains_ground_truth": False,
        "gpu_execution_verified": False,
    })
    return lock


def readme_text(n_windows: int, sequences: list | None = None) -> str:
    return f"""# VGGT off-box execution pack - cross-backbone v5 (pooled)

Status: AWAITING_USER_GPU_RUN.

Pack: `{v2.PACK_ID}`
Dataset: MARS-LVIG `{", ".join(sequences or [v2.SEQUENCE])}` (prospective; v5 pooled windows)
Contract: `configs/prospective_cross_backbone_v5.yaml` (frozen; sha256 in `model_lock.json`)
Windows: {n_windows} (SV test + validation + constant-cruise controls), 8 RGB frames each (16 s windows).

The pack contains ONLY selected RGB frames, their UTC timestamps and frozen model/source pins. There is
no evaluation ground truth, no reference trajectory and no metric cue value inside the pack.

Run (CUDA GPU host, Python 3.11, git): `python3.11 bootstrap.py` (Windows: `py -3.11 bootstrap.py`).
Validate inputs without a GPU: `python verify.py`  (expected: `PASS: {n_windows} frozen windows; outputs checked=False`).
Return the produced `outputs/` directory. Do not change frames, resolution, precision, seed, model or
source revision. Stop on OOM and report it as a blocker.
"""


def build(src: Path = SRC, pack: Path = PACK, template_pack: Path = TEMPLATE_PACK,
          contract: Path = v2.CONTRACT_PATH, check: bool = True) -> int:
    src, pack, template_pack = Path(src), Path(pack), Path(template_pack)
    contract_sha = v2.check_contract(contract, v2.CONTRACT_SHA, v2.REGISTRY_PATH if check else None)
    manifest_s1 = json.loads((src / "stage1_manifest.json").read_text())
    if manifest_s1.get("contract_sha256") != v2.CONTRACT_SHA:
        raise SystemExit("stage1 manifest contract sha != frozen v2 contract sha")
    wj = json.loads((src / "windows.json").read_text())
    wins = wj["windows"]
    if not wins:
        raise SystemExit("no stage1 windows")
    if (pack / "input_hashes.json").exists():
        raise SystemExit("input_hashes.json exists; pack is frozen (refusing rebuild)")

    files: dict[str, str] = {}
    seq_entries: dict[str, list] = {}
    windows_list = []
    for w in wins:
        t0 = float(w["t0"])
        seq = v2.seq_of(w)
        entry = f"{seq}/{window_folder(t0)}"
        if entry in windows_list:
            raise SystemExit(f"duplicate window folder {entry}")
        wdir = pack / entry
        wdir.mkdir(parents=True, exist_ok=True)
        images = []
        # pack frame_indices = indices into the time-sorted image list (strictly increasing by construction);
        # image files are stored under their bag index name in Stage 1.
        for i, k in enumerate(w["frame_indices_bag"]):
            name = f"f{i:02d}.png"
            dst = wdir / name
            shutil.copyfile(src / "frames" / v2.frame_relpath(w, k), dst)
            files[f"{entry}/{name}"] = v2.sha256_file(dst)
            images.append(name)
        ts = np.asarray(w["frame_times_utc"], dtype=np.float64)
        np.save(wdir / "timestamps.npy", ts)
        rec = {
            "t0": round(t0, 6),
            "t1": round(t0 + v2.WINDOW_DURATION_S, 6),
            "timestamps": [float(x) for x in ts],
            "frame_indices": [int(i) for i in w["frame_indices_sorted"]],
            "images": images,
            "pose_convention": "OpenCV w2c 3x4, C=-R^T t",
            "is_metric": False,
        }
        (wdir / "window.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        files[f"{entry}/window.json"] = v2.sha256_file(wdir / "window.json")
        files[f"{entry}/timestamps.npy"] = v2.sha256_file(wdir / "timestamps.npy")
        seq_entries.setdefault(seq, []).append(rec)
        windows_list.append(entry)

    sequences = list(seq_entries)
    for seq, recs in seq_entries.items():
        seq_manifest = {
            "sequence": seq,
            "role": "cross_backbone_prospective",
            "n_windows": len(recs),
            "instruction": (
                "Run frozen VGGT on images in order; save PredictionBundle npz with T_w2c (N,3,4) "
                "OpenCV w2c, timestamps, is_metric=false, model_id=vggt-1b/facebook/VGGT-1B. No GT. See README.md."
            ),
            "windows": recs,
        }
        (pack / seq / "manifest.json").write_text(json.dumps(seq_manifest, indent=2), encoding="utf-8")
        files[f"{seq}/manifest.json"] = v2.sha256_file(pack / seq / "manifest.json")

    for name in TEMPLATE_FILES:
        shutil.copyfile(template_pack / name, pack / name)
    lock = build_model_lock(json.loads((template_pack / "model_lock.json").read_text()),
                            Path(contract).read_text(encoding="utf-8"), contract_sha, "+".join(sequences))
    lock["runner_template_sha256"] = {n: v2.sha256_file(template_pack / n) for n in TEMPLATE_FILES}
    (pack / "model_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    (pack / "README.md").write_text(readme_text(len(windows_list), sequences), encoding="utf-8")

    manifest = {
        "pack_id": v2.PACK_ID,
        "dataset": "MARS-LVIG",
        "contains_ground_truth": False,
        "n_frames": v2.N_FRAMES,
        "n_windows": len(windows_list),
        "sequences": sequences,
        "windows": windows_list,
        "files": files,
    }
    (pack / "input_hashes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"pack built: {len(windows_list)} windows -> {pack}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build"])
    ap.parse_args()
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
