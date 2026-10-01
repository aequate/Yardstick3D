"""Site replication R1 - build the deterministic off-box VGGT pack.

Reuses the frozen v5 builder (scripts/build_vggt_pack_v2.py) through a PRIVATE importlib instance re-pointed
at the R1 identity (pack id vggt_offbox_site_replication_r1, R1 Stage-1 dir, R1 contract sha). Runner files
(bootstrap.py, run_vggt.py, verify.py, requirements.txt) are copied byte-for-byte from the v1 template pack
and the VGGT pins are UNCHANGED: the R1 contract states the backbones are "IDENTICAL to v5", so every pin is
asserted verbatim against the frozen v5 contract text (and that contract's sha is recorded in the lock).
The pack holds only RGB frames + UTC timestamps (no role, no cue, no reference).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yardstick3d.datasets import r1_contract as r1c  # noqa: E402
from yardstick3d.datasets import v2_scoring as v2  # noqa: E402
from yardstick3d.datasets.r1_compat import V2ViewR1  # noqa: E402

_spec = importlib.util.spec_from_file_location("r1_build_vggt_pack_v5", ROOT / "scripts/build_vggt_pack_v2.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)  # private instance; the frozen file is never edited

SRC = ROOT / r1c.STAGE1_REL
PACK = ROOT / r1c.PACK_REL
TEMPLATE_PACK = bp.TEMPLATE_PACK
PIN_CONTRACT = v2.CONTRACT_PATH  # frozen v5 contract: source of the VGGT pins ("IDENTICAL to v5")
_v5_build_model_lock = bp.build_model_lock


def build_model_lock_r1(template_lock: dict, contract_text: str, contract_sha: str, sequence: str = "") -> dict:
    """v5 lock logic, pins asserted against the FROZEN v5 contract text (R1 inherits them unchanged)."""
    v5_text = PIN_CONTRACT.read_text(encoding="utf-8")
    lock = _v5_build_model_lock(template_lock, v5_text, contract_sha, sequence)
    lock.update({"contract": "configs/prospective_site_replication_r1.yaml", "contract_sha256": contract_sha,
                 "pins_asserted_against": {"contract": "configs/prospective_cross_backbone_v5.yaml",
                                           "sha256": v2.sha256_file(PIN_CONTRACT)},
                 "experiment": r1c.EXPERIMENT_NAME})
    return lock


def readme_text_r1(n_windows: int, sequences: list | None = None) -> str:
    return f"""# VGGT off-box execution pack - site replication R1

Status: AWAITING_USER_GPU_RUN.

Pack: `{r1c.PACK_ID}`
Dataset: MARS-LVIG `{", ".join(sequences or [])}` (prospective R1 windows)
Contract: `configs/prospective_site_replication_r1.yaml` (frozen; sha256 in `model_lock.json`)
Windows: {n_windows}, 8 RGB frames each (16 s windows). VGGT pins IDENTICAL to v5.

The pack contains ONLY selected RGB frames, their UTC timestamps and frozen model/source pins. There is
no evaluation ground truth, no reference trajectory and no metric cue value inside the pack.

Run (CUDA GPU host, Python 3.11, git): `python3.11 bootstrap.py` (Windows: `py -3.11 bootstrap.py`).
Validate inputs without a GPU: `python verify.py`  (expected: `PASS: {n_windows} frozen windows; outputs checked=False`).
Return the produced `outputs/` directory. Do not change frames, resolution, precision, seed, model or
source revision. Stop on OOM and report it as a blocker.
"""


def build(src: Path = SRC, pack: Path = PACK, template_pack: Path = TEMPLATE_PACK,
          contract: Path = r1c.CONTRACT_PATH, registry: Path | None = r1c.REGISTRY_PATH, check: bool = True) -> int:
    """check=False skips the registry lookup (synthetic tests only); the Stage-1 manifest must still match."""
    man = json.loads((Path(src) / "stage1_manifest.json").read_text(encoding="utf-8"))
    if man.get("dry_run") is not False:
        raise SystemExit("refusing a dry-run (or unattested) Stage-1 manifest")
    view = V2ViewR1(contract, registry)
    bp.v2 = view
    bp.build_model_lock = build_model_lock_r1
    bp.readme_text = readme_text_r1
    return bp.build(src=src, pack=pack, template_pack=template_pack, contract=contract, check=check)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build"])
    ap.parse_args(argv)
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
