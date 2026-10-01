"""Site replication R1 - DA3-BASE arm (local GPU). Thin wrapper around the frozen v5 runner.

scripts/cross_backbone_v2_da3.py is loaded as a PRIVATE importlib instance and re-pointed at the R1 identity
(OUT = artifacts/site_replication_r1, R1 contract sha, R1 pool via r1_compat.V2ViewR1). The DA3 semantics
(DA3-BASE, process_res 378, from_pretrained, MetricGrounder diagnostics) are unchanged. Refuses unless the R1
contract is registered and the Stage-1 manifest is not a dry run.
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
from yardstick3d.datasets.r1_compat import V2ViewR1  # noqa: E402

_spec = importlib.util.spec_from_file_location("r1_cb_v2_da3", ROOT / "scripts/cross_backbone_v2_da3.py")
da3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(da3)  # private instance; the frozen file is never edited


def bind(out: Path = ROOT / r1c.OUT_REL, contract: Path = r1c.CONTRACT_PATH, registry: Path | None = r1c.REGISTRY_PATH):
    view = V2ViewR1(contract, registry)
    da3.v2 = view
    da3.OUT = Path(out)
    da3.CONTRACT = Path(contract)
    da3.CONTRACT_SHA = view.CONTRACT_SHA
    return view


def infer(out: Path = ROOT / r1c.OUT_REL) -> int:
    r1c.check_registered()
    bind(out)
    man = json.loads((Path(out) / "stage1" / "stage1_manifest.json").read_text(encoding="utf-8"))
    if man.get("dry_run") is not False:
        raise SystemExit("refusing a dry-run (or unattested) Stage-1 manifest")
    return da3.infer()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["infer"])
    ap.parse_args(argv)
    return infer()


if __name__ == "__main__":
    raise SystemExit(main())
