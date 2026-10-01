"""Site replication R1 - contract handle, frozen constants and topic allow-list.

Contract: configs/prospective_site_replication_r1.yaml (a NEW pre-registered experiment; NOT a v6 of the
cross-backbone chain). Every threshold below is transcribed from that file and asserted against its text by
tests/test_site_replication_r1_cues.py. This module never reads a bag, a cue value or a reference value.

Freeze = the sha256 of the contract file recorded in configs/experiment_registry.json. Nothing here
writes to the registry; `check_registered` only refuses to proceed when the current contract bytes are not
registered (mirrors v2_scoring.check_contract).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs/prospective_site_replication_r1.yaml"
REGISTRY_PATH = ROOT / "configs/experiment_registry.json"
EXPERIMENT_NAME = "prospective_site_replication_r1"

# ---- dataset.pool (fixed a priori, rank order) ----
POOL = ("HKairport_GNSS02", "HKairport_GNSS03")
V_CRUISE_TABLE_BY_SEQ = {"HKairport_GNSS02": 6.0, "HKairport_GNSS03": 9.0}

# ---- topics ----
CAMERA_TOPIC = "/left_camera/image/compressed"
LLA_TOPIC = "/ublox_driver/receiver_lla"     # primary cue + altitude arm
PVT_TOPIC = "/ublox_driver/receiver_pvt"     # speed (Doppler) arm
CUE_TOPICS_ALLOWED = (LLA_TOPIC, PVT_TOPIC)
FORBIDDEN_PREFIX = "/dji_osdk_ros/"
REF_TOPICS = ("/dji_osdk_ros/rtk_position", "/dji_osdk_ros/rtk_info_position")  # sealed sub-bag only
REQUIRED_TOPICS = (CAMERA_TOPIC, LLA_TOPIC, PVT_TOPIC, *REF_TOPICS)

# ---- timescale (published constants; never fitted) ----
GPST_MINUS_UTC_S = 18.0
GPS_EPOCH_UNIX_S = 315964800.0  # 1980-01-06T00:00:00Z
SECONDS_PER_WEEK = 604800.0

# ---- primary arm: pooled selection with the R1 minimum rule ----
N_TEST_TARGET = 4
N_TEST_MIN = 2
N_VALIDATION = 2
N_CC = 2
WINDOW_DURATION_S = 16.0
FRAMES_PER_WINDOW = 8

# ---- speed (Doppler) arm ----
PVT_GATE_MEDIAN_MAX_S = 0.25   # |median(t_utc - (record + delta))| < 0.25 s
PVT_GATE_SPREAD_MAX_S = 0.5    # (P99 - P1) of that residual < 0.5 s
SPEED_GAP_MAX_S = 0.5          # an interval is usable iff no sample gap > 0.5 s inside it
COVERAGE_MIN = 0.8             # window valid iff >= 80 % of its 7 intervals are usable
CUE_SIGMA = 1.0                # uniform variance model (as the primary cue)
MAX_DT_CUE_S = 0.06            # constraint frame match (as the primary cue)

# ---- altitude arm ----
ALT_DH_MIN_M = 5.0             # OBSERVABLE iff |dh| >= 5.0 m ...
ALT_RATIO_MIN = 0.2            # ... AND |dv| / ||C_last - C_first|| >= 0.2
ALT_GAP_MAX_S = 0.5            # no altitude interpolation across a gap > 0.5 s
VERTICAL_DH_MIN_M = 10.0       # V-window eligibility |dh_window| >= 10.0 m
VERTICAL_EDGE_S = 1.0          # dh_window = median over [t0+15, t0+16] - median over [t0, t0+1]
N_VERTICAL = 4
ALT_OBSERVABLE = "OBSERVABLE"
ALT_NOT_OBSERVABLE = "NOT_OBSERVABLE"
ALT_FAILURE = "FAILURE"

# ---- success thresholds (R1-6 / R1-7, same as the v1 prereg) ----
LEVERAGE_MIN_M = 0.25
R_MIN = 0.70
REDUCTION_MIN = 0.30
ORACLE_RATIO_MAX = 1.5
MIN_ARM_WINDOWS = 2            # secondary arms: INCONCLUSIVE below 2 valid / observable windows

# ---- roles / outputs ----
ROLE_TEST = "prospective_test"
ROLE_VALIDATION = "validation"
ROLE_CC = "cc_control"
ROLE_VERTICAL = "vertical"
SECTION_OF_ROLE = {ROLE_TEST: "sv_test", ROLE_VALIDATION: "validation", ROLE_CC: "cc_control",
                   ROLE_VERTICAL: "vertical"}
OUT_REL = "artifacts/site_replication_r1"
STAGE1_REL = "artifacts/site_replication_r1/stage1"
PACK_REL = "artifacts/vggt_pack_site_replication_r1"
PACK_ID = "vggt_offbox_site_replication_r1"


class ContractNotRegistered(RuntimeError):
    """The current R1 contract bytes are not recorded in the experiment registry (not frozen)."""


class ForbiddenCueTopic(ValueError):
    """A topic outside the R1 cue allow-list (or any /dji_osdk_ros/* topic) was offered as a cue."""


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def contract_sha(path: Path = CONTRACT_PATH) -> str:
    return sha256_file(Path(path))


def check_registered(path: Path = CONTRACT_PATH, registry: Path | None = REGISTRY_PATH) -> str:
    """Return the contract sha256; raise ContractNotRegistered unless it is recorded in the registry."""
    sha = contract_sha(path)
    reg = Path(registry) if registry is not None else None
    if reg is None or not reg.exists() or sha not in reg.read_text(encoding="utf-8"):
        raise ContractNotRegistered(f"R1 contract sha256 {sha} not recorded in {reg} (contract is not frozen)")
    return sha


def check_cue_topic(topic: str) -> str:
    """Allow-list guard for every cue constructor: /dji_osdk_ros/* (reference-bearing) is always refused."""
    t = str(topic)
    if t.startswith(FORBIDDEN_PREFIX) or "dji_osdk" in t or "rtk_" in t:
        raise ForbiddenCueTopic(f"forbidden cue topic {t!r}: DJI topics may carry RTK reference information")
    if t not in CUE_TOPICS_ALLOWED:
        raise ForbiddenCueTopic(f"cue topic {t!r} is outside the R1 allow-list {CUE_TOPICS_ALLOWED}")
    return t


_POOL_RE = re.compile(r"\{sequence: (\w+), bag_drive_id: ([\w-]+), table_duration_s: (\d+), "
                      r"v_cruise_table_mps: ([0-9.]+), table_altitude_m: (\d+)\}")


def _scalar(text: str, key: str) -> str | None:
    m = re.search(rf'^[ ]*{re.escape(key)}:[ ]*"?([^"#\n]*?)"?[ ]*(?:#.*)?$', text, re.M)
    return m.group(1).strip() if m else None


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    """Parse the fields Stage 1 needs (pool, rank order, status, sha). Does NOT check registration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"R1 contract missing: {path}")
    text = path.read_text(encoding="utf-8")
    if _scalar(text, "name") != EXPERIMENT_NAME:
        raise ValueError("not the R1 contract (name mismatch)")
    block = text.split("  pool:", 1)[1].split("  freshness:", 1)[0]
    pool = [{"sequence": m.group(1), "bag_drive_id": m.group(2), "table_duration_s": int(m.group(3)),
             "v_cruise_table_mps": float(m.group(4)), "table_altitude_m": int(m.group(5))}
            for m in _POOL_RE.finditer(block)]
    if tuple(p["sequence"] for p in pool) != POOL:
        raise ValueError(f"dataset.pool unreadable or not the R1 pool: {pool}")
    if {p["sequence"]: p["v_cruise_table_mps"] for p in pool} != V_CRUISE_TABLE_BY_SEQ:
        raise ValueError("pool cruise speeds != r1_contract.V_CRUISE_TABLE_BY_SEQ")
    return {"sha256": contract_sha(path), "status": _scalar(text, "status"), "pool": pool,
            "rank_order": [p["sequence"] for p in pool], "path": str(path)}


# ---------------------------------------------------------------- freeze pins
# Every repo file whose bytes determine R1 behaviour: the R1 contract + R1 code, the frozen v2/v5 scripts loaded via
# importlib, the frozen library modules they (transitively) import, the package __init__ files executed on import,
# the v5 contract the VGGT pins are asserted against, and the v1 VGGT template pack copied into the R1 pack.
# tests/test_site_replication_r1_cues.py checks (static AST scan) that this covers every repo file the R1 code
# imports or loads through importlib. Record freeze_pins() alongside the contract sha at freeze.
FREEZE_PIN_GLOBS = (
    "yardstick3d/datasets/r1_*.py",
    "scripts/*site_replication_r1*.py",
    "yardstick3d/datasets/v2_*.py",
)
FREEZE_PIN_FILES = (
    "configs/prospective_site_replication_r1.yaml",
    "configs/prospective_cross_backbone_v5.yaml",
    # frozen scripts loaded as private importlib instances (directly or by a loaded script)
    "scripts/cross_backbone_v2_stage1.py",
    "scripts/cross_backbone_v5_stage1.py",
    "scripts/cross_backbone_v2_score.py",
    "scripts/cross_backbone_v2_da3.py",
    "scripts/build_vggt_pack_v2.py",
    "scripts/claim1_gate_metrics_v2.py",
    # v1 VGGT template pack (runner + verify.py loaded by the scorer from the built pack + pinned model lock)
    "vggt_runner/bootstrap.py",
    "vggt_runner/run_vggt.py",
    "vggt_runner/verify.py",
    "vggt_runner/requirements.txt",
    "vggt_runner/model_lock.json",
    # frozen library modules (transitive imports)
    "yardstick3d/datasets/mars_lvig.py",
    "yardstick3d/datasets/timescale.py",
    "yardstick3d/datasets/advio.py",
    "yardstick3d/datasets/advio_cues.py",
    "yardstick3d/datasets/synthetic.py",
    "yardstick3d/optimization/grounder.py",
    "yardstick3d/optimization/scale_solver.py",
    "yardstick3d/experiments/advio_real.py",
    "yardstick3d/io/prediction_cache.py",
    "yardstick3d/types.py",
    "yardstick3d/adapters/base.py",
    "yardstick3d/adapters/da3.py",
    "yardstick3d/constraints/altitude.py",
    "yardstick3d/constraints/base.py",
    "yardstick3d/constraints/baseline.py",
    "yardstick3d/constraints/gnss.py",
    "yardstick3d/constraints/range.py",
    "yardstick3d/constraints/speed.py",
    "yardstick3d/evaluation/alignment.py",
    "yardstick3d/evaluation/leakage.py",
    "yardstick3d/evaluation/oracle.py",
    "yardstick3d/evaluation/scale.py",
    "yardstick3d/evaluation/taxonomy.py",
    "yardstick3d/evaluation/three_limits.py",
    "yardstick3d/evaluation/trajectory.py",
    "yardstick3d/geometry/cameras.py",
    "yardstick3d/geometry/rotations.py",
    "yardstick3d/geometry/sim3.py",
    "yardstick3d/geometry/trajectory.py",
    "yardstick3d/observability/metrics.py",
    "yardstick3d/sensors/geodesy.py",
    "yardstick3d/sensors/speed_integrate.py",
    "yardstick3d/uncertainty/robust.py",
    # imported by the package __init__ files below (executed on any yardstick3d.* import)
    "yardstick3d/adapters/dummy.py",
    "yardstick3d/adapters/vggt.py",
    "yardstick3d/constraints/camera_height.py",
    "yardstick3d/constraints/gravity.py",
    "yardstick3d/constraints/object_size.py",
    "yardstick3d/constraints/registry.py",
    "yardstick3d/experiments/synthetic_benchmark.py",
    "yardstick3d/geometry/se3.py",
    "yardstick3d/optimization/sim3_solver.py",
    "yardstick3d/visualization/plots.py",
    # package __init__ files executed on import
    "yardstick3d/__init__.py",
    "yardstick3d/adapters/__init__.py",
    "yardstick3d/constraints/__init__.py",
    "yardstick3d/datasets/__init__.py",
    "yardstick3d/evaluation/__init__.py",
    "yardstick3d/experiments/__init__.py",
    "yardstick3d/geometry/__init__.py",
    "yardstick3d/io/__init__.py",
    "yardstick3d/observability/__init__.py",
    "yardstick3d/optimization/__init__.py",
    "yardstick3d/sensors/__init__.py",
    "yardstick3d/uncertainty/__init__.py",
    "yardstick3d/visualization/__init__.py",
)


def freeze_pin_paths(root: Path = ROOT) -> list[str]:
    root = Path(root)
    rel = set(FREEZE_PIN_FILES)
    for g in FREEZE_PIN_GLOBS:
        rel |= {p.relative_to(root).as_posix() for p in root.glob(g) if p.is_file()}
    return sorted(rel)


def freeze_pins(root: Path = ROOT) -> dict[str, str]:
    """{repo-relative path: sha256} of every file R1 behaviour depends on. A listed file that is missing raises."""
    root = Path(root)
    pins = {}
    for rel in freeze_pin_paths(root):
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(f"freeze pin missing: {rel}")
        pins[rel] = sha256_file(p)
    return pins


def verify_freeze_pins(recorded: dict, root: Path = ROOT) -> list[str]:
    """Paths whose current sha256 differs from (or is absent in) a recorded freeze_pins() map; [] = all match."""
    now = freeze_pins(root)
    return sorted(k for k in set(now) | set(recorded) if now.get(k) != recorded.get(k))
