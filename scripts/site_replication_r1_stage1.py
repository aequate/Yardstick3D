"""Site replication R1 - host-runnable Stage 0/1 (contract: configs/prospective_site_replication_r1.yaml).

Derived from scripts/cross_backbone_v5_stage1.py, whose helpers are IMPORTED (importlib) rather than copied:
v4s.read_streams / census_stage1 / window_frames (UNCHANGED v5 gates + time_sync checks + window statistics),
s5.census_record / stats_from_census / copy_reference_topics (raw-byte reference copy, never deserialized).

Phases (each refuses a silent rerun; all outputs under <root>/artifacts/site_replication_r1/stage1):
  stage0   --sequence SEQ [--bag PATH] [--unavailable REASON]
           byte size + sha256 + rosbags index topic list, BEFORE any message is deserialized.
  census   --sequence SEQ [--bag PATH]
           v5 gates (STOP => R1 STOP), receiver_lla cue CSV (incl. altitude), receiver_pvt decode via the
           bag-embedded msgdef -> pvt_<SEQ>.csv + PVT timescale gate (FAIL => speed arm NOT RUN for SEQ),
           per-window statistics + vertical census (dh_window). Never reads a reference value.
  select   pooled primary selection with the R1 minimum rule + vertical windows -> windows.json
           (or selection_STOP.json, exit 2).
  frames   --sequence SEQ [--bag PATH]   selected frames -> frames/<SEQ>/<bag idx>.png + frames_<SEQ>.json
  refbag   --sequence SEQ [--bag PATH]   reference topics copied raw (never deserialized) -> ref_<SEQ>.bag
  finalize stage1_manifest.json with the sha256 of every Stage-1 output.

Refuses unless the contract sha256 is in configs/experiment_registry.json. --dry-contract-check skips that
check for SYNTHETIC test bags only: outputs are marked dry_run; the real output directory, any bag whose name
contains a pool sequence (case-insensitive), any bag under data/ and any bag > 50 MiB are refused; the scorer
and pack builder refuse dry-run manifests. No /dji_osdk_ros/* topic is ever a cue; the reference topics are only
header-parsed (bytes [0:12], v5 rtk recorder-clock gate) and raw-copied.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yardstick3d.datasets import r1_contract as r1c  # noqa: E402
from yardstick3d.datasets import r1_cues as cues  # noqa: E402
from yardstick3d.datasets import r1_selection as r1s  # noqa: E402
from yardstick3d.datasets import v2_scoring as v2  # noqa: E402
from yardstick3d.datasets import v2_sync_guards as sg  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v4s = _load("r1_v4s_stage1", "scripts/cross_backbone_v2_stage1.py")   # frozen; imported, never edited
s5 = _load("r1_v5_stage1", "scripts/cross_backbone_v5_stage1.py")      # frozen; imported, never edited
if tuple(s5.REF_TOPICS) != r1c.REF_TOPICS or v4s.CAMERA_TOPIC != r1c.CAMERA_TOPIC or v4s.CUE_TOPIC != r1c.LLA_TOPIC:
    raise RuntimeError("frozen v5 Stage-1 topics differ from the R1 contract topics")

CONTRACT = r1c.CONTRACT_PATH
REGISTRY = r1c.REGISTRY_PATH
PRIMARY_TOPICS = (r1c.CAMERA_TOPIC, r1c.LLA_TOPIC, *r1c.REF_TOPICS)  # absent => Stage-0 STOP (as v5)

sha256_file = v4s.sha256_file
dump = v4s.dump
DRY_BAG_MAX_BYTES = 50 * 2**20  # --dry-contract-check: synthetic test bags only (a real R1 bag is several GB)
RETRY_MIN_S = 24 * 3600.0         # contract dataset.stage0: at most one retry per id after >= 24 h


def _utcnow() -> float:
    """Wall-clock UTC seconds (module-level so tests can pin the retry clock)."""
    import time

    return time.time()


class Ctx:
    """Run context: output dir, contract handle, dry-run flag."""

    def __init__(self, root: Path = ROOT, dry: bool = False, contract: Path = CONTRACT, registry: Path = REGISTRY):
        self.root = Path(root)
        self.out = self.root / r1c.STAGE1_REL
        self.dry = bool(dry)
        c = r1c.load_contract(contract)
        if self.dry:
            if self.out.resolve() == (ROOT / r1c.STAGE1_REL).resolve():
                raise SystemExit("--dry-contract-check refuses the real R1 output directory (synthetic bags only)")
            c["registered"] = False
        else:
            try:
                r1c.check_registered(contract, registry)
            except r1c.ContractNotRegistered as exc:
                raise SystemExit(f"refusing: {exc}") from exc
            c["registered"] = True
        self.c = c

    @property
    def sha(self) -> str:
        return self.c["sha256"]

    def stamp(self) -> dict:
        return {"contract_sha256": self.sha, "dry_run": self.dry}

    def bag(self, seq: str, bag: Path | None) -> Path:
        p = Path(bag) if bag else self.root / "data/mars_lvig" / f"{seq}.bag"
        if self.dry:
            self.refuse_real_bag(p)
        return p

    @staticmethod
    def refuse_real_bag(p: Path) -> None:
        """--dry-contract-check must never touch a real R1 bag before freeze (contract header: freeze BEFORE any R1
        bag byte is read). Name/path and size checks only (stat, no byte read)."""
        name = p.name.lower()
        for seq in r1c.POOL:
            if seq.lower() in name:
                raise SystemExit(f"--dry-contract-check refuses {p.name}: name matches pool sequence {seq} "
                                 "(synthetic bags only)")
        if p.resolve().is_relative_to((ROOT / "data").resolve()):
            raise SystemExit(f"--dry-contract-check refuses a bag under data/: {p}")
        if p.exists() and p.stat().st_size > DRY_BAG_MAX_BYTES:
            raise SystemExit(f"--dry-contract-check refuses {p.name}: {p.stat().st_size} bytes > "
                             f"{DRY_BAG_MAX_BYTES} (synthetic bags only)")

    def check_seq(self, seq: str) -> None:
        if seq not in self.c["rank_order"]:
            raise SystemExit(f"{seq} is not in the R1 pool {self.c['rank_order']}")

    def read(self, name: str) -> dict:
        rec = json.loads((self.out / name).read_text(encoding="utf-8"))
        if rec.get("contract_sha256") != self.sha or bool(rec.get("dry_run")) != self.dry:
            raise SystemExit(f"{name}: contract sha / dry-run flag mismatch")
        return rec


def _refuse_existing(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            raise SystemExit(f"{p.name} exists; refusing silent rerun")


# --------------------------------- stage 0 -----------------------------------
def _attempts_path(ctx: Ctx, seq: str) -> Path:
    return ctx.out / f"stage0_attempts_{seq}.json"


def _record_attempt(ctx: Ctx, seq: str, bag: Path, error: str) -> list:
    p = _attempts_path(ctx, seq)
    rec = ctx.read(p.name) if p.exists() else {**ctx.stamp(), "sequence": seq, "attempts": []}
    rec["attempts"].append({"utc_s": _utcnow(), "bag_name": bag.name, "error": error})
    dump(p, rec)
    return rec["attempts"]


def stage0(ctx: Ctx, seq: str, bag: Path | None = None, unavailable: str | None = None) -> int:
    """Bytes + sha256 + index topic list BEFORE any message is deserialized.

    Access is ALWAYS attempted. A failed attempt is logged (stage0_attempts_<SEQ>.json, exit 4). UNAVAILABLE
    (--unavailable REASON) is recorded only when the bag is still inaccessible on a retry >= 24 h after the first
    failed attempt (contract dataset.stage0), and never once any census output exists.
    """
    ctx.check_seq(seq)
    out = ctx.out
    bag = ctx.bag(seq, bag)  # dry-run bag guard first (no byte read)
    _refuse_existing(out / f"stage0_{seq}.json", out / f"stage0_UNAVAILABLE_{seq}.json", out / f"stage0_STOP_{seq}.json")
    if unavailable and any(out.glob("census_*.json")):
        raise SystemExit("--unavailable refused: census output already exists (stage0 decisions precede any census)")
    from rosbags.rosbag1 import Reader

    error = None
    if not bag.exists():
        error = f"bag missing: {bag.name}"
    else:
        try:
            size, digest = bag.stat().st_size, sha256_file(bag)
            with Reader(bag) as reader:  # index/connection records only; no message is read
                topics = {t: {"msgtype": info.msgtype, "msgcount": int(info.msgcount)}
                          for t, info in reader.topics.items()}
        except Exception as exc:  # noqa: BLE001 - any access failure is logged as an attempt
            error = f"{type(exc).__name__}: {exc}"
    if error is not None:
        attempts = _record_attempt(ctx, seq, bag, error)
        if not unavailable:
            print(f"stage0 {seq}: ACCESS FAILED ({error}); attempt {len(attempts)} logged; retry after >= 24 h")
            return 4
        span = attempts[-1]["utc_s"] - attempts[0]["utc_s"]
        if len(attempts) < 2 or span < RETRY_MIN_S:
            raise SystemExit(f"--unavailable refused: needs a failed retry >= 24 h after the first failed attempt "
                             f"({len(attempts)} attempt(s), span {span:.0f} s)")
        dump(out / f"stage0_UNAVAILABLE_{seq}.json", {
            **ctx.stamp(), "sequence": seq, "status": "UNAVAILABLE", "reason": unavailable,
            "access_failure": error, "attempts": attempts,
            "attempts_sha256": sha256_file(_attempts_path(ctx, seq)),
            "rule": "R1 runs on the remaining bag (disclosed); minimum-window rule applies"})
        print(f"stage0 {seq}: UNAVAILABLE ({unavailable}; {error})")
        return 0
    if unavailable:
        raise SystemExit(f"--unavailable refused: {bag.name} is accessible (stage0 attempt succeeded)")
    missing = [t for t in r1c.REQUIRED_TOPICS if t not in topics]
    rec = {**ctx.stamp(), "sequence": seq, "bag_name": bag.name, "bag_bytes": size, "bag_sha256": digest,
           "topics": dict(sorted(topics.items())), "required_topics": list(r1c.REQUIRED_TOPICS),
           "missing_required_topics": missing, "messages_deserialized": False}
    missing_primary = [t for t in PRIMARY_TOPICS if t in missing]
    if missing_primary:
        dump(out / f"stage0_STOP_{seq}.json", {**rec, "status": "STOP",
                                                "reason": f"primary-arm topics absent: {missing_primary}"})
        print(f"STOP stage0 {seq}: {missing_primary}")
        return 2
    rec["status"] = "OK"
    rec["speed_arm_topic_present"] = r1c.PVT_TOPIC in topics
    dump(out / f"stage0_{seq}.json", rec)
    print(f"stage0 {seq}: {size} bytes sha256 {digest[:12]}..  missing={missing}")
    return 0


def require_stage0_decisions(ctx: Ctx) -> dict:
    """Census runs only after EVERY pool sequence has a Stage-0 decision: OK, or UNAVAILABLE with a recorded
    access failure and retry. A Stage-0 STOP on any pool sequence STOPS R1 (no census)."""
    out, dec = ctx.out, {}
    for s in ctx.c["rank_order"]:
        if (out / f"stage0_STOP_{s}.json").exists():
            raise SystemExit(f"stage0 STOP recorded for {s}: R1 STOPS; census refused")
        if (out / f"stage0_{s}.json").exists():
            if ctx.read(f"stage0_{s}.json").get("status") != "OK":
                raise SystemExit(f"stage0_{s}.json status != OK")
            dec[s] = "OK"
        elif (out / f"stage0_UNAVAILABLE_{s}.json").exists():
            u = ctx.read(f"stage0_UNAVAILABLE_{s}.json")
            if not u.get("access_failure") or len(u.get("attempts", [])) < 2:
                raise SystemExit(f"stage0_UNAVAILABLE_{s}.json lacks a recorded access failure / retry")
            if sha256_file(_attempts_path(ctx, s)) != u.get("attempts_sha256"):
                raise SystemExit(f"stage0 attempt log for {s} changed after the UNAVAILABLE decision")
            dec[s] = "UNAVAILABLE"
        else:
            raise SystemExit(f"census refused: no Stage-0 decision for pool sequence {s} (run stage0 for every "
                             "pool sequence first)")
    return dec


def verify_bag_stage0(ctx: Ctx, seq: str, bag: Path) -> str:
    s0 = ctx.read(f"stage0_{seq}.json")
    if not bag.exists():
        raise SystemExit(f"bag missing: {bag}")
    if bag.stat().st_size != s0["bag_bytes"]:
        raise SystemExit(f"{bag.name}: size != Stage-0 record")
    digest = sha256_file(bag)
    if digest != s0["bag_sha256"]:
        raise SystemExit(f"{bag.name}: sha256 {digest} != Stage-0 {s0['bag_sha256']}")
    return digest


# --------------------------------- PVT decode --------------------------------
def _field(m, *names):
    for n in names:
        if hasattr(m, n):
            return getattr(m, n)
    raise AttributeError(f"none of {names} in {type(m).__name__}")


def decode_pvt(reader, store) -> dict:
    """receiver_pvt -> {record_t, week, tow, vel_n, vel_e, vel_d} decoded with the bag-embedded msgdef.

    Robust to the nested gnss_comm/GnssTimeMsg `time` field (time.week / time.tow) and to a flat layout.
    Never raises: an undecodable topic returns {"decoded": False, reason} (=> speed arm NOT RUN).
    """
    r1c.check_cue_topic(r1c.PVT_TOPIC)
    conns = [x for x in reader.connections if x.topic == r1c.PVT_TOPIC]
    if not conns:
        return {"decoded": False, "reason": f"{r1c.PVT_TOPIC} absent"}
    cols = {k: [] for k in ("record_t", "week", "tow", "vel_n", "vel_e", "vel_d")}
    try:
        v4s._register_bag_types(store, conns)
        for conn, rec, raw in reader.messages(connections=conns):
            m = store.deserialize_ros1(raw, conn.msgtype)
            tm = getattr(m, "time", None)
            src = tm if tm is not None and hasattr(tm, "week") else m
            cols["record_t"].append(rec / 1e9)
            cols["week"].append(float(_field(src, "week")))
            cols["tow"].append(float(_field(src, "tow")))
            cols["vel_n"].append(float(_field(m, "vel_n")))
            cols["vel_e"].append(float(_field(m, "vel_e")))
            cols["vel_d"].append(float(_field(m, "vel_d")))
    except Exception as exc:  # noqa: BLE001 - any decode failure disables ONLY the speed arm
        return {"decoded": False, "reason": f"undecodable ({type(exc).__name__}: {exc})",
                "msgtype": conns[0].msgtype}
    out = {k: np.asarray(v, float) for k, v in cols.items()}
    out.update({"decoded": True, "msgtype": conns[0].msgtype, "n": int(out["record_t"].size)})
    if out["n"] == 0:
        return {"decoded": False, "reason": "no messages", "msgtype": conns[0].msgtype}
    return out


def write_pvt_csv(path: Path, record_t, t_utc, vn, ve, vd) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["record_t", "t_utc", "vel_n", "vel_e", "vel_d"])
        for row in zip(record_t, t_utc, vn, ve, vd):
            wr.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}"])


def load_pvt_csv(path: Path) -> dict:
    cols = {k: [] for k in ("record_t", "t_utc", "vel_n", "vel_e", "vel_d")}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.asarray(v, float) for k, v in cols.items()}


# ---------------------------------- census -----------------------------------
def vertical_census(stats, cue_t, alt, t_ref: float) -> list:
    """GT-free V-class inputs per grid window (frames matched, coverage, valid_frac, dh_window)."""
    return [{"k": int(w.k), "t0": float(w.t0), "frames_matched": bool(w.frames_matched),
             "cue_coverage": float(w.cue_coverage), "valid_frac": float(w.valid_frac),
             "dh_window": cues.vertical_dh_window(cue_t, alt, t_ref + float(w.t0))} for w in stats]


def census(ctx: Ctx, seq: str, bag: Path | None = None) -> int:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    ctx.check_seq(seq)
    out = ctx.out
    bag = ctx.bag(seq, bag)  # dry-run bag guard first (no byte read)
    _refuse_existing(out / f"census_{seq}.json", out / f"census_STOP_{seq}.json")
    if require_stage0_decisions(ctx)[seq] != "OK":
        raise SystemExit(f"{seq} is recorded UNAVAILABLE at Stage 0; census refused")
    digest = verify_bag_stage0(ctx, seq, bag)
    r1c.check_cue_topic(r1c.LLA_TOPIC)
    store = get_typestore(Stores.ROS1_NOETIC)
    try:
        with Reader(bag) as reader:
            conns = {x.topic: x for x in reader.connections}
            for t in v4s.GATE_TOPICS:
                if t not in conns:
                    raise sg.Stage1Stop(f"required topic absent from bag index: {t}")
            cam_h, cam_r, rtk_h, rtk_r, ch, cr, llh, sync_meta = v4s.read_streams(reader, store, conns)
            pvt = decode_pvt(reader, store)
        cen = v4s.census_stage1(cam_h, cam_r, ch, cr, llh, rtk_h, rtk_r)
    except sg.Stage1Stop as exc:
        dump(out / f"census_STOP_{seq}.json", {**ctx.stamp(), "sequence": seq, "status": "STOP", "reason": str(exc),
                                                "r1_consequence": "R1 STOPS (no fallback)", "bag_sha256": digest,
                                                "reference_values_read": False})
        print(f"STOP ({seq}): {exc}")
        return 2
    rec = s5.census_record(cen)
    delta = float(cen["gate"].recorder_offset_s)
    # primary cue CSV (incl. altitude for the altitude arm)
    cue_csv = out / v2.cue_csv_name(seq)
    cue_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(cue_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_utc", "lat_deg", "lon_deg", "alt_m"])
        for t, (la, lo, al) in zip(cen["cue_t"], cen["llh"]):
            wr.writerow([f"{t:.6f}", f"{la:.9f}", f"{lo:.9f}", f"{al:.4f}"])
    # speed arm: PVT decode + timescale gate (published 18.0 s; never fitted)
    pvt_rec = {"topic": r1c.PVT_TOPIC, "decoded": bool(pvt.get("decoded")), "msgtype": pvt.get("msgtype")}
    if pvt.get("decoded"):
        t_utc = cues.gnss_week_tow_to_utc(pvt["week"], pvt["tow"])
        gate = cues.pvt_timescale_gate(t_utc, pvt["record_t"], delta)
        pvt_csv = out / f"pvt_{seq}.csv"
        write_pvt_csv(pvt_csv, pvt["record_t"], t_utc, pvt["vel_n"], pvt["vel_e"], pvt["vel_d"])
        spd = cues.pvt_speed_3d(pvt["vel_n"], pvt["vel_e"], pvt["vel_d"])
        pvt_rec.update({"n": pvt["n"], "n_finite_speed": int(np.isfinite(spd).sum()), "gate": gate,
                        "csv": pvt_csv.name, "csv_sha256": sha256_file(pvt_csv),
                        "speed_arm": "RUN" if gate["status"] == "PASS" else "NOT_RUN",
                        "gps_minus_utc_s": r1c.GPST_MINUS_UTC_S})
    else:
        pvt_rec.update({"reason": pvt.get("reason"), "gate": None, "speed_arm": "NOT_RUN"})
    alt = np.asarray(cen["llh"], float)[:, 2]
    rec["vertical_census"] = vertical_census(cen["stats"], cen["cue_t"], alt, float(cen["image_t"][0]))
    dump(out / f"census_{seq}.json", {
        **ctx.stamp(), "sequence": seq, "bag_bytes": bag.stat().st_size, "bag_sha256": digest,
        "time_sync_metadata": sync_meta, "reference_values_read": False,
        "rtk_position_access": "header-only raw bytes [0:12]",
        "cue_topic": r1c.LLA_TOPIC, "cue_csv": cue_csv.name, "cue_csv_sha256": sha256_file(cue_csv),
        "pvt": pvt_rec, **rec,
    })
    print(f"census {seq}: {rec['class_counts']}  delta={delta:+.3f} s  speed_arm={pvt_rec['speed_arm']}")
    return 0


# ---------------------------------- select -----------------------------------
def select(ctx: Ctx) -> int:
    out = ctx.out
    _refuse_existing(out / "windows.json", out / "selection_STOP.json")
    rank_all = ctx.c["rank_order"]

    def stop(reason: str) -> int:
        dump(out / "selection_STOP.json", {**ctx.stamp(), "status": "STOP", "reason": reason,
                                           "reference_values_read": False})
        print(f"STOP: {reason}")
        return 2

    avail, unavailable, censuses = [], {}, {}
    for seq in rank_all:
        if (out / f"stage0_UNAVAILABLE_{seq}.json").exists():
            unavailable[seq] = ctx.read(f"stage0_UNAVAILABLE_{seq}.json")["reason"]
            continue
        for f in (f"stage0_STOP_{seq}.json", f"census_STOP_{seq}.json"):
            if (out / f).exists():
                return stop(f"{f} recorded: a Stage-0/1 STOP on any pool sequence STOPS R1")
        if not (out / f"census_{seq}.json").exists():
            raise SystemExit(f"census missing for {seq}")
        rec = ctx.read(f"census_{seq}.json")
        if rec["reference_values_read"] is not False:
            raise SystemExit(f"census {seq}: attestation mismatch")
        if sha256_file(out / rec["cue_csv"]) != rec["cue_csv_sha256"]:
            raise SystemExit(f"census {seq}: cue csv hash mismatch")
        if rec["pvt"].get("csv") and sha256_file(out / rec["pvt"]["csv"]) != rec["pvt"]["csv_sha256"]:
            raise SystemExit(f"census {seq}: pvt csv hash mismatch")
        avail.append(seq)
        censuses[seq] = rec
    if not avail:
        return stop("no pool bag available")
    sel = r1s.select_r1({s: s5.stats_from_census(censuses[s]) for s in avail}, avail)
    if sel.status != "OK":
        return stop(sel.reason)
    wins = []
    for s in sel.starts:
        fr = censuses[s["sequence"]]["candidate_frames"][str(s["k"])]
        wins.append({**s, **fr, "window_id": f"{s['sequence']}:k{s['k']}"})
    vert = r1s.select_vertical({s: censuses[s]["vertical_census"] for s in avail}, avail,
                               [(w["sequence"], w["k"]) for w in wins])
    for v in vert:
        fr = censuses[v["sequence"]]["candidate_frames"][str(v["k"])]
        wins.append({**v, **fr, "window_id": f"{v['sequence']}:k{v['k']}"})
    dump(out / "windows.json", {
        **ctx.stamp(), "rank_order": rank_all, "sequences_available": avail, "sequences_unavailable": unavailable,
        "n_test": sel.n_test, "n_test_target": r1c.N_TEST_TARGET, "minimum_rule_applied": sel.minimum_rule_applied,
        "n_sv_candidates_pooled": sel.n_sv_candidates, "n_vertical": len(vert),
        "speed_arm_by_sequence": {s: censuses[s]["pvt"]["speed_arm"] for s in avail},
        "census_sha256": {s: sha256_file(out / f"census_{s}.json") for s in avail},
        "t_ref_first_image_utc": {s: censuses[s]["t_ref_first_image_utc"] for s in avail},
        "windows": wins,
    })
    for w in wins:
        print(f"{w['role']:17s} {w['window_id']:24s} cls={w['cls']}")
    return 0


# ------------------------------ frames / refbag ------------------------------
def frames(ctx: Ctx, seq: str, bag: Path | None = None) -> int:
    import cv2
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    ctx.check_seq(seq)
    out = ctx.out
    rec_path = out / f"frames_{seq}.json"
    bag = ctx.bag(seq, bag)  # dry-run bag guard first (no byte read)
    _refuse_existing(rec_path)
    wj = ctx.read("windows.json")
    digest = verify_bag_stage0(ctx, seq, bag)
    want = {int(i): f"{int(i):08d}.png" for w in wj["windows"] if w["sequence"] == seq for i in w["frame_indices_bag"]}
    fdir = out / "frames" / seq
    fdir.mkdir(parents=True, exist_ok=True)
    store = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as reader:
        cn = [x for x in reader.connections if x.topic == r1c.CAMERA_TOPIC][0]
        for k, (_, _, raw) in enumerate(reader.messages(connections=[cn])):
            if k not in want:
                continue
            m = store.deserialize_ros1(raw, cn.msgtype)
            arr = cv2.imdecode(np.frombuffer(m.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise RuntimeError(f"failed to decode image {k}")
            cv2.imwrite(str(fdir / want[k]), arr)
    missing = [n for n in want.values() if not (fdir / n).exists()]
    if missing:
        raise RuntimeError(f"frames not written: {missing[:3]}")
    dump(rec_path, {**ctx.stamp(), "sequence": seq, "bag_sha256": digest,
                    "windows_json_sha256": sha256_file(out / "windows.json"),
                    "frame_sha256": {f"{seq}/{n}": sha256_file(fdir / n) for n in sorted(want.values())}})
    print(f"frames {seq}: {len(want)} images")
    return 0


def refbag(ctx: Ctx, seq: str, bag: Path | None = None) -> int:
    """Reference topics copied byte-for-byte (v5 copy_reference_topics; NEVER deserialized)."""
    ctx.check_seq(seq)
    out = ctx.out
    dst, rec_path = out / f"ref_{seq}.bag", out / f"refbag_{seq}.json"
    bag = ctx.bag(seq, bag)  # dry-run bag guard first (no byte read)
    _refuse_existing(dst, rec_path)
    digest = verify_bag_stage0(ctx, seq, bag)
    counts = s5.copy_reference_topics(bag, dst)
    dump(rec_path, {**ctx.stamp(), "sequence": seq, "source_bag_sha256": digest, "sub_bag": dst.name,
                    "sub_bag_bytes": dst.stat().st_size, "sub_bag_sha256": sha256_file(dst), "topics": counts,
                    "deserialized": False,
                    "rule": "opened ONLY by the one-shot R1 scorer after SCORING_RECEIPT.json + its sha256 file"})
    print(f"refbag {seq}: {counts}")
    return 0


# --------------------------------- finalize ----------------------------------
def finalize(ctx: Ctx) -> int:
    out = ctx.out
    man = out / "stage1_manifest.json"
    _refuse_existing(man)
    wj = ctx.read("windows.json")
    wj_sha = sha256_file(out / "windows.json")
    used = sorted({w["sequence"] for w in wj["windows"]}, key=ctx.c["rank_order"].index)
    per_seq, frame_sha = {}, {}
    for seq in wj["sequences_available"]:
        cen = ctx.read(f"census_{seq}.json")
        if sha256_file(out / f"census_{seq}.json") != wj["census_sha256"][seq]:
            raise SystemExit(f"census {seq} changed after selection")
        info = {"census_sha256": wj["census_sha256"][seq], "bag_sha256": cen["bag_sha256"], "gate": cen["gate"],
                "t_ref_first_image_utc": cen["t_ref_first_image_utc"], "class_counts": cen["class_counts"],
                "cue_csv": cen["cue_csv"], "cue_csv_sha256": cen["cue_csv_sha256"], "pvt": cen["pvt"],
                "v_cruise_table_mps": r1c.V_CRUISE_TABLE_BY_SEQ[seq]}
        if seq in used:
            fr = ctx.read(f"frames_{seq}.json")
            if fr["windows_json_sha256"] != wj_sha:
                raise SystemExit(f"frames_{seq}.json was built from a different windows.json")
            for name, dg in fr["frame_sha256"].items():
                if sha256_file(out / "frames" / name) != dg:
                    raise SystemExit(f"frame hash mismatch {name}")
            frame_sha.update(fr["frame_sha256"])
            if not (out / f"refbag_{seq}.json").exists():
                raise SystemExit(f"sealed reference sub-bag missing for {seq} (R1 requires one per used sequence)")
            r = ctx.read(f"refbag_{seq}.json")
            if sha256_file(out / r["sub_bag"]) != r["sub_bag_sha256"] or r["deserialized"] is not False:
                raise SystemExit(f"ref sub-bag {seq} hash/attestation mismatch")
            if r["source_bag_sha256"] != cen["bag_sha256"]:
                raise SystemExit(f"ref sub-bag {seq} was cut from a different bag")
            info["reference_source"] = {"kind": "sealed_sub_bag", "path": r["sub_bag"], "sha256": r["sub_bag_sha256"]}
        per_seq[seq] = info
    need = {f"{w['sequence']}/{int(i):08d}.png" for w in wj["windows"] for i in w["frame_indices_bag"]}
    if need != set(frame_sha):
        raise SystemExit("frame set != selected windows' frames")
    files = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file() and p != man}
    dump(man, {**ctx.stamp(), "experiment": r1c.EXPERIMENT_NAME, "rank_order": ctx.c["rank_order"],
               "sequences_used": used, "sequences_unavailable": wj["sequences_unavailable"],
               "n_test": wj["n_test"], "minimum_rule_applied": wj["minimum_rule_applied"],
               "reference_values_read": False, "rtk_position_access": "header-only raw bytes [0:12]",
               "windows_json_sha256": wj_sha, "sequences": per_seq, "frame_sha256": dict(sorted(frame_sha.items())),
               "output_sha256": files, "windows": wj["windows"]})
    print(f"stage1 R1 manifest: {len(wj['windows'])} windows over {used}; {len(files)} files hashed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["stage0", "census", "select", "frames", "refbag", "finalize"])
    ap.add_argument("--sequence")
    ap.add_argument("--bag", type=Path)
    ap.add_argument("--root", type=Path, default=ROOT, help="outputs go to <root>/artifacts/site_replication_r1/stage1")
    ap.add_argument("--unavailable", help="stage0 only: record the bag as UNAVAILABLE with this reason (requires a "
                    "failed access attempt now AND one logged >= 24 h earlier; refused once any census exists)")
    ap.add_argument("--dry-contract-check", action="store_true",
                    help="TESTS WITH SYNTHETIC BAGS ONLY: skip the registry check (outputs marked dry_run)")
    a = ap.parse_args(argv)
    if a.phase in ("stage0", "census", "frames", "refbag") and not a.sequence:
        ap.error("--sequence required")
    ctx = Ctx(a.root, a.dry_contract_check)
    if a.phase == "stage0":
        return stage0(ctx, a.sequence, a.bag, a.unavailable)
    if a.phase == "census":
        return census(ctx, a.sequence, a.bag)
    if a.phase == "select":
        return select(ctx)
    if a.phase == "frames":
        return frames(ctx, a.sequence, a.bag)
    if a.phase == "refbag":
        return refbag(ctx, a.sequence, a.bag)
    return finalize(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
