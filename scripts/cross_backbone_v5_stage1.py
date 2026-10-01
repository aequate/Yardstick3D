"""Cross-backbone v5 - pooled Stage 1 (contract: configs/prospective_cross_backbone_v5.yaml, amendment A3).

Phases (each refuses a silent rerun):
  census   --sequence SEQ [--bag PATH]  per pool sequence: UNCHANGED v4 gates + time_sync checks + per-window
                                        statistics (no selection). Writes census_<SEQ>.json + cue_<SEQ>.csv, or
                                        census_STOP_<SEQ>.json (=> v5 STOP) and exit 2.
  select                                pooled role assignment (ws.select_windows_pooled) over the 3 censuses
                                        -> windows.json, or selection_STOP.json (=> v5 STOP) and exit 2.
  frames   --sequence SEQ [--bag PATH]  decode that sequence's selected frames -> frames/<SEQ>/<bag idx>.png
                                        + frames_<SEQ>.json (sha256 per file).
  refbag   --sequence SEQ [--bag PATH]  remote-held bag only: copy the reference topics byte-for-byte
                                        (raw bytes + record time, NEVER deserialized) into ref_<SEQ>.bag.
  finalize                              stage1_manifest.json binding censuses, selection, frames, sub-bags.

Stage 1 never reads rtk_position values (header bytes [0:12] only) or any other forbidden topic value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yardstick3d.datasets import v2_sync_guards as sg  # noqa: E402
from yardstick3d.datasets import v2_window_selection as ws  # noqa: E402

_spec = importlib.util.spec_from_file_location("cb_v2_stage1", ROOT / "scripts/cross_backbone_v2_stage1.py")
v4s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4s)

CONTRACT = ROOT / "configs/prospective_cross_backbone_v5.yaml"
REGISTRY = ROOT / "configs/experiment_registry.json"
BAG_DIR = ROOT / "data/mars_lvig"
OUT = ROOT / "artifacts/cross_backbone_v5/stage1"
REF_TOPICS = ("/dji_osdk_ros/rtk_position", "/dji_osdk_ros/rtk_info_position")

dump = v4s.dump
sha256_file = v4s.sha256_file


# ------------------------------- contract ---------------------------------
def load_contract(path: Path = CONTRACT, registry: Path = REGISTRY) -> dict:
    """Frozen + registered v5 contract -> {sha256, pool: [{sequence, bag_bytes, bag_sha256, v_cruise_table_mps}]}."""
    if not path.exists():
        raise SystemExit(f"contract missing: {path}")
    text = path.read_text(encoding="utf-8")
    if v4s._yaml_scalar(text, "status") != "frozen_pre_data":
        raise SystemExit("v5 contract is not frozen_pre_data")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if not registry.exists() or sha not in registry.read_text(encoding="utf-8"):
        raise SystemExit(f"contract sha256 {sha} not recorded in {registry}")
    if "amendment_A3_pooled_sv:" not in text:
        raise SystemExit("amendment_A3_pooled_sv missing (fail closed)")
    pool = []
    block = text.split("  pool:", 1)[1].split("  bag_sha256:", 1)[0]
    for m in re.finditer(r"\{sequence: (\w+), bag_drive_id: [\w-]+, bag_bytes: (\d+), bag_sha256: ([0-9a-f]{64}), "
                         r"v_cruise_table_mps: ([0-9.]+)\}", block):
        pool.append({"sequence": m.group(1), "bag_bytes": int(m.group(2)), "bag_sha256": m.group(3),
                     "v_cruise_table_mps": float(m.group(4))})
    if [p["sequence"] for p in pool] != ["HKisland_GNSS02", "HKisland_GNSS03", "HKisland_GNSS01"]:
        raise SystemExit(f"dataset.pool unreadable or not the frozen pool: {pool}")
    return {"sha256": sha, "pool": pool, "rank_order": [p["sequence"] for p in pool]}


def pool_entry(c: dict, seq: str) -> dict:
    for p in c["pool"]:
        if p["sequence"] == seq:
            return p
    raise SystemExit(f"{seq} is not in the frozen v5 pool")


def verify_bag(entry: dict, bag: Path) -> str:
    """Size + sha256 must equal the frozen Stage-0 values (identity of the exact bytes)."""
    if not bag.exists():
        raise SystemExit(f"bag missing: {bag}")
    if bag.stat().st_size != entry["bag_bytes"]:
        raise SystemExit(f"{bag.name}: size {bag.stat().st_size} != frozen {entry['bag_bytes']}")
    digest = sha256_file(bag)
    if digest != entry["bag_sha256"]:
        raise SystemExit(f"{bag.name}: sha256 {digest} != frozen {entry['bag_sha256']}")
    return digest


# ---------------------------- census (pure) --------------------------------
def census_record(c: dict) -> dict:
    """JSON-able census from v4s.census_stage1 output: gate, stats, frames for every frame-matched candidate."""
    gate, image_t, stats = c["gate"], c["image_t"], c["stats"]
    frames = {str(w.k): v4s.window_frames(c, w.t0) for w in stats if w.frames_matched}
    return {
        "gate": {"classes": gate.classes, "stats": gate.stats, "cross_topic_max_diff_s": gate.cross_topic_max_diff_s,
                 "recorder_offset_s": gate.recorder_offset_s},
        "t_ref_first_image_utc": float(image_t[0]),
        "n_images": int(image_t.size),
        "n_cue_fixes": int(c["cue_t"].size),
        "candidate_stats": [w.__dict__ | {"reasons": list(w.reasons)} for w in stats],
        "class_counts": {k: sum(1 for w in stats if w.cls == k) for k in ("SV", "CC", "other", "ineligible")},
        "candidate_frames": frames,
    }


def stats_from_census(rec: dict) -> list:
    out = []
    for d in rec["candidate_stats"]:
        d = dict(d)
        d["reasons"] = tuple(d.get("reasons") or ())
        for f in ("cue_coverage", "valid_frac", "v_mean", "cv", "dv", "S"):
            d[f] = float("nan") if d[f] is None else float(d[f])
        out.append(ws.WindowStats(**d))
    return out


def pooled_windows(censuses: dict, rank_order: list) -> list:
    """Pure pooled selection -> window dicts with sequence + frames (raises SequenceIneligible => v5 STOP)."""
    sel = ws.select_windows_pooled({s: stats_from_census(censuses[s]) for s in rank_order}, rank_order)
    wins = []
    for s in sel.starts:
        fr = censuses[s["sequence"]]["candidate_frames"][str(s["k"])]
        wins.append({**s, **fr, "window_id": f"{s['sequence']}:k{s['k']}"})
    return wins, sel.n_sv_candidates


# --------------------------------- phases ----------------------------------
def _bag(seq: str, bag: Path | None) -> Path:
    return Path(bag) if bag else BAG_DIR / f"{seq}.bag"


def census(seq: str, bag: Path | None = None, out: Path = OUT) -> int:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    c = load_contract()
    entry = pool_entry(c, seq)
    bag = _bag(seq, bag)
    for f in (out / f"census_{seq}.json", out / f"census_STOP_{seq}.json"):
        if f.exists():
            raise SystemExit(f"{f.name} exists; refusing silent rerun")
    digest = verify_bag(entry, bag)
    store = get_typestore(Stores.ROS1_NOETIC)
    try:
        with Reader(bag) as reader:
            conns = {x.topic: x for x in reader.connections}
            for t in v4s.GATE_TOPICS:
                if t not in conns:
                    raise sg.Stage1Stop(f"required topic absent from bag index: {t}")
            cam_h, cam_r, rtk_h, rtk_r, ch, cr, llh, sync_meta = v4s.read_streams(reader, store, conns)
        cen = v4s.census_stage1(cam_h, cam_r, ch, cr, llh, rtk_h, rtk_r)
    except sg.Stage1Stop as exc:
        dump(out / f"census_STOP_{seq}.json", {"sequence": seq, "status": "STOP", "reason": str(exc),
                                                "v5_consequence": "v5 STOPS (no fallback)", "contract_sha256": c["sha256"],
                                                "bag_sha256": digest, "reference_values_read": False})
        print(f"STOP ({seq}): {exc}")
        return 2
    rec = census_record(cen)
    cue_csv = out / f"cue_{seq}.csv"
    cue_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(cue_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_utc", "lat_deg", "lon_deg", "alt_m"])
        for t, (la, lo, al) in zip(cen["cue_t"], cen["llh"]):
            wr.writerow([f"{t:.6f}", f"{la:.9f}", f"{lo:.9f}", f"{al:.4f}"])
    dump(out / f"census_{seq}.json", {
        "sequence": seq, "contract_sha256": c["sha256"], "bag_bytes": bag.stat().st_size, "bag_sha256": digest,
        "time_sync_metadata": sync_meta, "reference_values_read": False,
        "rtk_position_access": "header-only raw bytes [0:12]",
        "cue_csv": cue_csv.name, "cue_csv_sha256": sha256_file(cue_csv), **rec,
    })
    print(f"census {seq}: {rec['class_counts']}  recorder_offset={rec['gate']['recorder_offset_s']:+.3f} s")
    return 0


def select(out: Path = OUT) -> int:
    c = load_contract()
    if (out / "windows.json").exists() or (out / "selection_STOP.json").exists():
        raise SystemExit("selection output exists; refusing silent rerun")
    censuses = {}
    for seq in c["rank_order"]:
        if (out / f"census_STOP_{seq}.json").exists():
            raise SystemExit(f"census STOP recorded for {seq}: v5 STOPS")
        p = out / f"census_{seq}.json"
        if not p.exists():
            raise SystemExit(f"census missing for {seq}")
        rec = json.loads(p.read_text())
        if rec["contract_sha256"] != c["sha256"] or rec["reference_values_read"] is not False:
            raise SystemExit(f"census {seq}: contract/attestation mismatch")
        if sha256_file(out / rec["cue_csv"]) != rec["cue_csv_sha256"]:
            raise SystemExit(f"census {seq}: cue csv hash mismatch")
        censuses[seq] = rec
    try:
        wins, n_sv = pooled_windows(censuses, c["rank_order"])
    except ws.SequenceIneligible as exc:
        dump(out / "selection_STOP.json", {"status": "STOP", "reason": str(exc), "contract_sha256": c["sha256"],
                                           "reference_values_read": False})
        print(f"STOP: {exc}")
        return 2
    dump(out / "windows.json", {
        "contract_sha256": c["sha256"], "rank_order": c["rank_order"], "n_sv_candidates_pooled": n_sv,
        "census_sha256": {s: sha256_file(out / f"census_{s}.json") for s in c["rank_order"]},
        "t_ref_first_image_utc": {s: censuses[s]["t_ref_first_image_utc"] for s in c["rank_order"]},
        "windows": wins,
    })
    for w in wins:
        print(f"{w['role']:17s} {w['window_id']:22s} S={w['S']:.3f} cls={w['cls']}")
    return 0


def frames(seq: str, bag: Path | None = None, out: Path = OUT) -> int:
    import cv2
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    c = load_contract()
    entry = pool_entry(c, seq)
    rec_path = out / f"frames_{seq}.json"
    if rec_path.exists():
        raise SystemExit(f"{rec_path.name} exists; refusing silent rerun")
    wj = json.loads((out / "windows.json").read_text())
    if wj["contract_sha256"] != c["sha256"]:
        raise SystemExit("windows.json contract mismatch")
    bag = _bag(seq, bag)
    digest = verify_bag(entry, bag)
    want = {int(i): f"{int(i):08d}.png" for w in wj["windows"] if w["sequence"] == seq for i in w["frame_indices_bag"]}
    fdir = out / "frames" / seq
    fdir.mkdir(parents=True, exist_ok=True)
    store = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as reader:
        cn = [x for x in reader.connections if x.topic == v4s.CAMERA_TOPIC][0]
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
    dump(rec_path, {"sequence": seq, "contract_sha256": c["sha256"], "bag_sha256": digest,
                    "windows_json_sha256": sha256_file(out / "windows.json"),
                    "frame_sha256": {f"{seq}/{n}": sha256_file(fdir / n) for n in sorted(want.values())}})
    print(f"frames {seq}: {len(want)} images")
    return 0


def _conn_digest(conn) -> str | None:
    return getattr(conn, "digest", None) or getattr(getattr(conn, "ext", None), "md5sum", None)


def copy_reference_topics(src_bag: Path, dst: Path) -> dict:
    """Copy REF_TOPICS raw bytes + record times + connection msgdef/md5 into dst. Never deserializes."""
    from rosbags.rosbag1 import Reader, Writer

    counts = {}
    with Reader(src_bag) as reader, Writer(dst) as writer:
        src = [x for x in reader.connections if x.topic in REF_TOPICS]
        if sorted({x.topic for x in src}) != sorted(REF_TOPICS):
            raise RuntimeError("reference topics absent (fail closed)")
        for conn in src:
            md = getattr(conn, "msgdef", None)
            text = getattr(md, "data", md)
            wc = writer.add_connection(conn.topic, conn.msgtype, msgdef=text, md5sum=_conn_digest(conn))
            n = 0
            for _, rec, raw in reader.messages(connections=[conn]):
                writer.write(wc, rec, bytes(raw))  # raw bytes + record time; no deserialization
                n += 1
            counts[conn.topic] = counts.get(conn.topic, 0) + n
    return counts


def refbag(seq: str, bag: Path | None = None, out: Path = OUT) -> int:
    """Byte-for-byte copy of the reference topics into a sealed sub-bag (NEVER deserialized)."""
    c = load_contract()
    entry = pool_entry(c, seq)
    dst = out / f"ref_{seq}.bag"
    rec_path = out / f"refbag_{seq}.json"
    if dst.exists() or rec_path.exists():
        raise SystemExit("ref sub-bag exists; refusing silent rerun")
    bag = _bag(seq, bag)
    digest = verify_bag(entry, bag)
    counts = copy_reference_topics(bag, dst)
    dump(rec_path, {"sequence": seq, "contract_sha256": c["sha256"], "source_bag_sha256": digest,
                    "sub_bag": dst.name, "sub_bag_bytes": dst.stat().st_size, "sub_bag_sha256": sha256_file(dst),
                    "topics": counts, "deserialized": False,
                    "rule": "opened ONLY at one-time Stage-2 scoring after SCORING_RECEIPT.json (contract A3 remote_bag_handling)"})
    print(f"refbag {seq}: {counts}")
    return 0


def finalize(out: Path = OUT) -> int:
    c = load_contract()
    man = out / "stage1_manifest.json"
    if man.exists():
        raise SystemExit("stage1_manifest.json exists; refusing silent rerun")
    wj = json.loads((out / "windows.json").read_text())
    seqs_used = sorted({w["sequence"] for w in wj["windows"]}, key=c["rank_order"].index)
    frame_sha, per_seq = {}, {}
    for seq in c["rank_order"]:
        cen = json.loads((out / f"census_{seq}.json").read_text())
        if sha256_file(out / f"census_{seq}.json") != wj["census_sha256"][seq]:
            raise SystemExit(f"census {seq} changed after selection")
        info = {"census_sha256": wj["census_sha256"][seq], "bag_sha256": cen["bag_sha256"],
                "gate": cen["gate"], "t_ref_first_image_utc": cen["t_ref_first_image_utc"],
                "class_counts": cen["class_counts"], "cue_csv": cen["cue_csv"], "cue_csv_sha256": cen["cue_csv_sha256"],
                "v_cruise_table_mps": pool_entry(c, seq)["v_cruise_table_mps"]}
        if seq in seqs_used:
            fr = json.loads((out / f"frames_{seq}.json").read_text())
            if fr["windows_json_sha256"] != sha256_file(out / "windows.json"):
                raise SystemExit(f"frames_{seq}.json was built from a different windows.json")
            for name, dg in fr["frame_sha256"].items():
                if sha256_file(out / "frames" / name) != dg:
                    raise SystemExit(f"frame hash mismatch {name}")
            frame_sha.update(fr["frame_sha256"])
        rb = out / f"refbag_{seq}.json"
        if rb.exists():
            r = json.loads(rb.read_text())
            if sha256_file(out / r["sub_bag"]) != r["sub_bag_sha256"] or r["deserialized"] is not False:
                raise SystemExit(f"ref sub-bag {seq} hash/attestation mismatch")
            info["reference_source"] = {"kind": "sealed_sub_bag", "path": r["sub_bag"], "sha256": r["sub_bag_sha256"]}
        else:
            info["reference_source"] = {"kind": "local_bag", "path": f"data/mars_lvig/{seq}.bag", "sha256": cen["bag_sha256"]}
        per_seq[seq] = info
    need = {f"{w['sequence']}/{int(i):08d}.png" for w in wj["windows"] for i in w["frame_indices_bag"]}
    if need != set(frame_sha):
        raise SystemExit("frame set != selected windows' frames")
    dump(man, {"contract_sha256": c["sha256"], "amendment": "A3 pooled SV", "rank_order": c["rank_order"],
               "reference_values_read": False, "rtk_position_access": "header-only raw bytes [0:12]",
               "windows_json_sha256": sha256_file(out / "windows.json"), "sequences": per_seq,
               "frame_sha256": dict(sorted(frame_sha.items())), "windows": wj["windows"]})
    print(f"stage1 v5 manifest: {len(wj['windows'])} windows over {seqs_used}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["census", "select", "frames", "refbag", "finalize"])
    ap.add_argument("--sequence")
    ap.add_argument("--bag", type=Path)
    a = ap.parse_args()
    if a.phase in ("census", "frames", "refbag") and not a.sequence:
        ap.error("--sequence required")
    if a.phase == "census":
        return census(a.sequence, a.bag)
    if a.phase == "select":
        return select()
    if a.phase == "frames":
        return frames(a.sequence, a.bag)
    if a.phase == "refbag":
        return refbag(a.sequence, a.bag)
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
