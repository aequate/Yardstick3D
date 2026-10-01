"""Cross-backbone v2 - Stage 1 sensor-only extraction (contract: configs/prospective_cross_backbone_v2.yaml).

Refuses to run unless the contract is frozen (status: frozen_pre_data, sha256 recorded in
configs/experiment_registry.json) and dataset.bag_filename is set. Reads ONLY:
  * camera header stamps (first 12 bytes of each message; no image decode until windows are chosen),
  * receiver_lla cue fixes,
  * rtk_position HEADER STAMPS via raw bytes [0:12] (never deserialized),
  * time_sync topics (decoded with msg definitions embedded in the bag).
It NEVER reads rtk_position values, rtk_velocity, rtk_yaw, rtk_info_* or rtk_connection_status.

Order: timescale gate -> time_sync decode/checks -> cue-only window selection -> manifest.
Any Stage1Stop / SequenceIneligible writes stage1_STOP.json and exits 2 (sequence INELIGIBLE,
cue-only reason; never fitted or patched).
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yardstick3d.datasets import timescale as ts  # noqa: E402
from yardstick3d.datasets import v2_sync_guards as sg  # noqa: E402
from yardstick3d.datasets import v2_window_selection as ws  # noqa: E402

CONTRACT = ROOT / "configs/prospective_cross_backbone_v4.yaml"  # v4 = v3 + amendment A2 (recorder clock)
INELIGIBLE_PREFIX = "SEQUENCE_INELIGIBLE_v4_"  # v4 counts its own fallbacks; v3 records kept for audit
REGISTRY = ROOT / "configs/experiment_registry.json"
BAG_DIR = ROOT / "data/mars_lvig"
OUT = ROOT / "artifacts/cross_backbone_v2"

CAMERA_TOPIC = "/left_camera/image/compressed"
CUE_TOPIC = "/ublox_driver/receiver_lla"
RTK_POSITION_TOPIC = "/dji_osdk_ros/rtk_position"
GATE_TOPICS = (CAMERA_TOPIC, CUE_TOPIC, RTK_POSITION_TOPIC)
TIME_SYNC_TOPICS = (
    "/ublox_driver/time_pulse_info",
    "/dji_osdk_ros/time_sync_fc_time_utc",
    "/dji_osdk_ros/time_sync_gps_utc",
    "/dji_osdk_ros/time_sync_nmea_msg",
    "/dji_osdk_ros/time_sync_pps_source",
)
# Topics whose VALUES Stage 1 must never read.
FORBIDDEN_VALUE_TOPICS = (
    "/dji_osdk_ros/rtk_position",  # header-only raw parse is the sole permitted access
    "/dji_osdk_ros/rtk_velocity",
    "/dji_osdk_ros/rtk_yaw",
    "/dji_osdk_ros/rtk_info_position",
    "/dji_osdk_ros/rtk_info_yaw",
    "/dji_osdk_ros/rtk_connection_status",
)
TIME_SYNC_CHECK_TOL_S = 2.0  # v3 A1: integer-second PPS labels
TIME_SYNC_SPREAD_TOL_S = 0.5  # v3 A1: (P99 - P1) of residual


# ------------------------------- contract ---------------------------------
def _yaml_scalar(text: str, key: str, indent_min: int = 0) -> str | None:
    m = re.search(rf'^[ ]{{{indent_min},}}{re.escape(key)}:[ ]*"?([^"#\n]*?)"?[ ]*(?:#.*)?$', text, re.M)
    return m.group(1).strip() if m else None


def load_contract(path: Path = CONTRACT, registry: Path = REGISTRY) -> dict:
    """Read status + bag filename from the YAML; refuse unless frozen and registered."""
    if not path.exists():
        raise SystemExit(f"contract missing (v2 is not frozen yet): {path}")
    text = path.read_text(encoding="utf-8")
    status = _yaml_scalar(text, "status")
    if status != "frozen_pre_data":
        raise SystemExit(f"contract status is {status!r}, need 'frozen_pre_data'")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if not registry.exists() or sha not in registry.read_text(encoding="utf-8"):
        raise SystemExit(f"contract sha256 {sha} not recorded in {registry}")
    name = _yaml_scalar(text, "bag_filename", indent_min=1)
    if not name or "PLACEHOLDER" in name or "/" in name or "\\" in name or ".." in name:
        raise SystemExit(f"dataset.bag_filename invalid: {name!r}")
    if ts.GPST_MINUS_UTC_S != float(_yaml_scalar(text, "gpst_minus_utc_s", indent_min=1) or "nan"):
        raise SystemExit("gpst_minus_utc_s in contract != timescale.GPST_MINUS_UTC_S")
    # sequence_selection.cue_only_fallback: highest-ranked sequence without an INELIGIBLE record (max 2 fallbacks)
    if "rank_order:" not in text:
        raise SystemExit("sequence_selection.rank_order missing (fail closed)")
    ranks = re.findall(r"^\s+- (\w+_GNSS\d+)\s+#", text.split("rank_order:", 1)[1].split("rule:", 1)[0], re.M)
    if not ranks or ranks[0] + ".bag" != name:
        raise SystemExit(f"rank_order unreadable or rank 1 != dataset.bag_filename ({ranks[:1]} vs {name})")
    fallbacks = 0
    for seq in ranks:
        if (OUT / f"{INELIGIBLE_PREFIX}{seq}.json").exists():
            fallbacks += 1
            continue
        if fallbacks > 2:
            raise SystemExit("more than 2 cue-only fallbacks: experiment STOPS per contract")
        return {"sha256": sha, "bag_filename": seq + ".bag", "bag": BAG_DIR / (seq + ".bag"),
                "sequence": seq, "rank": ranks.index(seq) + 1, "fallbacks_used": fallbacks}
    raise SystemExit("no eligible sequence left in rank_order: experiment STOPS")


# ------------------------------ time_sync ---------------------------------
def _time_like(v):
    if hasattr(v, "sec") and (hasattr(v, "nanosec") or hasattr(v, "nsec")):
        return float(v.sec) + float(getattr(v, "nanosec", getattr(v, "nsec", 0))) * 1e-9
    return None


def _yymmdd_hhmmss(yymmdd: int, hhmmss: int) -> float:
    y, mo, d = 2000 + yymmdd // 10000, (yymmdd // 100) % 100, yymmdd % 100
    h, mi, se = hhmmss // 10000, (hhmmss // 100) % 100, hhmmss % 100
    return float(calendar.timegm((y, mo, d, h, mi, se, 0, 0, 0)))


def extract_utc_field(msg) -> float | None:
    """UTC epoch seconds carried in a time_sync message body (never its header/stamp)."""
    if hasattr(msg, "fc_utc_yymmdd") and hasattr(msg, "fc_utc_hhmmss"):  # dji FCTimeInUTC
        return _yymmdd_hhmmss(int(msg.fc_utc_yymmdd), int(msg.fc_utc_hhmmss))
    if hasattr(msg, "UTCTimeData"):  # dji GPSUTC, e.g. 'UTC 231024 075158 5 '
        m = re.match(r"\s*UTC\s+(\d{6})\s+(\d{6})", str(msg.UTCTimeData))
        return _yymmdd_hhmmss(int(m.group(1)), int(m.group(2))) if m else None
    for name in getattr(msg, "__dataclass_fields__", {}):
        if name in ("header", "stamp"):
            continue
        t = _time_like(getattr(msg, name))
        if t is not None:
            return t
    names = {n.lower(): n for n in getattr(msg, "__dataclass_fields__", {})}
    keys = [("year",), ("month",), ("day",), ("hour",), ("min", "minute", "minutes"), ("sec", "second", "seconds")]
    if all(any(a in names for a in k) for k in keys):
        v = [float(getattr(msg, names[next(a for a in k if a in names)])) for k in keys]
        return float(calendar.timegm((int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]), 0)) + v[5])
    return None


def gprmc_tod_s(sentence: str) -> float | None:
    m = re.search(r"\$?G[PNL]RMC,(\d{2})(\d{2})(\d{2}(?:\.\d*)?)", sentence)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def check_time_sync_topic(topic: str, header_s, record_s, decoded_utc, nmea_tod, recorder_offset_s: float = 0.0) -> dict:
    """Pure consistency checks. Raises Stage1Stop on exceedance or unverifiable content."""
    hs, rs = np.asarray(header_s, float), np.asarray(record_s, float)
    rec = {"present": True, "count": int(hs.size)}
    if hs.size == 0:
        raise sg.Stage1Stop(f"{topic}: present but no decodable header stamps")
    st = ts.header_offset_stats(hs, rs)
    cls = ts.classify_timescale(st["median"], sg.TOL_CLASS_S)
    rec.update({"header_minus_record": st, "timescale": cls, "rate_hz": float(hs.size / max(hs[-1] - hs[0], 1e-9)) if hs.size > 1 else None})
    if topic.endswith("time_pulse_info") or topic.endswith("time_sync_pps_source"):
        return rec  # disclosure only
    if cls == "unknown":
        raise sg.Stage1Stop(f"{topic}: header-record offset {st['median']:+.3f} s is unexplained")
    t_utc = ts.to_utc(hs, cls)
    if cls == "utc" and abs(st["median"]) < sg.RECORDER_TOPIC_MAX_S:  # v4 A2: stamped on the recorder clock
        t_utc = hs + recorder_offset_s
        rec["recorder_clock"] = True
        rec["recorder_offset_s"] = recorder_offset_s
    if topic.endswith("time_sync_nmea_msg"):
        idx = [i for i, x in enumerate(nmea_tod) if x is not None]  # pair each GPRMC with ITS OWN stamp
        if not idx:
            raise sg.Stage1Stop(f"{topic}: no GPRMC sentence decodable (fail closed)")
        tod = np.asarray([nmea_tod[i] for i in idx], float)
        d = (tod - (t_utc[idx] % 86400.0) + 43200.0) % 86400.0 - 43200.0
    else:  # time_sync_gps_utc / fc_time_utc
        idx = [i for i, x in enumerate(decoded_utc) if x is not None]  # pair with own stamp
        if not idx:
            raise sg.Stage1Stop(f"{topic}: no recognizable UTC field in message (fail closed)")
        d = np.asarray([decoded_utc[i] for i in idx], float) - t_utc[idx]
    med = float(np.median(d))
    spread = float(np.percentile(d, 99) - np.percentile(d, 1))
    rec.update({"decoded_residual_median_s": med, "decoded_residual_spread_p99_p1_s": spread,
                "rule": "v3 A1: |median| < 2.0 s and spread < 0.5 s"})
    if not (abs(med) < TIME_SYNC_CHECK_TOL_S and spread < TIME_SYNC_SPREAD_TOL_S):
        raise sg.Stage1Stop(f"{topic}: implied UTC residual median {med:+.3f} s / spread {spread:.3f} s violates v3 A1")
    return rec


# ------------------------------- planning ---------------------------------
def census_stage1(cam_header, cam_record, cue_header, cue_record, cue_llh, rtk_header, rtk_record) -> dict:
    """Gates + per-candidate window statistics (no selection). Pure; no bag I/O, no reference values.

    Raises Stage1Stop on a gate failure. Used by plan_stage1 (v2-v4) and the v5 per-sequence census.
    """
    from yardstick3d.datasets import mars_lvig as ml

    gate = sg.stage1_timescale_gate_v4(  # v4 A2
        CAMERA_TOPIC, (cam_header, cam_record),
        CUE_TOPIC, (cue_header, cue_record),
        {RTK_POSITION_TOPIC: (rtk_header, rtk_record)},
    )
    cam = sg.stamp_stream(CAMERA_TOPIC, cam_header, gate)
    cue = sg.stamp_stream(CUE_TOPIC, cue_header, gate)
    llh = np.asarray(cue_llh, float)
    ok = np.all(np.isfinite(llh[:, :2]), axis=1)
    cue_t = cue.t_utc[ok]
    order = np.argsort(cue_t, kind="stable")
    cue_t = cue_t[order]
    llh = llh[ok][order]
    alt = np.where(np.isfinite(llh[:, 2]), llh[:, 2], 0.0)
    cue_enu = ml.cue_enu_from_llh(llh[:, 0], llh[:, 1], alt)  # origin: first valid cue fix
    cam_order = np.argsort(cam.t_utc, kind="stable")
    image_t = cam.t_utc[cam_order]
    stats = ws.window_statistics(image_t, cue_t, cue_enu)
    return {"gate": gate, "image_t": image_t, "cam_order": cam_order, "stats": stats, "cue_t": cue_t, "llh": llh}


def window_frames(c: dict, t0: float) -> dict:
    """Frame indices/times for a window start (all 8 frames must match; asserted)."""
    image_t, cam_order = c["image_t"], c["cam_order"]
    idx, ideal = ws.frames_for_window(image_t, float(image_t[0]), t0)
    assert idx is not None
    return {
        "ideal_times_utc": [float(x) for x in ideal],
        "frame_indices_sorted": [int(i) for i in idx],
        "frame_indices_bag": [int(cam_order[i]) for i in idx],
        "frame_times_utc": [float(image_t[i]) for i in idx],
    }


def plan_stage1(cam_header, cam_record, cue_header, cue_record, cue_llh, rtk_header, rtk_record) -> dict:
    """Pure Stage-1 logic on arrays (no bag I/O, no reference values).

    cue_llh: (n,3) lat, lon, alt aligned with cue_header/cue_record. Raises Stage1Stop /
    SequenceIneligible; both mean the sequence is INELIGIBLE (cue-only reason).
    """
    c = census_stage1(cam_header, cam_record, cue_header, cue_record, cue_llh, rtk_header, rtk_record)
    gate, image_t, stats, cue_t, llh = c["gate"], c["image_t"], c["stats"], c["cue_t"], c["llh"]
    sel = ws.select_windows(stats)  # may raise SequenceIneligible
    t_ref = float(image_t[0])
    windows = [{**s, **window_frames(c, s["t0"])} for s in sel.starts]
    return {
        "gate": {"classes": gate.classes, "stats": gate.stats, "cross_topic_max_diff_s": gate.cross_topic_max_diff_s,
                 "recorder_offset_s": gate.recorder_offset_s},
        "t_ref_first_image_utc": t_ref,
        "n_images": int(image_t.size),
        "n_cue_fixes": int(cue_t.size),
        "candidate_stats": [w.__dict__ | {"reasons": list(w.reasons)} for w in stats],
        "n_sv_candidates": sel.n_sv_candidates,
        "windows": windows,
        "cue_t_utc": cue_t,
        "cue_llh": llh,
    }


# --------------------------------- I/O ------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, allow_nan=False), encoding="utf-8")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _guard_topic(topic: str) -> None:
    if topic in FORBIDDEN_VALUE_TOPICS and topic != RTK_POSITION_TOPIC:
        raise RuntimeError(f"Stage 1 must not read {topic}")


def _headers(reader, conn):
    """(header_s, record_s) for every message using ONLY raw bytes [0:12]."""
    _guard_topic(conn.topic)
    h, r = [], []
    for _, rec, raw in reader.messages(connections=[conn]):
        h.append(sg.rtk_header_stamp_s(raw))
        r.append(rec / 1e9)
    return np.asarray(h), np.asarray(r)


def _register_bag_types(store, conns):
    from rosbags.typesys import get_types_from_msg

    for c in conns:
        md = getattr(c, "msgdef", None)
        text = getattr(md, "data", md)
        if not isinstance(text, str):
            raise sg.Stage1Stop(f"{c.topic}: no msg definition embedded in bag connection (fail closed)")
        try:
            store.register(get_types_from_msg(text, c.msgtype))
        except Exception:
            pass  # already registered (standard types)


def read_streams(reader, store, conns) -> tuple:
    """All Stage-1 sensor reads (camera/rtk header bytes, cue fixes, time_sync decode+checks). No reference values."""
    cam_h, cam_r = _headers(reader, conns[CAMERA_TOPIC])
    rtk_h, rtk_r = _headers(reader, conns[RTK_POSITION_TOPIC])  # header-only bytes
    cue_conn = conns[CUE_TOPIC]
    _register_bag_types(store, [cue_conn])
    ch, cr, llh = [], [], []
    for _, rec, raw in reader.messages(connections=[cue_conn]):
        m = store.deserialize_ros1(raw, cue_conn.msgtype)
        ch.append(float(m.header.stamp.sec) + float(m.header.stamp.nanosec) * 1e-9)
        cr.append(rec / 1e9)
        llh.append((float(m.latitude), float(m.longitude), float(m.altitude)))
    sync_meta = {}
    for topic in TIME_SYNC_TOPICS:
        if topic not in conns:
            sync_meta[topic] = {"present": False}
            continue
        cn = conns[topic]
        try:
            _register_bag_types(store, [cn])
            hh, rr, dec, nm = [], [], [], []
            for _, rec, raw in reader.messages(connections=[cn]):
                m = store.deserialize_ros1(raw, cn.msgtype)
                if hasattr(m, "header"):
                    hh.append(sg.rtk_header_stamp_s(raw))
                elif hasattr(m, "stamp") and hasattr(m.stamp, "sec"):  # dji time_sync: top-level `time stamp`
                    hh.append(float(m.stamp.sec) + float(m.stamp.nanosec) * 1e-9)
                else:
                    hh.append(float("nan"))
                rr.append(rec / 1e9)
                dec.append(extract_utc_field(m))
                nm.append(gprmc_tod_s(str(getattr(m, "sentence", ""))))
            disclosure_only = topic.endswith("time_pulse_info") or topic.endswith("time_sync_pps_source")
            if not np.all(np.isfinite(hh)):
                if not disclosure_only:
                    raise sg.Stage1Stop(f"{topic}: message has no Header (fail closed)")
                # contract time_sync_decoding: "time_pulse_info: recorded as-is; disclosure only"
                rec_ = {"present": True, "count": len(rr), "has_header": False,
                        "decoded_time_fields_sample": [str(x) for x in dec[:3]],
                        "note": "no ROS Header; recorded as-is (disclosure only, not a gate)"}
            else:
                rec_ = check_time_sync_topic(topic, hh, rr, dec, nm, sg.recorder_clock_offset(cam_h, cam_r))
        except sg.Stage1Stop:
            raise
        except Exception as exc:
            raise sg.Stage1Stop(f"{topic}: present but undecodable ({type(exc).__name__}: {exc})") from exc
        rec_.update({"msgtype": cn.msgtype, "md5sum": getattr(getattr(cn, "ext", None), "md5sum", None)})
        sync_meta[topic] = rec_
    return cam_h, cam_r, rtk_h, rtk_r, ch, cr, np.asarray(llh), sync_meta


def stage1(contract_path: Path = CONTRACT) -> int:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    c = load_contract(contract_path)
    bag: Path = c["bag"]
    if not bag.exists():
        raise SystemExit(f"bag missing: {bag}")
    s1 = OUT / "stage1"
    if (s1 / "stage1_manifest.json").exists() or (s1 / "stage1_STOP.json").exists():
        raise SystemExit("stage1 output exists; refusing silent rerun")
    store = get_typestore(Stores.ROS1_NOETIC)
    try:
        with Reader(bag) as reader:
            conns = {x.topic: x for x in reader.connections}
            for t in GATE_TOPICS:
                if t not in conns:
                    raise sg.Stage1Stop(f"required topic absent from bag index: {t}")
            cam_h, cam_r, rtk_h, rtk_r, ch, cr, llh, sync_meta = read_streams(reader, store, conns)
            plan = plan_stage1(cam_h, cam_r, ch, cr, llh, rtk_h, rtk_r)
    except (sg.Stage1Stop, ws.SequenceIneligible) as exc:
        dump(s1 / "stage1_STOP.json", {"status": "STOP", "reason": str(exc), "sequence_ineligible_cue_only": True,
                                       "contract_sha256": c["sha256"], "reference_values_read": False})
        print(f"STOP: {exc}")
        return 2

    import cv2

    frames_dir = s1 / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    want = {i: f"{i:08d}.png" for w in plan["windows"] for i in w["frame_indices_bag"]}
    with Reader(bag) as reader:
        cn = [x for x in reader.connections if x.topic == CAMERA_TOPIC][0]
        for k, (_, _, raw) in enumerate(reader.messages(connections=[cn])):
            if k not in want:
                continue
            m = store.deserialize_ros1(raw, cn.msgtype)
            arr = cv2.imdecode(np.frombuffer(m.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise RuntimeError(f"failed to decode image {k}")
            cv2.imwrite(str(frames_dir / want[k]), arr)
    with open(s1 / "cue_receiver_lla.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_utc", "lat_deg", "lon_deg", "alt_m"])
        for t, (la, lo, al) in zip(plan.pop("cue_t_utc"), plan.pop("cue_llh")):
            wr.writerow([f"{t:.6f}", f"{la:.9f}", f"{lo:.9f}", f"{al:.4f}"])
    manifest = {
        "contract_sha256": c["sha256"],
        "bag": str(bag.relative_to(ROOT)),
        "bag_bytes": bag.stat().st_size,
        "bag_sha256": sha256_file(bag),
        "time_sync_metadata": sync_meta,
        "reference_values_read": False,
        "rtk_position_access": "header-only raw bytes [0:12]",
        "frame_sha256": {n: sha256_file(frames_dir / n) for n in sorted(want.values())},
        **plan,
    }
    dump(s1 / "stage1_manifest.json", manifest)
    dump(s1 / "windows.json", {"windows": plan["windows"], "t_ref_first_image_utc": plan["t_ref_first_image_utc"]})
    print(f"stage1 complete: {len(plan['windows'])} windows selected")
    return 0


def topics(contract_path: Path = CONTRACT) -> int:
    from rosbags.rosbag1 import Reader

    with Reader(load_contract(contract_path)["bag"]) as reader:
        for c in reader.connections:
            print(f"{c.topic}\t{c.msgtype}\t{c.msgcount}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["stage1", "topics"])
    ap.add_argument("--contract", type=Path, default=CONTRACT)
    a = ap.parse_args()
    return topics(a.contract) if a.phase == "topics" else stage1(a.contract)


if __name__ == "__main__":
    raise SystemExit(main())
