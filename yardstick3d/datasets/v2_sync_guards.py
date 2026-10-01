"""Cross-backbone v2 Stage-1 guards: timescale gate, UTC-only pairing, header-only rtk parser.

Contract: configs/prospective_cross_backbone_v2.DRAFT.yaml (timestamp_synchronization).
Nothing here reads reference positions: the rtk_position parser touches only the
12-byte ROS1 Header prefix [seq:u32, sec:u32, nsec:u32].
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from yardstick3d.datasets import timescale as ts

TOL_CLASS_S = 0.5
SPREAD_MAX_S = 0.5
CROSS_TOPIC_MAX_S = 0.25
RTK_HEADER_BYTES = 12  # seq, sec, nsec (u32 LE each)


class Stage1Stop(RuntimeError):
    """Stage 1 must halt: sequence INELIGIBLE (cue-only reason); never fitted or patched."""


# --------------------------------------------------------------------------
# header-only raw parser for /dji_osdk_ros/rtk_position
# --------------------------------------------------------------------------
def parse_ros1_header_only(raw) -> tuple[int, int, int]:
    """Return (seq, sec, nsec) from the first 12 bytes of a serialized ROS1 message body.

    Only the slice raw[0:12] is ever requested from the buffer; fields after the
    Header (position values) are never touched or deserialized.
    """
    head = bytes(raw[0:RTK_HEADER_BYTES])
    if len(head) != RTK_HEADER_BYTES:
        raise ValueError(f"message shorter than a ROS1 Header prefix ({len(head)} < {RTK_HEADER_BYTES})")
    seq, sec, nsec = struct.unpack("<III", head)
    if nsec >= 1_000_000_000:
        raise ValueError(f"invalid header nsec {nsec}")
    return int(seq), int(sec), int(nsec)


def rtk_header_stamp_s(raw) -> float:
    _, sec, nsec = parse_ros1_header_only(raw)
    return float(sec) + float(nsec) * 1e-9


# --------------------------------------------------------------------------
# Stage-1 timescale gate
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TimescaleGate:
    classes: dict  # topic -> 'utc' | 'gpst'
    stats: dict  # topic -> header_offset_stats + p1/p99/spread + utc_minus_record median
    cross_topic_max_diff_s: float
    recorder_offset_s: float = 0.0  # v4 A2: UTC = recorder_time + recorder_offset_s (0.0 under v2/v3)


def stage1_timescale_gate(topics: dict, tol_s: float = TOL_CLASS_S) -> TimescaleGate:
    """topics: {name: (header_s, record_s)} for ALL messages. Raises Stage1Stop on any failure.

    Rules (YAML): per-topic class via require_common_timescale; per-topic P99-P1 of d < 0.5 s;
    after to_utc the per-topic medians of (t_utc - record) agree pairwise within 0.25 s.
    """
    if not topics:
        raise Stage1Stop("no topics supplied to the timescale gate")
    stats: dict = {}
    for name, (h, r) in topics.items():
        try:
            s = ts.header_offset_stats(h, r)
        except ValueError as exc:
            raise Stage1Stop(f"{name}: {exc}") from exc
        d = np.asarray(h, float) - np.asarray(r, float)
        s["p1"], s["p99"] = float(np.percentile(d, 1)), float(np.percentile(d, 99))
        s["spread_p99_p1"] = s["p99"] - s["p1"]
        stats[name] = s
    try:
        classes = ts.require_common_timescale(topics, tol_s=tol_s)
    except RuntimeError as exc:
        raise Stage1Stop(str(exc)) from exc
    for name, s in stats.items():
        if not s["spread_p99_p1"] < SPREAD_MAX_S:
            raise Stage1Stop(f"{name}: P99-P1 of header-record offset {s['spread_p99_p1']:.3f} s >= {SPREAD_MAX_S} s (mid-bag switch)")
    meds = {}
    for name, (h, r) in topics.items():
        meds[name] = float(np.median(ts.to_utc(h, classes[name]) - np.asarray(r, float)))
        stats[name]["utc_minus_record_median"] = meds[name]
        stats[name]["timescale"] = classes[name]
    vals = list(meds.values())
    worst = float(max(vals) - min(vals)) if vals else 0.0
    if worst >= CROSS_TOPIC_MAX_S:
        raise Stage1Stop(f"cross-topic median (t_utc - record) disagree by {worst:.3f} s >= {CROSS_TOPIC_MAX_S} s")
    return TimescaleGate(classes=classes, stats=stats, cross_topic_max_diff_s=worst)


# --------------------------------------------------------------------------
# UTC-only pairing guard
# --------------------------------------------------------------------------
_TOKEN = object()


@dataclass(frozen=True)
class UtcStream:
    """Stamps proven to be in UTC. Construct ONLY via `stamp_stream` (private token)."""

    name: str
    header_s: np.ndarray
    timescale: str
    t_utc: np.ndarray
    _token: object = None

    def __post_init__(self):
        if self._token is not _TOKEN:
            raise TypeError("UtcStream must be built by stamp_stream(); raw stamps cannot be paired")
        if self.timescale != "recorder" and not np.array_equal(ts.to_utc(self.header_s, self.timescale), self.t_utc):
            raise ValueError(f"{self.name}: t_utc is not to_utc(header, {self.timescale})")


def stamp_stream(name: str, header_s, gate: TimescaleGate) -> UtcStream:
    """Convert a topic's header stamps to UTC using the class established by the Stage-1 gate."""
    if not isinstance(gate, TimescaleGate) or name not in gate.classes:
        raise Stage1Stop(f"{name}: not classified by the Stage-1 timescale gate; refusing to convert or pair")
    h = np.asarray(header_s, float)
    cls = gate.classes[name]
    t = h + gate.recorder_offset_s if cls == "recorder" else ts.to_utc(h, cls)  # v4 A2
    return UtcStream(name, h, cls, t, _TOKEN)


def pair_nearest(a: UtcStream, b: UtcStream, max_dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-sample pairing of a onto b. Refuses anything that is not a UtcStream.

    Returns (idx_b, ok): idx_b[i] nearest b sample to a[i]; ok[i] iff |dt| <= max_dt_s.
    """
    for s in (a, b):
        if not isinstance(s, UtcStream) or s._token is not _TOKEN:
            raise TypeError("pairing refused: both streams must be UtcStream (converted via timescale.to_utc)")
        if not np.array_equal(ts.to_utc(s.header_s, s.timescale), s.t_utc):
            raise ValueError(f"pairing refused: {s.name} stamps are not the UTC conversion of its header stamps")
    tb = b.t_utc
    if tb.size < 2:
        raise ValueError("pairing needs >= 2 samples in the target stream")
    order = np.argsort(tb, kind="stable")
    tbs = tb[order]
    j = np.clip(np.searchsorted(tbs, a.t_utc), 1, tbs.size - 1)
    pick = np.where(np.abs(a.t_utc - tbs[j - 1]) <= np.abs(tbs[j] - a.t_utc), j - 1, j)
    idx = order[pick]
    return idx, np.abs(a.t_utc - tb[idx]) <= max_dt_s


# --------------------------------------------------------------------------
# v4 amendment A2: recorder-clock offset measured from the hardware-timed camera
# --------------------------------------------------------------------------
RECORDER_OFFSET_MAX_S = 2.0      # A2: plausible recorder-clock offset bound
RECORDER_TOPIC_MAX_S = 0.05      # A2: |median(header - record)| for a topic stamped on the recorder clock


def recorder_clock_offset(cam_header, cam_record) -> float:
    """delta = median(camera header - bag record time); camera header is hardware UTC (A2)."""
    d = np.asarray(cam_header, float) - np.asarray(cam_record, float)
    if d.size == 0:
        raise Stage1Stop("camera: no stamps for recorder-clock offset")
    return float(np.median(d))


def stage1_timescale_gate_v4(camera_name: str, camera: tuple, cue_name: str, cue: tuple, recorder_topics: dict) -> TimescaleGate:
    """v4 A2 gate. camera/cue: (header_s, record_s) hardware-clock streams; recorder_topics:
    {name: (header_s, record_s)} stamped on the recorder clock (e.g. DJI rtk_position).

    Rules: every topic P99-P1 of (header - record) < 0.5 s; |delta| < 2.0 s;
    cue classified utc/gpst against (record + delta) and, after to_utc, agrees with the camera
    (|median(cue_utc - record - delta)| < 0.25 s); each recorder topic |median(header - record)| < 0.05 s.
    UTC for recorder topics = header + delta. delta is a sensor-only clock measurement, never GT-fitted.
    """
    topics = {camera_name: camera, cue_name: cue, **recorder_topics}
    stats = {}
    for name, (h, r) in topics.items():
        d = np.asarray(h, float) - np.asarray(r, float)
        if d.size == 0:
            raise Stage1Stop(f"{name}: no stamps")
        st = ts.header_offset_stats(h, r)
        st["p1"], st["p99"] = float(np.percentile(d, 1)), float(np.percentile(d, 99))
        st["spread_p99_p1"] = st["p99"] - st["p1"]
        if not st["spread_p99_p1"] < SPREAD_MAX_S:
            raise Stage1Stop(f"{name}: P99-P1 of header-record offset {st['spread_p99_p1']:.3f} s >= {SPREAD_MAX_S} s")
        stats[name] = st
    delta = recorder_clock_offset(*camera)
    if not abs(delta) < RECORDER_OFFSET_MAX_S:
        raise Stage1Stop(f"recorder-clock offset {delta:+.3f} s outside +/-{RECORDER_OFFSET_MAX_S} s")
    cue_h, cue_r = (np.asarray(x, float) for x in cue)
    cue_cls = ts.classify_timescale(float(np.median(cue_h - cue_r - delta)), TOL_CLASS_S)
    if cue_cls == "unknown":
        raise Stage1Stop(f"cue: offset vs camera-corrected record {float(np.median(cue_h - cue_r - delta)):+.3f} s unexplained")
    cue_vs_cam = float(np.median(ts.to_utc(cue_h, cue_cls) - cue_r - delta))
    if not abs(cue_vs_cam) < CROSS_TOPIC_MAX_S:
        raise Stage1Stop(f"cue vs camera disagree by {cue_vs_cam:+.3f} s >= {CROSS_TOPIC_MAX_S} s")
    classes = {camera_name: "utc", cue_name: cue_cls}
    for name in recorder_topics:
        m = stats[name]["median"]
        if not abs(m) < RECORDER_TOPIC_MAX_S:
            raise Stage1Stop(f"{name}: not on recorder clock (median header-record {m:+.3f} s)")
        classes[name] = "recorder"
    stats[cue_name]["utc_minus_camera_median"] = cue_vs_cam
    return TimescaleGate(classes=classes, stats=stats, cross_topic_max_diff_s=abs(cue_vs_cam), recorder_offset_s=delta)
