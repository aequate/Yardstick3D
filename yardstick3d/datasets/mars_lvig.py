"""MARS-LVIG prospective adapter (sensor-only extraction + cue/ref isolation).

Operates on EXTRACTED CSVs (no ROS dependency) with the topic contract from
configs/prospective_metric_grounding_v2.yaml. Bag parsing (rosbags + cv2) lives
in the stage-1 scripts and is deliberately NOT imported here so unit
tests and the solver path never touch bags or GT positions.

Isolation model (two-stage):
  Stage 1 (sensor-only): images + ublox cue topics + quality/time_sync topics.
    rtk_position/velocity/yaw values MUST NOT be opened.
  Stage 2 (one-time scoring): withheld rtk_* positions opened once via
    load_reference_positions(), recorded in the receipt.

All timestamps are float seconds in UTC(GPS) (ROS bag time after hardware
GPRMC+PPS discipline). Matching is nearest-sample within max_dt_match; no fitted
offsets exist anywhere in this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from yardstick3d.constraints.base import ConstraintSet
from yardstick3d.constraints.speed import PathDisplacementConstraint
from yardstick3d.sensors.geodesy import geodetic_to_enu
from yardstick3d.types import FloatArray

CUE_TOPICS = ("/ublox_driver/receiver_lla", "/ublox_driver/receiver_pvt")
REF_POSITION_TOPICS = (
    "/dji_osdk_ros/rtk_position",
    "/dji_osdk_ros/rtk_velocity",
    "/dji_osdk_ros/rtk_yaw",
)
REF_QUALITY_TOPICS = (
    "/dji_osdk_ros/rtk_connection_status",
    "/dji_osdk_ros/rtk_info_position",
    "/dji_osdk_ros/rtk_info_yaw",
)
TIME_SYNC_TOPICS = (
    "/ublox_driver/time_pulse_info",
    "/dji_osdk_ros/time_sync_fc_time_utc",
    "/dji_osdk_ros/time_sync_gps_utc",
    "/dji_osdk_ros/time_sync_nmea_msg",
    "/dji_osdk_ros/time_sync_pps_source",
)

DEV_SEQUENCE = "HKairport_GNSS01"
TEST_SEQUENCES = ("HKisland_GNSS01",)
DEGENERATE_SEQUENCES = (
    "AMvalley01", "AMvalley02", "AMvalley03",
    "Featureless_GNSS01", "Featureless_GNSS02", "Featureless_GNSS03",
)


def assert_cue_reference_isolation(cue_topics: list[str], ref_topics: list[str]) -> None:
    """Cue must come only from ublox; reference only from dji RTK. No overlap."""
    cue = set(cue_topics)
    ref = set(ref_topics)
    if cue & ref:
        raise ValueError(f"cue/reference topic overlap: {cue & ref}")
    if not cue <= set(CUE_TOPICS):
        raise ValueError(f"cue topic outside allowlist: {cue - set(CUE_TOPICS)}")
    if not ref <= set(REF_POSITION_TOPICS):
        raise ValueError(f"reference topic outside allowlist: {ref - set(REF_POSITION_TOPICS)}")
    for t in cue:
        if "rtk_" in t or "dji_osdk" in t:
            raise ValueError(f"reference-derived cue topic: {t}")
    for t in ref:
        if "ublox" in t:
            raise ValueError(f"cue-derived reference topic: {t}")


def enforce_sequence_role(sequence_id: str, role: str) -> None:
    """Dev adapter work may touch DEV only; scoring may touch TEST only."""
    if role == "development":
        if sequence_id != DEV_SEQUENCE:
            raise ValueError(f"development role restricted to {DEV_SEQUENCE}, got {sequence_id}")
    elif role == "prospective_test":
        if sequence_id not in TEST_SEQUENCES:
            raise ValueError(f"prospective role restricted to {TEST_SEQUENCES}, got {sequence_id}")
    else:
        raise ValueError(f"unknown role {role!r}")
    if sequence_id in DEGENERATE_SEQUENCES:
        raise ValueError(f"degenerate sequence excluded from primary: {sequence_id}")


def contract_sha256(contract_path: Path) -> str:
    return hashlib.sha256(contract_path.read_bytes()).hexdigest()


def check_contract_frozen(contract_path: Path, expected_sha256: str) -> None:
    actual = contract_sha256(contract_path)
    if actual != expected_sha256:
        raise ValueError(f"contract hash mismatch: {actual} != {expected_sha256}")


def nearest_index(ts: FloatArray, t: float, max_dt: float) -> int | None:
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0 or not np.isfinite(t):
        return None
    i = int(np.argmin(np.abs(ts - t)))
    if abs(float(ts[i]) - t) > max_dt:
        return None
    return i


def match_coverage(t_targets: FloatArray, t_source: FloatArray, max_dt: float) -> float:
    """Fraction of target times with a source sample within max_dt."""
    t_targets = np.asarray(t_targets, dtype=np.float64)
    t_source = np.asarray(t_source, dtype=np.float64)
    if t_targets.size == 0:
        return 0.0
    if t_source.size == 0:
        return 0.0
    d = np.abs(t_targets[:, None] - t_source[None, :]).min(axis=1)
    return float(np.mean(d <= max_dt))


@dataclass
class WindowSpec:
    sequence_id: str
    t_start: float
    t_end: float
    frame_times: FloatArray


def prospective_windows(
    sequence_id: str,
    t_bag0: float,
    duration_s: float = 16.0,
    starts_elapsed_s: tuple[float, ...] = (120.0, 240.0, 360.0, 480.0, 600.0),
    n_frames: int = 8,
) -> list[WindowSpec]:
    """Pre-registered sparse grid: elapsed offsets from first image timestamp."""
    out: list[WindowSpec] = []
    for s in starts_elapsed_s:
        t0 = float(t_bag0 + s)
        t1 = float(t0 + duration_s)
        ft = np.linspace(t0, t1, n_frames).round(6)
        out.append(WindowSpec(sequence_id=sequence_id, t_start=t0, t_end=t1, frame_times=ft))
    return out


def select_frame_indices_from_images(
    image_times: FloatArray, frame_times: FloatArray, max_dt_match: float
) -> np.ndarray:
    """Snap each ideal frame time to nearest extracted image timestamp."""
    idx = []
    for t in np.asarray(frame_times, dtype=np.float64):
        i = nearest_index(image_times, float(t), max_dt_match)
        if i is None:
            raise RuntimeError(f"no image within {max_dt_match}s of {t}")
        idx.append(i)
    return np.asarray(idx, dtype=int)


def cue_enu_from_llh(
    lat_deg: FloatArray,
    lon_deg: FloatArray,
    alt_m: FloatArray | None,
    lat0_deg: float | None = None,
    lon0_deg: float | None = None,
    alt0_m: float = 0.0,
) -> FloatArray:
    """ublox NavSatFix -> window-local ENU metres (standard WGS84, no GT fit)."""
    return geodetic_to_enu(lat_deg, lon_deg, alt_m, lat0_deg, lon0_deg, alt0_m)


def cue_polyline_lengths(
    cue_t: FloatArray,
    cue_enu: FloatArray,
    t_clip: FloatArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval cue path b_k via polyline + endpoint interpolation.

    Returns (b, coverage) with b_k the polyline length of cue fixes strictly
    inside (t_k, t_{k+1}] plus linearly interpolated endpoints, coverage_k the
    fraction of the interval spanned by real fixes (1.0 = fully covered).
    """
    cue_t = np.asarray(cue_t, dtype=np.float64)
    cue_enu = np.asarray(cue_enu, dtype=np.float64)
    t_clip = np.asarray(t_clip, dtype=np.float64)
    order = np.argsort(cue_t)
    cue_t, cue_enu = cue_t[order], cue_enu[order]
    b: list[float] = []
    cov: list[float] = []
    for k in range(len(t_clip) - 1):
        t0, t1 = float(t_clip[k]), float(t_clip[k + 1])
        if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
            b.append(float("nan"))
            cov.append(0.0)
            continue
        # endpoint interpolation of cue track at t0/t1
        try:
            p0 = _interp_track(cue_t, cue_enu, t0)
            p1 = _interp_track(cue_t, cue_enu, t1)
        except RuntimeError:
            b.append(float("nan"))
            cov.append(0.0)
            continue
        m = (cue_t > t0) & (cue_t <= t1)
        pts = [p0, *cue_enu[m].tolist(), p1] if False else [p0] + [cue_enu[i] for i in np.flatnonzero(m)] + [p1]
        pts = np.asarray(pts, dtype=np.float64)
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        b.append(float(segs.sum()))
        span = float(np.sum(np.diff(np.concatenate([[t0], cue_t[m], [t1]])))) if m.any() else 0.0
        cov.append(float(np.clip(span / (t1 - t0), 0.0, 1.0)))
    return np.asarray(b), np.asarray(cov)


def _interp_track(cue_t: FloatArray, cue_enu: FloatArray, t: float) -> np.ndarray:
    if t < cue_t[0] or t > cue_t[-1]:
        raise RuntimeError("cue track does not span interval endpoint")
    j = int(np.searchsorted(cue_t, t))
    if j == 0:
        return np.asarray(cue_enu[0], dtype=np.float64)
    if j >= len(cue_t):
        return np.asarray(cue_enu[-1], dtype=np.float64)
    t_a, t_b = float(cue_t[j - 1]), float(cue_t[j])
    w = 0.0 if t_b == t_a else (t - t_a) / (t_b - t_a)
    return (1 - w) * np.asarray(cue_enu[j - 1]) + w * np.asarray(cue_enu[j])


def constraints_from_ublox_displacement(
    t_clip: FloatArray,
    cue_t: FloatArray,
    cue_enu: FloatArray,
    max_dt_match: float = 0.06,
    min_coverage: float = 0.8,
    sigma: float = 1.0,
) -> tuple[ConstraintSet, dict[str, Any]]:
    """Build PathDisplacementConstraints s*a_k ~= b_k from ublox polyline.

    Intervals with coverage < min_coverage or non-finite b_k are DROPPED
    (missing-sensor policy), never zero-filled.
    """
    from yardstick3d.constraints.base import ConstraintSet as CS

    b, cov = cue_polyline_lengths(cue_t, cue_enu, np.asarray(t_clip, dtype=np.float64))
    cs = CS()
    kept, dropped = 0, 0
    for k in range(len(b)):
        if not np.isfinite(b[k]) or cov[k] < min_coverage:
            dropped += 1
            continue
        cs.add(
            PathDisplacementConstraint(
                t_i=float(t_clip[k]),
                t_j=float(t_clip[k + 1]),
                length_m=float(b[k]),
                sigma=float(sigma),
                max_dt_match=float(max_dt_match),
            )
        )
        kept += 1
    info = {
        "n_intervals": int(len(b)),
        "n_kept": int(kept),
        "n_dropped": int(dropped),
        "coverage": [float(c) for c in cov],
        "cue_lengths_m": [float(x) if np.isfinite(x) else None for x in b],
    }
    return cs, info


def interpolate_reference_at(
    ref_t: FloatArray, ref_enu: FloatArray, t_clip: FloatArray
) -> np.ndarray:
    """Linear interpolation of withheld reference ENU at frame times (eval only)."""
    ref_t = np.asarray(ref_t, dtype=np.float64)
    ref_enu = np.asarray(ref_enu, dtype=np.float64)
    t_clip = np.asarray(t_clip, dtype=np.float64)
    order = np.argsort(ref_t)
    ref_t, ref_enu = ref_t[order], ref_enu[order]
    out = np.zeros((len(t_clip), 3))
    for d in range(3):
        out[:, d] = np.interp(t_clip, ref_t, ref_enu[:, d], left=np.nan, right=np.nan)
    return out


@dataclass
class SyncManifest:
    sequence_id: str
    t_bag0: float
    n_images: int
    n_cue_fixes: int
    camera_cue_match_fraction: float
    time_sync_notes: dict[str, Any] = field(default_factory=dict)
    max_dt_match_camera_cue_s: float = 0.06
    max_dt_match_camera_ref_s: float = 0.15
    ref_positions_opened: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "t_bag0": float(self.t_bag0),
            "n_images": int(self.n_images),
            "n_cue_fixes": int(self.n_cue_fixes),
            "camera_cue_match_fraction": float(self.camera_cue_match_fraction),
            "max_dt_match_camera_cue_s": float(self.max_dt_match_camera_cue_s),
            "max_dt_match_camera_ref_s": float(self.max_dt_match_camera_ref_s),
            "time_sync_notes": self.time_sync_notes,
            "ref_positions_opened": bool(self.ref_positions_opened),
        }
