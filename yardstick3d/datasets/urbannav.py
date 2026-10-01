"""UrbanNav-HK Medium-Urban-1 prospective adapter (sensor-only + isolation).

Mirrors yardstick3d/datasets/mars_lvig.py interfaces for the v3 contract
(configs/prospective_metric_grounding_v3.yaml). Operates on EXTRACTED CSVs
(no ROS dependency here); bag parsing lives in scripts/urbannav_extract.py
(Stage 1 sensor-only / Stage 2 one-time scoring) and is never imported by
unit tests or the solver path.

Cue: ublox RINEX standalone SPP positions (RTKLIB rnx2rtkp, defaults, no base
corrections, no SPAN/IE inputs) as extracted CSV (t, lat_deg, lon_deg, alt_m).
Reference: SPAN-CPT+IE inspvax / GT-raw positions as extracted CSV
(t, lat_deg, lon_deg, alt_m) + quality/status fields. Topics/sources are
disjoint by construction; isolation enforced below.

Timestamps: float seconds, GPS-time domain after gps2utc discipline
(utctime = gps_week*604800 + gps_seconds - 18 + 315964800). Camera stamps are
ROS arrival times (software-sync limitation, disclosed in contract).
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

CUE_SOURCES = ("ublox_f9p_nmea_gga",)
REF_SOURCES = ("span_cpt_ie_inspvax", "span_cpt_ie_gt_raw")
CUE_TOPICS = ("/ublox_spp_csv",)
CUE_FILES = ("UrbanNav-HK-Medium-Urban-1.ublox.f9p.nmea",)
REF_TOPICS = ("/novatel_data/inspvax", "UrbanNav_TST_GT_raw.txt")
TIME_SYNC_TOPICS = ("/time_reference",)

NMEA_ALLOWED_QUALITY = (1, 2)
NMEA_MIN_SATS = 4

VALIDATION_STARTS = (90.0, 180.0)
TEST_STARTS = (270.0, 360.0, 450.0, 540.0, 630.0)


def assert_cue_reference_isolation(cue_sources: list[str], ref_sources: list[str]) -> None:
    cue, ref = set(cue_sources), set(ref_sources)
    if cue & ref:
        raise ValueError(f"cue/reference source overlap: {cue & ref}")
    if not cue <= set(CUE_SOURCES):
        raise ValueError(f"cue source outside allowlist: {cue - set(CUE_SOURCES)}")
    if not ref <= set(REF_SOURCES):
        raise ValueError(f"reference source outside allowlist: {ref - set(REF_SOURCES)}")
    for s in cue:
        if "span" in s or "inspvax" in s or "gt_raw" in s or "novatel" in s:
            raise ValueError(f"reference-derived cue source: {s}")


def enforce_window_role(start_elapsed_s: float, role: str) -> None:
    if role == "validation":
        if start_elapsed_s not in VALIDATION_STARTS:
            raise ValueError(f"validation restricted to {VALIDATION_STARTS}, got {start_elapsed_s}")
    elif role == "prospective_test":
        if start_elapsed_s not in TEST_STARTS:
            raise ValueError(f"test restricted to {TEST_STARTS}, got {start_elapsed_s}")
    else:
        raise ValueError(f"unknown role {role!r}")


def contract_sha256(contract_path: Path) -> str:
    return hashlib.sha256(contract_path.read_bytes()).hexdigest()


def check_contract_frozen(contract_path: Path, expected_sha256: str) -> None:
    actual = contract_sha256(contract_path)
    if actual != expected_sha256:
        raise ValueError(f"contract hash mismatch: {actual} != {expected_sha256}")


def gps2utc(gps_week: float, gps_seconds: float, leap_s: float = 18.0) -> float:
    """GETTING_STARTED.md conversion (May-2021 leap count frozen in contract)."""
    return float(gps_week * 604800.0 + gps_seconds - leap_s + 315964800.0)


def _nmea_dm_to_deg(dm: str, hemi: str, deg_width: int) -> float:
    if not dm or not hemi:
        raise ValueError("empty NMEA coordinate")
    deg = float(dm[:deg_width])
    minutes = float(dm[deg_width:])
    val = deg + minutes / 60.0
    if hemi in ("S", "W"):
        val = -val
    elif hemi not in ("N", "E"):
        raise ValueError(f"bad hemisphere {hemi!r}")
    return val


def _nmea_day_bounds(date_ddmmyy: str) -> tuple[int, int, int]:
    dd, mm, yy = int(date_ddmmyy[0:2]), int(date_ddmmyy[2:4]), int(date_ddmmyy[4:6])
    return (2000 + yy, mm, dd)


def parse_nmea_gga_cue(
    text: str,
    date_ddmmyy: str,
    allowed_quality: tuple[int, ...] = NMEA_ALLOWED_QUALITY,
    min_sats: int = NMEA_MIN_SATS,
) -> dict[str, np.ndarray]:
    """Parse GNGGA/GPGGA cue fixes (v4 contract). Gated epochs DROPPED (missing).

    Time base: GGA UTC hhmmss.ss + RMC calendar date (same drive day; midnight
    rollover handled by day carry when tod decreases). No GT, no fitting.
    """
    import calendar
    import datetime

    yyyy, mm, dd = _nmea_day_bounds(date_ddmmyy)
    day = datetime.date(yyyy, mm, dd)
    t_list, lat_list, lon_list, alt_list, q_list, ns_list = [], [], [], [], [], []
    prev_tod = None
    day_off = 0
    for ln in text.splitlines():
        if not ln.startswith("$"):
            continue
        tag = ln.split(",")[0]
        if not tag.endswith("GGA"):
            continue
        f = ln.split(",")
        if len(f) < 10:
            continue
        try:
            tod = float(f[1]) if f[1] else float("nan")
            lat = _nmea_dm_to_deg(f[2], f[3], 2)
            lon = _nmea_dm_to_deg(f[4], f[5], 3)
            q = int(f[6]) if f[6] else 0
            ns = int(f[7]) if f[7] else 0
            alt = float(f[9]) if f[9] else float("nan")
        except (ValueError, IndexError):
            continue
        if q not in allowed_quality or ns < min_sats:
            continue
        if not (np.isfinite(tod) and np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)):
            continue
        if prev_tod is not None and tod < prev_tod - 3600.0:
            day_off += 1  # UTC midnight rollover during drive
        prev_tod = tod
        hh, mi, ss = int(tod // 10000), int((tod // 100) % 100), tod % 100
        dt = datetime.datetime.combine(day + datetime.timedelta(days=day_off),
                                       datetime.time(hh, mi, 0)) + datetime.timedelta(seconds=ss)
        t_list.append(calendar.timegm(dt.timetuple()))
        lat_list.append(lat)
        lon_list.append(lon)
        alt_list.append(alt)
        q_list.append(q)
        ns_list.append(ns)
    return {
        "t": np.asarray(t_list, dtype=np.float64),
        "lat_deg": np.asarray(lat_list, dtype=np.float64),
        "lon_deg": np.asarray(lon_list, dtype=np.float64),
        "alt_m": np.asarray(alt_list, dtype=np.float64),
        "quality": np.asarray(q_list, dtype=int),
        "nsat": np.asarray(ns_list, dtype=int),
    }


def extract_rmc_date(text: str) -> str:
    """Drive calendar date (ddmmyy) from first valid RMC sentence (cue metadata)."""
    for ln in text.splitlines():
        if not ln.startswith("$"):
            continue
        if not ln.split(",")[0].endswith("RMC"):
            continue
        f = ln.split(",")
        if len(f) > 9 and len(f[9]) == 6 and f[9].isdigit():
            return f[9]
    raise RuntimeError("no RMC date in NMEA text")


def nearest_index(ts: FloatArray, t: float, max_dt: float) -> int | None:
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0 or not np.isfinite(t):
        return None
    i = int(np.argmin(np.abs(ts - t)))
    if abs(float(ts[i]) - t) > max_dt:
        return None
    return i


def match_coverage(t_targets: FloatArray, t_source: FloatArray, max_dt: float) -> float:
    t_targets = np.asarray(t_targets, dtype=np.float64)
    t_source = np.asarray(t_source, dtype=np.float64)
    if t_targets.size == 0 or t_source.size == 0:
        return 0.0
    d = np.abs(t_targets[:, None] - t_source[None, :]).min(axis=1)
    return float(np.mean(d <= max_dt))


@dataclass
class WindowSpec:
    t_start: float
    t_end: float
    frame_times: FloatArray
    role: str


def prospective_windows(
    t_bag0: float,
    duration_s: float = 16.0,
    n_frames: int = 8,
) -> list[WindowSpec]:
    out: list[WindowSpec] = []
    for s in VALIDATION_STARTS:
        t0, t1 = float(t_bag0 + s), float(t_bag0 + s + duration_s)
        out.append(WindowSpec(t0, t1, np.linspace(t0, t1, n_frames).round(6), "validation"))
    for s in TEST_STARTS:
        t0, t1 = float(t_bag0 + s), float(t_bag0 + s + duration_s)
        out.append(WindowSpec(t0, t1, np.linspace(t0, t1, n_frames).round(6), "prospective_test"))
    return out


def select_frame_indices_from_images(
    image_times: FloatArray, frame_times: FloatArray, max_dt_match: float
) -> np.ndarray:
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
    return geodetic_to_enu(lat_deg, lon_deg, alt_m, lat0_deg, lon0_deg, alt0_m)


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


def cue_polyline_lengths(
    cue_t: FloatArray,
    cue_enu: FloatArray,
    t_clip: FloatArray,
) -> tuple[np.ndarray, np.ndarray]:
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
        try:
            p0 = _interp_track(cue_t, cue_enu, t0)
            p1 = _interp_track(cue_t, cue_enu, t1)
        except RuntimeError:
            b.append(float("nan"))
            cov.append(0.0)
            continue
        m = (cue_t > t0) & (cue_t <= t1)
        pts = [p0] + [cue_enu[i] for i in np.flatnonzero(m)] + [p1]
        pts = np.asarray(pts, dtype=np.float64)
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        b.append(float(segs.sum()))
        span = float(np.sum(np.diff(np.concatenate([[t0], cue_t[m], [t1]])))) if m.any() else 0.0
        cov.append(float(np.clip(span / (t1 - t0), 0.0, 1.0)))
    return np.asarray(b), np.asarray(cov)


def constraints_from_ublox_spp_displacement(
    t_clip: FloatArray,
    cue_t: FloatArray,
    cue_enu: FloatArray,
    max_dt_match: float = 0.15,
    min_coverage: float = 0.8,
    sigma: float = 1.0,
) -> tuple[ConstraintSet, dict[str, Any]]:
    cs = ConstraintSet()
    b, cov = cue_polyline_lengths(cue_t, cue_enu, np.asarray(t_clip, dtype=np.float64))
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
    t_bag0: float
    n_images_left: int
    n_cue_fixes: int
    camera_cue_match_fraction: float
    time_reference_notes: dict[str, Any] = field(default_factory=dict)
    max_dt_match_camera_cue_s: float = 0.15
    max_dt_match_camera_ref_s: float = 0.6
    ref_positions_opened: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_bag0": float(self.t_bag0),
            "n_images_left": int(self.n_images_left),
            "n_cue_fixes": int(self.n_cue_fixes),
            "camera_cue_match_fraction": float(self.camera_cue_match_fraction),
            "max_dt_match_camera_cue_s": float(self.max_dt_match_camera_cue_s),
            "max_dt_match_camera_ref_s": float(self.max_dt_match_camera_ref_s),
            "time_reference_notes": self.time_reference_notes,
            "ref_positions_opened": bool(self.ref_positions_opened),
        }
