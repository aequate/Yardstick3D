"""ADVIO loader with hard inference / evaluation split.

IMU (accelerometer, gyro) is never attached to inference channels.
Ground-truth pose.csv, fixpoints, ARKit, ARCore, Tango are evaluation-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from yardstick3d.sensors.geodesy import geodetic_to_enu
from yardstick3d.types import FloatArray

IMU_FILENAMES = ("accelerometer.csv", "gyro.csv", "gyroscope.csv", "magnetometer.csv")
EVAL_ONLY = ("pose.csv", "poses.csv", "fixpoints.csv", "arkit.csv", "arcore.csv", "raw.csv", "area-learning.csv")


@dataclass
class InferenceChannels:
    sequence_id: str
    frame_timestamps: FloatArray
    frame_indices: FloatArray
    video_path: Path | None
    location_t: FloatArray
    lat: FloatArray
    lon: FloatArray
    hacc_m: FloatArray
    alt_m: FloatArray
    vacc_m: FloatArray
    speed_mps: FloatArray | None
    speed_column_present: bool
    baro_t: FloatArray
    baro_pressure: FloatArray
    baro_rel_alt_m: FloatArray
    imu_allowed: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def assert_imu_denied(self) -> None:
        if self.imu_allowed:
            raise RuntimeError("imu_allowed=True on flagship ADVIO run")
        if any(k in self.notes for k in ("accelerometer", "gyro", "gyroscope")):
            raise RuntimeError("IMU leaked into inference notes")


@dataclass
class EvaluationChannels:
    timestamps: FloatArray
    centers: FloatArray
    quat_wxyz: FloatArray
    source: str = "ground-truth/pose.csv"


@dataclass
class AdvioSequence:
    root: Path
    sequence_id: str
    inference: InferenceChannels
    evaluation: EvaluationChannels

    def frames_video_path(self) -> Path:
        if self.inference.video_path is None or not self.inference.video_path.exists():
            raise FileNotFoundError("iphone/frames.mov missing")
        return self.inference.video_path


def _load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def _find_seq_dir(root: Path, sequence_id: str) -> Path:
    if (root / "iphone").is_dir() or (root / "ground-truth").is_dir():
        return root
    name = f"advio-{int(sequence_id):02d}" if str(sequence_id).isdigit() else str(sequence_id)
    candidates = [root / name, root / f"advio-{sequence_id}", root / f"advio_{sequence_id}"]
    for c in candidates:
        if (c / "iphone").is_dir() or (c / "ground-truth").is_dir():
            return c
    raise FileNotFoundError(f"ADVIO sequence not found under {root} ({name})")


def load_advio(root: str | Path, sequence_id: str = "20", imu_allowed: bool = False) -> AdvioSequence:
    root = Path(root)
    seq = _find_seq_dir(root, sequence_id)
    iphone = seq / "iphone"
    frames = _load_csv(iphone / "frames.csv")
    loc_path = iphone / "platform-locations.csv"
    if not loc_path.exists():
        loc_path = iphone / "platform-location.csv"
    loc = _load_csv(loc_path)
    baro = _load_csv(iphone / "barometer.csv")
    speed_col = None
    speed_present = False
    if loc.shape[1] >= 7:
        speed_col = loc[:, 6]
        speed_present = True
        vacc = loc[:, 5]
        alt = loc[:, 4]
        hacc = loc[:, 3]
    elif loc.shape[1] == 6:
        # Observed on advio-20: t, lat, lon, hacc, alt, vacc. No Speed column.
        hacc, alt, vacc = loc[:, 3], loc[:, 4], loc[:, 5]
    else:
        raise ValueError(f"Unexpected platform-locations width {loc.shape[1]}")

    video = iphone / "frames.mov"
    inf = InferenceChannels(
        sequence_id=f"advio-{int(sequence_id):02d}" if str(sequence_id).isdigit() else str(sequence_id),
        frame_timestamps=frames[:, 0],
        frame_indices=frames[:, 1] if frames.shape[1] > 1 else np.arange(len(frames)),
        video_path=video if video.exists() else None,
        location_t=loc[:, 0],
        lat=loc[:, 1],
        lon=loc[:, 2],
        hacc_m=hacc,
        alt_m=alt,
        vacc_m=vacc,
        speed_mps=speed_col,
        speed_column_present=speed_present,
        baro_t=baro[:, 0],
        baro_pressure=baro[:, 1],
        baro_rel_alt_m=baro[:, 2],
        imu_allowed=bool(imu_allowed),
        notes={
            "location_columns": int(loc.shape[1]),
            "speed_column_present": speed_present,
            "imu_files_on_disk": [n for n in IMU_FILENAMES if (iphone / n).exists()],
            "imu_loaded": False,
        },
    )
    if not imu_allowed:
        inf.assert_imu_denied()

    gt_path = seq / "ground-truth" / "pose.csv"
    if not gt_path.exists():
        gt_path = seq / "ground-truth" / "poses.csv"
    gt = _load_csv(gt_path)
    ev = EvaluationChannels(
        timestamps=gt[:, 0],
        centers=gt[:, 1:4],
        quat_wxyz=gt[:, 4:8],
        source=str(gt_path.as_posix()),
    )
    return AdvioSequence(root=seq, sequence_id=inf.sequence_id, inference=inf, evaluation=ev)


def _enu_interp(t_q: float, t: FloatArray, xyz: FloatArray) -> FloatArray:
    return np.array([np.interp(t_q, t, xyz[:, k]) for k in range(3)], dtype=np.float64)


def corelocation_interval_distance(inf: InferenceChannels, t_i: float, t_j: float) -> float | None:
    """GPS polyline length on [t_i, t_j], interpolating ENU endpoints.

    Interior-only samples drop straddling segments when GPS is ~1 Hz and
    frames are farther apart. Always include interpolated p(t_i), p(t_j).
    """
    xyz = geodetic_to_enu(inf.lat, inf.lon, inf.alt_m)
    t = inf.location_t
    if t_j <= t_i:
        return None
    if t.size < 1:
        return None
    if t_i < t[0] - 1.0 or t_j > t[-1] + 1.0:
        return None
    p0 = _enu_interp(t_i, t, xyz)
    p1 = _enu_interp(t_j, t, xyz)
    interior = (t > t_i) & (t < t_j)
    if not np.any(interior):
        return float(np.linalg.norm(p1 - p0))
    pts = np.vstack([p0, xyz[interior], p1])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
