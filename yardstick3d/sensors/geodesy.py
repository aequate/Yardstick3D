from __future__ import annotations

import numpy as np

from yardstick3d.types import FloatArray

_A = 6378137.0
_E2 = 6.69437999014e-3


def geodetic_to_enu(
    lat_deg: FloatArray,
    lon_deg: FloatArray,
    alt_m: FloatArray | None = None,
    lat0_deg: float | None = None,
    lon0_deg: float | None = None,
    alt0_m: float = 0.0,
) -> FloatArray:
    """WGS84 geodetic → local ENU meters about the first finite sample (or given origin)."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    if alt_m is None:
        h = np.zeros_like(lat)
    else:
        h = np.asarray(alt_m, dtype=np.float64)
    if lat0_deg is None:
        lat0 = float(lat[np.isfinite(lat)][0])
        lon0 = float(lon[np.isfinite(lon)][0])
        h0 = float(h[np.isfinite(h)][0]) if np.any(np.isfinite(h)) else 0.0
    else:
        lat0 = np.radians(lat0_deg)
        lon0 = np.radians(lon0_deg)
        h0 = float(alt0_m)
    xyz = _geodetic_to_ecef(lat, lon, h)
    xyz0 = _geodetic_to_ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))[0]
    d = xyz - xyz0
    sL, cL = np.sin(lon0), np.cos(lon0)
    sB, cB = np.sin(lat0), np.cos(lat0)
    e = -sL * d[:, 0] + cL * d[:, 1]
    n = -sB * cL * d[:, 0] - sB * sL * d[:, 1] + cB * d[:, 2]
    u = cB * cL * d[:, 0] + cB * sL * d[:, 1] + sB * d[:, 2]
    return np.column_stack([e, n, u])


def _geodetic_to_ecef(lat: FloatArray, lon: FloatArray, h: FloatArray) -> FloatArray:
    s, c = np.sin(lat), np.cos(lat)
    n = _A / np.sqrt(1.0 - _E2 * s * s)
    x = (n + h) * c * np.cos(lon)
    y = (n + h) * c * np.sin(lon)
    z = (n * (1.0 - _E2) + h) * s
    return np.column_stack([x, y, z])


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(min(1.0, a))))
