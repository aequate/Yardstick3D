from __future__ import annotations

import numpy as np

from yardstick3d.sensors.speed_integrate import SPEED_INTEGRATION_METHOD, integrate_speed


def test_method_is_trapezoidal_frozen():
    assert SPEED_INTEGRATION_METHOD == "trapezoidal_linear"


def test_constant_speed_distance():
    t = np.linspace(0, 10, 11)
    v = np.full_like(t, 2.0)
    d, n = integrate_speed(0.0, 10.0, t, v)
    assert abs(d - 20.0) < 1e-9
    assert n >= 2


def test_irregular_samples():
    t = np.array([0.0, 0.5, 2.0, 5.0])
    v = np.array([1.0, 1.0, 1.0, 1.0])
    d, _ = integrate_speed(0.0, 5.0, t, v)
    assert abs(d - 5.0) < 1e-9


def test_missing_and_invalid_dropped():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    v = np.array([1.0, -1.0, 1.0, np.nan, 1.0])
    d, n = integrate_speed(0.0, 4.0, t, v)
    assert n >= 2
    assert d is not None and d > 0
    # must not treat invalid as 0 which would shrink the integral a lot
    assert d > 2.0
