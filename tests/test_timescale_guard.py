"""Header-timescale guard: the cross-backbone v1 GPST defect must be caught."""
from __future__ import annotations

import numpy as np
import pytest

from yardstick3d.datasets import timescale as ts


def test_v1_observed_offsets_are_classified():
    # values measured on HKairport_GNSS01 (artifacts/cross_backbone_v1/cue_timescale_check.json)
    assert ts.classify_timescale(18.028) == "gpst"
    assert ts.classify_timescale(-0.07) == "utc"


def test_gpst_converted_with_constant_not_fit():
    rec = np.arange(100.0, 110.0)
    hdr = rec + 18.03
    kinds = ts.require_common_timescale({"cue": (hdr, rec), "cam": (rec - 0.07, rec)})
    assert kinds == {"cue": "gpst", "cam": "utc"}
    assert np.allclose(ts.to_utc(hdr, "gpst") - rec, 0.03)


def test_unexplained_offset_refused():
    rec = np.arange(10.0)
    with pytest.raises(RuntimeError):
        ts.require_common_timescale({"cue": (rec + 5.0, rec)})
    with pytest.raises(ValueError):
        ts.to_utc(rec, "unknown")
