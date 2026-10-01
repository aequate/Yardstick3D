from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from yardstick3d.adapters.dummy import DummyFoundationModel
from yardstick3d.adapters.base import PredictRequest
from yardstick3d.adapters.da3 import DA3Adapter, architecture_name, hub_ids_for
from yardstick3d.adapters.vggt import VGGTAdapter, VGGT_BLOCKER


def test_dummy_adapter():
    m = DummyFoundationModel(seed=0, true_scale=2.2)
    m.load()
    imgs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(6)]
    pred = m.predict(PredictRequest(images=imgs))
    assert pred.T_w2c.shape[0] == 6
    assert pred.T_w2c.shape[-2:] == (3, 4)
    assert not pred.is_metric


def test_vggt_blocked():
    a = VGGTAdapter()
    with pytest.raises(RuntimeError, match="VGGT"):
        a.load()
    assert "vggt_runner" in VGGT_BLOCKER


def test_da3_hub_map_and_forbid_metric_nested():
    assert hub_ids_for("da3-small") == ("depth-anything/DA3-SMALL", "depth-anything/da3-small")
    assert hub_ids_for("da3-base") == ("depth-anything/DA3-BASE", "depth-anything/da3-base")
    assert architecture_name("depth-anything/DA3-SMALL") == "da3-small"
    for name in ("da3metric-large", "da3nested-giant-large", "da3-metric", "DA3METRIC"):
        with pytest.raises(RuntimeError, match="metric|nested"):
            architecture_name(name)
        a = DA3Adapter(model_name=name)
        with pytest.raises(RuntimeError, match="metric|nested"):
            a.load("cpu")


def test_da3_load_calls_from_pretrained(monkeypatch):
    calls: dict = {"from_pretrained": [], "ctor": 0}

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeDA3:
        def __init__(self, *args, **kwargs):
            calls["ctor"] += 1
            raise AssertionError("architecture-only DepthAnything3() must not be used")

        @classmethod
        def from_pretrained(cls, hub_id, **kwargs):
            calls["from_pretrained"].append({"hub_id": hub_id, "kwargs": kwargs})
            assert kwargs.get("model_name") == "da3-small"
            return FakeModel()

    fake_cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda
    api = types.ModuleType("depth_anything_3.api")
    api.DepthAnything3 = FakeDA3
    pkg = types.ModuleType("depth_anything_3")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "depth_anything_3", pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", api)

    a = DA3Adapter(model_name="da3-small")
    a.load("cuda")
    assert calls["ctor"] == 0
    assert len(calls["from_pretrained"]) == 1
    hid = calls["from_pretrained"][0]["hub_id"]
    assert hid in {"depth-anything/DA3-SMALL", "depth-anything/da3-small"}
    assert a.from_pretrained_ok is True
    assert a.hub_id == hid
    assert a._model is not None
