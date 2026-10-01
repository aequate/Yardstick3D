from __future__ import annotations


import numpy as np

from yardstick3d.adapters.base import AdapterCapabilities, FoundationModelAdapter, PredictRequest
from yardstick3d.geometry.cameras import camera_centers_from_w2c
from yardstick3d.types import PredictionBundle

DA3_SMALL_ID = "depth-anything/DA3-SMALL"
DA3_IS_METRIC = False  # any-view relative model; NOT DA3METRIC / NESTED

DA3_HUB_CANDIDATES: dict[str, tuple[str, ...]] = {
    "da3-small": ("depth-anything/DA3-SMALL", "depth-anything/da3-small"),
    "da3-base": ("depth-anything/DA3-BASE", "depth-anything/da3-base"),
    "da3-large": ("depth-anything/DA3-LARGE", "depth-anything/da3-large"),
}


def architecture_name(model_name: str) -> str:
    raw = model_name.strip()
    key = raw.lower().replace("_", "-")
    if "metric" in key or "nested" in key:
        raise RuntimeError(
            f"Refusing {model_name!r}: never load da3metric/nested as the relative backbone"
        )
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    if key not in DA3_HUB_CANDIDATES:
        raise RuntimeError(f"Unknown DA3 architecture {model_name!r}")
    return key


def hub_ids_for(model_name: str) -> tuple[str, ...]:
    arch = architecture_name(model_name)
    return DA3_HUB_CANDIDATES[arch]


class DA3Adapter(FoundationModelAdapter):
    caps = AdapterCapabilities(
        name="da3-small",
        min_vram_gb=2.0,
        max_views_on_6gb=8,
        poses=True,
        depth=True,
        metric=False,
        pose_kind="w2c",
        license="Apache-2.0",
        local=True,
        notes="Relative any-view DA3. Not DA3METRIC. Not nested-giant. Weights via from_pretrained.",
    )

    def __init__(self, model_name: str = "da3-small", process_res: int = 378) -> None:
        self.model_name = model_name
        self.process_res = int(process_res)
        self._model = None
        self.device = "cpu"
        self.hub_id: str | None = None
        self.from_pretrained_ok: bool = False
        self.arch: str | None = None

    def load(self, device: str = "cuda") -> None:
        arch = architecture_name(self.model_name)
        hubs = hub_ids_for(self.model_name)
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise RuntimeError(
                "DA3 not installed. Create .venv-da3 and pip install torch + depth_anything_3. "
                f"Original: {e}"
            ) from e
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.arch = arch
        last_err: Exception | None = None
        self._model = None
        self.hub_id = None
        self.from_pretrained_ok = False
        for hid in hubs:
            try:
                self._model = DepthAnything3.from_pretrained(hid, model_name=arch)
                self.hub_id = hid
                self.from_pretrained_ok = True
                break
            except Exception as e:
                last_err = e
                self._model = None
        if self._model is None:
            raise RuntimeError(
                f"DepthAnything3.from_pretrained failed for {hubs}: {last_err}"
            ) from last_err
        self._model = self._model.to(self.device)
        self._model.eval()

    def predict(self, req: PredictRequest) -> PredictionBundle:
        if self._model is None:
            raise RuntimeError("call load() first")
        import torch

        images = list(req.images)
        ts = (
            np.asarray(req.timestamps, dtype=np.float64)
            if req.timestamps is not None
            else np.arange(len(images), dtype=np.float64)
        )
        with torch.inference_mode():
            pred = self._model.inference(
                image=images,
                process_res=self.process_res,
                process_res_method="upper_bound_resize",
                ref_view_strategy="middle" if len(images) >= 3 else "first",
            )
        ext = np.asarray(pred.extrinsics, dtype=np.float64)
        if ext.ndim == 3 and ext.shape[-2:] == (4, 4):
            T = ext[:, :3, :4]
        else:
            T = ext
        K = np.asarray(pred.intrinsics, dtype=np.float64) if pred.intrinsics is not None else None
        depth = np.asarray(pred.depth, dtype=np.float32) if pred.depth is not None else None
        conf = np.asarray(pred.conf, dtype=np.float32) if getattr(pred, "conf", None) is not None else None
        hid = self.hub_id or "unloaded"
        bundle = PredictionBundle(
            timestamps=ts,
            T_w2c=T,
            K=K,
            depth_z=depth,
            camera_centers=camera_centers_from_w2c(T),
            depth_conf=conf,
            is_metric=DA3_IS_METRIC,
            model_id=f"{self.arch or self.model_name}/{hid}",
            aux={
                "process_res": self.process_res,
                "device": self.device,
                "hub_id": hid,
                "from_pretrained": bool(self.from_pretrained_ok),
                "is_metric": False,
            },
        )
        return bundle

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
