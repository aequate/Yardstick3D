from __future__ import annotations


from yardstick3d.adapters.base import AdapterCapabilities, FoundationModelAdapter, PredictRequest
from yardstick3d.datasets.synthetic import generate_scene
from yardstick3d.types import PredictionBundle


class DummyFoundationModel(FoundationModelAdapter):
    """Stand-in 3DFM: unknown-scale synthetic geometry. No network weights."""

    caps = AdapterCapabilities(
        name="dummy-synthetic",
        min_vram_gb=0.0,
        max_views_on_6gb=10**6,
        poses=True,
        depth=False,
        metric=False,
        pose_kind="w2c",
        license="MIT",
        local=True,
        notes="Generates a PredictionBundle with hidden Sim(3). Not a learned model.",
    )

    def __init__(self, seed: int = 0, true_scale: float = 2.5, regime: str = "line") -> None:
        self.seed = seed
        self.true_scale = true_scale
        self.regime = regime
        self.scene = None

    def load(self, device: str = "cpu") -> None:
        return None

    def predict(self, req: PredictRequest) -> PredictionBundle:
        n = len(req.images) if req.images else 20
        self.scene = generate_scene(n_frames=n, true_scale=self.true_scale, seed=self.seed, regime=self.regime)
        return self.scene.prediction
