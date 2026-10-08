"""CPU test of the token-level JEPA objective on the toy simulator."""

from __future__ import annotations

import numpy as np
import torch

from project import cache_latents, pretrain_mae
from project.common import load_encoder
from project.jepa import TokenJEPA
from project.tests.test_ls_benchmark import MAE_ARGS


def test_jepa_trains_saves_and_loads(tmp_path):
    out = tmp_path / "jepa"
    pretrain_mae.main(MAE_ARGS + ["--jepa", "--mask-mode", "blockplus", "--steps", "3", "--out", str(out)])
    assert (out / "mae.pt").is_file() and (out / "DONE").is_file()
    enc, meta = load_encoder(str(out / "mae.pt"), "cpu")
    assert isinstance(enc.model, TokenJEPA) and enc.kind == "mae"
    model = enc.model
    # the EMA target moved towards the context encoder but is not equal to it
    pc = next(model.context.transformer.parameters()).detach()
    pt = next(model.target.transformer.parameters()).detach()
    assert not torch.allclose(pc, pt) or model.ema == 1.0
    # a forward pass gives a finite loss and per-token latent targets of the encoder width
    cache_latents.main(["--ckpt", str(out / "mae.pt"), "--out", str(tmp_path / "latents.pt"), "--stride", "0.5", "--device", "cpu", "--workers", "0", "--batch-size", "16"])
    cache = torch.load(tmp_path / "latents.pt", map_location="cpu", weights_only=False)
    assert cache["z"].shape[1] == model.embed_dim and np.isfinite(cache["z"].float().numpy()).all()
    values = torch.randn(2, 12, meta.spec.n_channels)
    positions = torch.cat([torch.sort(torch.rand(2, 1, 12) * 5, -1).values + 1, torch.full((2, 1, 12), 472.0)], 1)
    with torch.no_grad():
        o = model(values, positions)
    assert torch.isfinite(o.loss) and o.pred.shape[-1] == model.embed_dim and o.pred.shape == o.target.shape
    assert "target_std" in model.last_stats
