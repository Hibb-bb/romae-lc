"""CPU test of the token-level JEPA objective on the toy simulator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from project import cache_latents, pretrain_mae
from project.common import load_encoder
from project.jepa import TokenJEPA
from project.tests._args import MAE_ARGS


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


def test_jepa_hybrid_and_warm_start(tmp_path):
    mae = tmp_path / "mae"
    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(mae)])
    out = tmp_path / "jepa"
    pretrain_mae.main(MAE_ARGS + ["--jepa", "--jepa-recon-weight", "0.5", "--jepa-init", str(mae / "mae.pt"), "--steps", "2", "--out", str(out)])
    enc, _ = load_encoder(str(out / "mae.pt"), "cpu")
    model = enc.model
    assert model.recon_head is not None and model.jepa_hparams["recon_weight"] == 0.5
    mae_model, _ = load_encoder(str(mae / "mae.pt"), "cpu")
    # the warm start copied the autoencoder's projection; two training steps keep them close
    a = mae_model.model.projection.weight.detach()
    b = model.context.projection.weight.detach()
    assert (a - b).abs().max() < 0.05 * a.abs().max() + 1e-3
    values = torch.randn(2, 12, 2)
    positions = torch.cat([torch.sort(torch.rand(2, 1, 12) * 5, -1).values + 1, torch.full((2, 1, 12), 472.0)], 1)
    with torch.no_grad():
        o = model(values, positions)
    assert torch.isfinite(o.loss) and "recon" in model.last_stats


def test_jepa_projection(tmp_path):
    out = tmp_path / "proj"
    pretrain_mae.main(MAE_ARGS + ["--jepa", "--jepa-proj", "8", "--jepa-proj-hidden", "16", "--steps", "3", "--out", str(out)])
    enc, _ = load_encoder(str(out / "mae.pt"), "cpu")
    model = enc.model
    assert model.proj is not None and model.jepa_hparams["proj_dim"] == 8
    values = torch.randn(2, 12, 2)
    positions = torch.cat([torch.sort(torch.rand(2, 1, 12) * 5, -1).values + 1, torch.full((2, 1, 12), 472.0)], 1)
    with torch.no_grad():
        o = model(values, positions)
    assert o.pred.shape[-1] == 8 and o.target.shape[-1] == 8 and torch.isfinite(o.loss)
    assert model.embed_dim == 24  # the probes read the encoder width, not the projection


def test_jepa_no_ema_views(tmp_path):
    """LeJEPA at the token level: no EMA, SIGReg, 2 global + 4 local views, spectral aux, phase rope."""
    out = tmp_path / "noema"
    pretrain_mae.main(MAE_ARGS + ["--jepa", "--jepa-no-ema", "--jepa-proj", "16", "--jepa-proj-hidden", "16", "--sig-slices", "64",
                                  "--spectral", "--spectral-aux", "0.5", "--phase-rope", "1", "--dec-width", "48", "--n-frames", "1", "--mask-ratio", "0.85",
                                  "--view-keep", "0.3", "--local-span", "1", "3", "--steps", "3", "--out", str(out)])
    enc, _ = load_encoder(str(out / "mae.pt"), "cpu")
    model = enc.model
    assert model.no_ema and not model.use_target and model.jepa_hparams["no_ema"]
    values = torch.randn(2, 12, 2)
    positions = torch.cat([torch.sort(torch.rand(2, 1, 12) * 5, -1).values + 1, torch.full((2, 1, 12), 472.0)], 1)
    from project.jepa import forward_views

    l_pred, l_sig, l_aux, st = forward_views(model, values, positions, torch.zeros(2, 12, dtype=torch.bool), torch.tensor([0.7, 3.0]),
                                             keep=0.3, local_span=(1.0, 3.0), window=5.0)
    assert torch.isfinite(l_pred) and torch.isfinite(l_sig) and torch.isfinite(l_aux) and st["target_share"] > 0 and "period_hit10_global" in st
    with torch.autocast("cpu", dtype=torch.bfloat16):  # the GPU run trains under autocast (job 3615321 failed here)
        l_pred, l_sig, _, _ = forward_views(model, values, positions, torch.zeros(2, 12, dtype=torch.bool), torch.tensor([0.7, 3.0]),
                                            keep=0.3, local_span=(1.0, 3.0), window=5.0)
    assert torch.isfinite(l_pred) and torch.isfinite(l_sig)
    with torch.no_grad():
        o = model(values, positions)
    assert o.pred.shape[-1] == 16 and torch.isfinite(o.loss)


def test_resume_refuses_removed_options():
    """A checkpoint saved with flags that this code no longer has resumes when they were off, and is refused when one was on."""
    old = dict(dyn=0, bottleneck=False, encoder="transformer", target="obs", window_range=None, mask_mode="blockplus", dyn_weight=1.0)
    pretrain_mae.check_resumable(dict(kind="jepa", args=old))
    pretrain_mae.check_resumable(dict(args={}))  # a plain masked autoencoder has no kind
    for bad in (dict(dyn=4), dict(bottleneck=True), dict(mask_mode="mix"), dict(window_range=[20.0, 80.0])):
        with pytest.raises(SystemExit, match="pre-cleanup"):
            pretrain_mae.check_resumable(dict(kind="jepa", args=dict(old, **bad)))
    with pytest.raises(SystemExit, match="pre-cleanup"):
        pretrain_mae.check_resumable(dict(kind="gruode", args=old))
