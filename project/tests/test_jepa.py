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


def test_smooth_targets_mae_and_jepa(tmp_path):
    out = tmp_path / "tpl"
    pretrain_mae.main(MAE_ARGS + ["--target", "mix", "--template-harmonics", "2", "--template-min-points", "6", "--steps", "2", "--out", str(out)])
    assert (out / "DONE").is_file()
    rows = [l for l in open(out / "log.jsonl") if '"eval"' in l]
    assert rows and '"template_share"' in rows[-1]
    out2 = tmp_path / "tpl_jepa"
    pretrain_mae.main(MAE_ARGS + ["--jepa", "--jepa-recon-weight", "0.5", "--target", "template", "--template-harmonics", "2", "--template-min-points", "6", "--steps", "2", "--out", str(out2)])
    assert (out2 / "DONE").is_file()


def test_jepa_clean_target(tmp_path):
    out = tmp_path / "clean"
    pretrain_mae.main(MAE_ARGS + ["--jepa", "--jepa-clean-target", "--target", "template", "--template-harmonics", "2", "--template-min-points", "6", "--steps", "2", "--out", str(out)])
    enc, _ = load_encoder(str(out / "mae.pt"), "cpu")
    assert enc.model.clean_target and enc.model.jepa_hparams["clean_target"]


def test_query_tokens_mae_and_clean_jepa(tmp_path):
    common = ["--target", "template", "--template-harmonics", "2", "--template-min-points", "6", "--query-extra", "6", "--query-delta", "0.5", "5", "--steps", "2"]
    out = tmp_path / "q_mae"
    pretrain_mae.main(MAE_ARGS + common + ["--out", str(out)])
    rows = [l for l in open(out / "log.jsonl") if '"eval"' in l]
    assert rows and '"query_delta_days"' in rows[-1]
    out2 = tmp_path / "q_jepa"
    pretrain_mae.main(MAE_ARGS + common + ["--jepa", "--jepa-clean-target", "--out", str(out2)])
    assert (out2 / "DONE").is_file()


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


def test_record_templates_and_training(tmp_path):
    import math

    from romae_lc.data import Record
    from project.templates import RecordTemplates, fit_record_templates

    rng = np.random.default_rng(0)
    recs = []
    for P in (0.7, 3.0):
        t = np.sort(rng.uniform(0, 900, 400)); band = rng.integers(0, 2, 400)
        y = np.sin(2 * math.pi * t / P) + 0.3 * np.cos(4 * math.pi * t / P) + band * 0.5 + 0.05 * rng.standard_normal(400)
        recs.append(Record(t=t.astype(np.float32), y=y.astype(np.float32), err=np.full(400, 0.05, np.float32), band=band.astype(np.int64), label=0, period=P, meta={}))
    fitted = fit_record_templates(recs, [0.7, 3.0], n_bands=2, workers=2, log=lambda *_: None)
    assert (fitted["harmonics"] >= 2).all()
    tpl = RecordTemplates(fitted, "cpu")
    t_abs = torch.rand(2, 30, dtype=torch.float64) * 900; band = torch.randint(0, 2, (2, 30))
    P = torch.tensor([0.7, 3.0], dtype=torch.float64)[:, None]
    truth = torch.sin(2 * math.pi * t_abs / P) + 0.3 * torch.cos(4 * math.pi * t_abs / P) + band * 0.5
    assert (tpl.eval(torch.tensor([0, 1]), t_abs, band) - truth.float()).abs().mean() < 0.05
    out = tmp_path / "rec"
    pretrain_mae.main(MAE_ARGS + ["--target", "template", "--template-fit", "record", "--template-workers", "2", "--query-extra", "6", "--query-delta", "0.5", "5", "--steps", "2", "--out", str(out)])
    assert (out / "templates.pt").is_file() and (out / "DONE").is_file()
    rows = [l for l in open(out / "log.jsonl") if '"eval"' in l]
    assert '"template_share"' in rows[-1] and '"query_delta_days"' in rows[-1]
