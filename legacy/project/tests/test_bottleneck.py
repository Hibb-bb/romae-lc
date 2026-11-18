"""CPU tests of the stage-1 bottleneck autoencoder (``project.bottleneck``)
on the toy simulator: the module forward, ``pretrain_mae --bottleneck`` with
resume, and the loaders and ``train_wm --init-backbone`` on its checkpoint.
About a minute in all."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch

from romae_lc import FrameConfig, FrameDataset, RoMAE, Tokens
from romae_lc.data import SimConfig, SurveyConfig, normalize, simulate

from project.bottleneck import (
    BottleneckAE,
    bottleneck_state,
    fuse_tokens,
    masked_decoder_loss,
)
from project.common import (
    TokenSpec,
    data_args_from,
    dense_ladder,
    encoder_config,
    err_stats,
    fuse_frames,
    load_data,
    load_encoder,
    load_mae,
    load_wm,
    rope_geometry,
    rope_layouts,
)
from project.decoder import decoder_loss

SURVEYS = (SurveyConfig("g", 480.7, 160, 0.05), SurveyConfig("r", 622.1, 120, 0.06))
SIM = SimConfig(baseline_days=400.0, logP_range=(-0.3, 1.0))
WL = {i: s.wavelength_nm for i, s in enumerate(SURVEYS)}

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip

WM_ARGS = [
    "--data", "sim", "--n-sim", "48", "--pred-depth", "1", "--pred-heads", "2",
    "--pred-dim-head", "8", "--pred-mlp", "32", "--proj-hidden", "32",
    "--n-slices", "16", "--window", "30", "--min-tokens", "4", "--max-tokens", "32",
    "--n-frames", "4", "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2",
    "--log-every", "2", "--probe-train", "16", "--probe-val", "8",
    "--shuffle-objects", "6", "--surprise-objects", "2", "--workers", "0",
    "--device", "cpu", "--steps", "4", "--no-eval-at-start",
]  # fmt: skip


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def records():
    recs = [normalize(r) for r in simulate(24, SURVEYS, SIM, seed=1)]
    for r in recs:
        r.meta["class_str"] = r.meta["superclass_str"] = f"c{r.label}"
    return recs


@pytest.fixture(scope="module")
def spec(records):
    return TokenSpec(dict(band_wavelengths=WL, time_scale=0.05), err_stats(records))


@pytest.fixture(scope="module")
def cfg():
    return FrameConfig(
        n_frames=3, window=30.0, min_tokens=4, max_tokens=64, with_err=True
    )


@pytest.fixture(scope="module")
def frames(records, spec, cfg):
    ds = FrameDataset(records, cfg, epoch_seed=False)
    return spec.collate()([ds[0], ds[1], ds[2]])["frames"]


def small_model(records, spec, loss_on="all", mask_ratio=0.5, **dec) -> BottleneckAE:
    """A 24-wide bottleneck autoencoder built the way pretrain_mae builds one."""
    args = argparse.Namespace(
        width=24, heads=2, depth=2, mlp_ratio=2.0, time_frac=0.875, p_rope=0.75, attention="softmax"
    )  # fmt: skip
    ladder = dense_ladder(
        records, rope_geometry(args), 30.0, "bands", "log", 0.75, 50.0, 8, 0, None, None
    )
    return BottleneckAE(
        encoder=encoder_config(args),
        n_channels=spec.n_channels,
        err_stats=spec.err_stats,
        rope=rope_layouts(args, ladder),
        decoder=dict(dict(d_model=24, nhead=2, depth=1), **dec),
        mask_ratio=mask_ratio,
        loss_on=loss_on,
    )


# ------------------------------------------------------------------- module


def test_denoise_resamples_the_encoder_input(records, spec, frames):
    model = small_model(records, spec)
    model.denoise = True
    tok = frames[0]
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    a = model(tok, generator=g1)
    model.denoise = False
    b = model(tok, generator=g2)
    assert torch.isfinite(a.loss) and torch.isfinite(b.loss)
    assert not torch.allclose(a.z, b.z)  # same drop, resampled magnitudes
    model.denoise = True
    model.eval()
    with torch.no_grad():
        c = model(tok, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(c.z, b.z, atol=1e-5)  # no resampling in eval mode
    assert model.bottleneck_hparams["denoise"] is True


def test_known_variance_loss(records, spec, frames):
    model = small_model(records, spec)
    tok = frames[0]
    g = torch.Generator().manual_seed(0)
    a = model(tok, generator=g)
    model.learned_var = False
    b = model(tok, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(b.loss) and not torch.isclose(a.loss, b.loss)
    # known error only: the loss is the chi-square term plus log sigma^2,
    # independent of the decoder's log-variance head
    sig2 = tok.extras.float().square().clamp_min(1e-8)
    m = tok.values[..., 0].float()
    expect = (0.5 * ((m - b.mu.float()).square() / sig2 + sig2.log()) * b.scored).sum() / b.scored.sum()
    assert torch.allclose(b.loss, expect, atol=1e-4)
    assert model.bottleneck_hparams["learned_var"] is False
    b.loss.backward()
    model.learned_var = "unit"
    u = model(tok, generator=torch.Generator().manual_seed(0))
    expect_u = (0.5 * (m - u.mu.float()).square() * u.scored).sum() / u.scored.sum()
    assert torch.allclose(u.loss, expect_u, atol=1e-4)  # plain squared error
    assert model.bottleneck_hparams["learned_var"] == "unit"


def test_forward_and_gradients(records, spec, frames):
    model = small_model(records, spec)
    tok = frames[0]
    out = model(tok)
    assert torch.isfinite(out.loss) and out.z.shape == (3, 24)
    assert out.mu.shape == out.logvar.shape == tok.pad_mask.shape
    assert torch.equal(out.scored, ~tok.pad_mask)  # loss_on="all"
    assert (out.hidden & tok.pad_mask).sum() == 0 and out.hidden.any()
    # at least 2 points stay visible in every window
    assert ((tok.n_real - out.hidden.sum(1)) >= torch.minimum(tok.n_real, torch.tensor(2))).all()
    out.loss.backward()
    enc = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    dec = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert enc and sum(g.abs().sum() for g in enc) > 0
    assert dec and sum(g.abs().sum() for g in dec) > 0
    # the encoder does not see the hidden points: their magnitudes do not move z
    model.zero_grad()
    values = tok.values.clone()
    values[..., 0][out.hidden] += 5.0
    z2 = model.latent(
        Tokens(values, tok.positions, tok.pad_mask | out.hidden, tok.extras)
    )
    z1 = model.latent(Tokens(tok.values, tok.positions, tok.pad_mask | out.hidden, tok.extras))
    assert torch.allclose(z1, z2, atol=1e-5)


def test_loss_variants_and_delegation(records, spec, frames):
    tok = frames[0]
    hidden_model = small_model(records, spec, loss_on="hidden")
    out = hidden_model(tok)
    assert torch.equal(out.scored, out.hidden) and torch.isfinite(out.loss)
    # with everything scored the masked loss is decoder_loss itself
    model = small_model(records, spec, mask_ratio=0.0)
    z = model.latent(tok)
    loss, _, _ = masked_decoder_loss(model.decoder, z, tok, ~tok.pad_mask)
    assert torch.allclose(loss, decoder_loss(model.decoder, z, tok))
    assert not model(tok).hidden.any()
    # the encoder's face
    assert model.embed_dim == 24 and model.use_cls and model.per_layer_rope
    assert model.rope_layout == model.encoder.rope_layout
    assert model.transformer is model.encoder.transformer
    assert model.projection is model.encoder.projection
    x, pad = model.encode(*tok)
    assert x.shape == (3, tok.values.shape[1] + 1, 24) and pad.shape[1] == x.shape[1]
    bb = model.backbone("cls")
    assert isinstance(bb, RoMAE) and bb.rope_layout == model.rope_layout
    assert torch.allclose(bb(*tok), z)
    with pytest.raises(ValueError, match="head_dim"):
        small_model(records, spec, nhead=3)
    with pytest.raises(ValueError, match="loss_on"):
        small_model(records, spec, loss_on="visible")
    with pytest.raises(ValueError, match="extras"):
        model(Tokens(tok.values, tok.positions, tok.pad_mask, None))


def test_fuse_tokens_and_checkpoint_roundtrip(records, spec, cfg, frames, tmp_path):
    fused = fuse_tokens(frames)
    values, positions, pad = fuse_frames(frames)
    assert torch.equal(fused.values, values) and torch.equal(fused.pad_mask, pad)
    b = frames[0].values.shape[0]
    for i, f in enumerate(frames):
        k = f.extras.shape[1]
        assert torch.equal(fused.extras[i * b : (i + 1) * b, :k], f.extras)
    assert (fused.extras[fused.pad_mask] == 0).all()
    with pytest.raises(ValueError, match="extras"):
        fuse_tokens([Tokens(f.values, f.positions, f.pad_mask, None) for f in frames])

    model = small_model(records, spec, loss_on="hidden", mask_ratio=0.3)
    ladder = argparse.Namespace()  # only to_dict is needed
    ladder.to_dict = lambda: dict(time_scale=0.05, timescales=[1.0], wavelengths=[0.3], lam_min=0.3, lam_max=0.3, spacing="log")  # fmt: skip
    state = bottleneck_state(
        model, spec, cfg, ladder, ("a", "b"), dict(bottleneck=True), 3, dict(loss=1.0)
    )
    assert state["kind"] == "bottleneck" and "mae" not in state
    assert state["bottleneck"]["loss_on"] == "hidden"
    assert state["bottleneck"]["mask_ratio"] == pytest.approx(0.3)
    assert state["bottleneck"]["err_stats"] == pytest.approx(list(spec.err_stats))
    assert state["backbone"]["encoder"]["d_model"] == 24
    torch.save(state, tmp_path / "bn.pt")
    twin = BottleneckAE.from_checkpoint(
        torch.load(tmp_path / "bn.pt", map_location="cpu", weights_only=False)
    )
    assert twin.loss_on == "hidden" and twin.mask_ratio == pytest.approx(0.3)
    tok = frames[0]
    assert torch.allclose(twin.latent(tok), model.latent(tok))
    assert all(
        torch.equal(a, b)
        for a, b in zip(twin.state_dict().values(), model.state_dict().values())
    )


# --------------------------------------------------------------- end to end


def test_pretrain_bottleneck_then_init_backbone(tmp_path):
    from project import pretrain_mae, train_wm

    out = tmp_path / "bn"
    common = MAE_ARGS + ["--bottleneck", "--out", str(out)]
    pretrain_mae.main(common + ["--steps", "4"])
    assert (out / "mae.pt").is_file() and (out / "DONE").is_file()
    ckpt = torch.load(out / "mae.pt", map_location="cpu", weights_only=False)
    assert ckpt["kind"] == "bottleneck" and "mae" not in ckpt
    assert ckpt["bottleneck"]["loss_on"] == "all" and ckpt["args"]["bottleneck"]
    pretrain_mae.main(common + ["--steps", "6"])  # resume from last.pt
    lines = [json.loads(l) for l in open(out / "log.jsonl")]
    assert max(l["step"] for l in lines) == 6 and lines[0]["kind"] == "baseline"
    evals = [l for l in lines if l["kind"] == "eval"]
    assert evals[0]["step"] == 0 and all(np.isfinite(e["val_loss"]) for e in evals)
    assert all(np.isfinite(l["loss"]) for l in lines if l["kind"] == "train")

    mae, meta = load_mae(out / "mae.pt")
    assert isinstance(mae, BottleneckAE) and meta.step == 6
    assert isinstance(mae.backbone("cls"), RoMAE)
    assert meta.ladder.layers == 2 and meta.ladder.heads == 2 and mae.per_layer_rope

    enc, emeta = load_encoder(out / "mae.pt")
    assert enc.kind == "mae" and enc.dim == 24 and emeta.step == 6
    data = load_data(data_args_from(emeta.args))
    ds = FrameDataset(data["validation"][:3], emeta.cfg, epoch_seed=False)
    frames = emeta.spec.collate()([ds[i] for i in range(3)])["frames"]
    z = enc.encode(frames)
    assert z.shape == (3, emeta.cfg.n_frames, 24) and z.dtype == torch.float32
    assert torch.isfinite(z).all()
    wm0, wmeta0 = load_wm(out / "mae.pt")
    assert wm0.encoder_only and wmeta0.args["encoder_kind"] == "mae"
    assert torch.allclose(wm0.encode(frames), z, atol=1e-5)

    wm = tmp_path / "wm"
    train_wm.main(WM_ARGS + ["--init-backbone", str(out / "mae.pt"), "--out", str(wm)])
    model, wmeta = load_wm(wm / "wm.pt")
    assert wmeta.args["width"] == 24 and wmeta.args["depth"] == 2
    assert wmeta.ladder.to_dict() == meta.ladder.to_dict()
    assert model.backbone.rope_layout == mae.rope_layout


def test_bottleneck_refuses_without_err_channel(tmp_path):
    from project import pretrain_mae

    with pytest.raises(SystemExit):
        pretrain_mae.main(
            MAE_ARGS
            + ["--bottleneck", "--no-err-channel", "--steps", "2", "--out", str(tmp_path / "x")]
        )
