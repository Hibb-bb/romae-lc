"""CPU tests of the frozen-encoder loading path on the toy simulator: a tiny
stage-1 autoencoder (``pretrain_mae``) and a tiny stage-2 world model
(``train_wm``) are trained for a few steps once per module, then
:func:`project.common.load_encoder`, the autoencoder branch of
:func:`project.common.load_wm` and ``train_decoder`` (stage 3) are checked on
them. A minute or two in all."""

from __future__ import annotations

import pytest
import torch

from romae_lc import FrameDataset

from project.common import (
    FrameEncoder,
    LatentEncoder,
    PooledEncoder,
    checkpoint_kind,
    data_args_from,
    load_data,
    load_encoder,
    load_mae,
    load_wm,
    wm_state,
)

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

DEC_ARGS = [
    "--kind", "mse", "--steps", "2", "--batch-size", "4", "--width", "24",
    "--heads", "2", "--depth", "1", "--eval-every", "2", "--ckpt-every", "2",
    "--val-objects", "6", "--baselines", "--sample-steps", "2", "--workers", "0",
    "--device", "cpu",
]  # fmt: skip


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def mae_ckpt(tmp_path_factory):
    from project import pretrain_mae

    out = tmp_path_factory.mktemp("mae")
    pretrain_mae.main(MAE_ARGS + ["--steps", "4", "--out", str(out)])
    assert (out / "mae.pt").is_file()
    return out / "mae.pt"


@pytest.fixture(scope="module")
def wm_ckpt(mae_ckpt, tmp_path_factory):
    from project import train_wm

    out = tmp_path_factory.mktemp("wm")
    train_wm.main(WM_ARGS + ["--init-backbone", str(mae_ckpt), "--out", str(out)])
    assert (out / "wm.pt").is_file()
    return out / "wm.pt"


def batch_frames(meta, n=3):
    """``T`` token batches of up to ``n`` validation sequences cut and
    tokenised the way the checkpoint was trained."""
    data = load_data(data_args_from(meta.args), splits=("validation",))
    ds = FrameDataset(data["validation"], meta.cfg, epoch_seed=False)
    assert len(ds) >= 1
    batch = meta.spec.collate()([ds[i] for i in range(min(n, len(ds)))])
    return batch["frames"]


def assert_frozen(enc: LatentEncoder):
    assert not enc.training and not enc.model.training
    assert all(not p.requires_grad for p in enc.parameters())
    enc.train()
    assert not enc.training and not enc.backbone.training
    enc.train(True)
    assert not enc.training


def test_load_encoder_mae(mae_ckpt):
    enc, meta = load_encoder(mae_ckpt)
    assert enc.kind == "mae" and enc.dim == 24 and enc.history == 3
    assert enc.backbone.embed_dim == 24 and enc.backbone.rope_layout
    assert_frozen(enc)
    frames = batch_frames(meta)
    z = enc.encode(frames)
    assert z.shape == (len(frames[0].values), meta.cfg.n_frames, 24)
    assert z.dtype == torch.float32 and torch.isfinite(z).all()
    assert torch.equal(z, enc.encode(frames, project=False))  # ignored for an AE
    mae, meta2 = load_mae(mae_ckpt)
    ref = FrameEncoder(PooledEncoder(mae)).encode(frames)
    assert torch.allclose(z, ref, atol=1e-6)
    assert meta2.step == meta.step == 4 and meta.args["width"] == 24
    ckpt = torch.load(mae_ckpt, map_location="cpu", weights_only=False)
    assert checkpoint_kind(ckpt) == "mae"
    assert LatentEncoder(mae, "mae").encode(frames).shape == z.shape
    with pytest.raises(ValueError):
        LatentEncoder(mae, "vae")


def test_load_wm_accepts_mae(mae_ckpt, tmp_path):
    model, meta = load_wm(mae_ckpt)
    assert model.encoder_only is True and not model.training
    assert meta.args["encoder_frozen"] is True and meta.args["encoder_kind"] == "mae"
    assert meta.args["width"] == 24 and meta.args["depth"] == 2  # the MAE args
    assert isinstance(model.projector, torch.nn.Identity)
    assert model.history == 3 and model.lamb == 0.0 and model.embed_dim == 24
    frames = batch_frames(meta)
    enc, _ = load_encoder(mae_ckpt)
    z = model.encode(frames)
    assert torch.allclose(z, enc.encode(frames), atol=1e-6)
    assert torch.allclose(z, model.encode(frames, project=False), atol=1e-6)
    # the (untrained) predictor runs, so probes and gates can call surprise
    assert torch.isfinite(model.surprise(frames)).all()
    # re-saved through wm_state it is an ordinary stage-2 checkpoint
    path = tmp_path / "wm_from_mae.pt"
    torch.save(
        wm_state(model, meta.spec, meta.cfg, meta.ladder, meta.classes, meta.args, 0),
        path,
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint_kind(ckpt) == "wm" and ckpt["hparams"]["proj_hidden"] == 0
    again, meta2 = load_wm(path)
    assert again.encoder_only is False and meta2.args["encoder_kind"] == "mae"
    assert torch.allclose(again.encode(frames), z, atol=1e-6)
    enc2, _ = load_encoder(path)
    assert enc2.kind == "wm" and torch.allclose(enc2.encode(frames), z, atol=1e-6)


def test_load_encoder_wm(wm_ckpt):
    enc, meta = load_encoder(wm_ckpt)
    model, meta2 = load_wm(wm_ckpt)
    assert enc.kind == "wm" and enc.dim == 24 and enc.history == model.history
    assert enc.backbone is enc.model.backbone
    assert model.encoder_only is False and meta.step == meta2.step == 4
    assert_frozen(enc)
    frames = batch_frames(meta)
    z = enc.encode(frames)
    assert z.shape == (len(frames[0].values), meta.cfg.n_frames, 24)
    assert torch.allclose(z, model.encode(frames), atol=1e-6)
    feats = enc.encode(frames, project=False)
    assert torch.allclose(feats, model.encode(frames, project=False), atol=1e-6)
    assert not torch.allclose(feats, z)  # the projector is not the identity


def test_train_decoder_on_mae(mae_ckpt, tmp_path):
    from project import train_decoder

    out = tmp_path / "dec"
    train_decoder.main(["--ckpt", str(mae_ckpt), "--out", str(out)] + DEC_ARGS)
    assert (out / "dec.pt").is_file() and (out / "DONE").is_file()
    ckpt = torch.load(out / "dec.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 2
