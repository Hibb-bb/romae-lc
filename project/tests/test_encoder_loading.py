"""CPU test of the frozen-encoder loading path on the toy simulator: a tiny autoencoder
(``project.train``) is trained for a few steps, then :func:`project.checkpoint.load_encoder`
and :func:`project.checkpoint.load_mae` are checked on it."""

from __future__ import annotations

import pytest
import torch

from romae_lc import FrameDataset

from project.checkpoint import FrameEncoder, LatentEncoder, load_encoder, load_mae, PooledEncoder
from project.data import data_args_from, load_data
from project.tests._args import MAE_ARGS


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def mae_ckpt(tmp_path_factory):
    from project import train

    out = tmp_path_factory.mktemp("mae")
    train.main(MAE_ARGS + ["--steps", "4", "--out", str(out)])
    assert (out / "mae.pt").is_file()
    return out / "mae.pt"


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
    assert enc.kind == "mae" and enc.dim == 24
    assert enc.backbone.embed_dim == 24 and enc.backbone.rope_layout
    assert_frozen(enc)
    frames = batch_frames(meta)
    z = enc.encode(frames)
    assert z.shape == (len(frames[0].values), meta.cfg.n_frames, 24)
    assert z.dtype == torch.float32 and torch.isfinite(z).all()
    mae, meta2 = load_mae(mae_ckpt)
    ref = FrameEncoder(PooledEncoder(mae)).encode(frames)
    assert torch.allclose(z, ref, atol=1e-6)
    assert meta2.step == meta.step == 4 and meta.args["width"] == 24
    assert LatentEncoder(mae).encode(frames).shape == z.shape
    # a checkpoint of a kind whose code moved to legacy/ is refused with a clear message
    ckpt = torch.load(mae_ckpt, map_location="cpu", weights_only=False)
    with pytest.raises(ValueError, match="pre-cleanup"):
        load_mae(dict(ckpt, kind="bottleneck"))
