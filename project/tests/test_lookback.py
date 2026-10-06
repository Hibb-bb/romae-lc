"""CPU tests of the lookback decoder (toy simulator): shapes, the latent
switch, the context time axis, a short training run, and the forecast
script on its checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from project import lookback
from project.common import flat_layout, load_encoder

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip


@pytest.fixture(scope="module")
def mae(tmp_path_factory):
    from project import pretrain_mae

    out = tmp_path_factory.mktemp("lb")
    pretrain_mae.main(MAE_ARGS + ["--steps", "4", "--out", str(out / "mae")])
    return out / "mae" / "mae.pt"


@pytest.fixture(scope="module")
def loaded(mae):
    enc, meta = load_encoder(str(mae), "cpu")
    return enc, meta


def test_dataset_and_context(loaded):
    import argparse

    from project.common import data_args_from, load_data

    enc, meta = loaded
    cfg, spec = meta.cfg, meta.spec
    data = load_data(data_args_from(meta.args, argparse.Namespace(data="sim", max_rows=None, n_sim=48)), splits=("validation",))
    ds = lookback.ForecastDataset(data["validation"], cfg, 0.5, cfg.max_tokens, 3, [1, 2], seed=0, epoch_seed=False)
    items = [ds[i] for i in range(len(ds))]
    items = [x for x in items if x is not None]
    assert items, "no record gave a forecast sample"
    for x in items:
        assert x["horizon"] in (1, 2) and 1 <= len(x["ctx"]) <= 3
        for frame, off in x["ctx"]:
            # every context window ends at or before the target's start
            assert off <= -cfg.window * x["horizon"] + 1e-6
            assert len(frame[0]) >= cfg.min_tokens
    same = ds[items[0]["index"]]
    assert same["start"] == items[0]["start"] and same["horizon"] == items[0]["horizon"]
    ds.set_epoch(0)
    batch = lookback.make_collate(spec, 3)(items[:4])
    tgt, z, ctx = lookback.batch_context(enc, spec, batch, torch.device("cpu"))
    feats, pos, pad = ctx
    assert feats.shape[0] == 4 and feats.shape[-1] == enc.backbone.embed_dim + 2 and pos.shape[1] == tgt.positions.shape[1]
    assert pad.shape == feats.shape[:2] and (~pad).any(1).all()
    ts = spec.tokenize["time_scale"]
    # real context tokens sit before the target window on its time axis
    assert ((pos[:, 0] - 1.0) * ts)[~pad].max() <= 0.0 + 1e-3
    assert z.shape == (4, enc.dim)


def test_model_switch_and_training(loaded, tmp_path, mae):
    enc, meta = loaded
    spec = meta.spec
    flat = flat_layout(enc.backbone)
    dec = lookback.LookbackDecoder(enc.dim, enc.backbone.embed_dim, flat, spec.err_stats or (0.0, 1.0), d_model=24, nhead=2, depth=1)
    torch.manual_seed(0)
    b, n, m = 2, 5, 7
    z = torch.randn(b, enc.dim)
    pos = torch.rand(b, len(flat), n) * 3
    pad = torch.zeros(b, n, dtype=torch.bool)
    ctx = (torch.randn(b, m, enc.backbone.embed_dim + 2), torch.rand(b, len(flat), m) * 3 - 3, torch.zeros(b, m, dtype=torch.bool))
    mu1, lv1 = dec(z, pos, pad, ctx)
    assert mu1.shape == (b, n) and lv1.shape == (b, n)
    mu2, _ = dec(z, pos, pad, ctx, torch.tensor([True, False]))
    assert torch.allclose(mu1[0], mu2[0]) and not torch.allclose(mu1[1], mu2[1])
    mu3, _ = dec(torch.randn(b, enc.dim), pos, pad, ctx, torch.tensor([False, False]))
    assert torch.allclose(mu2[1], mu3[1])  # without the latent, z does not matter
    assert lookback.LookbackDecoder(**dec.hparams).kind == "lookback"
    out = tmp_path / "lb"
    lookback.main(["--ckpt", str(mae), "--out", str(out), "--steps", "3", "--batch-size", "4", "--width", "24",
                   "--heads", "2", "--depth", "1", "--eval-every", "3", "--ckpt-every", "3", "--log-every", "1",
                   "--val-objects", "8", "--workers", "0", "--device", "cpu", "--stride", "0.5",
                   "--data", "sim", "--n-sim", "48"])  # fmt: skip
    assert (out / "dec.pt").is_file() and (out / "DONE").is_file() and lookback.is_lookback(out / "dec.pt")
    dec2, meta2 = lookback.load_lookback(out / "dec.pt", "cpu")
    assert meta2["n_ctx"] == 3 and meta2["metrics"]["n"] > 0
    for k in ("nll_model", "nll_no_latent", "nll_const", "skill_model", "skill_no_latent"):
        assert np.isfinite(meta2["metrics"][k]), k
    # resume picks the run up and stops at once
    lookback.main(["--ckpt", str(mae), "--out", str(out), "--steps", "3", "--workers", "0", "--device", "cpu"])
