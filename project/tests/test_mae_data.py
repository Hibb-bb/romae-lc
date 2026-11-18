"""CPU tests of the masks of :mod:`project.mae_data` (stretches of time hidden
instead of single points); then the scripts end to end on the toy simulator."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from project.mae_data import block_mask, longest_gap, make_mask

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


def runs(row: torch.Tensor) -> int:
    """How many separate runs of True a row has."""
    r = row.int()
    return int(((r[1:] - r[:-1]) == 1).sum() + r[0])


def test_block_mask_hides_stretches():
    g = torch.Generator().manual_seed(0)
    b, n = 16, 60
    t = torch.sort(torch.rand(b, n, generator=g) * 250, 1).values
    pad = torch.zeros(b, n, dtype=torch.bool)
    pad[3, 40:] = True  # a shorter row
    pad[7, 20:] = True
    m = block_mask(t, pad, 0.5, (1, 3), g)
    k = ((~pad).sum(1) * 0.5).ceil().long()
    assert m.shape == (b, n)
    assert ((m & ~pad).sum(1) == k).all()  # the exact share of the real points
    assert (m.sum(1) == k.max()).all()  # every row hides the same number of tokens
    for i in range(b):
        real = ~pad[i]
        assert 1 <= runs(m[i][real]) <= 3  # a few stretches, not scattered points
    # random points leave no long gap, stretches do
    rand = make_mask(t, pad, 0.5, "random", generator=g)
    full = ~pad.any(1)
    gap_b = longest_gap(t[full], (~m & ~pad)[full])
    gap_r = longest_gap(t[full], (~rand & ~pad)[full])
    # (a stretch at the edge of a window leaves no gap between visible points,
    # so only the mean is compared)
    assert gap_b.mean() > 1.5 * gap_r.mean()
    # one stretch hides one run
    one = block_mask(t, pad, 0.5, (1, 1), g)
    assert all(runs(one[i][~pad[i]]) == 1 for i in range(b))
    with pytest.raises(ValueError):
        block_mask(t, pad, 0.5, (0, 2), g)
    with pytest.raises(ValueError):
        make_mask(t, pad, 0.5, "stripes")


def test_pretrain_with_stretches_and_two_cache_lengths(tmp_path):
    from project import cache_latents, period_probe, pretrain_mae
    from project.common import load_encoder

    out = tmp_path / "mae"
    common = MAE_ARGS + ["--out", str(out), "--mask-mode", "block", "--mask-blocks", "1", "3"]
    pretrain_mae.main(common + ["--steps", "4"])
    assert (out / "mae.pt").is_file()
    pretrain_mae.main(common + ["--steps", "6"])  # resume keeps the settings
    lines = [json.loads(l) for l in open(out / "log.jsonl")]
    evals = [l for l in lines if l["kind"] == "eval"]
    assert max(l["step"] for l in lines) == 6
    assert all(np.isfinite(e["val_loss"]) and np.isfinite(e["val_loss_block"]) for e in evals)
    args = json.load(open(out / "args.json"))
    assert args["mask_mode"] == "block" and args["mask_blocks"] == [1, 3]
    enc, meta = load_encoder(out / "mae.pt")
    assert enc.kind == "mae" and meta.cfg.window == 30.0

    # the same encoder cached at two window lengths, joined in the period probe
    for w in (30, 60):
        cache_latents.main(
            ["--ckpt", str(out / "mae.pt"), "--out", str(tmp_path / f"lat{w}.pt"), "--window", str(w),
             "--stride", "0.5", "--device", "cpu", "--workers", "0", "--batch-size", "16"]  # fmt: skip
        )
    c30 = torch.load(tmp_path / "lat30.pt", weights_only=False)
    c60 = torch.load(tmp_path / "lat60.pt", weights_only=False)
    assert c30["meta"]["window"] == 30.0 and c60["meta"]["window"] == 60.0
    assert c60["z"].shape[0] < c30["z"].shape[0]  # fewer, longer windows
    res = period_probe.run(
        period_probe.parse_args(
            ["--latents", str(tmp_path / "lat30.pt"), "--extra-latents", str(tmp_path / "lat60.pt"),
             "--out", str(tmp_path / "probe"), "--n-ls", "4", "--mlp-steps", "30", "--bins", "10",
             "--no-fold", "--device", "cpu", "--data", "sim", "--n-sim", "48"]  # fmt: skip
        )
    )
    assert res["models"]["multi/ridge"]["dim"] == 24 + 25
    assert 0.0 <= res["models"]["multi/bins"]["rec10"] <= 1.0


def test_absolute_time_features():
    import torch

    from romae_lc import RoMAE, RoMAEForPreTraining

    torch.manual_seed(0)
    enc, dec = dict(d_model=16, nhead=2, depth=1), dict(d_model=16, nhead=2, depth=1)
    model = RoMAEForPreTraining(decoder=dec, encoder=enc, n_channels=2, target_channels=1, abs_timescales=[0.5, 2.0, 7.0])
    assert model.abs_proj is not None and model.hparams["abs_timescales"] == [0.5, 2.0, 7.0]
    b, n = 2, 10
    values, positions = torch.randn(b, n, 2), torch.rand(b, 2, n) * 20 + 1
    pad = torch.zeros(b, n, dtype=torch.bool)
    out = model(values, positions, pad)
    assert torch.isfinite(out.loss)
    # the features change with the time since the window's start; the plain model's embedding does not
    plain = RoMAEForPreTraining(decoder=dec, encoder=enc, n_channels=2, target_channels=1)
    shifted = positions.clone()
    shifted[:, 0] += 3.0
    assert not torch.allclose(model.embed(values, positions), model.embed(values, shifted))
    assert torch.allclose(plain.embed(values, positions), plain.embed(values, shifted))
    # the backbone carries the feature projection and gives the same CLS feature
    bb = model.backbone()
    assert isinstance(bb, RoMAE) and bb.abs_proj is not None
    x, _ = model.encode(values, positions, pad)
    assert torch.allclose(bb(values, positions, pad), x[:, 0], atol=1e-5)
    rebuilt = RoMAE(**bb.hparams)
    rebuilt.load_state_dict(bb.state_dict())
    assert torch.allclose(rebuilt(values, positions, pad), x[:, 0], atol=1e-5)


def test_block_position_and_blockplus():
    import torch

    from project.mae_data import block_mask, make_mask

    torch.manual_seed(0)
    b, n = 3, 40
    t = torch.sort(torch.rand(b, n) * 100, 1).values
    pad = torch.zeros(b, n, dtype=torch.bool)
    pad[0, 30:] = True
    g = torch.Generator().manual_seed(1)
    last = block_mask(t, pad, 0.25, (1, 1), g, position="last")
    for i in range(b):
        real = ~pad[i]
        hidden = torch.nonzero(last[i] & real).flatten()
        k = int(torch.ceil(real.sum().float() * 0.25))
        assert len(hidden) == k and hidden.min() == real.sum() - k  # the last k real points
    bp = make_mask(t, pad, 0.4, "blockplus", generator=g, position="last", block_share=0.5)
    total = bp.sum(1)  # rows are equalised with trailing padding tokens, like gen_mask
    assert (total == total[0]).all()
    for i in range(b):
        real = ~pad[i]
        assert int((bp[i] & real).sum()) == int(torch.ceil(real.sum().float() * 0.4))
    # the stretch at the end is hidden and some random points elsewhere too
    assert bp[1, -8:].all() and bp[1, :20].any() and not bp[1, :20].all()
