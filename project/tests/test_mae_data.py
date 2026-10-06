"""CPU tests of the two stage-1 changes of :mod:`project.mae_data`: stretches
of time hidden instead of single points, and training windows of many
lengths; then the scripts end to end on the toy simulator."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from romae_lc import FrameConfig
from romae_lc.data import SimConfig, SurveyConfig, normalize, simulate

from project.mae_data import MultiWindowDataset, block_mask, longest_gap, make_mask

SURVEYS = (SurveyConfig("g", 480.7, 160, 0.05), SurveyConfig("r", 622.1, 120, 0.06))
SIM = SimConfig(baseline_days=400.0, logP_range=(-0.3, 1.0))

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


def test_mix_has_both_kinds_of_row():
    g = torch.Generator().manual_seed(1)
    t = torch.sort(torch.rand(64, 50, generator=g) * 100, 1).values
    pad = torch.zeros(64, 50, dtype=torch.bool)
    m = make_mask(t, pad, 0.5, "mix", (1, 2), 0.5, g)
    assert (m.sum(1) == 25).all()
    n_runs = np.array([runs(row) for row in m])
    assert (n_runs <= 2).sum() > 10 and (n_runs > 5).sum() > 10
    all_blocks = make_mask(t, pad, 0.5, "mix", (1, 2), 1.0, g)
    assert all(runs(row) <= 2 for row in all_blocks)


def test_multi_window_dataset():
    recs = [normalize(r) for r in simulate(12, SURVEYS, SIM, seed=1)]
    cfg = FrameConfig(n_frames=6, window=30.0, min_tokens=4, max_tokens=24, with_err=True)
    ds = MultiWindowDataset(recs, cfg, (20.0, 200.0), seed=0, epoch_seed=False)
    assert len(ds) == len(ds.indices) == 12
    lengths = []
    for i in range(len(ds)):
        item = ds[i]
        assert len(item["frames"]) == 6 and item["actions"].shape == (6, 1)
        for f, w in zip(item["frames"], item["lengths"]):
            assert len(f) == 4 and len(f[0]) <= 24
            assert (f[0] >= 0).all() and (f[0] < w + 1e-3).all()  # counted from the window start
        lengths += item["lengths"].tolist()
    lengths = np.array(lengths)
    assert lengths.min() >= 20.0 - 1e-6 and lengths.max() <= 200.0 + 1e-6
    assert lengths.min() < 40 and lengths.max() > 100  # short and long windows both occur
    a, b = ds[0], ds[0]
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a["frames"], b["frames"]))  # deterministic
    with pytest.raises(ValueError):
        MultiWindowDataset(recs, cfg, (5000.0, 6000.0))
    with pytest.raises(ValueError):
        MultiWindowDataset(recs, cfg, (50.0, 20.0))


def test_pretrain_with_stretches_and_window_range(tmp_path):
    from project import cache_latents, period_probe, pretrain_mae
    from project.common import load_encoder

    out = tmp_path / "mae"
    common = MAE_ARGS + ["--out", str(out), "--mask-mode", "mix", "--mask-blocks", "1", "3",
                         "--window-range", "20", "80"]  # fmt: skip
    pretrain_mae.main(common + ["--steps", "4"])
    assert (out / "mae.pt").is_file()
    pretrain_mae.main(common + ["--steps", "6"])  # resume keeps the settings
    lines = [json.loads(l) for l in open(out / "log.jsonl")]
    evals = [l for l in lines if l["kind"] == "eval"]
    assert max(l["step"] for l in lines) == 6
    assert all(np.isfinite(e["val_loss"]) and np.isfinite(e["val_loss_block"]) for e in evals)
    args = json.load(open(out / "args.json"))
    assert args["mask_mode"] == "mix" and args["window_range"] == [20.0, 80.0] and args["lam_max"] == 160.0
    ladder = json.load(open(out / "ladder.json"))
    assert ladder["lam_max"] == 160.0  # the longest window fits the ladder
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


def test_stretches_refused_for_the_bottleneck(tmp_path):
    from project import pretrain_mae

    with pytest.raises(SystemExit):
        pretrain_mae.main(MAE_ARGS + ["--out", str(tmp_path / "bn"), "--steps", "2", "--bottleneck",
                                      "--mask-mode", "block"])  # fmt: skip


def test_error_weights_and_weighted_loss():
    import torch

    from project.pretrain_mae import error_weights
    from romae_lc import RoMAEForPreTraining
    from romae_lc.model import gen_mask

    torch.manual_seed(0)
    b, n = 3, 12
    values = torch.randn(b, n, 2)
    pad = torch.zeros(b, n, dtype=torch.bool)
    pad[0, 9:] = True
    w = error_weights(values, pad, (0.0, 1.0), cap=20.0)
    assert w.shape == (b, n) and (w[pad] == 0).all() and (w[~pad] > 0).all()
    real = (~pad).float()
    assert torch.allclose((w * real).sum(1) / real.sum(1), torch.ones(b), atol=1e-5)  # mean 1 per window
    assert w[1][values[1, :, 1].argmin()] >= w[1][values[1, :, 1].argmax()]  # precise points weigh more
    # the cap: one absurdly precise point cannot take the window over
    v2 = values.clone()
    v2[2, 0, 1] = -20.0
    w2 = error_weights(v2, pad, (0.0, 1.0), cap=20.0)
    assert w2[2, 0] <= 20.0 * w2[2, 1:].median() * (n / (w2[2] * real[2]).sum()) + 1e-4 and w2[2, 0] < n
    # uniform weights give the plain loss; the weighted loss follows the weights
    model = RoMAEForPreTraining(decoder=dict(d_model=16, nhead=2, depth=1), encoder=dict(d_model=16, nhead=2, depth=1),
                                n_channels=2, target_channels=1)
    positions = torch.rand(b, 2, n) * 10 + 1
    mask = gen_mask(0.5, pad, torch.Generator().manual_seed(1))
    plain = model(values, positions, pad, mask).loss
    same = model(values, positions, pad, mask, torch.ones(b, n)).loss
    assert torch.allclose(plain, same, atol=1e-6)
    heavy = torch.ones(b, n)
    heavy[:, :1] = 50.0
    assert torch.isfinite(model(values, positions, pad, mask, heavy).loss)


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
