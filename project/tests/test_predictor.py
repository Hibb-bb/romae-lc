"""CPU tests of stage 2 on cached latents (toy simulator): a tiny stage-1
autoencoder (``pretrain_mae``, 4 steps) is trained once per module, its
window grid is cached with :mod:`project.cache_latents`, and the flow and
MSE predictors of :mod:`project.train_predictor` are trained, resumed and
reloaded on the cache. Unit tests cover the anchored log density, sampling
on a zero velocity field and the advance rounding. About a minute in all."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from project import cache_latents, train_predictor
from project.cache_latents import object_windows
from project.common import data_args_from, grid_starts, load_data, load_encoder
from project.train_predictor import (
    FlowPredictor,
    LatentStore,
    MsePredictor,
    anchored_log_density,
    load_predictor,
    round_advance,
)

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip

STRIDE = 0.5


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def mae_ckpt(tmp_path_factory):
    from project import pretrain_mae

    out = tmp_path_factory.mktemp("mae")
    pretrain_mae.main(MAE_ARGS + ["--steps", "4", "--out", str(out)])
    assert (out / "mae.pt").is_file()
    return out / "mae.pt"


@pytest.fixture(scope="module")
def latents(mae_ckpt):
    out = mae_ckpt.parent / "latents.pt"
    cache_latents.main(
        [
            "--ckpt", str(mae_ckpt), "--out", str(out), "--stride", str(STRIDE),
            "--device", "cpu", "--workers", "0", "--batch-size", "16",
        ]  # fmt: skip
    )
    assert out.is_file() and out.with_suffix(".json").is_file()
    return out


# ------------------------------------------------------------------- cache


def test_cache_layout_and_starts(latents, mae_ckpt):
    c = torch.load(latents, map_location="cpu", weights_only=False)
    z, n = c["z"], c["z"].shape[0]
    assert z.dtype == torch.float16 and z.shape[1] == 24 and n > 0
    assert torch.isfinite(z.float()).all()
    for key, dtype in dict(obj=torch.int32, win=torch.int16, start=torch.float32, n_tokens=torch.int16).items():
        assert c[key].dtype == dtype and c[key].shape == (n,), key
    assert set(c["splits"]) == {"train", "validation"}
    assert sum(cnt for _, cnt in c["splits"].values()) == n
    meta = c["meta"]
    assert meta["stride"] == STRIDE and meta["window"] == 30.0 and meta["dim"] == 24
    assert meta["min_tokens"] == 4 and meta["cap"] == 32 and meta["encoder_kind"] == "mae"
    assert (c["n_tokens"] >= 0).all() and (c["n_tokens"] >= meta["min_tokens"]).sum() > n // 2
    info = json.load(open(latents.with_suffix(".json")))
    assert info["counts"]["windows_total"] == n and info["stride"] == STRIDE

    enc, emeta = load_encoder(mae_ckpt)
    data = load_data(data_args_from(emeta.args))
    for split, (off, cnt) in c["splits"].items():
        ptr, objs = c["ptr"][split], c["objects"][split]
        n_obj = len(ptr) - 1
        assert ptr.dtype == torch.int64 and ptr[0] == off and ptr[-1] == off + cnt
        assert (ptr[1:] > ptr[:-1]).all()
        assert len(objs["label"]) == len(objs["period"]) == len(objs["superclass"]) == len(objs["index"]) == n_obj
        assert n_obj == len(data[split])  # no --max-objects: every record
        for i in range(n_obj):
            rows = slice(int(ptr[i]), int(ptr[i + 1]))
            rec = data[split][int(objs["index"][i])]
            starts = grid_starts(rec, emeta.cfg, advance=STRIDE)
            assert len(starts) == ptr[i + 1] - ptr[i]
            assert np.allclose(c["start"][rows].numpy(), starts, atol=1e-3)
            assert (c["win"][rows].long() == torch.arange(len(starts))).all()
            assert (c["obj"][rows] == objs["index"][i]).all()
            assert objs["label"][i] == rec.label

    # a stored row is what the frozen encoder gives the same window
    split, i, w = "validation", 1, 3
    ptr, objs = c["ptr"][split], c["objects"][split]
    rec = data[split][int(objs["index"][i])]
    starts, frames, n_tokens = object_windows(
        rec, emeta.cfg, STRIDE, emeta.cfg.max_tokens, meta["created"]["seed"], int(objs["index"][i])
    )
    row = int(ptr[i]) + w
    assert c["n_tokens"][row] == n_tokens[w] and len(frames[w][0]) <= 32
    ref = enc.encode([emeta.spec.tokens([frames[w]])])[0, 0]
    assert torch.allclose(z[row].float(), ref, atol=1e-2, rtol=1e-2)


def test_cache_max_objects_and_seed(latents, mae_ckpt):
    # With --max-objects the obj column must still be the data[split]
    # position (what objects[split]["index"] holds), and --seed picks the
    # subset and the cap only: the records stay the checkpoint's.
    out = mae_ckpt.parent / "latents_sub.pt"
    cache_latents.main(
        [
            "--ckpt", str(mae_ckpt), "--out", str(out), "--stride", str(STRIDE),
            "--splits", "train", "--max-objects", "5", "--seed", "3",
            "--device", "cpu", "--workers", "0", "--batch-size", "16",
        ]  # fmt: skip
    )
    c = torch.load(out, map_location="cpu", weights_only=False)
    _, emeta = load_encoder(mae_ckpt)
    data = load_data(data_args_from(emeta.args))  # the checkpoint's seed
    ptr, objs = c["ptr"]["train"], c["objects"]["train"]
    idx = objs["index"]
    assert len(ptr) - 1 == 5 and c["meta"]["created"]["seed"] == 3
    assert (idx[1:] > idx[:-1]).all() and not torch.equal(idx, torch.arange(5))
    for i in range(5):
        rows = slice(int(ptr[i]), int(ptr[i + 1]))
        assert (c["obj"][rows] == idx[i]).all()
        rec = data["train"][int(idx[i])]
        assert objs["label"][i] == rec.label
        assert objs["period"][i] == pytest.approx(float(rec.period), rel=1e-6)
        assert len(grid_starts(rec, emeta.cfg, advance=STRIDE)) == ptr[i + 1] - ptr[i]


def test_latent_store_draw(latents):
    c = torch.load(latents, map_location="cpu", weights_only=False)
    store = LatentStore(c, "train", "cpu", 4)
    assert store.z.shape == (c["splits"]["train"][1], 24) and store.z.dtype == torch.float32
    gen = torch.Generator().manual_seed(0)
    rows, adv = store.draw(32, (1.0, 1.5), 3, STRIDE, gen, extra=2)
    assert rows.shape == (32, 6) and adv.shape == (32, 5)
    k = (rows[:, 1:] - rows[:, :-1]).float()
    assert torch.allclose(k * STRIDE, adv) and (k >= 2).all() and (k <= 3).all()
    assert (adv[:, -1] == adv[:, -2]).all() and (adv[:, -2] == adv[:, 2]).all()
    assert (store.valid[rows]).all() and (rows[:, -1] < store.end[rows[:, 0]]).all()
    with pytest.raises(RuntimeError, match="acceptance"):
        store.draw(8, (1000.0, 1000.0), 3, STRIDE, gen)


# ---------------------------------------------------------------- predictor


EVAL_KEYS = (
    "val_loss",
    "val_mse",
    "val_mse_persist",
    "val_mse_histmean",
    "val_mse_ratio",
    "val_nll",
    "val_nll_persist_gauss",
    "val_rollout_mse_h1",
    "val_rollout_persist_h1",
    "val_rollout_norm_drift_h1",
    "val_rollout_mse_h4",
    "val_sample_std",
    "val_true_std",
)


@pytest.mark.parametrize("kind", ["flow", "mse"])
def test_train_predictor_end_to_end(latents, kind, tmp_path):
    out = tmp_path / kind
    common = [
        "--latents", str(latents), "--out", str(out), "--kind", kind,
        "--batch-size", "8", "--eval-every", "10", "--ckpt-every", "10",
        "--log-every", "5", "--val-sequences", "16", "--eval-samples", "2",
        "--hidden", "16", "--depth", "1", "--history", "2", "--n-euler", "4",
        "--device", "cpu",
    ]  # fmt: skip
    train_predictor.main(common + ["--steps", "20"])
    assert (out / "pred.pt").is_file() and (out / "DONE").is_file()
    lines = [json.loads(l) for l in open(out / "log.jsonl")]
    evals = [l for l in lines if l["kind"] == "eval"]
    assert [e["step"] for e in evals] == [10, 20]
    for e in evals:
        for key in EVAL_KEYS:
            assert key in e, key
        for key in ("val_mse", "val_mse_persist", "val_mse_ratio", "val_nll_persist_gauss", "val_rollout_mse_h1", "val_rollout_norm_drift_h1", "val_true_std"):
            assert math.isfinite(e[key]), key
        assert e["val_rollout_norm_drift_h1"] > 0
        if kind == "mse":
            assert math.isnan(e["val_nll"]) and math.isnan(e["val_sample_std"])
        else:
            assert math.isfinite(e["val_nll"]) and e["val_sample_std"] > 0
    trains = [l for l in lines if l["kind"] == "train"]
    assert [t["step"] for t in trains] == [5, 10, 15, 20]
    assert all("lr" in t and "advance_lo" in t and "s_per_step" in t for t in trains)

    train_predictor.main(common + ["--steps", "30"])  # resume from last.pt
    lines = [json.loads(l) for l in open(out / "log.jsonl")]
    assert max(l["step"] for l in lines) == 30
    assert [l["step"] for l in lines if l["kind"] == "eval"] == [10, 20, 30]

    model, meta = load_predictor(out / "pred.pt")
    assert meta["step"] == 30 and meta["kind"] == kind and not model.training
    assert meta["latent_meta"]["stride"] == STRIDE and meta["latent_meta"]["dim"] == 24
    assert meta["hparams"]["history"] == 2 and "opt" not in meta
    c = torch.load(latents, map_location="cpu", weights_only=False)
    store = LatentStore(c, "validation", "cpu", meta["latent_meta"]["min_tokens"])
    rows, adv = store.draw(6, (1.0, 1.5), 2, STRIDE, torch.Generator().manual_seed(0))
    z = store.z[rows]
    hist, target = z[:, :2], z[:, 2]
    model2, _ = load_predictor(out / "pred.pt")
    p1 = model.predict_mean(hist, adv, 2, generator=torch.Generator().manual_seed(1))
    p2 = model2.predict_mean(hist, adv, 2, generator=torch.Generator().manual_seed(1))
    assert p1.shape == (6, 24) and torch.allclose(p1, p2)
    lp = model.log_prob(target, hist, adv)
    assert lp.shape == (6,) and (torch.isfinite(lp).all() if kind == "flow" else torch.isnan(lp).all())
    assert not torch.equal(model.mu, torch.zeros(24))  # standardisation was set


# --------------------------------------------------------------------- units


def gauss_logpdf(x, mean, sigma):
    d = x.shape[-1]
    return -0.5 * (((x - mean) / sigma).square().sum(-1) + d * math.log(2 * math.pi)) - d * math.log(sigma)


def test_anchored_log_density_zero_field():
    x1, anchor = torch.randn(5, 3), torch.randn(5, 3)
    zero = lambda x, s: torch.zeros_like(x)
    lp = anchored_log_density(zero, x1, anchor, 0.3, n_steps=4)
    assert torch.allclose(lp, gauss_logpdf(x1, anchor, 0.3), atol=1e-5)
    # a fresh FlowPredictor has a zero-initialised output: the same field
    m = FlowPredictor(3, 2, hidden=8, depth=1, sigma0=0.3, n_euler=4)
    hist, adv, z_next = torch.randn(5, 2, 3), torch.full((5, 2), 1.25), torch.randn(5, 3)
    lp = m.log_prob(z_next, hist, adv)
    assert torch.allclose(lp, gauss_logpdf(z_next, hist[:, -1], 0.3), atol=1e-5)
    # standardisation enters as its log Jacobian
    m.sd.fill_(2.0)
    lp2 = m.log_prob(z_next, hist, adv)
    assert torch.allclose(lp2, gauss_logpdf(z_next / 2, hist[:, -1] / 2, 0.3) - 3 * math.log(2.0), atol=1e-5)


def test_anchored_log_density_linear_field():
    # A zero field has zero divergence and cannot pin the sign of the
    # integrated divergence. v(x, s) = x - a maps x_0 ~ N(a, s0^2 I) to
    # x_1 = a + e (x_0 - a) ~ N(a, e^2 s0^2 I); the Jacobian is I, so the
    # Rademacher estimate of div v is exact and only the Euler error remains.
    d, sigma0 = 3, 0.7
    x1, anchor = torch.randn(4, d), torch.randn(4, d)
    linear = lambda x, s: x - anchor
    lp = anchored_log_density(linear, x1, anchor, sigma0, n_steps=1000)
    expect = gauss_logpdf(x1, anchor, math.e * sigma0)
    assert torch.allclose(lp, expect, atol=1e-2)
    # and a contracting field (div v < 0) raises the density, not lowers it
    # (x1 near the anchor: the Euler error grows with the distance)
    contract = lambda x, s: anchor - x
    x1c = anchor + 0.3 * torch.randn(4, d)
    lp_c = anchored_log_density(contract, x1c, anchor, sigma0, n_steps=1000)
    expect_c = gauss_logpdf(x1c, anchor, sigma0 / math.e)
    assert torch.allclose(lp_c, expect_c, atol=2e-2)
    assert (lp_c - expect_c).abs().max() < 0.1 * 2 * d  # the flipped sign is off by 2d


def test_sample_zero_field_returns_x0():
    m = FlowPredictor(3, 2, hidden=8, depth=1, sigma0=0.3)
    hist, adv = torch.randn(5, 2, 3), torch.full((5, 2), 1.0)
    x = m.sample(hist, adv, n_euler=3, generator=torch.Generator().manual_seed(0))
    eps = torch.randn(5, 3, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(x, hist[:, -1] + 0.3 * eps)
    mean = m.predict_mean(hist, adv, 4, n_euler=2, generator=torch.Generator().manual_seed(0))
    assert mean.shape == (5, 3)
    proj = FlowPredictor(3, 2, hidden=8, depth=1, reproject="norm")
    y = proj.sample(hist, adv, n_euler=2)
    assert torch.allclose(y.norm(dim=-1), proj.norm_mean.expand(5), atol=1e-5)
    with pytest.raises(ValueError):
        FlowPredictor(3, 2, reproject="sphere")
    mse = MsePredictor(3, 2, hidden=8, depth=1)
    assert torch.allclose(mse.predict_mean(hist, adv), hist[:, -1])  # zero output
    assert torch.isfinite(mse.loss(hist, torch.randn(5, 3), adv))
    loss = m.loss(hist, torch.randn(5, 3), adv)
    loss.backward()
    assert m.out.weight.grad is not None


def test_round_advance():
    k = round_advance(torch.tensor([0.05, 0.24, 0.26, 1.0, 1.37, 1.5]), 0.25)
    assert k.tolist() == [1, 1, 1, 4, 5, 6] and k.dtype == torch.int64
    assert (round_advance(torch.rand(100) * 0.1, 0.25) >= 1).all()
