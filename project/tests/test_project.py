"""CPU tests of the project modules on the toy simulator (seconds each)."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch

from romae_lc import ARPredictor, FrameConfig, LeWorldModel, RoMAE, lewm_mlp
from romae_lc.data import SimConfig, SurveyConfig, normalize, simulate

from project import baselines as bl
from project import flow
from project.common import (
    TokenSpec,
    data_args_from,
    err_channel,
    err_stats,
    grid_item,
    grid_starts,
    subset,
)
from project.decoder import (
    QueryDecoder,
    decode_mean,
    decode_samples,
    decoder_loss,
    drop_tokens,
    hide_tokens,
    load_decoder,
    decoder_state,
)
from project.diagnostics import grid_surprise, shuffle_score, surprise_summary
from project.energy import (
    PathEnergy,
    init_path,
    interpolate_path,
    langevin,
    map_path,
    periodicity_energy,
)
from project.inject import KINDS, _auroc, band_templates, inject, summarize
from project.residual import (
    FlowResidual,
    GaussianResidual,
    bin_edges,
    bimodality,
    collect,
    load_residual,
    residual_state,
)

SURVEYS = (SurveyConfig("g", 480.7, 160, 0.05), SurveyConfig("r", 622.1, 120, 0.06))
SIM = SimConfig(baseline_days=400.0, logP_range=(-0.3, 1.0))
WL = {i: s.wavelength_nm for i, s in enumerate(SURVEYS)}


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
def model():
    backbone = RoMAE(
        encoder=dict(d_model=24, nhead=2, depth=1),
        n_channels=2,
        rope_timescales=[1.0, 30.0],
    )
    pred = ARPredictor(24, n_frames=2, depth=1, heads=2, dim_head=8, mlp_dim=32)
    m = LeWorldModel(
        backbone,
        projector=lewm_mlp(24, 32),
        predictor=pred,
        pred_proj=lewm_mlp(24, 32),
        history=2,
        n_slices=16,
    )
    # a few steps so BatchNorm running statistics are not the defaults
    return m.eval()


# ------------------------------------------------------------------ common


def test_err_channel_and_collate(records, spec, cfg):
    from romae_lc import FrameDataset

    ds = FrameDataset(records, cfg, epoch_seed=False)
    batch = spec.collate()([ds[0], ds[1]])
    for f in batch["frames"]:
        assert f.values.shape[-1] == 2 and f.extras is not None
        assert (f.values[f.pad_mask] == 0).all()
        real = ~f.pad_mask
        assert torch.isfinite(f.values[real]).all() and (f.extras[real] > 0).all()
    tok = spec.tokens([ds[0]["frames"][0], ds[1]["frames"][0]])
    assert tok.values.shape[-1] == 2
    single = TokenSpec(spec.tokenize, None)
    assert (
        single.n_channels == 1
        and single.collate()([ds[0]])["frames"][0].values.shape[-1] == 1
    )
    assert TokenSpec.from_dict(spec.to_dict()).err_stats == spec.err_stats
    with pytest.raises(ValueError):
        err_channel(
            type(tok)(tok.values, tok.positions, tok.pad_mask, None), spec.err_stats
        )


def test_grid_item_and_starts(records, cfg):
    r = records[0]
    item = grid_item(r, cfg, cap=10, seed=0)
    starts = grid_starts(r, cfg)
    assert len(item["frames"]) == len(starts) and item["actions"].shape == (
        len(starts),
        1,
    )
    for f in item["frames"]:
        assert len(f) == 4 and len(f[0]) <= 10
    assert (
        subset(records, 5, 0) == subset(records, 5, 0) and len(subset(records, 5)) == 5
    )


def test_data_args_from():
    saved = dict(data="x", classes=None, max_rows=3, min_points=8, n_sim=4, seed=1)
    ns = data_args_from(saved, argparse.Namespace(data=None, max_rows=7))
    assert ns.data == "x" and ns.max_rows == 7 and ns.seed == 1


# ------------------------------------------------------------- diagnostics


def test_diagnostics(records, spec, cfg, model):
    s = shuffle_score(model, records, cfg, spec, "cpu", n=6, batch_size=3)
    assert 0.0 <= s <= 1.0 + 1e-6
    g = grid_surprise(model, records[0], cfg, spec, "cpu")
    k = len(g["starts"])
    assert (
        g["scores"].shape == (k - 1,)
        and g["z"].shape == (k, 24)
        and g["n_tokens"].shape == (k,)
    )
    summary = surprise_summary(model, records, cfg, spec, "cpu", n=4)
    assert summary["n_objects"] <= 4 and set(summary["per_superclass"]) <= {
        f"c{i}" for i in range(5)
    }


# ------------------------------------------------------------------ inject


def test_inject_kinds(records):
    r = records[1]
    rng = np.random.default_rng(0)
    t_star = float(np.median(r.t))
    templates, resid = band_templates(r, r.period)
    assert set(templates) == set(np.unique(r.band).tolist())
    for kind in KINDS:
        inj, params = inject(r, kind, t_star, rng)
        assert params["kind"] == kind and inj.n == r.n and inj.meta["inject"] == params
        before = r.t < t_star - 4 * params.get("width", 0.0)  # the bump has a width
        assert np.allclose(inj.y[before], r.y[before], atol=1e-3)
        assert not np.allclose(inj.y, r.y)
    with pytest.raises(ValueError):
        inject(r, "nope", t_star, rng)


def test_auroc_and_summarize():
    assert _auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert _auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert abs(_auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) - 0.5) < 1e-9
    res = [dict(s0=np.zeros(6), s1=np.array([0, 0, 1.0, 0.5, 0, 0]), k_star=3)]
    s = summarize(res)
    assert s["n"] == 1 and s["hit_rate"] == 1.0 and s["auroc_delta"] > 0.8


# ---------------------------------------------------------------- residual


def test_gaussian_residual():
    rng = torch.Generator().manual_seed(0)
    dt = torch.rand(400, generator=rng) + 1
    r = torch.randn(400, 6, generator=rng) * dt[:, None]
    edges = bin_edges(dt, 3)
    assert len(edges) == 2 and edges[0] < edges[1]
    for full in (False, True):
        g = GaussianResidual(6, edges, full=full).fit(r, dt)
        assert g.count.sum() == 400 and g.width()[0] < g.width()[-1]
        e = g.energy(r, dt)
        assert e.shape == (400,) and (e >= 0).all()
        assert torch.isfinite(g.nll(r, dt)).all()
        s = g.sample(dt, generator=rng)
        assert s.shape == (400, 6) and abs(s.std() - r.std()) < 0.3
        x = r[:5].clone().requires_grad_(True)
        g.energy(x, dt[:5]).sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


def test_flow_residual_and_roundtrip(tmp_path):
    rng = torch.Generator().manual_seed(0)
    dt = torch.rand(64, generator=rng) + 1
    r, z = torch.randn(64, 6, generator=rng), torch.randn(64, 8, generator=rng)
    f = FlowResidual(6, 8, hidden=16, depth=1)
    f.set_stats(r, dt)
    loss = f.loss(r, z, dt)
    assert torch.isfinite(loss)
    loss.backward()
    assert f.sample(z, dt, n_steps=2).shape == (64, 6)
    e = f.energy(r, z, dt)
    assert e.shape == (64,)
    e_exact = f.energy(r[:4], z[:4], dt[:4], exact=True, n_steps=2)
    assert torch.isfinite(e_exact).all()
    g = GaussianResidual(6, bin_edges(dt, 2)).fit(r, dt)
    torch.save(residual_state(g, f, dict(a=1)), tmp_path / "res.pt")
    g2, f2, meta = load_residual(tmp_path / "res.pt")
    assert torch.allclose(g2.var, g.var) and f2 is not None and meta["a"] == 1
    assert (
        bimodality(
            np.concatenate([np.random.randn(500) - 4, np.random.randn(500) + 4])
        )["bc"]
        > 5 / 9
    )


def test_collect(records, spec, cfg, model):
    from project.common import frame_loader

    loader = frame_loader(records[:8], cfg, spec, 4)
    out = collect(model, loader, "cpu")
    n = len(loader.dataset) * (cfg.n_frames - 1)
    assert (
        out["r"].shape == (n, 24)
        and out["dt"].shape == (n,)
        and out["z"].shape == (n, 24)
    )


# ----------------------------------------------------------------- decoder


@pytest.mark.parametrize("kind", ["flow", "mse"])
def test_decoder(records, spec, cfg, model, kind, tmp_path):
    from romae_lc import FrameDataset

    ds = FrameDataset(records, cfg, epoch_seed=False)
    batch = spec.collate()([ds[0], ds[1], ds[2]])
    tok = batch["frames"][0]
    z = model.encode([tok])[:, 0].detach()
    dec = QueryDecoder(
        24,
        model.backbone.rope.layout,
        spec.err_stats,
        kind,
        d_model=24,
        nhead=2,
        depth=1,
    )
    loss = decoder_loss(dec, z, tok)
    assert torch.isfinite(loss)
    loss.backward()
    mu = decode_mean(dec, z, tok, n_steps=3)
    assert mu.shape == tok.pad_mask.shape and torch.isfinite(mu).all()
    assert (
        decode_samples(dec, z, tok, n_samples=2, n_steps=2).shape
        == (2,) + tok.pad_mask.shape
    )
    zz = z.clone().requires_grad_(True)
    decode_mean(dec, zz, tok, n_steps=2, grad=True).sum().backward()
    assert zz.grad is not None and zz.grad.abs().sum() > 0
    dropped = drop_tokens(tok, 0.5, keep_min=2)
    assert (dropped.n_real >= torch.minimum(tok.n_real, torch.tensor(2))).all()
    assert (dropped.n_real <= tok.n_real).all() and (dropped.n_real < tok.n_real).any()
    hide = ~tok.pad_mask & (torch.arange(tok.pad_mask.shape[1])[None] % 2 == 0)
    assert (hide_tokens(tok, hide).n_real == tok.n_real - hide.sum(1)).all()
    torch.save(decoder_state(dec, dict(kind=kind), 3), tmp_path / "dec.pt")
    dec2, meta = load_decoder(tmp_path / "dec.pt")
    assert meta["step"] == 3 and torch.allclose(
        decode_mean(dec2, z, tok, n_steps=3), mu
    )
    with pytest.raises(ValueError):
        QueryDecoder(
            24, model.backbone.rope.layout, spec.err_stats, kind, d_model=24, nhead=3
        )


# -------------------------------------------------------------------- flow


def test_flow_utils():
    x1, eps = torch.randn(4, 3), torch.randn(4, 3)
    s = torch.tensor([0.0, 0.5, 1.0, 0.25])
    xs = flow.interpolate(x1, eps, s[:, None])
    assert torch.allclose(xs[0], eps[0]) and torch.allclose(xs[2], x1[2], atol=1e-3)
    assert flow.fm_loss(flow.target(x1, eps), x1, eps) == 0
    zero = lambda x, s: torch.zeros_like(x)
    assert torch.allclose(flow.sample(zero, eps, 4), eps)
    assert torch.allclose(flow.sample(zero, eps, 4, "midpoint"), eps)
    lp = flow.log_density(zero, x1, n_steps=2)
    expect = -0.5 * (x1.square().sum(-1) + 3 * np.log(2 * np.pi))
    assert torch.allclose(lp, expect, atol=1e-5)
    # the sign of the integrated divergence: v(x, s) = x sends N(0, I) to
    # N(0, e^2 I), so log p_1(x_1) = log N(x_1; 0, I e^2) = base(x_1 / e) - d
    linear = lambda x, s: x
    lp = flow.log_density(linear, x1, n_steps=1000)
    expect = -0.5 * ((x1 / np.e).square().sum(-1) + 3 * np.log(2 * np.pi)) - 3
    assert torch.allclose(lp, expect, atol=1e-2)


# ----------------------------------------------------------------- baselines


def test_baselines():
    t = np.linspace(0, 10, 21)
    y = 2 * t + 1
    e = np.full_like(t, 0.1)
    tq = np.array([2.5, 7.5])
    mu, var = bl.linear(t, y, e, tq)
    assert np.allclose(mu, 2 * tq + 1) and (var > 0).all()
    mu_c, _ = bl.constant(t, y, e, tq)
    mu_g, var_g = bl.gp_rbf(t, np.sin(t), e, tq)
    assert np.abs(mu_g - np.sin(tq)).max() < np.abs(mu_c - np.sin(tq)).max()
    mu_p, _ = bl.gp_periodic(t, np.sin(t), e, tq, period=2 * np.pi)
    assert np.abs(mu_p - np.sin(tq)).max() < 0.2
    band = np.array([0, 1] * 10 + [0])
    mu_b, var_b = bl.per_band(bl.linear, t, y, e, band, tq, np.array([0, 1]))
    assert mu_b.shape == (2,) and (var_b > 0).all()
    assert bl.gaussian_nll(np.zeros(2), np.zeros(2), np.ones(2)).sum() == pytest.approx(
        np.log(2 * np.pi)
    )


# ------------------------------------------------------------------- energy


def test_energy_and_inference(records, spec, cfg, model):
    dec = QueryDecoder(
        24,
        model.backbone.rope.layout,
        spec.err_stats,
        "mse",
        d_model=24,
        nhead=2,
        depth=1,
    )
    energy = PathEnergy(model, dec, None, w_dyn=1.0, w_obs=1.0, w_prior=0.1)
    item = grid_item(records[2], cfg, cap=16)
    batch = spec.collate()([item])
    frames, actions = batch["frames"], batch["actions"][0]
    K = len(frames)
    assert K >= 3
    observed = np.ones(K, bool)
    observed[1] = False
    Z0 = init_path(model, frames, actions, observed)
    assert Z0.shape == (1, K, 24) and torch.isfinite(Z0).all()
    windows = [
        (k, frames[k])
        for k in range(K)
        if observed[k] and batch["n_tokens"][0, k] >= cfg.min_tokens
    ]
    e, parts = energy.total(Z0, actions, windows)
    assert (
        e.shape == (1,)
        and parts["dyn"].shape == (1, K - 1)
        and parts["obs"].shape == (1, len(windows))
    )
    Zp = (Z0 + 0.5 * torch.randn_like(Z0)).requires_grad_(True)
    e2, _ = energy.total(Zp, actions, windows)
    e2.sum().backward()
    assert Zp.grad.abs().sum() > 0
    Zmap, trace = map_path(energy, Zp.detach(), actions, windows, steps=30, lr=2e-2)
    assert trace[-1] < trace[0]
    samples, ltrace = langevin(
        energy, Zmap, actions, windows, n_chains=3, steps=3, eta=1e-3
    )
    assert samples.shape == (3, K, 24) and len(ltrace) == 3
    # residual-based dynamics energy
    gauss = GaussianResidual(24, [1.5], full=False)
    gauss.var.fill_(0.5)
    e_res = PathEnergy(model, None, gauss).dyn(Zmap, actions)
    assert e_res.shape == (1, K - 1) and (e_res >= 0).all()
    # periodicity
    t = np.arange(6) * 30.0
    Zc = torch.ones(6, 4)
    assert periodicity_energy(Zc, t, 45.0) == 0
    Zl = torch.arange(6.0)[:, None].expand(6, 4)
    assert torch.allclose(
        interpolate_path(Zl, t, np.array([15.0, 150.0])),
        torch.tensor([[0.5] * 4, [5.0] * 4]),
    )
    assert torch.isnan(periodicity_energy(Zc, t, 1e4))


# ------------------------------------------------------------ end to end


def test_train_scripts_end_to_end(tmp_path):
    from project import inject as inj_mod
    from project import residual as res_mod
    from project import train_decoder, train_wm

    out = tmp_path / "wm"
    common = [
        "--data",
        "sim",
        "--n-sim",
        "48",
        "--width",
        "24",
        "--depth",
        "1",
        "--heads",
        "2",
        "--pred-depth",
        "1",
        "--pred-heads",
        "2",
        "--pred-dim-head",
        "8",
        "--pred-mlp",
        "32",
        "--proj-hidden",
        "32",
        "--n-slices",
        "16",
        "--window",
        "30",
        "--min-tokens",
        "4",
        "--max-tokens",
        "32",
        "--batch-size",
        "4",
        "--eval-every",
        "4",
        "--ckpt-every",
        "2",
        "--log-every",
        "2",
        "--probe-train",
        "16",
        "--probe-val",
        "8",
        "--shuffle-objects",
        "6",
        "--surprise-objects",
        "2",
        "--ladder-curves",
        "8",
        "--workers",
        "0",
        "--device",
        "cpu",
        "--out",
        str(out),
        "--n-frames",
        "4",
        "--recon-weight",
        "0.5",
        "--recon-width",
        "24",
        "--recon-heads",
        "2",
        "--recon-depth",
        "1",
    ]
    train_wm.main(common + ["--steps", "4"])
    assert (out / "wm.pt").is_file() and (out / "DONE").is_file()
    train_wm.main(common + ["--steps", "6"])  # resume from last.pt
    lines = [json.loads(l) for l in open(out / "log.jsonl")]
    assert max(l["step"] for l in lines) == 6
    kinds = [l["kind"] for l in lines]
    assert (
        kinds[0] == "baseline" and lines[1]["kind"] == "eval" and lines[1]["step"] == 0
    )
    evals = [l for l in lines if l["kind"] == "eval"]
    for key in ("val_sigreg_bn_train", "val_recon", "probe_macro_f1", "logP_r2_within"):
        assert all(key in e for e in evals)
    assert all(l["recon"] is not None for l in lines if l["kind"] == "train")
    ckpt = torch.load(out / "wm.pt", map_location="cpu", weights_only=False)
    assert (
        "recon" in ckpt
        and ckpt["ladder"]["heads"] == 2
        and ckpt["args"]["n_frames"] == 4
    )
    train_decoder.main(
        [
            "--ckpt",
            str(out / "wm.pt"),
            "--kind",
            "mse",
            "--out",
            str(tmp_path / "dec"),
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--width",
            "24",
            "--heads",
            "2",
            "--depth",
            "1",
            "--eval-every",
            "2",
            "--ckpt-every",
            "2",
            "--val-objects",
            "6",
            "--baselines",
            "--sample-steps",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert (tmp_path / "dec" / "dec.pt").is_file()
    res_mod.main(
        [
            "--ckpt",
            str(out / "wm.pt"),
            "--out",
            str(tmp_path / "res"),
            "--batch-size",
            "4",
            "--n-chains",
            "3",
            "--horizons",
            "1",
            "--workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert (tmp_path / "res" / "residual.pt").is_file()
    inj_mod.main(
        [
            "--ckpt",
            str(out / "wm.pt"),
            "--n-objects",
            "3",
            "--kinds",
            "phase",
            "bump",
            "--out",
            str(tmp_path / "inj"),
            "--device",
            "cpu",
        ]
    )
    s = json.load(open(tmp_path / "inj" / "summary.json"))
    assert set(s["results"]) == {"phase", "bump"}


def test_fused_encode_matches_per_frame(records, spec, cfg, model):
    from romae_lc import FrameDataset

    from project.common import use_fused_encode

    ds = FrameDataset(records, cfg, epoch_seed=False)
    batch = spec.collate()([ds[0], ds[1], ds[2]])
    frames = batch["frames"]
    use_fused_encode(model, False)
    ref = model.encode(frames)
    ref_feats = model.encode(frames, project=False)
    use_fused_encode(model, True)
    fused = model.encode(frames)
    fused_feats = model.encode(frames, project=False)
    use_fused_encode(model, False)
    assert fused.shape == ref.shape == (3, cfg.n_frames, 24)
    assert torch.allclose(fused_feats, ref_feats, atol=1e-5)
    assert torch.allclose(fused, ref, atol=1e-5)
    # the training forward and surprise go through the same method
    use_fused_encode(model, True)
    out = model(frames, batch["actions"])
    s = model.surprise(frames, batch["actions"])
    use_fused_encode(model, False)
    assert torch.isfinite(out.loss) and s.shape == (3, cfg.n_frames - 1)


def test_pretrain_mae_then_init_backbone(tmp_path):
    from project import pretrain_mae, train_wm
    from project.common import load_mae, load_wm

    out = tmp_path / "mae"
    common = [
        "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
        "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
        "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
        "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
        "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
        "--ladder-curves", "8", "--workers", "0", "--device", "cpu", "--out", str(out),
    ]  # fmt: skip
    pretrain_mae.main(common + ["--steps", "4"])
    assert (out / "mae.pt").is_file() and (out / "DONE").is_file()
    pretrain_mae.main(common + ["--steps", "6"])  # resume from last.pt
    lines = [json.loads(l) for l in open(out / "log.jsonl")]
    assert max(l["step"] for l in lines) == 6 and lines[0]["kind"] == "baseline"
    assert [l["step"] for l in lines if l["kind"] == "eval"][0] == 0
    mae, meta = load_mae(out / "mae.pt")
    assert (
        meta.ladder.layers == 2 and meta.ladder.heads == 2 and meta.ladder.n_rungs == 12
    )
    assert mae.per_layer_rope and mae.rope.blocks[0].per_head

    wm = tmp_path / "wm"
    train_wm.main(
        [
            "--data", "sim", "--n-sim", "48", "--pred-depth", "1", "--pred-heads", "2",
            "--pred-dim-head", "8", "--pred-mlp", "32", "--proj-hidden", "32",
            "--n-slices", "16", "--window", "30", "--min-tokens", "4", "--max-tokens", "32",
            "--n-frames", "4", "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2",
            "--log-every", "2", "--probe-train", "16", "--probe-val", "8",
            "--shuffle-objects", "6", "--surprise-objects", "2", "--workers", "0",
            "--device", "cpu", "--steps", "4", "--no-eval-at-start",
            "--init-backbone", str(out / "mae.pt"), "--out", str(wm),
        ]  # fmt: skip
    )
    model, wmeta = load_wm(wm / "wm.pt")
    assert (
        wmeta.args["width"] == 24
        and wmeta.args["depth"] == 2
        and wmeta.args["heads"] == 2
    )
    assert wmeta.ladder.to_dict() == meta.ladder.to_dict()
    assert model.backbone.rope_layout == mae.rope_layout
    enc = mae.backbone("cls")
    for (n, a), (m, b) in zip(
        enc.named_parameters(), model.backbone.named_parameters()
    ):
        assert (
            n == m and a.shape == b.shape
        )  # same architecture as the pretrained encoder


def test_freeze_backbone(tmp_path):
    """Stage-2 MSE ablation on frozen stage-1 latents: the encoder is
    untouched, the projector an identity, SIGReg off, baselines logged."""
    from project import pretrain_mae, train_wm
    from project.common import load_mae, load_wm

    out = tmp_path / "mae"
    common = [
        "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
        "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
        "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
        "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
        "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
        "--ladder-curves", "8", "--workers", "0", "--device", "cpu", "--out", str(out),
    ]  # fmt: skip
    pretrain_mae.main(common + ["--steps", "4"])
    mae, _ = load_mae(out / "mae.pt")
    enc = mae.backbone("cls")

    wm = tmp_path / "wm"
    wm_args = [
        "--data", "sim", "--n-sim", "48", "--pred-depth", "1", "--pred-heads", "2",
        "--pred-dim-head", "8", "--pred-mlp", "32", "--proj-hidden", "32",
        "--n-slices", "16", "--window", "30", "--min-tokens", "4", "--max-tokens", "32",
        "--n-frames", "4", "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2",
        "--log-every", "2", "--probe-train", "16", "--probe-val", "8",
        "--shuffle-objects", "6", "--surprise-objects", "2", "--workers", "0",
        "--device", "cpu", "--no-eval-at-start", "--init-backbone", str(out / "mae.pt"),
        "--freeze-backbone",
    ]  # fmt: skip

    def check(steps):
        ckpt = torch.load(wm / "wm.pt", map_location="cpu", weights_only=False)
        assert ckpt["step"] == steps
        assert ckpt["args"]["freeze_backbone"] is True and ckpt["args"]["lamb"] == 0.0
        assert ckpt["hparams"]["proj_hidden"] == 0 and ckpt["hparams"]["lamb"] == 0.0
        model, _ = load_wm(wm / "wm.pt")
        assert isinstance(model.projector, torch.nn.Identity)
        assert not model.encoder_only
        names = [n for n, _ in enc.named_parameters()]
        assert names == [n for n, _ in model.backbone.named_parameters()]
        for (_, a), (_, b) in zip(enc.named_parameters(), model.backbone.named_parameters()):
            assert torch.equal(a, b)  # the frozen encoder is the MAE encoder
        evals = [
            json.loads(l) for l in open(wm / "log.jsonl") if '"eval"' in l
        ]
        evals = [e for e in evals if e["kind"] == "eval"]
        assert evals and evals[-1]["step"] == steps
        for e in evals:
            for k in ("val_pred_persist", "val_pred_histmean", "val_pred_ratio"):
                assert np.isfinite(e[k])
            assert e["val_pred_ratio"] == pytest.approx(
                e["val_pred"] / e["val_pred_persist"], rel=1e-6
            )
        return model

    train_wm.main(wm_args + ["--steps", "4", "--out", str(wm)])
    check(4)
    train_wm.main(wm_args + ["--steps", "6", "--out", str(wm)])  # resume
    check(6)

    with pytest.raises((SystemExit, ValueError)):
        train_wm.main(
            [a for a in wm_args if a not in ("--init-backbone", str(out / "mae.pt"))]
            + ["--steps", "4", "--out", str(tmp_path / "bad_no_init")]
        )
    with pytest.raises((SystemExit, ValueError)):
        train_wm.main(
            wm_args
            + ["--recon-weight", "0.5", "--steps", "4", "--out", str(tmp_path / "bad_recon")]
        )


def test_prediction_baselines():
    from project.train_wm import prediction_baselines

    z = torch.randn(3, 5, 7)
    persist, hist = prediction_baselines(z, history=3)
    assert persist == pytest.approx((z[:, :-1] - z[:, 1:]).square().mean().item())
    # t = 0: mean of z[:, :1]; t = 3: mean of z[:, 1:4]
    m3 = z[:, 1:4].mean(1)
    manual = torch.stack([z[:, 0], z[:, :2].mean(1), z[:, :3].mean(1), m3], 1)
    assert hist == pytest.approx((manual - z[:, 1:]).square().mean().item())
    assert prediction_baselines(z, history=1) == pytest.approx((persist, persist))


# --------------------------------------------------------- ladders and probes


def test_dense_ladder_dealing_and_geometry():
    from project.common import (
        Ladder,
        RopeGeometry,
        _nest,
        check_ladder,
        deal_ladder,
        resolve_model_args,
        rope_geometry,
    )

    w = np.geomspace(0.1, 100.0, 12)
    bands = np.array(deal_ladder(w, 2, 2, 3, "bands"))
    comb = np.array(deal_ladder(w, 2, 2, 3, "comb"))
    for dealt in (bands, comb):
        assert dealt.shape == (2, 2, 3)
        assert np.allclose(np.sort(dealt.ravel()), w)  # every rung exactly once
        for layer in dealt:  # every layer spans the whole range
            assert layer.min() <= w[1] and layer.max() >= w[-2]
    assert bands[0, 0].max() < bands[0, 1].min()  # bands: contiguous per head
    assert comb[0, 0].max() > comb[0, 1].min()  # comb: heads interleave
    assert np.array(deal_ladder(w[:3], 1, 1, 3)).shape == (1, 1, 3)
    with pytest.raises(ValueError):
        deal_ladder(w, 2, 2, 2, "bands")
    with pytest.raises(ValueError):
        deal_ladder(w, 2, 2, 3, "spiral")

    def ns(size):
        return resolve_model_args(
            argparse.Namespace(
                size=size,
                width=None,
                heads=None,
                depth=None,
                time_frac=0.875,
                p_rope=0.75,
            )
        )

    geo = rope_geometry(ns("light"))
    assert (geo.layers, geo.heads, geo.head_dim, geo.time_dim, geo.n_angles) == (
        6,
        3,
        64,
        56,
        21,
    )
    assert geo.n_rungs == 378 and rope_geometry(ns("wide")).n_rungs == 756
    assert (
        rope_geometry(
            argparse.Namespace(width=192, heads=3, depth=6, time_frac=None, p_rope=0.75)
        ).n_angles
        == 12
    )  # the old even split
    ladder = Ladder(
        0.01,
        _nest(bands / 0.0628, 2, 2),
        _nest(bands, 2, 2),
        0.1,
        100.0,
        "log",
        "",
        2,
        2,
        "bands",
    )
    assert ladder.n_rungs == 12 and ladder.per_head == 3
    assert ladder.flat == pytest.approx(w.tolist())
    check_ladder(ladder, RopeGeometry(2, 2, 12, 8, 3))
    with pytest.raises(ValueError, match="layers"):
        check_ladder(ladder, RopeGeometry(3, 2, 12, 8, 3))
    with pytest.raises(ValueError, match="heads"):
        check_ladder(ladder, RopeGeometry(2, 3, 12, 8, 3))
    with pytest.raises(ValueError, match="do not fit"):
        check_ladder(ladder, RopeGeometry(2, 2, 4, 4, 1))
    old = Ladder.from_dict(
        dict(time_scale=0.01, timescales=[1.0, 2.0], wavelengths=[0.06, 0.13], lam_min=0.06, lam_max=0.13, spacing="log")
    )  # fmt: skip
    assert old.layers == old.heads == 1 and old.deal == "shared" and old.n_rungs == 2
    assert _nest(np.ones((1, 1, 3)), 1, 1) == [1.0, 1.0, 1.0]
    assert np.array(_nest(np.ones((1, 2, 3)), 1, 2)).shape == (2, 3)


def test_dense_backbone_and_flat_layout(records, spec):
    from project.common import build_backbone, dense_ladder, flat_layout, rope_geometry

    args = argparse.Namespace(
        width=24, heads=2, depth=2, mlp_ratio=4.0, time_frac=0.875, p_rope=0.75, attention="softmax"
    )  # fmt: skip
    geo = rope_geometry(args)
    assert (geo.time_dim, geo.n_angles, geo.n_rungs) == (8, 3, 12)
    ladder = dense_ladder(
        records, geo, 30.0, "bands", "log", 0.75, 50.0, 8, 0, None, None
    )
    assert ladder.layers == 2 and ladder.heads == 2 and ladder.n_rungs == 12
    assert ladder.lam_max == 60.0 and ladder.deal == "bands"
    backbone = build_backbone(args, ladder, 2)
    assert backbone.per_layer_rope and len(backbone.rope_layers) == 2
    assert backbone.rope.blocks[0].per_head and backbone.rope.dims == [8, 4]
    got = sorted(
        float(v)
        for b in backbone.rope_layers
        for v in b.blocks[0].timescale[torch.isfinite(b.blocks[0].timescale)]
    )
    assert got == pytest.approx(sorted(np.asarray(ladder.timescales).ravel().tolist()))
    flat = flat_layout(backbone)
    assert len(flat[0]["timescales"]) == 3 and flat[0]["p"] == 0.75
    dec = QueryDecoder(24, flat, spec.err_stats, "mse", d_model=36, nhead=3, depth=1)
    assert dec.rope.blocks[0].per_head is False
    shared = build_backbone(
        args, dense_ladder(records, rope_geometry(argparse.Namespace(**dict(vars(args), depth=1))), 30.0, "comb", "log", 0.75, 50.0, 8, 0, None, None), 2
    )  # fmt: skip
    assert shared.per_layer_rope is False and shared.rope.blocks[0].per_head
    with pytest.raises(ValueError, match="layers"):
        build_backbone(argparse.Namespace(**dict(vars(args), depth=3)), ladder, 2)


def test_probe_metrics_and_baseline(records):
    from project.common import (
        balanced_accuracy,
        baseline_probe,
        describe_by_class,
        describe_probe,
        hand_features,
        macro_f1,
        probe_metrics,
    )

    y, pred = np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 1, 0])
    assert macro_f1(y, pred) == pytest.approx((4 / 6 + 4 / 5 + 0.0) / 3)
    assert balanced_accuracy(y, pred) == pytest.approx((2 / 3 + 1.0 + 0.0) / 3)
    tr, va = records[:16], records[16:]
    bands = sorted({int(b) for r in records for b in r.bands})
    x = hand_features(tr[0], bands)
    assert x.shape == (4 + 10 * len(bands),) and np.isfinite(x).all()
    x_tr = np.stack([hand_features(r, bands) for r in tr])
    x_va = np.stack([hand_features(r, bands) for r in va])
    m = probe_metrics(x_tr, tr, x_va, va, 5, min_train=1, min_val=1)
    for k in (
        "acc",
        "train_acc",
        "macro_f1",
        "balanced_acc",
        "majority_acc",
        "r2",
        "r2_within",
    ):
        assert k in m and np.isfinite(m[k])
    assert 0 <= m["acc"] <= 1 and 0 <= m["macro_f1"] <= 1 and m["n_val"] == len(va)
    assert set(m["n_by_superclass"]) == {f"c{r.label}" for r in va}
    assert sum(m["n_by_superclass"].values()) == len(va)
    assert baseline_probe(tr, va, bands, 5).keys() == m.keys()
    line = describe_by_class(
        {"ECL": 0.5, "ROT": 0.1, "X": 0.2}, {"ECL": 10, "ROT": 5, "X": 1}
    )
    assert line == "ROT 0.10/5 ECL 0.50/10 X 0.20/1"
    assert describe_probe(m).startswith("probe acc")


def test_batch_stats_context(model):
    from project.common import batch_stats

    bn = [m for m in model.projector.modules() if isinstance(m, torch.nn.BatchNorm1d)][
        0
    ]
    running = bn.running_mean.clone()
    x = torch.randn(8, 24)
    model.eval()
    with torch.no_grad():
        y_eval = model.projector(x)
        with batch_stats(model.projector):
            assert bn.training and bn.momentum == 0.0
            y_bn = model.projector(x)
        assert not bn.training and bn.momentum == 0.1
    assert torch.equal(bn.running_mean, running)
    assert not torch.allclose(y_eval, y_bn)
