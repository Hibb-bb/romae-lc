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
    ]
    train_wm.main(common + ["--steps", "4"])
    assert (out / "wm.pt").is_file() and (out / "DONE").is_file()
    train_wm.main(common + ["--steps", "6"])  # resume from last.pt
    lines = [json.loads(l) for l in open(out / "log.jsonl")]
    assert max(l["step"] for l in lines) == 6
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
