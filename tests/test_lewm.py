"""LeWorldModel: predictor, action encoder, losses, rollout, surprise."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from romae_lc.data import SimConfig, SurveyConfig, simulate
from romae_lc.lejepa import EppsPulley, SlicedEppsPulley, mlp
from romae_lc.lewm import (
    ActionEncoder,
    ARPredictor,
    LeWorldModel,
    lewm_mlp,
    straightness,
)
from romae_lc.model import RoMAE
from romae_lc.tokenize import tokenize

D = 32
WL = {0: 480.0, 1: 620.0}


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def backbone(**kw):
    return RoMAE(encoder=dict(d_model=D, nhead=2, depth=1, **kw.pop("enc", {})), **kw)


def predictor():
    return ARPredictor(D, n_frames=3, depth=2, heads=2, dim_head=8, mlp_dim=64)


def model(bb=None, **kw):
    kw.setdefault("predictor", predictor())
    kw.setdefault("projector", lewm_mlp(D, 64))
    kw.setdefault("pred_proj", lewm_mlp(D, 64))
    kw.setdefault("n_slices", 32)
    return LeWorldModel(bb if bb is not None else backbone(), **kw)


def frames(T, B=4, seed=0, n_min=3, n_max=8):
    """``T`` token batches of ``B`` windows with ``n_min..n_max`` points each."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(T):
        lengths = torch.randint(n_min, n_max + 1, (B,), generator=g).tolist()
        times = [torch.rand(m, generator=g) * 30 for m in lengths]
        values = [torch.randn(m, generator=g) for m in lengths]
        bands = [torch.randint(0, 2, (m,), generator=g) for m in lengths]
        out.append(tokenize(times, values, bands, band_wavelengths=WL))
    return out


def actions_for(T, B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 1 + torch.rand(B, T, 1, generator=g)


def test_lewm_mlp_layers():
    net = lewm_mlp(32)
    assert [type(m) for m in net] == [nn.Linear, nn.BatchNorm1d, nn.GELU, nn.Linear]
    assert (net[0].in_features, net[0].out_features) == (32, 2048)
    assert net[1].num_features == 2048
    assert (net[3].in_features, net[3].out_features) == (2048, 32)
    small = lewm_mlp(32, 64, 16)
    assert (small[-1].in_features, small[-1].out_features) == (64, 16)
    old = mlp([4, 8, 2])
    assert [type(m) for m in old] == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Linear]
    with pytest.raises(ValueError):
        mlp([4, 8, 2], activation="tanh")
    assert EppsPulley(n_points=8).t.shape == (8,)
    with pytest.raises(ValueError):
        EppsPulley(n_points=1)


def test_action_encoder_shapes():
    enc = ActionEncoder(1, 32)
    out = enc(actions_for(4))
    assert out.shape == (4, 4, 32)
    assert isinstance(enc.patch_embed, nn.Conv1d)
    assert enc.patch_embed.in_channels == 1 and enc.patch_embed.out_channels == 10
    assert enc.patch_embed.kernel_size == (1,)
    assert isinstance(enc.embed[0], nn.Linear)
    assert (enc.embed[0].in_features, enc.embed[0].out_features) == (10, 128)
    half = enc(actions_for(4).half())
    assert half.dtype == torch.float32 and half.shape == (4, 4, 32)


def test_predictor_shapes_and_defaults():
    p = ARPredictor(192)
    assert len(p.layers) == 6
    blk = p.layers[0]
    assert blk.attn.heads == 16
    assert blk.attn.to_qkv.out_features == 3 * 16 * 64
    assert blk.mlp.net[1].out_features == 2048
    assert blk.attn.to_out[1].p == 0.1
    assert blk.mlp.net[3].p == 0.1 and blk.mlp.net[5].p == 0.1
    assert p.dropout.p == 0.0
    assert p.pos_embedding.shape == (1, 3, 192)
    for name in ("input_proj", "cond_proj", "output_proj"):
        assert isinstance(getattr(p, name), nn.Identity)
    wide = ARPredictor(192, hidden_dim=256)
    for name in ("input_proj", "cond_proj", "output_proj"):
        assert isinstance(getattr(wide, name), nn.Linear)
    cond = ARPredictor(32, cond_dim=8, depth=1, heads=2, dim_head=8, mlp_dim=64)
    assert isinstance(cond.cond_proj, nn.Linear)
    assert (cond.cond_proj.in_features, cond.cond_proj.out_features) == (8, 32)
    assert isinstance(cond.input_proj, nn.Identity)
    assert cond(torch.randn(2, 3, 32), torch.randn(2, 3, 8)).shape == (2, 3, 32)
    p2 = predictor()
    with pytest.raises(ValueError, match="n_frames"):
        p2(torch.randn(2, p2.n_frames + 1, D))


def test_predictor_is_identity_at_init():
    p = predictor().eval()
    x = torch.randn(4, 3, D)
    c1, c2 = torch.randn(4, 3, D), torch.randn(4, 3, D)
    out = p(x, c1)
    ref = F.layer_norm(x + p.pos_embedding[:, :3], (D,), p.norm.weight, p.norm.bias)
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(out, p(x, c2), atol=1e-6)
    for blk in p.layers:
        assert torch.all(blk.adaLN_modulation[-1].weight == 0)
        assert torch.all(blk.adaLN_modulation[-1].bias == 0)


def test_predictor_is_causal():
    p = predictor().eval()
    for blk in p.layers:
        nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.1)
        nn.init.normal_(blk.adaLN_modulation[-1].bias, std=0.1)
    x, c = torch.randn(4, 3, D), torch.randn(4, 3, D)
    x2, c2 = x.clone(), c.clone()
    x2[:, 2] += 1
    c2[:, 2] += 1
    out, out2 = p(x, c), p(x2, c2)
    assert torch.allclose(out[:, :2], out2[:, :2], atol=1e-6)
    assert not torch.allclose(out[:, 2], out2[:, 2])


def test_adaln_zero_gradients_at_init():
    m = model()
    out = m(frames(4), actions_for(4))
    out.loss.backward()
    for blk in m.predictor.layers:
        for sub in (blk.attn, blk.mlp):
            for name, prm in sub.named_parameters():
                assert prm.grad is not None, name
                assert torch.all(prm.grad == 0), name
    for name, prm in m.action_encoder.named_parameters():
        assert prm.grad is not None and torch.all(prm.grad == 0), name
    p = m.predictor
    assert p.layers[0].adaLN_modulation[-1].weight.grad.abs().sum() > 0
    assert p.pos_embedding.grad.abs().sum() > 0
    assert p.norm.weight.grad.abs().sum() > 0


def test_forward_losses_and_shapes():
    m = model()
    out = m(frames(4), actions_for(4))
    assert torch.allclose(out.loss, out.pred_loss + m.lamb * out.sigreg_loss, atol=1e-6)
    assert out.embedding.shape == (4, 4, D) and not out.embedding.requires_grad
    assert out.predicted.shape == (4, 3, D) and not out.predicted.requires_grad
    assert out.features.shape == (4, 4, D) and out.features.requires_grad
    assert out.straightness.shape == () and torch.isfinite(out.straightness)
    out.loss.backward()
    assert m.backbone.projection.weight.grad is not None
    assert m.projector[0].weight.grad is not None
    for name, prm in m.predictor.named_parameters():
        assert prm.grad is not None, name
    assert m.pred_proj[0].weight.grad is not None
    for name, prm in m.action_encoder.named_parameters():
        assert prm.grad is not None, name
    assert int(m.sigreg.step) == 1


def test_target_receives_gradient():
    m = model(lamb=0.0)
    out = m(frames(4), actions_for(4))
    out.features.retain_grad()
    out.loss.backward()
    assert out.features.grad[:, -1].abs().sum() > 0
    assert out.features.grad[:, 0].abs().sum() > 0


def test_sigreg_matches_official_semantics():
    B, T, Dz, M = 64, 4, 16, 128
    emb = torch.randn(B, T, Dz) * 1.7 + 0.3
    proj = emb.transpose(0, 1)  # [T, B, D]

    # literal port of le-wm module.SIGReg with an injected direction matrix
    knots = 17
    t = torch.linspace(0, 3, knots)
    dt = 3 / (knots - 1)
    w = torch.full((knots,), 2 * dt)
    w[[0, -1]] = dt
    phi = torch.exp(-t.square() / 2)
    w = w * phi
    g = torch.Generator().manual_seed(0)
    A = torch.randn(Dz, M, generator=g)
    A = A / A.norm(dim=0)
    x_t = (proj @ A).unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
    official = ((err @ w) * proj.size(-2)).mean()

    s = SlicedEppsPulley(n_slices=M, t_max=3.0, n_points=17)
    s.step.zero_()
    ours = s(proj)  # generator manual_seed(0) -> the same A on CPU
    assert torch.allclose(official, ours, atol=1e-5)
    vals = []
    for i in range(T):
        s.step.zero_()
        vals.append(s(emb[:, i]))
    assert torch.allclose(torch.stack(vals).mean(), ours, atol=1e-5)


def test_actions_none_path():
    m = model().eval()
    fr = frames(4)
    out = m(fr)
    assert torch.isfinite(out.loss)
    emb = m.encode(fr)[:, :3]
    with torch.no_grad():
        ref = m.pred_proj(m.predictor(emb, torch.zeros(4, 3, D)).flatten(0, 1))
        assert torch.allclose(m.predict(emb), ref.unflatten(0, (4, 3)), atol=1e-6)
    acts = actions_for(4)
    with pytest.raises(ValueError):
        m(fr, acts[:, :-1])
    with pytest.raises(ValueError):
        m(fr, acts.repeat(1, 1, 2))
    with pytest.raises(ValueError):
        m.rollout(fr[:2])


def test_forward_validation():
    m = model()
    with pytest.raises(ValueError):
        m(frames(1))
    with pytest.raises(ValueError):
        m(frames(5))  # T - 1 = 4 > n_frames = 3
    short = ARPredictor(D, n_frames=2, depth=1, heads=2, dim_head=8, mlp_dim=64)
    with pytest.raises(ValueError):
        model(predictor=short, history=3)
    with pytest.raises(ValueError):
        model(history=0)


def test_causality_end_to_end():
    """Exact only in eval mode: train-mode BatchNorm couples the rows (as in
    the official code)."""
    m = model().eval()
    fr, acts = frames(4), actions_for(4)
    fr2 = list(fr)
    fr2[2] = frames(1, seed=7)[0]
    out, out2 = m(fr, acts), m(fr2, acts)
    assert torch.allclose(out.predicted[:, :2], out2.predicted[:, :2], atol=1e-5)
    assert not torch.allclose(out.predicted[:, 2], out2.predicted[:, 2])


def test_rollout_lengths():
    m = model().train()
    fr, acts = frames(7), actions_for(7)
    step = int(m.sigreg.step)
    z = m.rollout(fr[:2], acts[:, :6])
    assert z.shape == (4, 7, D) and m.training and int(m.sigreg.step) == step
    assert m.rollout(fr[:2], n_steps=3).shape == (4, 5, D)
    m.eval()
    ctx = m.encode(fr[:2])
    m.train()
    assert torch.allclose(z[:, :2], ctx, atol=1e-6)
    assert m.rollout(fr[:5], n_steps=2).shape == (4, 7, D)
    with pytest.raises(ValueError):
        m.rollout(fr[:2], acts[:, :1])
    with pytest.raises(ValueError):
        m.rollout(fr[:2], acts[:, :3], n_steps=4)


def test_surprise_and_straightness():
    m = model().eval()
    s = m.surprise(frames(6))
    assert s.shape == (4, 5) and torch.isfinite(s).all() and (s >= 0).all()
    fr, acts = frames(4), actions_for(4)
    out = m(fr, acts)
    ref = ((out.predicted - out.embedding[:, 1:]) ** 2).mean(-1)
    assert torch.allclose(m.surprise(fr, acts), ref, atol=1e-5)
    z0, v = torch.randn(3, 1, D), torch.randn(3, 1, D)
    steps = torch.arange(5.0)[None, :, None]
    line = z0 + steps * v
    assert line.shape == (3, 5, D)
    assert straightness(line).shape == ()
    assert torch.allclose(straightness(line), torch.tensor(1.0), atol=1e-5)
    zig = torch.cat([z0, z0 + v, z0, z0 + v], 1)
    assert torch.allclose(straightness(zig), torch.tensor(-1.0), atol=1e-5)
    assert torch.isnan(straightness(torch.randn(2, 2, D)))


def _cut_frames(records, window, n_frames, rng):
    """Consecutive ``window``-day frames of every record (a plain contiguous
    cut, so this test does not depend on :mod:`romae_lc.frames`)."""
    fr, acts = [[] for _ in range(n_frames)], []
    for r in records:
        d = rng.uniform(1.0, 2.0, size=n_frames)
        need = window * (1 + d[:-1].sum())
        s = r.t.min() + rng.uniform(0, max(r.t.max() - r.t.min() - need, 0))
        for i in range(n_frames):
            sel = (r.t >= s) & (r.t < s + window)
            fr[i].append((r.t[sel] - s, r.y[sel], r.band[sel]))
            s += d[i] * window
        acts.append(d)
    tok = [
        tokenize(
            [torch.from_numpy(t) for t, _, _ in f],
            [torch.from_numpy(y) for _, y, _ in f],
            [torch.from_numpy(b) for _, _, b in f],
            band_wavelengths={0: 480.7, 1: 622.1},
            time_scale=0.5,
        )
        for f in fr
    ]
    return tok, torch.tensor(np.stack(acts), dtype=torch.float32)[..., None]


def test_loss_decreases_no_collapse():
    surveys = [SurveyConfig("g", 480.7, 80, 0.02), SurveyConfig("r", 622.1, 50, 0.03)]
    records = simulate(32, surveys, SimConfig(baseline_days=200), seed=0)
    fr, acts = _cut_frames(records, 20.0, 4, np.random.default_rng(0))
    assert len(fr) == 4 and fr[0].values.shape[0] == 32 and acts.shape == (32, 4, 1)
    m = model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        out = m(fr, acts)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    assert losses[-1] < 0.7 * losses[0]
    emb = out.embedding.reshape(-1, D)
    assert emb.std(0).mean() > 0.1
    assert torch.linalg.matrix_rank(emb) > D // 2


@pytest.mark.parametrize("rope", ["axial", "simplex"])
@pytest.mark.parametrize("attention", ["softmax", "linear"])
@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_backbone_variants(rope, attention, pool):
    bb = backbone(rope=rope, pool=pool, enc=dict(attention=attention))
    m = model(bb)
    out = m(frames(4), actions_for(4))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    assert m.rollout(frames(2), n_steps=2).shape == (4, 4, D)


def test_backbone_without_cls():
    m = model(backbone(use_cls=False, pool="mean"))
    out = m(frames(4, n_min=3), actions_for(4))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert m.rollout(frames(2, n_min=3), n_steps=2).shape == (4, 4, D)


def test_empty_window_with_cls():
    fr = frames(4)
    empty = [torch.zeros(0), torch.rand(5) * 30, torch.rand(4) * 30, torch.rand(6) * 30]
    fr[1] = tokenize(
        empty,
        [torch.randn(t.numel()) for t in empty],
        [torch.randint(0, 2, (t.numel(),)) for t in empty],
        band_wavelengths=WL,
    )
    single = [torch.rand(3) * 30, torch.rand(1) * 30, torch.rand(4) * 30, torch.rand(5)]
    fr[2] = tokenize(
        single,
        [torch.randn(t.numel()) for t in single],
        [torch.randint(0, 2, (t.numel(),)) for t in single],
        band_wavelengths=WL,
    )
    assert fr[1].pad_mask[0].all() and fr[2].n_real[1] == 1
    m = model(backbone(use_cls=True))
    out = m(fr, actions_for(4))
    assert torch.isfinite(out.loss)
    assert torch.isfinite(m.encode(fr)).all()


def test_empty_step_without_cls_raises():
    fr = frames(4)
    fr[1] = tokenize(
        [torch.zeros(0)] * 4,
        [torch.zeros(0)] * 4,
        [torch.zeros(0).long()] * 4,
        band_wavelengths=WL,
    )
    assert fr[1].values.shape[1] == 0
    m = model(backbone(use_cls=False, pool="mean"))
    with pytest.raises(ValueError, match="use_cls=False"):
        m(fr, actions_for(4))
    with pytest.raises(ValueError, match="use_cls=False"):
        m.encode(fr)


def test_forward_losses_float32_under_autocast():
    m = model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(frames(4), actions_for(4))
    assert out.pred_loss.dtype == torch.float32
    assert out.sigreg_loss.dtype == torch.float32
    assert out.loss.dtype == torch.float32
    assert torch.isfinite(out.loss)


def test_eval_mode_forward():
    m = model()
    m(frames(4), actions_for(4))  # populate BN running stats
    m.eval()
    out = m(frames(4, seed=1), actions_for(4))
    assert torch.isfinite(out.loss)
    assert int(m.sigreg.step) == 2


def test_hparams_roundtrip():
    p = predictor()
    p2 = ARPredictor(**p.hparams)
    assert p2.pos_embedding.shape == p.pos_embedding.shape
    assert len(p2.layers) == len(p.layers)
    assert p2.layers[0].attn.to_qkv.weight.shape == p.layers[0].attn.to_qkv.weight.shape
    assert p2.layers[0].mlp.net[1].out_features == 64
    m = model()
    hp = m.hparams
    assert set(hp) == {
        "embed_dim",
        "action_dim",
        "history",
        "lamb",
        "n_slices",
        "t_max",
        "n_points",
        "predictor",
        "proj_hidden",
        "pred_proj_hidden",
    }
    assert hp["proj_hidden"] == 64 and hp["pred_proj_hidden"] == 64
    assert hp["predictor"] == p.hparams
    m2 = LeWorldModel.from_hparams(backbone(), hp)
    m2.load_state_dict(m.state_dict())
    m.eval()
    m2.eval()
    fr = frames(4)
    assert torch.allclose(m.encode(fr), m2.encode(fr), atol=1e-6)
    with pytest.raises(ValueError):
        LeWorldModel.from_hparams(backbone(), dict(hp, proj_hidden=None))


def test_identity_projector_roundtrip():
    """A world model over a frozen encoder has no projector: ``nn.Identity``
    reports ``proj_hidden`` 0 and rebuilds through ``from_hparams``."""
    m = model(projector=nn.Identity(), pred_proj=nn.Identity())
    hp = m.hparams
    assert hp["proj_hidden"] == 0 and hp["pred_proj_hidden"] == 0
    m2 = LeWorldModel.from_hparams(backbone(), hp)
    assert isinstance(m2.projector, nn.Identity)
    assert isinstance(m2.pred_proj, nn.Identity)
    m2.load_state_dict(m.state_dict())
    m.eval()
    m2.eval()
    fr = frames(4)
    z = m.encode(fr)
    assert torch.allclose(z, m2.encode(fr), atol=1e-6)
    assert torch.allclose(z, m.encode(fr, project=False), atol=1e-6)
    # a mixed model: Identity projector next to an MLP pred_proj
    mixed = model(projector=nn.Identity())
    hp = mixed.hparams
    assert hp["proj_hidden"] == 0 and hp["pred_proj_hidden"] == 64
    m3 = LeWorldModel.from_hparams(backbone(), hp)
    m3.load_state_dict(mixed.state_dict())
    with pytest.raises(ValueError, match="equal widths"):
        LeWorldModel.from_hparams(backbone(), dict(hp, embed_dim=D + 1))
