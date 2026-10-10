"""Queries in cycles, the warm start, and the phasor features at the model's own period."""
import math

import torch

from project.common import warm_start
from project.templates import sample_queries_cycles
from romae_lc import RoMAEForPreTraining
from romae_lc.model import gen_mask


def _window(b=3, n=40, span=50.0, time_scale=10.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.sort(torch.rand(b, n, generator=g) * span, 1).values
    positions = torch.stack([t / time_scale + 1.0, torch.full((b, n), 0.6)], 1)
    pad = torch.zeros(b, n, dtype=torch.bool)
    pad[0, -5:] = True
    band = torch.zeros(b, n, dtype=torch.long)
    return positions, pad, band, t


def test_queries_in_cycles_stay_within_the_cap_and_the_window():
    positions, pad, band, t = _window()
    period = torch.tensor([0.3, 2.0, float("nan")])
    g = torch.Generator().manual_seed(1)
    t_q, b_q = sample_queries_cycles(positions, pad, band, 64, period, 0.1, 5.0, 10.0, random_share=0.25, generator=g)
    assert t_q.shape == (3, 64) and b_q.shape == (3, 64)
    t_max = (t * ~pad).max(1).values
    assert (t_q >= 0).all() and (t_q <= t_max[:, None] + 1e-6).all()
    # the drawn queries (after the random share) sit within cap x period of SOME real token (clipping only brings them closer)
    for i, p in enumerate((0.3, 2.0)):
        real_t = t[i][~pad[i]]
        d = (t_q[i, 16:, None] - real_t[None]).abs().min(1).values
        assert (d <= 5.0 * p + 1e-4).all()
    # the random share is spread over the window, not tied to a token
    assert t_q[:, :16].std() > 5.0
    # the log-uniform draw covers small offsets too
    real_t = t[1][~pad[1]]
    d = (t_q[1, 16:, None] - real_t[None]).abs().min(1).values
    assert (d < 0.2).any() and (d > 0.5).any()  # dense sampling: a seen point is never far, but the offsets do spread


def _spectral_model(phasor):
    return RoMAEForPreTraining(decoder=dict(d_model=16, nhead=2, depth=1), encoder=dict(d_model=16, nhead=2, depth=2),
                               n_channels=2, target_channels=1, phasor=phasor,
                               spectral=dict(time_scale=10.0, p_min=0.5, p_max=50.0, rel=0.05, channels=2, reader=8, depth=1, after_layer=1))


def test_phasor_features_run_and_train_the_period_estimate():
    torch.manual_seed(0)
    model = _spectral_model(4)
    positions, pad, band, t = _window()
    values = torch.randn(3, 40, 2)
    mask = gen_mask(0.5, pad, torch.Generator().manual_seed(2))
    out = model(values, positions, pad, mask)
    assert torch.isfinite(out.loss)
    assert model.phasor.last_freq.shape == (3,) and (model.phasor.last_freq > 0).all()
    # at step 0 the zero-initialised map leaves the decoder unchanged ...
    plain = _spectral_model(0)
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("phasor.")})
    assert torch.allclose(plain(values, positions, pad, mask).loss, out.loss, atol=1e-6)
    # ... and the loss reaches the map and, through it, the spectral reader (the period estimate)
    out.loss.backward()
    assert model.phasor.proj.weight.grad.abs().sum() > 0
    assert model.spectral.logit.weight.grad is not None and model.spectral.logit.weight.grad.abs().sum() > 0
    # the checkpoint carries the setting
    from dataclasses import asdict
    state = dict(state_dict=model.state_dict(), backbone=dict(model.hparams, encoder=asdict(model.cfg)),
                 mae=dict(decoder=asdict(model.dec_cfg), mask_ratio=model.mask_ratio, target_channels=model.target_channels, phasor=model.phasor.harmonics))
    rebuilt = RoMAEForPreTraining(**state["mae"], **state["backbone"])  # what load_mae does
    rebuilt.load_state_dict(state["state_dict"])
    assert rebuilt.phasor is not None and rebuilt.phasor.harmonics == 4
    assert torch.allclose(rebuilt(values, positions, pad, mask).loss, out.loss, atol=1e-6)


def test_phasor_needs_the_spectral_layer():
    import pytest
    with pytest.raises(ValueError):
        RoMAEForPreTraining(decoder=dict(d_model=16, nhead=2, depth=1), encoder=dict(d_model=16, nhead=2, depth=1), n_channels=2, phasor=2)


def test_warm_start_copies_matching_tensors_only():
    torch.manual_seed(0)
    src = _spectral_model(0)
    dst = _spectral_model(4)
    w = warm_start(dst, dict(state_dict=src.state_dict()))
    assert w["copied"] == len(src.state_dict()) and all(k.startswith("phasor.") for k in w["left_names"])
    assert torch.equal(dst.transformer.state_dict()["layers.0.attn.qkv.weight"] if "layers.0.attn.qkv.weight" in dst.transformer.state_dict() else next(iter(dst.transformer.parameters())),
                       src.transformer.state_dict()["layers.0.attn.qkv.weight"] if "layers.0.attn.qkv.weight" in src.transformer.state_dict() else next(iter(src.transformer.parameters())))
