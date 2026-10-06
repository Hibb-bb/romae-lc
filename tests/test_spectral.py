"""The spectral layer: Lomb-Scargle as its special case, shapes and
gradients, and the encoder round trip through the backbone."""

from __future__ import annotations

import numpy as np
import torch

from romae_lc import RoMAE, RoMAEForPreTraining
from romae_lc.spectral import SpectralLayer, spectral_aux_loss

ENC = dict(d_model=16, nhead=2, depth=2)


def test_spectrum_matches_lomb_scargle_peak():
    rng = np.random.default_rng(0)
    n, period = 60, 0.7
    t = np.sort(rng.uniform(0, 30, n))
    y = np.sin(2 * np.pi * t / period) + 0.05 * rng.standard_normal(n)
    layer = SpectralLayer(16, time_scale=1.0, p_min=0.2, p_max=5.0, rel=2e-3, channels=1, chunk=300)
    # at initialisation channel 0 reads the raw brightness (value channel 0): a Lomb-Scargle-like periodogram
    x = torch.zeros(1, n, 16)
    values = torch.zeros(1, n, 2)
    values[0, :, 0] = torch.tensor(y, dtype=torch.float32)
    real = torch.ones(1, n, dtype=torch.bool)
    spec = layer.spectrum(x, torch.tensor(t, dtype=torch.float32)[None], real, values)
    power = spec[0, 0].detach().numpy()
    f_peak = layer.freqs[int(power.argmax())].item()
    assert abs(1.0 / f_peak - period) < 0.01
    # against the generalised Lomb-Scargle of the probe (same statistic up to the sampling correction)
    from project.period_probe import gls_power

    ref = gls_power(t, y - y.mean(), np.ones(n), layer.freqs.numpy().astype(np.float64))
    assert abs(1.0 / layer.freqs[int(ref.argmax())].item() - period) < 0.01
    assert np.corrcoef(power, ref)[0, 1] > 0.95


def test_layer_shapes_grad_and_aux():
    torch.manual_seed(0)
    layer = SpectralLayer(16, time_scale=0.5, p_min=0.3, p_max=20.0, rel=5e-3, channels=4, reader=16, depth=2, chunk=200, after_layer=1)
    b, n = 3, 12
    x = torch.randn(b, 1 + n, 16, requires_grad=True)
    pos = torch.rand(b, 2, 1 + n) * 40 + 1
    pos[:, :, 0] = 0.0
    pad = torch.zeros(b, 1 + n, dtype=torch.bool)
    pad[0, 9:] = True
    out = layer(x, pos, pad, has_cls=True, values=torch.randn(b, n, 2))
    assert out.shape == x.shape and torch.allclose(out[:, 1:], x[:, 1:])  # only the CLS changes
    assert layer.last_logits.shape == (b, layer.n_bins)
    loss = spectral_aux_loss(layer, torch.tensor([0.9, 3.0, float("nan")]))
    (out.sum() + loss).backward()
    assert torch.isfinite(loss) and x.grad is not None and torch.isfinite(x.grad).all()
    assert SpectralLayer(16, **layer.hparams).n_bins == layer.n_bins


def test_mae_with_spectral_layer_and_backbone_round_trip():
    torch.manual_seed(0)
    spec = dict(time_scale=0.5, p_min=0.3, p_max=20.0, rel=5e-3, channels=4, reader=16, depth=1, chunk=200, after_layer=1)
    model = RoMAEForPreTraining(decoder=dict(d_model=16, nhead=2, depth=1), encoder=ENC, n_channels=2, target_channels=1, spectral=spec)
    assert model.hparams["spectral"]["rel"] == 5e-3
    b, n = 2, 10
    values, positions = torch.randn(b, n, 2), torch.rand(b, 2, n) * 40 + 1
    pad = torch.zeros(b, n, dtype=torch.bool)
    out = model(values, positions, pad)
    assert torch.isfinite(out.loss) and model.spectral.last_logits.shape[0] == b
    out.loss.backward()
    assert model.spectral.proj.weight.grad is not None
    bb = model.backbone()
    assert isinstance(bb, RoMAE) and bb.spectral is not None
    x, _ = model.encode(values, positions, pad)
    assert torch.allclose(bb(values, positions, pad), x[:, 0], atol=1e-5)
    rebuilt = RoMAE(**bb.hparams)
    rebuilt.load_state_dict(bb.state_dict())
    assert torch.allclose(rebuilt(values, positions, pad), x[:, 0], atol=1e-5)
