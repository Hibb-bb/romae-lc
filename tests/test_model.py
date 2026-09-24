"""RoMAE models: token masking, masked autoencoding, backbone extraction, heads."""

from __future__ import annotations

import math

import pytest
import torch

from romae_lc.model import RoMAE, RoMAEForClassification, RoMAEForPreTraining, gen_mask
from romae_lc.rope import layout
from romae_lc.transformer import TransformerConfig

ENCODER = dict(d_model=32, nhead=2, depth=2)
DECODER = dict(d_model=32, nhead=2, depth=1)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def pad(n_real):
    """Padding mask ``[B, max(n_real)]`` with the real tokens first."""
    mask = torch.zeros(len(n_real), max(n_real), dtype=torch.bool)
    for i, n in enumerate(n_real):
        mask[i, n:] = True
    return mask


def batch(n_axes=2, n_real=(12, 9, 5), seed=0):
    g = torch.Generator().manual_seed(seed)
    pad_mask = pad(n_real)
    values = torch.randn(*pad_mask.shape, 1, generator=g)
    positions = torch.rand(len(n_real), n_axes, max(n_real), generator=g) * 10 + 1
    return values, positions * (~pad_mask)[:, None, :], pad_mask


@pytest.mark.parametrize("ratio", [0.0, 0.5, 0.75, 1.0])
def test_gen_mask_counts(ratio):
    n_real = (20, 13, 7, 1)
    pad_mask = pad(n_real)
    mask = gen_mask(ratio, pad_mask)
    k = torch.tensor([math.ceil(n * ratio) for n in n_real])
    assert torch.equal((mask & ~pad_mask).sum(1), k)
    assert (mask.sum(1) == k.max()).all()
    trailing = torch.arange(20)[None] >= (20 - (k.max() - k))[:, None]
    assert torch.equal(mask & pad_mask, trailing)


def test_gen_mask_is_reproducible_with_a_generator():
    pad_mask = pad((10, 6))
    a = gen_mask(0.5, pad_mask, torch.Generator().manual_seed(3))
    b = gen_mask(0.5, pad_mask, torch.Generator().manual_seed(3))
    c = gen_mask(0.5, pad_mask, torch.Generator().manual_seed(4))
    assert torch.equal(a, b) and not torch.equal(a, c)
    with pytest.raises(ValueError):
        gen_mask(1.5, pad_mask)


def test_mae_loss_is_finite_and_trains_the_encoder():
    model = RoMAEForPreTraining(decoder=DECODER, encoder=ENCODER)
    values, positions, pad_mask = batch()
    out = model(values, positions, pad_mask)
    k = int(out.mask.sum(1)[0])
    assert torch.isfinite(out.loss)
    assert out.pred.shape == out.target.shape == (3, k, 1)
    assert (out.mask.sum(1) == k).all()
    out.loss.backward()
    assert model.projection.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.transformer.parameters())


@pytest.mark.parametrize("attention", ["softmax", "linear"])
def test_mae_loss_ignores_padded_values(attention):
    model = RoMAEForPreTraining(
        decoder=dict(DECODER, attention=attention),
        encoder=dict(ENCODER, attention=attention),
    ).eval()
    values, positions, pad_mask = batch()
    mask = gen_mask(0.5, pad_mask, torch.Generator().manual_seed(0))
    loss = model(values, positions, pad_mask, mask).loss
    corrupted = values.masked_fill(pad_mask[..., None], 100.0)
    assert torch.allclose(model(corrupted, positions, pad_mask, mask).loss, loss)


@pytest.mark.parametrize("rope", ["axial", "simplex"])
def test_backbone_reproduces_the_encoder(rope):
    model = RoMAEForPreTraining(decoder=DECODER, encoder=ENCODER, n_axes=3, rope=rope)
    values, positions, pad_mask = batch(n_axes=3)
    backbone = model.eval().backbone("mean").eval()
    assert isinstance(backbone, RoMAE) and backbone.hparams == model.hparams
    tokens, _ = model.encode(values, positions, pad_mask)
    tokens_bb, _ = backbone.encode(values, positions, pad_mask)
    assert torch.allclose(tokens, tokens_bb, atol=1e-6)
    assert backbone(values, positions, pad_mask).shape == (3, 32)


@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_classification_and_pooling_shapes(pool):
    values, positions, pad_mask = batch()
    clf = RoMAEForClassification(n_classes=5, pool=pool, encoder=ENCODER)
    assert clf(values, positions, pad_mask).shape == (3, 5)
    model = RoMAE(pool=pool, encoder=ENCODER).eval()
    z, tokens = model(values, positions, pad_mask, return_tokens=True)
    assert z.shape == (3, 32) and tokens.shape == (3, 12, 32)
    assert torch.equal(z, model(values, positions, pad_mask))


def test_mean_pooling_ignores_padding():
    model = RoMAE(pool="mean", encoder=ENCODER, use_cls=False).eval()
    values, positions, pad_mask = batch()
    z = model(values, positions, pad_mask)
    alone = model(values[2:, :5], positions[2:, :, :5])
    assert torch.allclose(z[2], alone[0], atol=1e-5)


@pytest.mark.parametrize("cls", [RoMAE, RoMAEForClassification])
def test_constructor_validation(cls):
    kwargs = dict(n_classes=3) if cls is RoMAEForClassification else {}
    with pytest.raises(ValueError):
        cls(pool="cls", encoder=ENCODER, use_cls=False, **kwargs)
    with pytest.raises(ValueError):
        cls(pool="max", encoder=ENCODER, **kwargs)
    with pytest.raises(ValueError):
        cls(encoder=ENCODER, n_axes=1, rope=layout(16, 2), **kwargs)


def test_default_config_builds_a_model():
    values, positions, pad_mask = batch()
    for encoder in ({}, dict(depth=1), TransformerConfig(depth=1)):
        model = RoMAE(encoder=encoder).eval()
        assert model(values, positions, pad_mask).shape == (3, 432)


def test_mae_rejects_an_empty_mask():
    with pytest.raises(ValueError):
        RoMAEForPreTraining(decoder=DECODER, encoder=ENCODER, mask_ratio=0.0)
    model = RoMAEForPreTraining(decoder=DECODER, encoder=ENCODER)
    values, positions, pad_mask = batch()
    with pytest.raises(ValueError):
        model(values, positions, pad_mask, torch.zeros_like(pad_mask))
    assert model.mask_token.abs().sum() > 0  # not the RMSNorm singularity


def test_explicit_time_ladder_reaches_the_decoder_and_the_checkpoint():
    ladder = [1.0, 7.0, 90.0, 40000.0]  # time block: 8 of 16 channels, 4 angles
    model = RoMAEForPreTraining(
        decoder=dict(d_model=16, nhead=2, depth=1),  # head_dim 8: 2 time angles
        encoder=ENCODER,
        rope_timescales=ladder,
    )
    enc, dec = model.rope.layout[0], model.decoder_rope.layout[0]
    assert enc["timescales"] == ladder and enc["p"] == 1.0
    assert dec["dim"] == 4 and dec["timescales"] == pytest.approx([1.0, 4e4])
    values, positions, pad_mask = batch()
    assert torch.isfinite(model(values, positions, pad_mask).loss)
    backbone = model.backbone()
    assert backbone.hparams == model.hparams
    rebuilt = RoMAE(**model.hparams)
    assert rebuilt.rope.layout[0]["timescales"] == ladder
    with pytest.raises(ValueError, match="string layouts"):
        RoMAE(encoder=ENCODER, rope=model.rope.layout, rope_timescales=ladder)
    with pytest.raises(ValueError, match="do not fit"):
        RoMAE(encoder=ENCODER, rope_timescales=list(range(1, 20)))
