"""Transformer: config resolution, relativity through attention, padding."""

from __future__ import annotations

import pytest
import torch

from romae_lc.rope import BlockRope, layout
from romae_lc.transformer import (
    SIZES,
    Transformer,
    TransformerConfig,
    attention_mask,
    config,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def build(attention="softmax", kind="axial", p=0.75, n_axes=2):
    cfg = TransformerConfig(d_model=32, nhead=2, depth=2, attention=attention)
    blocks = layout(cfg.head_dim, n_axes, kind, base=100.0, theta=100.0, p=p)
    return Transformer(cfg).eval(), BlockRope(cfg.head_dim, cfg.nhead, blocks)


def batch(n_axes=2, seed=0):
    """Two rows of 8 tokens; the second row has 3 padding tokens."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 8, 32, generator=g)
    positions = torch.rand(2, n_axes, 8, generator=g) * 10 + 1
    pad = torch.zeros(2, 8, dtype=torch.bool)
    pad[1, 5:] = True
    return x, positions * (~pad)[:, None, :], pad


def test_config_resolution():
    assert config("tiny-shallow") == TransformerConfig(**SIZES["tiny-shallow"])
    cfg = config(dict(d_model=32, nhead=2), depth=1)
    assert (cfg.d_model, cfg.nhead, cfg.depth, cfg.head_dim) == (32, 2, 1, 16)
    assert config(cfg, attention="linear").attention == "linear"
    assert config(None) == TransformerConfig()
    with pytest.raises(ValueError):
        config("huge")
    assert TransformerConfig().head_dim % 2 == 0
    with pytest.raises(ValueError):
        TransformerConfig(d_model=30, nhead=4)
    with pytest.raises(ValueError):
        TransformerConfig(d_model=342, nhead=6)  # odd head_dim
    with pytest.raises(ValueError):
        TransformerConfig(attention="flash")


@pytest.mark.parametrize("attention", ["softmax", "linear"])
@pytest.mark.parametrize("kind", ["axial", "simplex"])
def test_global_position_shift_is_invisible(attention, kind):
    net, rope = build(attention, kind, n_axes=3)
    x, positions, pad = batch(n_axes=3)
    mask = attention_mask(pad)
    y = net(x, rope.prepare(positions), mask)
    shift = torch.tensor([3.7, -1.2, 0.4])[None, :, None]
    y_shift = net(x, rope.prepare(positions + shift), mask)
    assert y.shape == (2, 8, 32)
    assert torch.allclose(y[~pad], y_shift[~pad], atol=1e-4)


def test_axis_permutation_changes_the_output():
    net, rope = build()
    x, positions, pad = batch()
    y = net(x, rope.prepare(positions), attention_mask(pad))
    y_perm = net(x, rope.prepare(positions.flip(1)), attention_mask(pad))
    assert not torch.allclose(y[~pad], y_perm[~pad], atol=1e-3)


@pytest.mark.parametrize("kind", ["axial", "simplex"])
def test_p_rope_zero_is_position_blind(kind):
    net, rope = build(kind=kind, p=0.0, n_axes=3)
    x, positions, pad = batch(n_axes=3)
    y = net(x, rope.prepare(positions), attention_mask(pad))
    y_other = net(x, rope.prepare(batch(n_axes=3, seed=1)[1]), attention_mask(pad))
    assert torch.allclose(y, y_other, atol=1e-6)


@pytest.mark.parametrize("attention", ["softmax", "linear"])
def test_padded_row_matches_unpadded_run(attention):
    net, rope = build(attention)
    x, positions, pad = batch()
    y = net(x, rope.prepare(positions), attention_mask(pad))
    alone = net(x[1:, :5], rope.prepare(positions[1:, :, :5]))
    assert torch.allclose(y[1, :5], alone[0], atol=1e-5)


def test_attention_mask_shapes():
    assert attention_mask(None) is None
    pad = torch.tensor([[False, False, True]])
    mask = attention_mask(pad)
    assert mask.shape == (1, 1, 3, 3) and mask.dtype == torch.bool
    assert torch.equal(mask[0, 0], (~pad).expand(3, 3))
    assert attention_mask(pad, length=2).shape == (1, 1, 2, 3)


def test_pos_dropout_is_softmax_only():
    x, positions, _ = batch()
    for attention, active in (("softmax", True), ("linear", False)):
        cfg = TransformerConfig(32, 2, 1, attention=attention, pos_dropout=0.5)
        net = Transformer(cfg).train()
        rope = BlockRope(cfg.head_dim, cfg.nhead, layout(cfg.head_dim, 2))
        rot = rope.prepare(positions)
        assert torch.equal(net(x, rot), net(x, rot)) is not active


def test_linear_attention_rejects_query_dependent_masks():
    net, rope = build("linear")
    x, positions, _ = batch()
    causal = torch.ones(8, 8).tril().bool()[None, None].expand(2, 1, 8, 8)
    with pytest.raises(NotImplementedError):
        net(x, rope.prepare(positions), causal)


def test_per_layer_rotations():
    net, rope_a = build()
    _, rope_b = build(p=1.0)
    x, positions, pad = batch()
    mask = attention_mask(pad)
    ra, rb = rope_a.prepare(positions), rope_b.prepare(positions)
    y = net(x, [ra, rb], mask)
    h = net.layers[1](net.layers[0](x, ra, mask), rb, mask)
    assert torch.allclose(y, h, atol=1e-6)
    assert not torch.allclose(y, net(x, ra, mask), atol=1e-4)
    assert torch.allclose(net(x, (ra, ra), mask), net(x, ra, mask), atol=1e-6)
    with pytest.raises(ValueError, match="rotations"):
        net(x, [ra], mask)
