"""Rotary encodings: wavelength ladders, layouts, relativity."""

from __future__ import annotations

import math

import pytest
import torch

from romae_lc.rope import (
    AxialRope,
    BlockRope,
    SimplexRope,
    layout,
    min_axial_dim,
    min_simplex_scales,
    resample_ladder,
    simplex_directions,
)


def test_axial_wavelengths_follow_the_geometric_ladder():
    dim, base, p = 16, 1e4, 0.75
    rope = AxialRope(dim, base, p)
    n_ang = int(p * dim // 2)
    expected = 2 * math.pi * base ** (2 * torch.arange(n_ang) / dim)
    assert rope.wavelengths.shape == (n_ang,)
    assert torch.allclose(rope.wavelengths, expected.float())
    assert torch.isinf(rope.timescale[n_ang:]).all()
    assert AxialRope(dim, base, p=0.0).wavelengths.numel() == 0


def test_axial_angles_shape_and_validation():
    rope = AxialRope(8, p=0.5)
    assert rope.angles(torch.rand(2, 5)).shape == (2, 5, 1, 4)
    with pytest.raises(ValueError):
        AxialRope(7)
    with pytest.raises(ValueError):
        AxialRope(8, p=1.5)


def test_simplex_scales_and_wave_vectors():
    rope = SimplexRope(dim=12, n_axes=2, nhead=2, theta=100.0)
    mag = torch.tensor([1.0, 100.0**-0.5])
    assert torch.allclose(rope.mag, mag)
    assert torch.allclose(rope.wavelengths, 2 * math.pi / mag)
    assert rope.wave_vectors.shape == (2, 6, 2)
    norms = rope.wave_vectors.norm(dim=-1).reshape(2, 3, 2)
    assert torch.allclose(norms, mag.expand(2, 3, 2))
    # Unit simplex directions with pairwise cosine -1/n, rotated per head.
    freqs = rope.freqs
    assert torch.allclose(freqs.norm(dim=-1), torch.ones(2, 3))
    off_diag = (freqs @ freqs.transpose(1, 2))[:, ~torch.eye(3, dtype=torch.bool)]
    assert torch.allclose(off_diag, torch.full_like(off_diag, -0.5), atol=1e-6)
    assert not torch.allclose(freqs[0], freqs[1])
    assert rope.angles(torch.rand(3, 2, 7)).shape == (3, 7, 2, 6)


def test_simplex_directions_are_pinned_and_backend_independent():
    # The Gram matrix (-1/n off-diagonal) is satisfied by every orientation of
    # the simplex, so pin the coordinates themselves: the closed-form Helmert
    # construction must not drift with the LAPACK build or the default dtype.
    s3, s6, s18 = math.sqrt(3), math.sqrt(6), math.sqrt(18)
    expected = {
        1: [[1.0]],
        2: [[s3 / 2, 0.5], [-s3 / 2, 0.5], [0.0, -1.0]],
        3: [
            [2 / s6, 2 / s18, 1 / 3],
            [-2 / s6, 2 / s18, 1 / 3],
            [0.0, -4 / s18, 1 / 3],
            [0.0, 0.0, -1.0],
        ],
    }
    for n, coords in expected.items():
        got = simplex_directions(n)
        assert got.dtype == torch.float32
        assert torch.allclose(got, torch.tensor(coords), atol=1e-6)
    for n in range(2, 7):
        d = simplex_directions(n)
        assert torch.allclose(d.norm(dim=1), torch.ones(n + 1))
        gram = (d @ d.T)[~torch.eye(n + 1, dtype=torch.bool)]
        assert torch.allclose(gram, torch.full_like(gram, -1 / n), atol=1e-6)
    torch.set_default_dtype(torch.float64)
    try:
        under_f64 = simplex_directions(4)
    finally:
        torch.set_default_dtype(torch.float32)
    assert torch.equal(under_f64, simplex_directions(4))


def test_simplex_p_rope_and_validation():
    rope = SimplexRope(dim=12, n_axes=2, nhead=1, p=0.5, rotate=False)
    assert (rope.mag[1:] == 0).all()
    assert rope.wave_vectors.shape == (1, 3, 2)
    assert rope.wavelengths.shape == (1,)
    assert (SimplexRope(dim=6, n_axes=2, nhead=1, p=0.0).mag == 0).all()
    with pytest.raises(ValueError):
        SimplexRope(dim=10, n_axes=2, nhead=1)
    for bad_p in (1.5, -0.5):
        with pytest.raises(ValueError, match="p must be in"):
            SimplexRope(dim=12, n_axes=2, nhead=1, p=bad_p)


def test_blocks_with_no_active_angle_are_rejected():
    # p=0 is a legal NoPE block; p>0 must leave at least one angle rotating.
    assert torch.isinf(AxialRope(2, p=0.0).timescale).all()
    with pytest.raises(ValueError, match="no active rotary angle"):
        AxialRope(2, p=0.75)
    with pytest.raises(ValueError, match="no active rotary scale"):
        SimplexRope(dim=6, n_axes=2, nhead=1, p=0.4)
    assert [min_axial_dim(p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)] == [2, 8, 4, 4, 2]
    assert [min_simplex_scales(p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)] == [
        1,
        3,
        2,
        1,
        1,
    ]
    for p in (0.25, 0.5, 0.75, 1.0):
        d = min_axial_dim(p)
        assert AxialRope(d, p=p).wavelengths.numel() == 1
        assert d == 2 or int(p * (d - 2) // 2) == 0
    # layout() fails with its own message instead of building dead blocks.
    with pytest.raises(ValueError, match="too small"):
        layout(8, 3, "axial")
    with pytest.raises(ValueError, match="too small"):
        layout(8, 3, "simplex")
    with pytest.raises(ValueError, match="too small"):
        layout(12, 3, "simplex", p=0.4)
    with pytest.raises(ValueError, match="too small"):
        layout(2, 1)
    assert [b["dim"] for b in layout(8, 3, "axial", p=1.0)] == [4, 2, 2]
    for head_dim, n_axes, kind in [(12, 3, "axial"), (16, 3, "simplex")]:
        rope = BlockRope(head_dim, 1, layout(head_dim, n_axes, kind))
        for block in rope.blocks:
            assert block.wavelengths.numel() >= 1


def test_layout_dims_sum_to_head_dim():
    assert layout(16, 1) == [dict(kind="axial", axes=[0], dim=16, base=1e4, p=0.75)]
    blocks = layout(16, 3, "axial", time_base=1e7, base=100.0)
    assert [b["dim"] for b in blocks] == [6, 6, 4]
    assert [b["axes"] for b in blocks] == [[0], [1], [2]]
    assert (blocks[0]["base"], blocks[1]["base"]) == (1e7, 100.0)
    blocks = layout(16, 3, "simplex", time_frac=0.5)
    assert [b["kind"] for b in blocks] == ["axial", "simplex"]
    assert [b["dim"] for b in blocks] == [10, 6]
    assert blocks[1]["axes"] == [1, 2]
    cases = [(30, 4, "simplex"), (24, 2, "simplex"), (20, 5, "axial")]
    for head_dim, n_axes, kind in cases:
        blocks = layout(head_dim, n_axes, kind)
        assert sum(b["dim"] for b in blocks) == head_dim
        assert BlockRope(head_dim, 2, blocks).n_axes == n_axes


def test_layout_and_blockrope_validation():
    with pytest.raises(ValueError):
        layout(15, 2)
    with pytest.raises(ValueError):
        layout(16, 2, "spiral")
    with pytest.raises(ValueError):
        layout(4, 3, "simplex")
    with pytest.raises(ValueError):
        BlockRope(16, 1, [dict(kind="axial", axes=[0, 1], dim=16)])
    with pytest.raises(ValueError):
        BlockRope(16, 1, [dict(kind="spiral", axes=[0], dim=16)])
    with pytest.raises(ValueError):
        BlockRope(16, 1, [dict(kind="axial", axes=[0], dim=8)])


@pytest.mark.parametrize("kind", ["axial", "simplex"])
def test_rotation_is_relative_and_norm_preserving(kind):
    rope = BlockRope(16, 2, layout(16, 3, kind, p=1.0))
    g = torch.Generator().manual_seed(0)
    q = torch.randn(4, 1, 2, 16, generator=g)
    k = torch.randn(4, 1, 2, 16, generator=g)
    pos_q = torch.rand(4, 3, 1, generator=g) * 5
    pos_k = torch.rand(4, 3, 1, generator=g) * 5
    rq = rope.prepare(pos_q)(q)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    lhs = (rq * rope.prepare(pos_k)(k)).sum(-1)
    rhs = (rope.prepare(pos_q - pos_k)(q) * k).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-4)
    assert torch.allclose(rope.prepare(torch.zeros(4, 3, 1))(q), q)


def test_nope_channels_are_not_rotated():
    rope = BlockRope(8, 1, [dict(kind="axial", axes=[0], dim=8, p=0.5)])
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 3, 1, 8, generator=g)
    rq = rope.prepare(torch.rand(2, 1, 3, generator=g) * 10)(q)
    nope, active = [2, 3, 6, 7], [0, 1, 4, 5]  # pairs (i, i + 4); angles 2, 3 idle
    assert torch.equal(rq[..., nope], q[..., nope])
    assert not torch.allclose(rq[..., active], q[..., active])


def test_axial_explicit_timescales_in_any_spacing():
    ladder = [0.5, 0.7, 4.0, 300.0]  # neither log- nor linearly spaced
    rope = AxialRope(12, timescales=ladder)
    assert rope.timescales == ladder and rope.p == 4 / 6
    assert torch.allclose(rope.timescale[:4], torch.tensor(ladder))
    assert torch.isinf(rope.timescale[4:]).all()
    assert torch.allclose(rope.wavelengths, 2 * math.pi * torch.tensor(ladder))
    assert rope.angles(torch.rand(2, 5)).shape == (2, 5, 1, 6)
    full = AxialRope(8, timescales=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert full.p == 1.0 and full.wavelengths.numel() == 4
    with pytest.raises(ValueError, match="do not fit"):
        AxialRope(4, timescales=[1.0, 2.0, 3.0])
    for bad in ([], [0.0, 1.0], [1.0, float("inf")], [-1.0]):
        with pytest.raises(ValueError):
            AxialRope(8, timescales=bad)


def test_simplex_explicit_scales():
    rope = SimplexRope(dim=18, n_axes=2, nhead=2, scales=[1.0, 0.3])
    assert rope.scales == [1.0, 0.3] and rope.p == 2 / 3
    assert torch.allclose(rope.mag, torch.tensor([1.0, 0.3, 0.0]))
    assert rope.wave_vectors.shape == (2, 6, 2)
    assert torch.allclose(rope.wavelengths, 2 * math.pi / torch.tensor([1.0, 0.3]))
    with pytest.raises(ValueError, match="do not fit"):
        SimplexRope(dim=12, n_axes=2, nhead=1, scales=[1.0, 0.5, 0.25])
    with pytest.raises(ValueError):
        SimplexRope(dim=12, n_axes=2, nhead=1, scales=[1.0, 0.0])


def test_resample_ladder_keeps_the_ends_in_log_space():
    assert resample_ladder([1.0, 10.0, 100.0], 5) == pytest.approx(
        [1.0, 10**0.5, 10.0, 10**1.5, 100.0]
    )
    assert resample_ladder([1.0, 100.0], 3) == pytest.approx([1.0, 10.0, 100.0])
    assert resample_ladder([2.0, 3.0, 50.0], 1) == [2.0]
    assert resample_ladder([7.0], 3) == [7.0, 7.0, 7.0]
    with pytest.raises(ValueError):
        resample_ladder([1.0, 2.0], 0)


def test_layout_and_blockrope_carry_explicit_ladders():
    blocks = layout(16, 2, time_timescales=[1.0, 3.0, 10.0])
    assert blocks[0]["timescales"] == [1.0, 3.0, 10.0] and blocks[0]["p"] == 0.75
    assert "timescales" not in blocks[1]
    rope = BlockRope(16, 1, blocks)
    assert rope.layout[0]["timescales"] == [1.0, 3.0, 10.0]
    assert torch.allclose(rope.blocks[0].timescale[:3], torch.tensor([1.0, 3.0, 10.0]))
    # numpy / tensor ladders are stored as plain floats so the layout serialises
    blocks[0]["timescales"] = torch.tensor([1.0, 3.0, 10.0])
    assert BlockRope(16, 1, blocks).layout[0]["timescales"] == [1.0, 3.0, 10.0]
    with pytest.raises(ValueError, match="do not fit"):
        layout(16, 2, time_timescales=list(range(1, 10)))
    simplex = layout(16, 3, "simplex", time_timescales=[2.0])
    assert simplex[0]["timescales"] == [2.0] and simplex[1]["kind"] == "simplex"
    one = layout(8, 1, time_timescales=[0.5, 5.0])
    assert one == [
        dict(kind="axial", axes=[0], dim=8, base=1e4, p=0.5, timescales=[0.5, 5.0])
    ]


def test_explicit_ladder_rotation_is_relative():
    blocks = layout(16, 2, time_timescales=[0.3, 1.0, 4.5, 20.0])
    rope = BlockRope(16, 2, blocks)
    g = torch.Generator().manual_seed(1)
    q = torch.randn(4, 1, 2, 16, generator=g)
    k = torch.randn(4, 1, 2, 16, generator=g)
    pos_q = torch.rand(4, 2, 1, generator=g) * 5
    pos_k = torch.rand(4, 2, 1, generator=g) * 5
    lhs = (rope.prepare(pos_q)(q) * rope.prepare(pos_k)(k)).sum(-1)
    rhs = (rope.prepare(pos_q - pos_k)(q) * k).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_per_head_ladders():
    from romae_lc.rope import ladder_rows

    rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    rope = AxialRope(8, timescales=rows, nhead=3)
    assert rope.per_head and rope.timescales == rows and rope.p == 0.5
    assert rope.timescale.shape == (3, 4) and torch.isinf(rope.timescale[:, 2:]).all()
    expected = 2 * math.pi * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert torch.allclose(rope.wavelengths, expected)
    ang = rope.angles(torch.rand(2, 5))
    assert ang.shape == (2, 5, 3, 4)
    assert torch.allclose(ang[:, :, 1, 0], ang[:, :, 0, 0] / 3)
    assert not AxialRope(8, timescales=rows[0]).per_head
    with pytest.raises(ValueError, match="rows"):
        AxialRope(8, timescales=rows, nhead=2)
    with pytest.raises(ValueError):
        AxialRope(8, timescales=[[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError, match="do not fit"):
        AxialRope(4, timescales=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert ladder_rows([1.0, 2.0]) == [[1.0, 2.0]]
    assert ladder_rows(torch.tensor(rows)) == rows
    with pytest.raises(ValueError):
        ladder_rows([[[1.0]]])
    with pytest.raises(ValueError):
        ladder_rows([[1.0, 0.0], [2.0, 3.0]])


def test_per_head_rotation_is_relative_and_the_layout_serialises():
    rows = [[0.3, 1.0], [4.5, 20.0]]
    blocks = layout(16, 2, time_timescales=rows)
    assert blocks[0]["timescales"] == rows and blocks[0]["p"] == 0.5
    rope = BlockRope(16, 2, blocks)
    assert rope.layout[0]["timescales"] == rows and rope.blocks[0].per_head
    g = torch.Generator().manual_seed(2)
    q = torch.randn(4, 1, 2, 16, generator=g)
    k = torch.randn(4, 1, 2, 16, generator=g)
    pos_q = torch.rand(4, 2, 1, generator=g) * 5
    pos_k = torch.rand(4, 2, 1, generator=g) * 5
    rq = rope.prepare(pos_q)(q)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    lhs = (rq * rope.prepare(pos_k)(k)).sum(-1)
    rhs = (rope.prepare(pos_q - pos_k)(q) * k).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-4)
    # head 0 rotates as with the shared ladder rows[0]; head 1 does not
    shared = BlockRope(16, 2, layout(16, 2, time_timescales=rows[0]))
    r_shared = shared.prepare(pos_q)(q)
    assert torch.allclose(r_shared[:, :, 0], rq[:, :, 0], atol=1e-6)
    assert not torch.allclose(r_shared[:, :, 1], rq[:, :, 1], atol=1e-3)
    with pytest.raises(ValueError, match="rows"):
        BlockRope(16, 3, blocks)


def test_collapse_layout():
    from romae_lc.rope import collapse_layout

    flat = layout(16, 2, time_timescales=[1.0, 3.0, 10.0])
    assert collapse_layout(flat) == flat
    per_head = layout(16, 2, time_timescales=[[1.0, 2.0], [3.0, 4.0]])
    c = collapse_layout(per_head)
    assert c[0]["timescales"] == pytest.approx([1.0, 4.0]) and c[0]["p"] == 0.5
    assert c[1] == per_head[1]
    per_layer = [
        layout(16, 2, time_timescales=[[1.0, 2.0], [3.0, 4.0]]),
        layout(16, 2, time_timescales=[[5.0, 6.0], [7.0, 8.0]]),
    ]
    c2 = collapse_layout(per_layer)
    assert c2[0]["timescales"] == pytest.approx([1.0, 8.0])
    kept = collapse_layout(layout(16, 2, time_timescales=[[1.0, 2.0], [1.0, 2.0]]))
    assert kept[0]["timescales"] == [1.0, 2.0]
    assert BlockRope(16, 5, c2).blocks[0].per_head is False
    with pytest.raises(ValueError):
        collapse_layout([])
    with pytest.raises(ValueError, match="same blocks"):
        collapse_layout([per_layer[0], layout(16, 1, time_timescales=[1.0])])
