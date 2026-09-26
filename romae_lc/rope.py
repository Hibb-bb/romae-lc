"""Continuous rotary position encodings for irregular, multi-axis positions.

Positions are real-valued coordinates ``[B, n_axes, N]`` (for light curves:
time, then one or more wavelength coordinates). The head dimension is cut into
contiguous channel blocks and each block is rotated by its own subset of axes
with its own scheme:

* :class:`AxialRope` - standard continuous p-RoPE on **one** axis (RoMAE,
  Zivanovic et al. 2025). A fraction ``p`` of the rotation angles are active,
  the rest are NoPE channels (infinite timescale, angle 0). An explicit
  ladder may differ per head (a nested ``[nhead][n]`` list), so that the
  heads of one layer resolve different timescales.
* :class:`SimplexRope` - nD-RoPE (Li et al. 2026, arXiv:2606.12146) over a
  **group** of axes: every angle is the inner product between the whole
  position vector and a wave vector drawn from the centroid-to-vertex
  directions of a regular simplex, replicated over geometric scales.
* :class:`BlockRope` - the composer that owns the layout. ``rotate`` is
  applied to queries and keys inside every attention layer.

Tables are prepared once per forward with :meth:`BlockRope.prepare` and the
resulting :class:`Rotation` is passed down the transformer, so there is no
mutable cache to reset between forwards. A model may hold one
:class:`BlockRope` per layer (a per-layer list of layouts, see
:class:`~romae_lc.model.RoMAEBase`); :func:`collapse_layout` folds such
per-head, per-layer ladders back into one shared layout for a decoder.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tables = tuple[torch.Tensor, torch.Tensor]
Ladder = "torch.Tensor | np.ndarray | list[float] | tuple[float, ...]"


def rotate(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rotate pairs ``(x[..., i], x[..., i + d/2])`` by the given angles."""
    first, second = torch.tensor_split(x, 2, dim=-1)
    out = torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)
    return out.to(x.dtype)


def as_ladder(values, name: str = "timescales") -> list[float]:
    """Validate an explicit ladder: a flat list of finite, positive floats."""
    ts = np.asarray(values, dtype=np.float64).flatten()
    if ts.size < 1:
        raise ValueError(f"{name} must hold at least one value")
    if not (np.isfinite(ts).all() and (ts > 0).all()):
        raise ValueError(f"{name} must be finite and positive, got {values}")
    return [float(v) for v in ts]


def ladder_rows(values, name: str = "timescales") -> list[list[float]]:
    """Validate an explicit ladder that is either flat (one ladder shared by
    every head) or per-head (a nested list with one row per head, all of the
    same length) and return its rows, a single row for a flat ladder. Values
    must be finite and positive."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} rows must all have the same length") from e
    if arr.ndim == 1:
        arr = arr[None]
    elif arr.ndim != 2:
        raise ValueError(
            f"{name} must be a flat or per-head (2-D) ladder, got shape {arr.shape}"
        )
    if arr.shape[1] < 1:
        raise ValueError(f"{name} must hold at least one value")
    if not (np.isfinite(arr).all() and (arr > 0).all()):
        raise ValueError(f"{name} must be finite and positive, got {values}")
    return [[float(v) for v in row] for row in arr]


def resample_ladder(values, n: int) -> list[float]:
    """``n`` values along the log-interpolated ladder ``values`` (its ends
    kept), for carrying a ladder to a block with a different number of
    active angles (the MAE decoder). ``n = 1`` keeps the first value."""
    ts = np.asarray(values, dtype=np.float64).flatten()
    if n < 1:
        raise ValueError(f"need n >= 1, got {n}")
    if n == 1 or ts.size == 1:
        return [float(ts[0])] * n
    x = np.linspace(0.0, 1.0, n)
    xp = np.linspace(0.0, 1.0, ts.size)
    return [float(v) for v in np.exp(np.interp(x, xp, np.log(ts)))]


class AxialRope(nn.Module):
    """Continuous p-RoPE on one position axis.

    The ``dim / 2`` rotation angles have timescales ``timescale[i]`` (in
    position units; the channel turns once per lag ``2 pi timescale[i]``).
    By default they form the geometric ladder ``base ** (2i / dim)`` of which
    a fraction ``p`` is active, the rest NoPE (infinite timescale). An
    explicit ``timescales`` ladder replaces it: any positive values in any
    spacing (log, linear, quantiles of a period distribution, ...), one per
    active angle, at most ``dim / 2`` of them; the remaining angles are NoPE
    and ``p`` becomes the active fraction. Build such ladders with
    :func:`romae_lc.analysis.rotary_ladder`. A nested ``[nhead][n]`` ladder
    gives every head its own timescales (``timescale`` is then ``[nhead, dim
    / 2]`` and the angles carry a head axis), so the heads of a layer can
    tile a band of timescales between them instead of sharing one ladder.

    Args:
        dim: Channels of this block (even).
        base: Base of the geometric timescale ladder ``base ** (2i / dim)``.
        p: Fraction of the ``dim / 2`` angles that rotate; the rest are NoPE.
        timescales: Explicit ladder in position units, flat or per head
            (overrides ``base`` and ``p``).
        nhead: With a per-head ladder, the number of heads it must match;
            ``None`` accepts any row count.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        p: float = 1.0,
        timescales: Ladder | None = None,
        nhead: int | None = None,
    ):
        super().__init__()
        if dim % 2:
            raise ValueError(f"AxialRope dim must be even, got {dim}")
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.dim, self.base = dim, float(base)
        if timescales is not None:
            rows = ladder_rows(timescales)
            n_rope = len(rows[0])
            if n_rope > dim // 2:
                raise ValueError(
                    f"{n_rope} timescales do not fit dim={dim} ({dim // 2} angles)"
                )
            if len(rows) > 1 and nhead is not None and len(rows) != nhead:
                raise ValueError(
                    f"per-head ladder has {len(rows)} rows for nhead={nhead}"
                )
            self.timescales = rows[0] if len(rows) == 1 else rows
            self.p = n_rope / (dim // 2)
            ts = torch.tensor(rows, dtype=torch.float32)  # [H, n_rope]
            if len(rows) == 1:
                ts = ts[0]
        else:
            self.timescales = None
            self.p = p
            n_rope = int(p * dim // 2)
            if p > 0 and n_rope < 1:
                raise ValueError(f"no active rotary angle for dim={dim}, p={p}")
            ts = base ** (2.0 * torch.arange(n_rope) / dim)
        timescale = F.pad(ts, (0, dim // 2 - n_rope), value=torch.inf)
        # [dim // 2] for a shared ladder, [nhead, dim // 2] for a per-head one
        self.register_buffer("timescale", timescale, persistent=False)

    @property
    def per_head(self) -> bool:
        """Whether every head has its own ladder."""
        return self.timescale.ndim == 2

    @property
    def wavelengths(self) -> torch.Tensor:
        """Position lag at which each active angle completes one turn (all
        heads' angles flattened, head-major, for a per-head ladder)."""
        ts = self.timescale
        return 2 * math.pi * ts[torch.isfinite(ts)]

    def angles(self, positions: torch.Tensor) -> torch.Tensor:
        """``[B, N]`` positions -> ``[B, N, 1, dim // 2]`` angles, or
        ``[B, N, nhead, dim // 2]`` with a per-head ladder."""
        if self.per_head:
            return positions[..., None, None] / self.timescale
        return (positions[..., None] / self.timescale)[:, :, None, :]


def simplex_directions(n: int) -> torch.Tensor:
    """Unit vectors from the centroid to the ``n + 1`` vertices of a regular
    simplex in ``R^n`` (``[[1.0]]`` for ``n == 1``).

    Vertex ``i`` is the ``i``-th coordinate of the Helmert basis of the
    sum-zero hyperplane of ``R^(n + 1)``: column ``k`` (1-based) is
    ``(1, ..., 1, -k, 0, ..., 0) / sqrt(k (k + 1))`` with ``k`` ones. This is
    pure arithmetic in float64, so the orientation of the simplex is the same
    on every LAPACK build and under any default dtype (an SVD of the centred
    identity has degenerate singular values and would leave it to the backend).
    """
    if n == 1:
        return torch.tensor([[1.0]])
    k = torch.arange(1, n + 1, dtype=torch.float64)  # (n,)
    i = torch.arange(n + 1, dtype=torch.float64)[:, None]  # (n + 1, 1)
    red = (i < k).double() - k * (i == k).double()  # (n + 1, n)
    red = red / torch.sqrt(k * (k + 1))
    return (red / red.norm(dim=1, keepdim=True)).float()


class SimplexRope(nn.Module):
    """nD-RoPE over a group of ``n_axes`` position axes.

    The block has ``dim = 2 * M * S`` channels: ``M = n_axes + 1`` simplex
    directions times ``S`` geometric scales ``theta ** (-s / S)``. Each head
    optionally gets its own random rotation of the simplex (``rotate=True``,
    as in the reference implementation), drawn once from ``seed``. ``p``
    mirrors p-RoPE: only the ``round(p * S)`` highest-frequency scales rotate.
    An explicit ``scales`` ladder (wave-vector magnitudes in inverse position
    units, any spacing, at most ``S`` of them; the rest are NoPE and ``p``
    becomes the active fraction) replaces the geometric one.

    Args:
        dim: Channels of this block (multiple of ``2 * (n_axes + 1)``).
        n_axes: Number of position axes in the group.
        nhead: Number of attention heads (per-head rotations).
        theta: Base of the scale ladder.
        p: Fraction of scales that rotate.
        rotate: Random per-head rotation of the simplex.
        seed: RNG seed for the rotations.
        scales: Explicit magnitudes (overrides ``theta`` and ``p``).
    """

    def __init__(
        self,
        dim: int,
        n_axes: int,
        nhead: int,
        theta: float = 100.0,
        p: float = 1.0,
        rotate: bool = True,
        seed: int = 0,
        scales: Ladder | None = None,
    ):
        super().__init__()
        m = n_axes + 1 if n_axes > 1 else 1
        if dim % (2 * m):
            raise ValueError(f"SimplexRope dim {dim} must be a multiple of {2 * m}")
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.dim, self.n_axes, self.nhead, self.m = dim, n_axes, nhead, m
        self.theta, self.p = float(theta), p
        self.n_scales = dim // (2 * m)
        if scales is not None:
            self.scales = as_ladder(scales, "scales")
            n_act = len(self.scales)
            if n_act > self.n_scales:
                raise ValueError(
                    f"{n_act} scales do not fit dim={dim} ({self.n_scales} scales)"
                )
            self.p = n_act / self.n_scales
            mag = torch.tensor(self.scales, dtype=torch.float32)
            mag = F.pad(mag, (0, self.n_scales - n_act), value=0.0)
        else:
            self.scales = None
            if p > 0 and int(round(p * self.n_scales)) < 1:
                raise ValueError(
                    f"no active rotary scale for dim={dim}, n_axes={n_axes}, p={p}"
                )
            s = torch.arange(self.n_scales, dtype=torch.float32)
            mag = theta ** (-s / max(self.n_scales, 1))
            mag[int(round(p * self.n_scales)) :] = 0.0
        base = simplex_directions(n_axes)  # (M, n)
        g = torch.Generator().manual_seed(seed)
        freqs = []
        for _ in range(nhead):
            if rotate and n_axes > 1:
                q, _ = torch.linalg.qr(torch.randn(n_axes, n_axes, generator=g))
                if torch.linalg.det(q) < 0:
                    q[:, 0] = -q[:, 0]
                freqs.append(base @ q.T)
            else:
                freqs.append(base)
        self.register_buffer("freqs", torch.stack(freqs), persistent=False)  # (H,M,n)
        self.register_buffer("mag", mag, persistent=False)  # (S,)

    @property
    def wave_vectors(self) -> torch.Tensor:
        """All wave vectors ``[nhead, M * S, n_axes]`` (active scales only)."""
        active = self.mag[self.mag > 0]
        return (self.freqs[:, :, None, :] * active[None, None, :, None]).reshape(
            self.nhead, -1, self.n_axes
        )

    @property
    def wavelengths(self) -> torch.Tensor:
        """Position lag along a wave vector for one full turn, per scale."""
        return 2 * math.pi / self.mag[self.mag > 0]

    def angles(self, positions: torch.Tensor) -> torch.Tensor:
        """``[B, n_axes, N]`` positions -> ``[B, N, nhead, dim // 2]`` angles."""
        b, _, n = positions.shape
        pos = positions.transpose(1, 2).float()  # (B, N, n)
        proj = torch.einsum("bnd,hmd->bnhm", pos, self.freqs)  # (B, N, H, M)
        return (proj[..., None] * self.mag).reshape(b, n, self.nhead, -1)


class Rotation:
    """Sin/cos tables for one batch of positions; rotates ``[B, N, H, D]``."""

    def __init__(self, tables: list[Tables], dims: list[int]):
        self.tables, self.dims = tables, dims

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outs, lo = [], 0
        for (sin, cos), dim in zip(self.tables, self.dims):
            outs.append(rotate(x[..., lo : lo + dim], sin, cos))
            lo += dim
        return torch.cat(outs, dim=-1)


class BlockRope(nn.Module):
    """Rotary encoding with an explicit head-dimension layout.

    ``layout`` is a list of dicts ``{"kind", "axes", "dim", ...}`` whose
    ``dim`` values sum to ``head_dim``; extra keys go to the block constructor
    (``base``/``p``/``timescales`` for ``"axial"``,
    ``theta``/``p``/``rotate``/``seed``/``scales`` for ``"simplex"``). Example
    for ``(time, l10, l50, l90)`` positions at ``head_dim=60``::

        [{"kind": "axial", "axes": [0], "dim": 36, "p": 0.75, "base": 1e7},
         {"kind": "simplex", "axes": [1, 2, 3], "dim": 24, "p": 0.75}]

    An explicit ladder frees the spacing of the time block::

        [{"kind": "axial", "axes": [0], "dim": 36, "timescales": [0.5, 1, 4, 30]},
         ...]

    Explicit ladders are kept as lists of floats in ``layout`` (nested per
    head when they are, see :class:`AxialRope`) so that the layout stays a
    plain, serialisable description of the encoding.

    Use :func:`layout` to build the standard light-curve layouts.
    """

    def __init__(self, head_dim: int, nhead: int, layout: list[dict]):
        super().__init__()
        self.head_dim, self.nhead = head_dim, nhead
        self.layout = [dict(b) for b in layout]
        for spec in self.layout:
            if spec.get("timescales") is not None:
                rows = ladder_rows(spec["timescales"])
                spec["timescales"] = rows[0] if len(rows) == 1 else rows
            if spec.get("scales") is not None:
                spec["scales"] = as_ladder(spec["scales"], "scales")
        self.axes, self.dims, blocks = [], [], []
        for spec in self.layout:
            spec = dict(spec)
            kind, axes, dim = spec.pop("kind"), list(spec.pop("axes")), spec.pop("dim")
            if kind == "axial":
                if len(axes) != 1:
                    raise ValueError("an axial block takes exactly one axis")
                blocks.append(AxialRope(dim, nhead=nhead, **spec))
            elif kind == "simplex":
                blocks.append(SimplexRope(dim, len(axes), nhead, **spec))
            else:
                raise ValueError(f"unknown rope block kind {kind!r}")
            self.axes.append(axes)
            self.dims.append(dim)
        if sum(self.dims) != head_dim:
            raise ValueError(f"block dims {self.dims} must sum to head_dim={head_dim}")
        self.blocks = nn.ModuleList(blocks)

    @property
    def n_axes(self) -> int:
        """Number of position axes the layout refers to."""
        return 1 + max(a for axes in self.axes for a in axes)

    def prepare(self, positions: torch.Tensor) -> Rotation:
        """Precompute the rotation for ``positions [B, n_axes, N]``."""
        tables = []
        for block, axes in zip(self.blocks, self.axes):
            pos = (
                positions[:, axes[0]]
                if isinstance(block, AxialRope)
                else positions[:, axes]
            )
            ang = block.angles(pos)
            tables.append((torch.sin(ang), torch.cos(ang)))
        return Rotation(tables, self.dims)


def min_axial_dim(p: float) -> int:
    """Smallest even block dim with at least one active angle at fraction ``p``
    (2 when ``p`` is 0 or 1; 4 at the default ``p = 0.75``)."""
    d = 2
    while p > 0 and int(p * d // 2) < 1:
        d += 2
    return d


def min_simplex_scales(p: float) -> int:
    """Smallest number of simplex scales with at least one active scale."""
    s = 1
    while p > 0 and int(round(p * s)) < 1:
        s += 1
    return s


def _even_split(total: int, k: int) -> list[int]:
    """Split ``total`` (even) into ``k`` even parts, as equal as possible."""
    pairs = total // 2
    base, extra = divmod(pairs, k)
    return [2 * (base + (1 if i < extra else 0)) for i in range(k)]


def layout(
    head_dim: int,
    n_axes: int,
    kind: str = "axial",
    time_frac: float | None = None,
    time_base: float = 10000.0,
    base: float = 10000.0,
    theta: float = 100.0,
    p: float = 0.75,
    seed: int = 0,
    time_timescales: Ladder | None = None,
) -> list[dict]:
    """Standard rotary layouts for ``(time, *wavelength)`` positions.

    Axis 0 (time) always gets its own axial block with base ``time_base``, or
    with the explicit ladder ``time_timescales`` (position units, any
    spacing; it must fit the time block, i.e. hold at most ``time_dim / 2``
    values, and it fixes the block's active fraction instead of ``p``; a
    nested ``[nhead][n]`` ladder gives every head its own, see
    :class:`AxialRope`).
    The remaining axes share the rest of the head either as equal axial
    slices (``kind="axial"``, base ``base``) or as one nD-RoPE simplex block
    (``kind="simplex"``, base ``theta``).

    Args:
        head_dim: Channels per head.
        n_axes: Number of position axes (1 = time only).
        kind: ``"axial"`` or ``"simplex"`` for the non-time axes.
        time_frac: Fraction of the head rotated by time. ``None`` splits the
            head equally over all axes.
        time_base: Rotary base of the time axis.
        base: Rotary base of the axial wavelength blocks.
        theta: Scale base of the simplex block.
        p: p-RoPE fraction for every block.
        seed: Seed of the simplex per-head rotations.
        time_timescales: Explicit time ladder, see above.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    d_min = min_axial_dim(p)

    def time_block(dim: int) -> dict:
        block = dict(kind="axial", axes=[0], dim=dim, base=time_base, p=p)
        if time_timescales is not None:
            rows = ladder_rows(time_timescales)
            if len(rows[0]) > dim // 2:
                raise ValueError(
                    f"{len(rows[0])} time timescales do not fit the time block "
                    f"(dim {dim}, {dim // 2} angles)"
                )
            block["timescales"] = rows[0] if len(rows) == 1 else rows
            block["p"] = len(rows[0]) / (dim // 2)
        return block

    if n_axes == 1:
        if head_dim < d_min:
            raise ValueError(f"head_dim {head_dim} too small at p={p}")
        return [time_block(head_dim)]
    n_wave = n_axes - 1
    if time_frac is None:
        time_dim = _even_split(head_dim, n_axes)[0]
    else:
        time_dim = 2 * (int(round(head_dim * time_frac)) // 2)
    wave_dim = head_dim - time_dim
    if kind == "simplex":
        unit = 2 * (n_wave + 1) if n_wave > 1 else 2
        wave_dim = max(unit * min_simplex_scales(p), (wave_dim // unit) * unit)
    else:
        wave_dim = max(d_min * n_wave, wave_dim)
    time_dim = head_dim - wave_dim
    if time_dim < d_min:
        raise ValueError(
            f"head_dim {head_dim} too small for {n_wave} wavelength axes at p={p}"
        )
    blocks = [time_block(time_dim)]
    if kind == "simplex":
        wave_axes = list(range(1, n_axes))
        blocks.append(
            dict(
                kind="simplex",
                axes=wave_axes,
                dim=wave_dim,
                theta=theta,
                p=p,
                seed=seed,
            )
        )
    elif kind == "axial":
        for i, d in enumerate(_even_split(wave_dim, n_wave)):
            blocks.append(dict(kind="axial", axes=[1 + i], dim=d, base=base, p=p))
    else:
        raise ValueError(f"kind must be 'axial' or 'simplex', got {kind!r}")
    return blocks


def collapse_layout(layouts) -> list[dict]:
    """One shared layout from a layout or a per-layer list of layouts whose
    axial blocks may carry per-head ladders.

    Every explicit ``timescales`` ladder becomes the sorted union of all its
    rows over heads and layers, resampled in log space
    (:func:`resample_ladder`) to the block's own number of angles when the
    union is larger; a flat ladder of a single layout is returned as it is.
    Blocks without an explicit ladder and every other key come from the
    first layout. This is what a decoder with its own head count takes from
    an encoder whose heads and layers tile a dense ladder between them.
    """
    if not layouts:
        raise ValueError("collapse_layout needs at least one layout")
    if isinstance(layouts[0], dict):
        layouts = [layouts]
    if any(len(lay) != len(layouts[0]) for lay in layouts):
        raise ValueError("per-layer layouts must have the same blocks")
    out = [dict(b) for b in layouts[0]]
    for i, spec in enumerate(out):
        if spec.get("timescales") is None:
            continue
        rows = []
        for lay in layouts:
            rows += ladder_rows(lay[i]["timescales"])
        n = len(rows[0])
        if len(rows) == 1:
            spec["timescales"] = rows[0]
            continue
        union = sorted({v for row in rows for v in row})
        spec["timescales"] = (
            resample_ladder(union, n) if len(union) > n else [float(v) for v in union]
        )
        spec["p"] = len(spec["timescales"]) / (spec["dim"] // 2)
    return out
