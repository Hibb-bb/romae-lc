"""The encoder's size presets and the rotary time ladder.

``--size`` picks the width, heads and depth (:data:`PRESETS`). The time
ladder (:class:`Ladder`) is the set of timescales that the rotary position
encoding of every head uses. It is measured on the training curves
(:func:`get_ladder`), or given, and it is saved in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch.nn as nn

from romae_lc import resample_ladder, suggest_time_encoding, time_series, timescales_for
from romae_lc import layout as rope_layout


# ------------------------------------------------------------------ encoder size


PRESETS = {
    "light": dict(width=192, heads=3, depth=6),  # 378 rungs
    "wide": dict(width=384, heads=6, depth=6),  # 756 rungs
    "xwide": dict(width=512, heads=8, depth=6),  # 1344 rungs
    "xxwide": dict(width=768, heads=12, depth=6),  # 2016 rungs
}


def add_model_args(parser) -> None:
    g = parser.add_argument_group("model (the encoder; --size light by default)")
    g.add_argument(
        "--size",
        choices=tuple(PRESETS),
        default="light",
        help="encoder preset: light = 192 wide, 3 heads, 6 deep (378 rungs in a "
        "dense ladder); wide = 384 wide, 6 heads, 6 deep (756 rungs); xwide = 512 wide, 8 heads (1344 rungs); xxwide = 768 wide, 12 heads (2016 rungs); --width, "
        "--heads and --depth override it",
    )
    g.add_argument("--width", type=int, default=None, help="encoder width")
    g.add_argument("--depth", type=int, default=None, help="encoder blocks")
    g.add_argument("--heads", type=int, default=None)
    g.add_argument("--mlp-ratio", type=float, default=4.0, help="encoder MLP width")
    g.add_argument("--p-rope", type=float, default=0.75)
    g.add_argument(
        "--time-frac",
        type=float,
        default=0.875,
        help="fraction of every head's channels rotated by time, the rest by "
        "wavelength (ZTF has two bands, so 8 of 64 channels suffice for them)",
    )
    g.add_argument("--attention", choices=("softmax", "linear"), default="softmax")
    g.add_argument(
        "--no-err-channel",
        action="store_true",
        help="single-channel encoder (ablation)",
    )


def resolve_model_args(args) -> argparse.Namespace:
    """Fill ``--width --heads --depth`` from ``--size`` where not given."""
    for k, v in PRESETS[args.size].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def encoder_config(args) -> dict:
    return dict(
        d_model=args.width,
        nhead=args.heads,
        depth=args.depth,
        attention=args.attention,
        mlp_ratio=getattr(args, "mlp_ratio", 4.0),
    )


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ------------------------------------------------------------------ the rotary time ladder


@dataclass
class Ladder:
    """A rotary time ladder: ``time_scale`` for the tokenizer, ``timescales``
    (position units) for the time block of every head, ``wavelengths`` the
    same in days for reading.

    A *shared* ladder is a flat list (``layers == heads == 1``). A *dense*
    ladder is nested: ``[heads][n]`` when the heads of every layer tile it
    (``layers == 1``) or ``[layers][heads][n]`` when the layers do too, and
    ``deal`` says how the rungs were dealt (:func:`deal_ladder`). Old
    ``ladder.json`` files load as shared ladders.
    """

    time_scale: float
    timescales: list
    wavelengths: list
    lam_min: float
    lam_max: float
    spacing: str
    summary: str = ""
    layers: int = 1
    heads: int = 1
    deal: str = "shared"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Ladder":
        return cls(**d)

    @property
    def flat(self) -> list[float]:
        """Every distinct wavelength (days), ascending."""
        return [float(v) for v in np.unique(np.asarray(self.wavelengths, float))]

    @property
    def n_rungs(self) -> int:
        return len(self.flat)

    @property
    def per_head(self) -> int:
        """Active time angles per head."""
        return int(np.asarray(self.timescales, float).shape[-1])


@dataclass
class RopeGeometry:
    """How many rotary time angles a backbone has: ``layers`` x ``heads``
    heads of ``n_angles`` active angles each in a time block of ``time_dim``
    of the ``head_dim`` channels."""

    layers: int
    heads: int
    head_dim: int
    time_dim: int
    n_angles: int

    @property
    def n_rungs(self) -> int:
        """Distinct timescales a dense ladder gives this backbone."""
        return self.layers * self.heads * self.n_angles


def _n_active(dim: int, p: float) -> int:
    return int(p * dim // 2)


def rope_geometry(args) -> RopeGeometry:
    """The rotary geometry of the backbone that ``rope_layouts(args, ...)``
    builds (``--width --heads --depth --time-frac --p-rope``)."""
    head_dim = args.width // args.heads
    blocks = rope_layout(head_dim, 2, time_frac=args.time_frac, p=args.p_rope)
    time_dim = blocks[0]["dim"]
    return RopeGeometry(
        args.depth, args.heads, head_dim, time_dim, _n_active(time_dim, args.p_rope)
    )


DEALS = ("bands", "comb")


def deal_ladder(wavelengths, layers: int, heads: int, n: int, deal: str = "bands"):
    """Deal ``layers * heads * n`` ascending wavelengths to the heads of every
    layer, ``n`` each, every rung to exactly one head, as a nested
    ``[layers][heads][n]`` list.

    ``"bands"``: rung ``i`` goes to layer ``i mod layers``, so every layer
    holds every ``layers``-th rung of the grid and the layers are offset by
    one grid step; within a layer the heads take contiguous bands of that
    sub-grid, so each head is a narrow-band filter over its own range of
    timescales and the heads of one layer cover the whole band between them.
    ``"comb"``: the same across layers, but within a layer the rungs are
    dealt round-robin over the heads, so every head spans the whole band at
    ``layers * heads`` times the grid spacing.
    """
    w = np.sort(np.asarray(wavelengths, dtype=np.float64).ravel())
    m = layers * heads * n
    if w.size != m:
        raise ValueError(f"{w.size} wavelengths for {layers} x {heads} x {n} slots")
    if deal not in DEALS:
        raise ValueError(f"deal must be one of {DEALS}, got {deal!r}")
    i = np.arange(m)
    layer, j = i % layers, i // layers
    if deal == "bands":
        head, angle = j // n, j % n
    else:
        head, angle = j % heads, j // heads
    out = np.empty((layers, heads, n), dtype=np.float64)
    out[layer, head, angle] = w
    return out.tolist()


def _nest(rows: np.ndarray, layers: int, heads: int):
    """Drop the axes of size 1 of a ``[layers][heads][n]`` array in the way
    :class:`Ladder` stores it: ``[n]``, ``[heads][n]`` or all three."""
    arr = np.asarray(rows, dtype=np.float64)
    if layers == 1:
        arr = arr[0]
        if heads == 1:
            arr = arr[0]
    return arr.tolist()


def add_ladder_args(parser) -> None:
    g = parser.add_argument_group(
        "time ladder (measured on the training curves, or explicit)"
    )
    g.add_argument(
        "--time-spacing", choices=("log", "linear", "quantile"), default="quantile"
    )
    g.add_argument("--time-mix", type=float, default=0.75, help="quantile-vs-log blend")
    g.add_argument("--max-freq", type=float, default=50.0, help="periodogram cap, 1/d")
    g.add_argument("--ladder-curves", type=int, default=128, help="curves sampled")
    g.add_argument("--lam-min", type=float, default=None, help="override, days")
    g.add_argument(
        "--lam-max", type=float, default=None, help="override, days (2 x window)"
    )
    g.add_argument("--ladder", default=None, help="reuse this ladder.json")
    g.add_argument(
        "--ladder-mode",
        choices=("dense", "shared"),
        default="dense",
        help="dense: one distinct rung per active time angle of every head of "
        "every layer, so the encoder resolves layers x heads x angles "
        "timescales (the folding resolution a period needs is about 1 / (4 x "
        "cycles per window)); shared: the same ladder in every head and layer",
    )
    g.add_argument(
        "--ladder-deal",
        choices=DEALS,
        default="bands",
        help="dense ladders: bands = each head a contiguous band of timescales, "
        "comb = each head spans the whole band (see deal_ladder)",
    )
    g.add_argument(
        "--rope-wavelengths",
        type=float,
        nargs="+",
        default=None,
        help="explicit ladder in days (e.g. the PC_matches census: 0.01 0.053 "
        "0.107 0.218 0.428 0.835 1.64 3.43 6.98 14.3 34.9 6000); needs --time-scale",
    )
    g.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help="days per position unit for --rope-wavelengths (census: 0.0015915)",
    )


def measure_wavelengths(
    records,
    n: int,
    window,
    spacing="quantile",
    mix=0.75,
    max_freq=50.0,
    max_curves=128,
    seed=0,
    lam_min=None,
    lam_max=None,
):
    """``n`` ascending wavelengths (days) measured on the periodograms of
    ``records`` by :func:`~romae_lc.suggest_time_encoding` (``lam_max``
    defaults to twice the window), with the report."""
    times, values, bands = time_series(records)
    report = suggest_time_encoding(
        times,
        values,
        2 * n,  # a time block of 2 n channels at p = 1 has n angles
        p=1.0,
        bands=bands,
        errors=[r.err for r in records],
        spacing=spacing,
        mix=mix,
        max_freq=max_freq,
        max_curves=max_curves,
        seed=seed,
        lam_min=lam_min,
        lam_max=2.0 * window if lam_max is None else lam_max,
    )
    w = np.unique(np.asarray(report.wavelengths, dtype=np.float64))
    if w.size < n:  # rungs that piled up on the same timescale
        w = np.asarray(resample_ladder(w, n), dtype=np.float64)
    return w, report


def measure_ladder(records, n: int, window, *measure_args) -> Ladder:
    """A shared ladder of ``n`` measured wavelengths (see
    :func:`measure_wavelengths` for the remaining arguments)."""
    w, report = measure_wavelengths(records, n, window, *measure_args)
    return Ladder(
        float(report.time_scale),
        timescales_for(w, float(report.time_scale)),
        [float(x) for x in w],
        float(report.lam_min),
        float(report.lam_max),
        report.spacing,
        str(report),
    )


def dense_ladder(records, geo: RopeGeometry, window, deal="bands", *measure_args):
    """A dense ladder: ``geo.n_rungs`` measured wavelengths dealt to the
    heads of every layer (:func:`deal_ladder`), so no two heads share a
    timescale."""
    w, report = measure_wavelengths(records, geo.n_rungs, window, *measure_args)
    days = deal_ladder(w, geo.layers, geo.heads, geo.n_angles, deal)
    ts = np.asarray(days, dtype=np.float64) / (2 * np.pi * report.time_scale)
    return Ladder(
        float(report.time_scale),
        _nest(ts, geo.layers, geo.heads),
        _nest(days, geo.layers, geo.heads),
        float(report.lam_min),
        float(report.lam_max),
        report.spacing,
        f"dense ladder of {geo.n_rungs} wavelengths ({geo.layers} layers x "
        f"{geo.heads} heads x {geo.n_angles} angles, {deal}) from {report}",
        geo.layers,
        geo.heads,
        deal,
    )


def check_ladder(ladder: Ladder, geo: RopeGeometry) -> None:
    """Raise unless ``ladder`` fits the backbone geometry ``geo``."""
    if ladder.layers not in (1, geo.layers):
        raise ValueError(f"ladder has {ladder.layers} layers, the model {geo.layers}")
    if ladder.heads not in (1, geo.heads):
        raise ValueError(f"ladder has {ladder.heads} heads, the model {geo.heads}")
    if ladder.per_head > geo.time_dim // 2:
        raise ValueError(
            f"{ladder.per_head} timescales per head do not fit the time block "
            f"({geo.time_dim} channels, {geo.time_dim // 2} angles)"
        )


def get_ladder(args, records, out_dir: Path, geo: RopeGeometry) -> Ladder:
    """``--rope-wavelengths`` with ``--time-scale`` (an explicit shared
    ladder), else ``--ladder`` or ``out_dir/ladder.json`` when present, else
    measured on ``records`` (dense over the heads and layers of ``geo`` with
    ``--ladder-mode dense``, shared otherwise); always written to
    ``out_dir/ladder.json``."""
    out_dir = Path(out_dir)
    path = Path(args.ladder) if args.ladder else out_dir / "ladder.json"
    measure_args = (
        args.time_spacing,
        args.time_mix,
        args.max_freq,
        args.ladder_curves,
        args.seed,
        args.lam_min,
        args.lam_max,
    )
    if args.rope_wavelengths:
        if not args.time_scale:
            raise ValueError("--rope-wavelengths needs --time-scale")
        w = [float(x) for x in args.rope_wavelengths]
        if len(w) > geo.n_angles:
            raise ValueError(
                f"{len(w)} wavelengths but the time block has {geo.n_angles} angles"
            )
        ladder = Ladder(
            float(args.time_scale),
            timescales_for(np.array(w), float(args.time_scale)),
            w,
            min(w),
            max(w),
            "explicit",
            f"explicit ladder of {len(w)} wavelengths, time_scale {args.time_scale}",
        )
    elif path.is_file():
        ladder = Ladder.from_dict(json.load(open(path)))
        check_ladder(ladder, geo)
        return ladder
    elif args.ladder_mode == "dense":
        ladder = dense_ladder(
            records, geo, args.window, args.ladder_deal, *measure_args
        )
    else:
        ladder = measure_ladder(records, geo.n_angles, args.window, *measure_args)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(ladder.to_dict(), open(out_dir / "ladder.json", "w"), indent=1)
    return ladder


def rope_layouts(args, ladder: Ladder):
    """The rotary layout (shared) or per-layer layouts of the backbone: the
    time block of ``--time-frac`` of every head carries the ladder, flat or
    per head, and the wavelength block takes the rest."""
    check_ladder(ladder, rope_geometry(args))
    head_dim = args.width // args.heads

    def lay(ts):
        return rope_layout(
            head_dim, 2, time_frac=args.time_frac, p=args.p_rope, time_timescales=ts
        )

    if ladder.layers > 1:
        return [lay(ts) for ts in ladder.timescales]
    return lay(ladder.timescales)
