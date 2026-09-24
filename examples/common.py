"""Shared helpers for the example scripts.

Argument groups (data, model, training), record loading, the backbone the
arguments describe (``--auto-time`` reads the rotary time band off the
training curves), token-batch loaders, embedding, a logistic linear probe, a
closed-form ridge probe on log period, and checkpoints of a backbone or of a
whole :class:`~romae_lc.LeWorldModel`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from romae_lc import (
    DEFAULT_SURVEYS,
    PHOEBE_BANDS,
    FrameConfig,
    LeWorldModel,
    LightCurveDataset,
    PC_BANDS,
    Record,
    RoMAE,
    TimeEncodingReport,
    ViewConfig,
    collate,
    load_pc,
    load_phoebe,
    normalize,
    simulate,
    suggest_time_encoding,
    time_series,
    timescales_for,
    wavelengths_for,
)


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """``--data``, ``--n``, ``--split-frac``, ``--seed``."""
    g = parser.add_argument_group("data")
    g.add_argument(
        "--data",
        default="sim",
        help="sim | PHOEBE HF repo id or save_to_disk dir | PC_matches sub-dataset "
        "dir (e.g. /projects/bfrf/data/PC_matches/ZTFxPC)",
    )
    g.add_argument("--n", type=int, default=2048, help="simulated stars")
    g.add_argument("--max-rows", type=int, help="cap rows per split (real data)")
    g.add_argument(
        "--label",
        choices=("superclass", "class"),
        default="superclass",
        help="PC_matches label field for the probe (8 superclasses or fine classes)",
    )
    g.add_argument(
        "--min-points", type=int, default=8, help="PC_matches: drop shorter records"
    )
    g.add_argument("--split-frac", type=float, default=0.2, help="simulated val split")
    g.add_argument("--seed", type=int, default=0)


def add_model_args(
    parser: argparse.ArgumentParser, depth_help: str = "encoder blocks"
) -> None:
    """Encoder size, rotary layout, attention, time unit and device;
    ``depth_help`` lets a script that overrides the ``--depth`` default say
    so."""
    g = parser.add_argument_group("model")
    g.add_argument("--width", type=int, default=192)
    g.add_argument("--depth", type=int, default=4, help=depth_help)
    g.add_argument("--heads", type=int, default=3)
    g.add_argument("--rope", choices=("axial", "simplex"), default="axial")
    g.add_argument("--attention", choices=("softmax", "linear"), default="softmax")
    g.add_argument("--p-rope", type=float, default=0.75)
    g.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="days per position; with the default base the shortest rotary "
        "wavelength is 2*pi*time_scale days (the language-model band), so pass "
        "--auto-time or explicit values for sub-day periods such as the simulator's",
    )
    g.add_argument(
        "--rope-base",
        type=float,
        default=1e4,
        help="time-axis base; the language-model default, see --time-scale",
    )
    g.add_argument(
        "--auto-time",
        action="store_true",
        help="time band from the data (overrides --time-scale and --rope-base)",
    )
    g.add_argument(
        "--auto-time-max-freq",
        type=float,
        default=50.0,
        help="--auto-time periodogram cap in cycles/day (periods down to 0.02 d); "
        "the uncapped grid on ZTF-like cadence takes an hour per report",
    )
    g.add_argument(
        "--auto-time-curves",
        type=int,
        default=128,
        help="--auto-time: curves sampled for the periodograms",
    )
    g.add_argument(
        "--time-spacing",
        choices=("log", "linear", "quantile"),
        default="log",
        help="spacing of the --auto-time ladder: log (geometric, what "
        "--rope-base encodes), linear, or quantile (channel density follows "
        "the timescales measured on the training curves)",
    )
    g.add_argument(
        "--time-mix",
        type=float,
        default=0.75,
        help="quantile-versus-log blend of a quantile ladder (1 = pure quantile)",
    )
    g.add_argument(
        "--rope-wavelengths",
        type=float,
        nargs="+",
        help="explicit time ladder in days (any spacing, at most one value per "
        "angle of the time block); replaces --rope-base",
    )
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def add_train_args(parser: argparse.ArgumentParser, out: str) -> None:
    """Optimisation schedule, loader and checkpoint arguments."""
    g = parser.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--wd", type=float, default=0.05)
    g.add_argument("--eval-every", type=int, default=5, help="epochs between probes")
    g.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    g.add_argument("--out", default=out, help="checkpoint path")


def device(args) -> torch.device:
    """The ``--device`` argument as a ``torch.device``."""
    return torch.device(args.device)


def cosine_schedule(total: int, warmup_frac: float = 0.05):
    """Learning-rate multiplier for ``LambdaLR``: linear warmup over
    ``warmup_frac`` of the ``total`` steps, then cosine decay to zero."""
    warmup = max(1, int(warmup_frac * total))

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return multiplier


def build_records(args) -> tuple[list[Record], list[Record], dict[int, float]]:
    """``(train, val, band_wavelengths)`` of per-band standardised records."""
    if args.data == "sim":
        records = [normalize(r) for r in simulate(args.n, seed=args.seed)]
        order = np.random.default_rng(args.seed).permutation(len(records))
        n_val = int(round(args.split_frac * len(records)))
        train = [records[i] for i in order[n_val:]]
        val = [records[i] for i in order[:n_val]]
        return train, val, {i: s.wavelength_nm for i, s in enumerate(DEFAULT_SURVEYS)}
    if is_pc_dataset(args.data):
        field = f"{args.label}_str"
        classes = None
        if field == "class_str":  # one vocabulary for both splits
            import datasets

            dd = datasets.load_from_disk(args.data)
            classes = tuple(
                sorted(set(dd["train"][field]) | set(dd["validation"][field]))
            )
        kw = dict(
            label_field=field,
            classes=classes,
            max_rows=getattr(args, "max_rows", None),
            min_points=getattr(args, "min_points", 1),
        )
        load = lambda split: [normalize(r) for r in load_pc(args.data, split, **kw)]
        return load("train"), load("validation"), wavelengths_for(PC_BANDS)
    load = lambda split: [
        normalize(r) for r in load_phoebe(args.data, split, max_rows=args.max_rows)
    ]
    return load("train"), load("validation"), wavelengths_for(PHOEBE_BANDS)


def is_pc_dataset(source: str) -> bool:
    """A ``save_to_disk`` directory whose train split has a ``lightcurve``
    struct column (the PC_matches layout)."""
    info = Path(source) / "train" / "dataset_info.json"
    if not info.is_file():
        return False
    import json

    return "lightcurve" in json.load(open(info)).get("features", {})


def model_kwargs(args, n_axes: int = 2) -> dict:
    """:class:`~romae_lc.model.RoMAEBase` keywords the arguments describe.

    An explicit time ladder comes from ``args.rope_timescales`` (position
    units, set by :func:`auto_time` with a non-log ``--time-spacing``) or
    from ``--rope-wavelengths`` (days, converted with ``--time-scale``).
    """
    encoder = dict(d_model=args.width, nhead=args.heads, depth=args.depth)
    rope = dict(rope=args.rope, rope_base=args.rope_base, p_rope=args.p_rope)
    timescales = getattr(args, "rope_timescales", None)
    if timescales is None and getattr(args, "rope_wavelengths", None):
        timescales = timescales_for(args.rope_wavelengths, args.time_scale)
    if timescales is not None:
        rope["rope_timescales"] = list(timescales)
    return dict(encoder=dict(encoder, attention=args.attention), n_axes=n_axes, **rope)


def build_backbone(args, n_axes: int = 2) -> RoMAE:
    return RoMAE(**model_kwargs(args, n_axes))


def time_report(
    args, records: list[Record], n_axes: int = 2, **kw
) -> TimeEncodingReport:
    """:func:`~romae_lc.analysis.suggest_time_encoding` for the time block of
    the backbone the arguments describe; ``**kw`` (e.g. ``lam_max``) is
    passed on."""
    dim = build_backbone(args, n_axes).rope.dims[0]
    times, values, bands = time_series(records)
    errors = [r.err for r in records]
    kw.setdefault("spacing", getattr(args, "time_spacing", "log"))
    kw.setdefault("mix", getattr(args, "time_mix", 1.0))
    kw.setdefault("max_freq", getattr(args, "auto_time_max_freq", None))
    kw.setdefault("max_curves", getattr(args, "auto_time_curves", 128))
    return suggest_time_encoding(
        times,
        values,
        dim,
        p=args.p_rope,
        bands=bands,
        errors=errors,
        seed=args.seed,
        **kw,
    )


def auto_time(args, train: list[Record], **kw) -> None:
    """With ``--auto-time``, print the report for the training curves and
    overwrite ``args.time_scale`` and ``args.rope_base`` with its advice (and
    ``args.rope_timescales`` with the ladder when ``--time-spacing`` is not
    ``log``); ``**kw`` goes to :func:`time_report`."""
    if args.auto_time:
        report = time_report(args, train, **kw)
        print(report)
        args.time_scale, args.rope_base = report.time_scale, report.base
        args.rope_timescales = None if report.spacing == "log" else report.timescales


def make_loader(
    records, tokenize_kwargs, batch_size, train=False, view_cfg=None, workers=0, seed=0
) -> DataLoader:
    """Padded :class:`~romae_lc.tokenize.Tokens` batches; a ``train`` loader
    shuffles, drops the last batch and re-draws the views every epoch."""
    ds = LightCurveDataset(records, view_cfg, seed=seed, epoch_seed=train)
    fn = partial(collate, **tokenize_kwargs)
    return DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=train and len(records) > batch_size,
        num_workers=workers,
        collate_fn=fn,
    )


@torch.no_grad()
def embed(model_fn, records, tokenize_kwargs, batch_size=256, device="cpu"):
    """``[n_records, D]`` embeddings of the full curves as a numpy array."""
    loader = make_loader(records, tokenize_kwargs, batch_size)
    zs = [model_fn(*batch["full"].to(device)).float().cpu() for batch in loader]
    return torch.cat(zs).numpy()


def _standardize(z_tr, z_va):
    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-6
    return (z_tr - mu) / sd, (z_va - mu) / sd


def linear_probe(z_tr, y_tr, z_va, y_va, steps: int = 300, wd: float = 1e-3) -> dict:
    """Multinomial logistic regression on standardised features, fitted with
    full-batch L-BFGS. Returns ``dict(acc, train_acc)``."""
    x_tr, x_va = (torch.from_numpy(z).float() for z in _standardize(z_tr, z_va))
    y_tr, y_va = (torch.as_tensor(y, dtype=torch.long) for y in (y_tr, y_va))
    head = nn.Linear(x_tr.shape[1], int(max(y_tr.max(), y_va.max())) + 1)
    opt = torch.optim.LBFGS(
        head.parameters(), max_iter=steps, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(head(x_tr), y_tr) + wd * head.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = (head(x_va).argmax(1) == y_va).float().mean().item()
        train_acc = (head(x_tr).argmax(1) == y_tr).float().mean().item()
    return dict(acc=acc, train_acc=train_acc)


def ridge_r2(z_tr, t_tr, z_va, t_va, alpha: float = 0.1) -> float:
    """Validation R2 of a closed-form ridge regression (mean-loss penalty
    ``alpha``) from standardised features to a target such as log period."""
    x_tr, x_va = (x.astype(np.float64) for x in _standardize(z_tr, z_va))
    t_tr, t_va = np.asarray(t_tr, dtype=np.float64), np.asarray(t_va, dtype=np.float64)
    n, d = x_tr.shape
    a = x_tr.T @ x_tr / n + alpha * np.eye(d)
    w = np.linalg.solve(a, x_tr.T @ (t_tr - t_tr.mean()) / n)
    resid = t_va - t_tr.mean() - x_va @ w
    return float(1.0 - (resid**2).sum() / ((t_va - t_va.mean()) ** 2).sum())


def evaluate(
    backbone,
    train,
    val,
    tokenize_kwargs,
    batch_size,
    device,
    embed_fn: Callable[[list[Record]], np.ndarray] | None = None,
) -> dict:
    """Class probe accuracy and log-period R2 of the backbone's embeddings.

    ``embed_fn(records) -> [len(records), D]`` numpy is called once for
    ``train`` and once for ``val``; by default it is :func:`embed` with the
    backbone switched to eval mode and restored afterwards. A given
    ``embed_fn`` owns the modes of its modules: nothing is switched here."""
    if embed_fn is None:
        was_training = backbone.training
        backbone.eval()
        z_tr, z_va = (
            embed(backbone, r, tokenize_kwargs, batch_size, device)
            for r in (train, val)
        )
        backbone.train(was_training)
    else:
        z_tr, z_va = embed_fn(train), embed_fn(val)
    y_tr, y_va = (np.array([r.label for r in rs]) for rs in (train, val))
    p_tr, p_va = (np.log10([r.period for r in rs]) for rs in (train, val))
    out = linear_probe(z_tr, y_tr, z_va, y_va)
    return dict(out, r2=ridge_r2(z_tr, p_tr, z_va, p_va))


def save_backbone(path: str, backbone: RoMAE, tokenize_kwargs: dict) -> None:
    """Checkpoint with the weights, the constructor keywords and the
    tokenizer keywords the backbone was trained with."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    hparams = dict(backbone.hparams, encoder=asdict(backbone.cfg), pool=backbone.pool)
    sd = backbone.state_dict()
    torch.save(dict(state_dict=sd, hparams=hparams, tokenize=tokenize_kwargs), path)


def load_backbone(path: str, device="cpu") -> tuple[RoMAE, dict]:
    """``(backbone in eval mode, tokenize_kwargs)`` from :func:`save_backbone`."""
    ckpt = torch.load(path, map_location=device)
    backbone = RoMAE(**ckpt["hparams"]).to(device)
    backbone.load_state_dict(ckpt["state_dict"])
    return backbone.eval(), ckpt["tokenize"]


def load_world_model(path: str, device="cpu") -> tuple[LeWorldModel, dict, FrameConfig]:
    """``(model in eval mode, tokenize_kwargs, FrameConfig)`` from the
    ``_wm.pt`` checkpoint of ``examples/train_lewm.py``: the backbone is
    rebuilt from ``ckpt["backbone"]`` (the :func:`save_backbone` dict), the
    world model from ``ckpt["hparams"]`` via
    :meth:`~romae_lc.LeWorldModel.from_hparams`. ``ckpt["no_actions"]``
    (``torch.load(path)``) records whether the predictor was trained
    unconditioned (``--no-actions``); on such a checkpoint the AdaLN weights
    are exactly zero, so passing actions to it is a no-op."""
    ckpt = torch.load(path, map_location=device)
    backbone = RoMAE(**ckpt["backbone"])
    model = LeWorldModel.from_hparams(backbone, ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["tokenize"], FrameConfig(**ckpt["frames"])
