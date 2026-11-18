"""Records: loading the data, the splits, the class names, the data flags.

Records come from :func:`~romae_lc.load_pc` with the fine ``class_str`` label
(or from the toy simulator with ``--data sim``). Every record is standardised
per band. :class:`Data` holds the records of each split and the class
vocabulary that all splits share.

The catalogue period of a record is never a model input. It is a training
target (spectral aux loss, phase-band mask) and a probe target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from romae_lc import CLASSES, DEFAULT_SURVEYS, PC_BANDS, Record, load_pc, normalize, simulate, wavelengths_for


DATA_ROOT = "/projects/bfrf/data/PC_matches/ZTFxPC"


LABEL_FIELD = "class_str"


@dataclass
class Data:
    """Per-band standardised records per split, the shared fine-class
    vocabulary (``record.label`` indexes it) and the tokenizer band table."""

    splits: dict[str, list[Record]]
    classes: tuple[str, ...]
    wavelengths: dict[int, float]

    def __getitem__(self, split: str) -> list[Record]:
        return self.splits[split]


def add_data_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("data")
    g.add_argument(
        "--data",
        default=DATA_ROOT,
        help="PC_matches sub-dataset directory, or 'sim' for the toy simulator",
    )
    g.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="keep only these class_str labels (default: every class)",
    )
    g.add_argument("--max-rows", type=int, default=None, help="cap rows per split")
    g.add_argument("--min-points", type=int, default=8, help="drop shorter records")
    g.add_argument(
        "--norm-days",
        type=float,
        default=None,
        help="normalise every star with the median and scatter of its FIRST this many days only "
        "(per band), so nothing later leaks into the statistics; default: the whole light curve",
    )
    g.add_argument("--n-sim", type=int, default=256, help="--data sim: stars")
    g.add_argument("--seed", type=int, default=0)


def class_vocabulary(root: str, field: str = LABEL_FIELD) -> tuple[str, ...]:
    """Sorted ``class_str`` values over every split: one vocabulary shared by
    train, validation and test."""
    import datasets

    dd = datasets.load_from_disk(root)
    vocab: set[str] = set()
    for split in dd:
        vocab |= set(dd[split][field])
    return tuple(sorted(vocab))


def superclass(record: Record) -> str:
    return str(record.meta.get("superclass_str", "?"))


def fine_class(record: Record) -> str:
    return str(record.meta.get("class_str", "?"))


#: Fine classes kept out of the class probe (decision of 2026-09-26): they are
#: anomaly classes (Blazhko modulation, O'Connell-effect eclipsing binaries),
#: the kind of object the anomaly scores are meant to find, so a classifier
#: must not be trained to name them. Their periods still count in every
#: period metric and they stay in the training data.
ANOMALY_CLASSES = ("RRab-Blazhko", "RRc-Blazhko", "EW/EB-OC")


def load_data(args, splits: Sequence[str] = ("train", "validation")) -> Data:
    """Load and standardise the requested splits (see :class:`Data`)."""
    if args.data == "sim":
        records = simulate(args.n_sim, seed=args.seed)
        for r in records:
            r.meta["class_str"] = r.meta["superclass_str"] = CLASSES[r.label]
        order = np.random.default_rng(args.seed).permutation(len(records))
        n_val = max(1, len(records) // 5)
        parts = {
            "validation": order[:n_val],
            "test": order[n_val : 2 * n_val],
            "train": order[2 * n_val :],
        }
        nd = getattr(args, "norm_days", None)
        out = {s: [normalize(records[i], ref_days=nd) for i in parts[s]] for s in splits}
        wl = {i: s.wavelength_nm for i, s in enumerate(DEFAULT_SURVEYS)}
        return Data(out, CLASSES, wl)
    classes = class_vocabulary(args.data)
    keep = None if not args.classes else set(args.classes)
    if keep and not keep <= set(classes):
        raise ValueError(f"unknown classes {sorted(keep - set(classes))}")
    out = {}
    for split in splits:
        recs = load_pc(
            args.data,
            split,
            label_field=LABEL_FIELD,
            classes=classes,
            max_rows=args.max_rows,
            min_points=args.min_points,
        )
        if keep:
            recs = [r for r in recs if classes[r.label] in keep]
        out[split] = [normalize(r, ref_days=getattr(args, "norm_days", None)) for r in recs]
    return Data(out, classes, wavelengths_for(PC_BANDS))


DATA_KEYS = ("data", "classes", "max_rows", "min_points", "n_sim", "seed", "norm_days")


def data_args_from(saved: dict, overrides=None) -> argparse.Namespace:
    """The data arguments a checkpoint was trained with, with any non-None
    attribute of ``overrides`` (a Namespace) taking precedence."""
    ns = argparse.Namespace(**{k: saved.get(k) for k in DATA_KEYS})
    for k in DATA_KEYS:
        v = getattr(overrides, k, None) if overrides is not None else None
        if v is not None:
            setattr(ns, k, v)
    return ns


def subset(records: Sequence, n: int | None, seed: int = 0) -> list:
    """A seeded random subset of at most ``n`` records (order kept)."""
    if n is None or n >= len(records):
        return list(records)
    idx = np.sort(np.random.default_rng(seed).choice(len(records), n, replace=False))
    return [records[i] for i in idx]
