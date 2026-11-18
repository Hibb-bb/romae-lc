"""Encode a grid of windows once with a frozen encoder.

The latent of a window is the pooled CLS feature of the encoder, ``[D]`` per
window with ``D`` the width. This script lays a grid of windows over every
record of every split (``--stride`` window lengths between consecutive window
starts, the window length and token bounds from the checkpoint's
:class:`~romae_lc.FrameConfig`), encodes every window once through
:func:`project.checkpoint.load_encoder` and writes them to one ``latents.pt``.
The probes read that file (:mod:`project.period_probe`,
:mod:`project.eval.card_class`, :mod:`project.eval.card_period`,
:mod:`project.eval.classify_rf`).

Windows are cut with :func:`~romae_lc.frame_grid` (``fill=True``: every
window that fits, sparse or empty ones included, so a reader can skip them
by ``n_tokens``) and their starts agree with
:func:`project.windows.grid_starts` by construction (asserted once). A window
above ``--cap`` points is subsampled deterministically (one rng per object
seeded from ``(seed, record index)``, applied to the windows in order), so
:func:`project.windows.object_windows` reproduces any stored row.

The file is one dict:

- ``z`` float16 ``[N, D]``: the latents, rows grouped by split, then by
  object, then in window order;
- ``obj`` int32 ``[N]``: the record's position in ``data[split]``;
- ``win`` int16 ``[N]``: the window's index in its object's grid;
- ``start`` float32 ``[N]``: the window start in days (record time);
- ``n_tokens`` int16 ``[N]``: real points in the window before the cap;
- ``splits``: ``{split: [offset, count]}``, the rows of every split;
- ``objects``: ``{split: dict(label int64, period float32, superclass
  [str], index int64)}``, one entry per encoded object in cache order;
- ``ptr``: ``{split: int64 [n_objects + 1]}``, CSR offsets into the
  **global** rows: object ``i`` of a split owns rows ``ptr[i]:ptr[i + 1]``,
  and ``ptr[0]`` is the split's offset;
- ``meta``: ``ckpt``, ``encoder_kind``, ``dim``, ``stride``, ``window``,
  ``min_tokens`` (the checkpoint's; a window with fewer real points does
  not count as valid downstream), ``cap``, ``frames`` (the FrameConfig),
  ``created`` (encoder step and the arguments of this run).

A ``latents.json`` next to it repeats the meta with the counts.

    python -m project.eval.encode --ckpt project/runs/enc_j1/mae.pt
    python -m project.eval.encode --ckpt project/runs/enc_j1/mae.pt \\
        --out project/runs/enc_j1/latents_test.pt --splits test
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from project.checkpoint import load_encoder
from project.data import data_args_from, load_data, superclass
from project.utils import add_device_arg, dump_json, get_device
from project.windows import WindowDataset, make_collate


def _pick(n_records: int, n: int | None, seed: int) -> list[int]:
    """The record positions :func:`project.data.subset` keeps (same rng)."""
    if n is None or n >= n_records:
        return list(range(n_records))
    return np.sort(np.random.default_rng(seed).choice(n_records, n, replace=False)).tolist()


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True, help="the encoder checkpoint (mae.pt)")
    p.add_argument("--out", default=None, help="default <ckpt dir>/latents.pt")
    p.add_argument(
        "--stride",
        type=float,
        default=0.25,
        help="window-start spacing in window units (the window length comes "
        "from the checkpoint)",
    )
    p.add_argument(
        "--window",
        type=float,
        default=None,
        help="window length in days (default: the checkpoint's); an encoder "
        "trained on windows of many lengths is cached once per length",
    )
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--data", default=None, help="override the checkpoint's data")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-objects", type=int, default=None, help="cap per split")
    p.add_argument("--batch-size", type=int, default=512, help="windows per batch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--cap",
        type=int,
        default=None,
        help="max tokens per window (default: the checkpoint's max_tokens)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="cap subsample seed (default: the checkpoint's data seed)",
    )
    add_device_arg(p)
    args = p.parse_args(argv)
    t_start = time.time()
    dev = get_device(args)
    out = Path(args.out or (Path(args.ckpt).parent / "latents.pt"))
    out.parent.mkdir(parents=True, exist_ok=True)

    enc, meta = load_encoder(args.ckpt, dev)
    cfg, spec = meta.cfg, meta.spec
    if args.window:
        cfg = replace(cfg, window=float(args.window))
    cap = cfg.max_tokens if args.cap is None else args.cap
    # Only the data source may be overridden: --seed is the cap / subset seed
    # and must not re-simulate or re-split the checkpoint's data.
    data_ns = data_args_from(
        meta.args, argparse.Namespace(data=args.data, max_rows=args.max_rows)
    )
    seed = args.seed if args.seed is not None else int(data_ns.seed or 0)
    data = load_data(data_ns, splits=tuple(args.splits))
    splits = []
    for split in args.splits:
        records = data[split]
        idx = _pick(len(records), args.max_objects, seed)
        splits.append((split, [records[i] for i in idx], idx))
    print(
        f"encoder {enc.kind} (step {meta.step}, dim {enc.dim}) from {args.ckpt}; "
        f"window {cfg.window} d, stride {args.stride}, cap {cap}; "
        + ", ".join(f"{s} {len(r)} objects" for s, r, _ in splits)
        + f"; loaded in {time.time() - t_start:.0f}s"
    )
    ds = WindowDataset(splits, cfg, args.stride, cap, seed)
    loader = DataLoader(
        ds,
        args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=make_collate(spec),
    )
    n = ds.total
    z = torch.empty(n, enc.dim, dtype=torch.float16)
    cols = dict(
        split=torch.empty(n, dtype=torch.int64),
        obj=torch.empty(n, dtype=torch.int64),
        win=torch.empty(n, dtype=torch.int64),
        start=torch.empty(n, dtype=torch.float32),
        n_tokens=torch.empty(n, dtype=torch.int64),
    )
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    t0, row = time.time(), 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            with amp:
                zb = enc.encode([batch["tokens"].to(dev)])[:, 0]
            zb = zb.float().cpu().half()
            widx = batch["widx"]
            z[widx] = zb
            for k, v in cols.items():
                v[widx] = batch[k]
            row += len(widx)
            if (i + 1) % 50 == 0:
                print(f"  {row} / {n} windows, {time.time() - t0:.0f}s", flush=True)
    assert row == n

    # The dataset walks splits, objects and windows in order, so the rows
    # are already grouped; the CSR pointers follow from the counts.
    bounds, objects, ptr = {}, {}, {}
    for s, (split, records, idx) in enumerate(splits):
        off = ds.offsets[s]
        bounds[split] = [int(off[0]), int(off[-1] - off[0])]
        ptr[split] = torch.from_numpy(off.astype(np.int64))
        objects[split] = dict(
            label=torch.tensor([r.label for r in records], dtype=torch.int64),
            period=torch.tensor(
                [float(r.period) if r.period is not None else float("nan") for r in records],
                dtype=torch.float32,
            ),
            superclass=[superclass(r) for r in records],
            index=torch.tensor(idx, dtype=torch.int64),
        )
        assert (cols["split"][off[0] : off[-1]] == s).all()
    valid = int((cols["n_tokens"] >= cfg.min_tokens).sum())
    info = dict(
        ckpt=str(args.ckpt),
        encoder_kind=enc.kind,
        dim=int(enc.dim),
        stride=float(args.stride),
        window=float(cfg.window),
        min_tokens=int(cfg.min_tokens),
        cap=int(cap),
        frames=asdict(cfg),
        created=dict(encoder_step=meta.step, args=vars(args), seed=seed),
    )
    state = dict(
        z=z,
        obj=cols["obj"].to(torch.int32),
        win=cols["win"].clamp(max=32767).to(torch.int16),
        start=cols["start"],
        n_tokens=cols["n_tokens"].clamp(max=32767).to(torch.int16),
        splits=bounds,
        objects=objects,
        ptr=ptr,
        meta=info,
    )
    torch.save(state, out)
    counts = dict(
        objects={s: len(r) for s, r, _ in splits},
        windows={s: b[1] for s, b in bounds.items()},
        windows_total=n,
        windows_valid=valid,
        gb=out.stat().st_size / 2**30,
        seconds=time.time() - t_start,
    )
    dump_json(dict(info, counts=counts), out.with_suffix(".json"))
    print(
        f"saved {out}: {sum(counts['objects'].values())} objects, {n} windows "
        f"({valid} with >= {cfg.min_tokens} points), "
        f"{counts['gb']:.2f} GB, "
        f"{counts['seconds']:.0f}s"
    )


if __name__ == "__main__":
    main()
