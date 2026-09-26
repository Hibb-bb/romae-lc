"""Encode a grid of windows once with a frozen encoder (the input of stage 2).

Stage 1 is the autoencoder (``pretrain_mae.py``, ``mae.pt``); its latent is
the pooled CLS feature of the RoMAE encoder, ``[D]`` per window with ``D``
the width (192 for ``--size light``). Stage 2 trains the predictor on those
latents with the encoder frozen, so nothing is gained by re-encoding the
same windows for hundreds of thousands of steps: this script lays a grid of
windows over every record of every split (``--stride`` window lengths
between consecutive window starts, the window length and token bounds from
the checkpoint's :class:`~romae_lc.FrameConfig`), encodes every window once
through :func:`project.common.load_encoder` (``mae.pt`` or ``wm.pt``) and
writes them to one ``latents.pt`` that :mod:`project.train_predictor` loads
onto the GPU whole.

Windows are cut with :func:`~romae_lc.frame_grid` (``fill=True``: every
window that fits, sparse or empty ones included, so a sequence sampler can
skip them by ``n_tokens``) and their starts agree with
:func:`project.common.grid_starts` by construction (asserted once). A window
above ``--cap`` points is subsampled deterministically, the way
:func:`project.common.grid_item` does it (one rng per object seeded from
``(seed, record index)``, applied to the windows in order), so
:func:`object_windows` reproduces any stored row.

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

    python -m project.cache_latents --ckpt project/runs/mae_w250/mae.pt
    python -m project.cache_latents --ckpt project/runs/mae_w250/mae.pt \\
        --stride 0.25 --splits train validation --batch-size 512 --workers 8
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from romae_lc import FrameConfig, Record, frame_grid

from project.common import (
    TokenSpec,
    add_device_arg,
    data_args_from,
    dump_json,
    get_device,
    grid_starts,
    load_data,
    load_encoder,
    subsample_frame,
    superclass,
)


def object_windows(
    record: Record, cfg: FrameConfig, stride: float, cap: int | None, seed: int, index: int
):
    """The grid windows of one record as the cache stores them: ``(starts
    float64 [K], frames, n_tokens int [K])`` with ``frames`` the ``K``
    ``(t, y, band, err)`` windows after the deterministic cap (rng seeded
    from ``(seed, index)`` like :func:`project.common.grid_item`) and
    ``n_tokens`` their point counts before it."""
    starts = grid_starts(record, cfg, advance=stride)
    frames, _ = frame_grid(record, cfg, advance=stride, fill=True)
    if len(frames) != len(starts):
        raise AssertionError(
            f"frame_grid gave {len(frames)} windows, grid_starts {len(starts)}"
        )
    n_tokens = [len(f[0]) for f in frames]
    if cap is not None:
        rng = np.random.default_rng([seed, index])
        frames = [subsample_frame(f, cap, rng) for f in frames]
    return starts, frames, n_tokens


class WindowDataset(Dataset):
    """Every window of every object of every split as one flat index. An
    item is ``(split_i, obj_i, win)`` resolved lazily: the object's grid is
    cut once and kept while consecutive items address it (the sampler is
    sequential, so a worker's batch mostly hits one or two objects)."""

    def __init__(self, splits: list[tuple[str, list[Record], list[int]]], cfg, stride, cap, seed):
        self.splits, self.cfg, self.stride, self.cap, self.seed = (
            splits,
            cfg,
            stride,
            cap,
            seed,
        )
        self.counts = []  # windows per object, per split
        self.offsets = []  # first flat index of every object, per split
        total = 0
        for _, records, _ in splits:
            n = np.array(
                [len(grid_starts(r, cfg, advance=stride)) for r in records],
                dtype=np.int64,
            )
            self.counts.append(n)
            self.offsets.append(total + np.concatenate([[0], np.cumsum(n)]))
            total += int(n.sum())
        self.total = total
        self._key, self._cached = None, None

    def __len__(self) -> int:
        return self.total

    def locate(self, i: int) -> tuple[int, int, int]:
        """``(split_i, obj_i, win)`` of flat index ``i``."""
        for s, off in enumerate(self.offsets):
            if i < off[-1]:
                o = int(np.searchsorted(off, i, side="right") - 1)
                return s, o, int(i - off[o])
        raise IndexError(i)

    def windows(self, s: int, o: int):
        if self._key != (s, o):
            _, records, indices = self.splits[s]
            self._cached = object_windows(
                records[o], self.cfg, self.stride, self.cap, self.seed, indices[o]
            )
            self._key = (s, o)
        return self._cached

    def __getitem__(self, i: int) -> dict:
        s, o, w = self.locate(i)
        starts, frames, n_tokens = self.windows(s, o)
        return dict(
            split=s,
            obj=self.splits[s][2][o],  # the data[split] position, not the subset's
            win=w,
            start=float(starts[w]),
            n_tokens=int(n_tokens[w]),
            frame=frames[w],
        )


def make_collate(spec: TokenSpec):
    """Tokenise a list of single windows into one padded batch."""

    def collate(items: list[dict]) -> dict:
        return dict(
            tokens=spec.tokens([it["frame"] for it in items]),
            split=torch.tensor([it["split"] for it in items], dtype=torch.int64),
            obj=torch.tensor([it["obj"] for it in items], dtype=torch.int64),
            win=torch.tensor([it["win"] for it in items], dtype=torch.int64),
            start=torch.tensor([it["start"] for it in items], dtype=torch.float32),
            n_tokens=torch.tensor([it["n_tokens"] for it in items], dtype=torch.int64),
        )

    return collate


def _pick(n_records: int, n: int | None, seed: int) -> list[int]:
    """The record positions :func:`project.common.subset` keeps (same rng)."""
    if n is None or n >= n_records:
        return list(range(n_records))
    return np.sort(np.random.default_rng(seed).choice(n_records, n, replace=False)).tolist()


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True, help="mae.pt (stage 1) or wm.pt")
    p.add_argument("--out", default=None, help="default <ckpt dir>/latents.pt")
    p.add_argument(
        "--stride",
        type=float,
        default=0.25,
        help="window-start spacing in window units (the window length comes "
        "from the checkpoint)",
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
    cap = cfg.max_tokens if args.cap is None else args.cap
    # Only the data source may be overridden: --seed is the cap / subset seed
    # and must not re-simulate or re-split the checkpoint's data (gate.py does
    # the same).
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
    n = len(ds)
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
            b = batch["obj"].shape[0]
            with amp:
                zb = enc.encode([batch["tokens"].to(dev)])[:, 0]
            z[row : row + b] = zb.float().cpu().half()
            for k, v in cols.items():
                v[row : row + b] = batch[k]
            row += b
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
        f"({valid} with >= {cfg.min_tokens} points), {counts['gb']:.2f} GB, "
        f"{counts['seconds']:.0f}s"
    )


if __name__ == "__main__":
    main()
