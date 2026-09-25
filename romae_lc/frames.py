"""Light-curve "frames" for LeWorldModel: consecutive time windows and actions.

A LeWM frame is one observation ``o_t`` of a trajectory; here it is one
time window of a light curve, all bands together, tokenised by
:func:`~romae_lc.tokenize.tokenize` with time re-zeroed at the window start.
The action that follows a frame is the advance to the next window in window
units, the only exogenous variable of a passive time series. Windows share
the same time origin because rotary attention is relative but the CLS token
sits at position 0 and the tokenizer offsets every position by +1: two
windows are comparable only through that anchor, and the window start (not
the first epoch inside it) keeps the phase relation between consecutive
windows exactly what the action describes.

:class:`FrameConfig` sets the geometry, :func:`sample_frames` draws a random
sequence of windows from a :class:`~romae_lc.data.Record` (with redraws when
windows are too sparse), :func:`frame_grid` lays deterministic windows for
evaluation, and :class:`FrameDataset` with :func:`collate_frames` turn records
into per-time-step padded :class:`~romae_lc.tokenize.Tokens` batches.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import Record
from .tokenize import Tokens, tokenize


@dataclass
class FrameConfig:
    """How :func:`sample_frames` cuts a sequence of consecutive time windows
    ("frames") from a record.

    The model defaults are those of the paper and the released code
    (``history_size 3 + num_preds 1`` frames per sub-trajectory); the window,
    advance and token bounds are the light-curve translation, tuned for the
    simulator of :func:`~romae_lc.data.simulate` (in-season density ~0.9
    points per day, 240-d seasons): a 30-d window holds ~18 points on
    average, a 4-frame sequence spans 120-210 d and fits inside one season
    often enough that a draw succeeds with probability ~0.6. For other data
    choose ``window`` so that ``window * (in-season points per day) >=
    min_tokens`` in the sparsest band and ``window <= span / (1 + (n_frames
    - 1) * advance[0])`` so the record can host a sequence at all
    (:class:`FrameDataset` drops the ones that cannot). With ``window`` many
    periods long the next window's phase is unpredictable and the world model
    learns period/shape/amplitude dynamics, not phase; make ``window`` a few
    periods when you want phase.

    ``use_cls=True`` (the :class:`~romae_lc.model.RoMAE` default) is what
    anchors a window's phase: without the CLS token at position 0 the
    encoder is purely relative in time, a window's embedding is invariant to
    a time shift of its contents and only shift-invariant window statistics
    can be learnt. With ``use_cls=False, pool="mean"`` every window must also
    be non-empty: a step whose every row is empty raises ``ValueError`` in the
    encoder.

    Attributes:
        n_frames: Windows per sequence, history 3 + 1 target.
        window: Window length in days.
        advance: ``(lo, hi)`` of the uniform start-to-start advance between
            consecutive windows in window units; 1 = contiguous, ``< 1``
            overlapping (allowed, e.g. ``(0.5, 1.0)`` for dense cadences).
            A random advance keeps the action informative; ``(1.0, 1.0)``
            reproduces a fixed frame skip.
        min_tokens: Minimum points per window (all bands) for a draw to
            succeed; below it the draw is repeated.
        max_tokens: Random subsample cap per window.
        max_tries: Redraws before the best draw is kept.
        resample: Redraw ``y`` from ``N(y, err)``; off, the official pipeline
            has no augmentation.
        with_err: Frames also carry the per-point one-sigma errors: every
            frame is a ``(t, y, band, err)`` quadruple instead of a triple
            and :func:`collate_frames` passes ``err`` to the tokenizer as
            ``extras`` (``Tokens.extras [B, N]``), for encoders that take the
            uncertainty as an input channel and for decoders that need it.
    """

    n_frames: int = 4
    window: float = 30.0
    advance: tuple[float, float] = (1.0, 2.0)
    min_tokens: int = 8
    max_tokens: int = 512
    max_tries: int = 20
    resample: bool = False
    with_err: bool = False

    def __post_init__(self):
        if self.n_frames < 2:
            raise ValueError(f"n_frames must be >= 2, got {self.n_frames}")
        if self.window <= 0:
            raise ValueError(f"window must be > 0, got {self.window}")
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {self.min_tokens}")
        if self.max_tokens < self.min_tokens:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) must be >= min_tokens "
                f"({self.min_tokens})"
            )
        if self.max_tries < 1:
            raise ValueError(f"max_tries must be >= 1, got {self.max_tries}")
        lo, hi = self.advance
        if not (0 < lo <= hi):
            raise ValueError(f"advance must satisfy 0 < lo <= hi, got {self.advance}")


def _offsets(d: np.ndarray) -> np.ndarray:
    """Window starts in window units for advances ``d``: ``0, d_1, d_1 + d_2, ...``
    (the last advance follows the last window and moves nothing)."""
    return np.concatenate([[0.0], np.cumsum(d[:-1], dtype=np.float64)])


def _window_masks(t: np.ndarray, starts: np.ndarray, window: float) -> list:
    """Half-open membership ``s <= t < s + window`` of every window."""
    return [(t >= s) & (t < s + window) for s in starts]


def _cut(record: Record, t: np.ndarray, masks, starts, cfg, rng, augment: bool):
    """``(t, y, band)`` float32/float32/int64 triples of the masked windows
    (``(t, y, band, err)`` quadruples with ``cfg.with_err``), times relative
    to the window start; ``augment`` applies the ``max_tokens`` subsample and
    ``resample`` of ``cfg``."""
    frames = []
    for mask, start in zip(masks, starts):
        idx = np.flatnonzero(mask)
        if augment and len(idx) > cfg.max_tokens:
            idx = np.sort(rng.choice(idx, size=cfg.max_tokens, replace=False))
        y = record.y[idx]
        if augment and cfg.resample:
            y = y + rng.standard_normal(len(idx)) * record.err[idx]
        frame = (
            (t[idx] - start).astype(np.float32),
            y.astype(np.float32),
            record.band[idx].astype(np.int64),
        )
        if cfg.with_err:
            frame = frame + (record.err[idx].astype(np.float32),)
        frames.append(frame)
    return frames


def sample_frames(record: Record, cfg: FrameConfig, rng: np.random.Generator):
    """One random sequence of consecutive time windows of a record.

    One draw takes advances ``d_1..d_T ~ U(*cfg.advance)`` (one per frame:
    the last is the action of the last frame, as in the official layout
    where every frame carries the action block that follows it), fails when
    the record cannot host them (``span < window * (1 + sum(d_1..d_{T-1}))``)
    and otherwise places the first window uniformly in the free span and the
    next ones at ``s_{i+1} = s_i + d_i * window``. Window ``i`` holds the
    points with ``s_i <= t < s_i + window`` in every band (half-open, so
    contiguous windows partition the points). A draw succeeds when every
    window has at least ``cfg.min_tokens`` points; up to ``cfg.max_tries``
    draws are made and the first success returned, otherwise the feasible
    draw whose sparsest window is largest (first on ties), so sparse or empty
    windows are possible after ``max_tries`` failures. When no draw was
    feasible a last one uses ``d_i = advance[0]`` for every ``i``, which fits
    every record kept by :class:`FrameDataset`. Windows above
    ``cfg.max_tokens`` are randomly subsampled (order kept) and, with
    ``cfg.resample``, ``y`` is redrawn from ``N(y, err)``. Deterministic
    given ``rng``.

    Returns:
        ``(frames, actions)``: ``frames`` is a list of ``cfg.n_frames``
        ``(t, y, band)`` float32/float32/int64 triples (``(t, y, band, err)``
        with ``cfg.with_err``) with ``t`` relative to the window start (in
        ``[0, window)``), ``actions`` a float32 ``[n_frames, 1]`` array with
        ``actions[i] = d_i``.

    Raises:
        ValueError: The record span cannot host ``n_frames`` windows even
            with the smallest advance.
    """
    t = record.t.astype(np.float64)
    n_frames, window = cfg.n_frames, float(cfg.window)
    t_min, span = (t.min(), t.max() - t.min()) if record.n else (0.0, 0.0)
    best = None  # (min count, advances, starts, masks)
    for _ in range(cfg.max_tries):
        d = rng.uniform(*cfg.advance, size=n_frames)
        span_free = span - window * (1.0 + d[:-1].sum())
        if span_free < 0:
            continue
        starts = t_min + rng.uniform(0, span_free) + window * _offsets(d)
        masks = _window_masks(t, starts, window)
        score = min(int(m.sum()) for m in masks)
        if score >= cfg.min_tokens:
            best = (score, d, starts, masks)
            break
        if best is None or score > best[0]:
            best = (score, d, starts, masks)
    if best is None:
        d = np.full(n_frames, cfg.advance[0], dtype=np.float64)
        span_free = span - window * (1.0 + d[:-1].sum())
        if span_free < 0:
            raise ValueError(
                f"record span {span:.1f} d cannot host {n_frames} windows of "
                f"{cfg.window} d with advance >= {cfg.advance[0]}"
            )
        starts = t_min + rng.uniform(0, span_free) + window * _offsets(d)
        masks = _window_masks(t, starts, window)
        best = (0, d, starts, masks)
    _, d, starts, masks = best
    frames = _cut(record, t, masks, starts, cfg, rng, augment=True)
    return frames, d.astype(np.float32).reshape(n_frames, 1)


def frame_grid(
    record: Record,
    cfg: FrameConfig,
    start: float | None = None,
    n_frames: int | None = None,
    advance: float | None = None,
    fill: bool = False,
):
    """Deterministic consecutive windows of a record for evaluation.

    Window ``i`` starts at ``start + i * advance * window``. No redraws, no
    subsampling, no augmentation: sparse or empty windows are kept (the
    caller chose the grid); empty windows in the season gaps give CLS-only
    embeddings, mask them with ``n_tokens`` from :func:`collate_frames`.

    Args:
        record: The light curve.
        cfg: Supplies ``window``, the default ``n_frames`` and the default
            ``advance``.
        start: First window start, default the first epoch.
        n_frames: Number of windows, default ``cfg.n_frames``; ignored with
            ``fill``.
        advance: Start-to-start advance in window units, default
            ``cfg.advance[0]`` (contiguous windows with the default config,
            the lower edge of the training distribution).
        fill: Use every window that fits instead: ``n = max(1, floor((t.max()
            - start - window) / (advance * window)) + 1)``, so the last window
            ends at or before the last epoch (when even the first does not
            fit, a single data-truncated window).

    Returns:
        ``(frames, actions)`` like :func:`sample_frames`, with
        ``actions[i] = advance`` for every ``i``.
    """
    t = record.t.astype(np.float64)
    window = float(cfg.window)
    advance = cfg.advance[0] if advance is None else float(advance)
    if advance <= 0:
        raise ValueError(f"advance must be > 0, got {advance}")
    start = float(t.min()) if start is None else float(start)
    if fill:
        n = max(1, int(np.floor((t.max() - start - window) / (advance * window))) + 1)
    else:
        n = cfg.n_frames if n_frames is None else int(n_frames)
        if n < 1:
            raise ValueError(f"n_frames must be >= 1, got {n}")
    starts = start + advance * window * np.arange(n, dtype=np.float64)
    masks = _window_masks(t, starts, window)
    frames = _cut(record, t, masks, starts, cfg, rng=None, augment=False)
    return frames, np.full((n, 1), advance, dtype=np.float32)


class FrameDataset(Dataset):
    """Records as a map-style dataset yielding frame sequences.

    Each item is ``dict(frames, actions, label, index)`` from
    :func:`sample_frames`: ``frames`` a list of ``cfg.n_frames`` ``(t, y,
    band)`` triples, ``actions`` a float32 ``[n_frames, 1]`` array and
    ``index`` the record's position in the ``records`` argument (so labels
    and periods stay addressable after records are dropped). Seeding is that
    of :class:`~romae_lc.data.LightCurveDataset`: the rng is seeded from
    ``(seed, index)`` plus, with ``epoch_seed``, the number of times the item
    was requested and ``torch.initial_seed()`` (refreshed by the DataLoader
    in every worker each epoch), so sequences differ from epoch to epoch;
    ``epoch_seed=False`` makes every item deterministic.

    Records whose span is shorter than ``window * (1 + (n_frames - 1) *
    advance[0])`` cannot host a sequence even with the smallest advance and
    are dropped at construction with a warning; ``len(ds) == len(indices)``.

    Args:
        records: The light curves.
        cfg: Frame sampling parameters, default :class:`FrameConfig`.
        seed: Base seed.
        epoch_seed: Re-randomise items on every request.

    Attributes:
        records: The kept records, in order.
        indices: Their positions in the input.
        cfg: The frame configuration.

    Raises:
        ValueError: No record can host a sequence.
    """

    def __init__(
        self,
        records: Sequence[Record],
        cfg: FrameConfig | None = None,
        seed: int = 0,
        epoch_seed: bool = True,
    ):
        cfg = cfg or FrameConfig()
        need = cfg.window * (1.0 + (cfg.n_frames - 1) * cfg.advance[0])
        spans = np.array(
            [float(r.t.max() - r.t.min()) if r.n else 0.0 for r in records],
            dtype=np.float64,
        )
        keep = [i for i, s in enumerate(spans) if s >= need]
        n, k = len(spans), len(spans) - len(keep)
        if k:
            med = float(np.median(spans)) if n else 0.0
            warnings.warn(
                f"{k} of {n} records shorter than {need:.1f} d (median span "
                f"{med:.1f} d) dropped; reduce FrameConfig.window or advance"
            )
        if not keep:
            raise ValueError(
                f"no record can host {cfg.n_frames} windows of {cfg.window} d "
                f"with advance >= {cfg.advance[0]} (needs a span >= {need:.1f} d)"
            )
        self.records = [records[i] for i in keep]
        self.indices = keep
        self.cfg, self.seed, self.epoch_seed = cfg, seed, epoch_seed
        self.calls = [0] * len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        j = self.indices[i]
        entropy = [self.seed, j]
        if self.epoch_seed:
            self.calls[i] += 1
            entropy += [self.calls[i], torch.initial_seed()]
        rng = np.random.default_rng(entropy)
        record = self.records[i]
        frames, actions = sample_frames(record, self.cfg, rng)
        return dict(frames=frames, actions=actions, label=record.label, index=j)


def collate_frames(batch: list[dict], **tokenize_kwargs) -> dict:
    """DataLoader ``collate_fn`` for :class:`FrameDataset` items; use
    ``functools.partial(collate_frames, band_wavelengths=..., time_scale=...)``
    to pass the :func:`~romae_lc.tokenize.tokenize` keywords, exactly like
    :func:`romae_lc.data.collate`. Frame ``t`` of every item is tokenized
    together, one padded :class:`~romae_lc.tokenize.Tokens` per time step, so
    windows with different point counts pad within their step. Frames cut
    with ``FrameConfig(with_err=True)`` carry a fourth ``err`` array, which
    is passed to the tokenizer as ``extras`` and comes back as
    ``Tokens.extras [B, N]`` (padding entries are 0).

    Returns:
        ``dict(frames=[Tokens] * T, actions FloatTensor [B, T, 1], n_tokens
        LongTensor [B, T] (points per window, ``frames[t].n_real``), label
        LongTensor [B], index LongTensor [B])``.
    """
    n_frames = len(batch[0]["frames"])
    if any(len(item["frames"]) != n_frames for item in batch):
        raise ValueError("every item must carry the same number of frames")

    def tok(tuples) -> Tokens:
        cols = [
            [torch.from_numpy(np.ascontiguousarray(a)) for a in c] for c in zip(*tuples)
        ]
        if len(cols) == 4:  # (t, y, band, err): the errors ride along as extras
            return tokenize(*cols[:3], extras=cols[3], **tokenize_kwargs)
        return tokenize(*cols, **tokenize_kwargs)

    frames = [tok([item["frames"][t] for item in batch]) for t in range(n_frames)]
    actions = np.stack(
        [np.asarray(item["actions"], dtype=np.float32) for item in batch]
    )
    return dict(
        frames=frames,
        actions=torch.from_numpy(actions),
        n_tokens=torch.stack([f.n_real for f in frames], 1).long(),
        label=torch.tensor([item["label"] for item in batch], dtype=torch.long),
        index=torch.tensor([item["index"] for item in batch], dtype=torch.long),
    )
