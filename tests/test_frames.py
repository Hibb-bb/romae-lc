"""Frames: config validation, window geometry, redraws, grids, dataset, collate."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from romae_lc.data import Record, SimConfig, SurveyConfig, normalize, simulate
from romae_lc.frames import (
    FrameConfig,
    FrameDataset,
    collate_frames,
    frame_grid,
    sample_frames,
)
from romae_lc.tokenize import Tokens

SURVEYS = (SurveyConfig("g", 480.7, 120, 0.02), SurveyConfig("r", 622.1, 80, 0.03))
CFG = SimConfig(baseline_days=400.0)
WL = {i: s.wavelength_nm for i, s in enumerate(SURVEYS)}


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def records():
    return simulate(12, SURVEYS, CFG, seed=0)


def frame_cfg(**kw) -> FrameConfig:
    """The test default: 40-d windows, at least 4 points each."""
    return FrameConfig(**{"window": 40.0, "min_tokens": 4, **kw})


def dense_record() -> Record:
    """Two bands sampled daily / every other day over 100 d, no gaps."""
    t = np.concatenate([np.arange(101.0), np.arange(0.5, 100.0, 2.0)])
    y = np.random.default_rng(0).normal(size=t.size).astype(np.float32)
    band = np.repeat([0, 1], [101, 50])
    return Record(t.astype(np.float32), y, np.full_like(y, 0.1), band, 0, 1.0)


def source_indices(record: Record, y: np.ndarray) -> np.ndarray:
    """Indices of the record points an unaugmented window was cut from."""
    return np.array([np.flatnonzero(record.y == v)[0] for v in y], dtype=np.int64)


def window_start(record: Record, t: np.ndarray, y: np.ndarray) -> float:
    """The window start ``s`` such that ``record.t[src] == s + t``."""
    offsets = record.t[source_indices(record, y)] - t
    assert np.allclose(offsets, offsets[0], atol=1e-4)
    return float(offsets[0])


def test_frame_config_validation():
    for bad in (
        dict(n_frames=0),
        dict(min_tokens=0),
        dict(advance=(0, 1)),
        dict(advance=(2, 1)),
        dict(window=0),
        dict(max_tokens=2, min_tokens=4),
    ):
        with pytest.raises(ValueError):
            FrameConfig(**bad)
    assert FrameConfig(advance=(0.5, 1.0)).advance == (0.5, 1.0)


def test_sample_frames_geometry():
    r = dense_record()
    cfg = FrameConfig(window=10.0, advance=(1, 2), min_tokens=4, max_tokens=1000)
    rng = np.random.default_rng(0)
    for _ in range(10):
        frames, actions = sample_frames(r, cfg, rng)
        assert len(frames) == cfg.n_frames
        assert actions.shape == (cfg.n_frames, 1) and actions.dtype == np.float32
        assert (1 <= actions).all() and (actions <= 2).all()
        starts = []
        for t, y, band in frames:
            assert (t.dtype, y.dtype, band.dtype) == ("f4", "f4", "i8")
            assert len(t) == len(y) == len(band) >= cfg.min_tokens
            assert (0 <= t).all() and (t < 10).all()
            src = source_indices(r, y)
            assert np.array_equal(r.band[src], band)
            assert set(band.tolist()) == {0, 1}
            s = window_start(r, t, y)
            starts.append(s)
            inside = np.flatnonzero((r.t >= s) & (r.t < s + 10))
            assert (r.t[src] >= s).all() and (r.t[src] < s + 10).all()
            assert np.array_equal(np.sort(src), inside)  # nothing left out
        gaps = np.diff(starts)
        assert np.allclose(gaps, actions[:-1, 0] * cfg.window, atol=1e-4)
    capped = FrameConfig(window=10.0, advance=(1, 2), min_tokens=4, max_tokens=5)
    for t, y, band in sample_frames(r, capped, rng)[0]:
        assert len(t) == 5 and (np.diff(source_indices(r, y)) > 0).all()
    noisy = FrameConfig(window=10.0, advance=(1, 2), min_tokens=4, resample=True)
    for t, y, band in sample_frames(r, noisy, rng)[0]:
        assert not np.isin(y, r.y).any() and np.isfinite(y).all()
    a = sample_frames(r, cfg, np.random.default_rng(3))
    b = sample_frames(r, cfg, np.random.default_rng(3))
    assert np.array_equal(a[1], b[1])
    for fa, fb in zip(a[0], b[0]):
        assert all(np.array_equal(x, z) for x, z in zip(fa, fb))


def test_sample_frames_best_draw():
    t = np.concatenate([np.linspace(0, 25, 50), np.linspace(50, 75, 50)])
    y = np.arange(100, dtype=np.float32)
    gapped = Record(
        t.astype(np.float32), y, np.full_like(y, 0.1), np.zeros(100, np.int64), 0, 1.0
    )
    cfg = FrameConfig(
        window=10, n_frames=2, advance=(1, 1), min_tokens=4, max_tries=200
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        frames, actions = sample_frames(gapped, cfg, rng)
        assert len(frames) == 2 and (actions == 1).all()
        assert all(len(f[0]) >= 4 for f in frames)
    hopeless = FrameConfig(
        window=10, n_frames=2, advance=(1, 1), min_tokens=1000, max_tokens=1000
    )
    frames, actions = sample_frames(gapped, hopeless, rng)
    assert len(frames) == 2 and actions.shape == (2, 1)
    assert all(len(f[0]) < 1000 for f in frames)
    t = np.linspace(0, 15, 30, dtype=np.float32)
    short = Record(t, t.copy(), np.full_like(t, 0.1), np.zeros(30, np.int64), 0, 1.0)
    with pytest.raises(ValueError, match="cannot host"):
        sample_frames(short, FrameConfig(window=10, n_frames=2, advance=(1, 1)), rng)


def test_frame_grid():
    r = dense_record()
    cfg = FrameConfig(window=10.0, advance=(1, 2), min_tokens=4)
    frames, actions = frame_grid(r, cfg, start=0, n_frames=3, advance=1.0)
    assert len(frames) == 3 and actions.shape == (3, 1)
    assert actions.dtype == np.float32 and (actions == 1.0).all()
    starts = [window_start(r, t, y) for t, y, _ in frames]
    assert np.allclose(starts, [0.0, 10.0, 20.0])
    assert [len(f[0]) for f in frames] == [15, 15, 15]  # 10 daily + 5 alternate
    for t, y, band in frames:
        assert (0 <= t).all() and (t < 10).all()
        assert (np.diff(source_indices(r, y)) > 0).all()  # record order kept
    frames, actions = frame_grid(r, cfg)  # defaults: t.min(), cfg.n_frames, lo
    assert len(frames) == cfg.n_frames and (actions == cfg.advance[0]).all()
    assert np.isclose(window_start(r, *frames[0][:2]), r.t.min())
    half = FrameConfig(window=10.0, advance=(0.5, 1.0), min_tokens=4)
    frames, actions = frame_grid(r, half, start=0, n_frames=3)
    assert (actions == 0.5).all()
    assert np.allclose([window_start(r, t, y) for t, y, _ in frames], [0, 5, 10])
    frames, actions = frame_grid(r, cfg, start=0, fill=True)
    n = int(np.floor((r.t.max() - 10.0) / 10.0)) + 1
    assert len(frames) == n == 10 and actions.shape == (n, 1)
    starts = [window_start(r, t, y) for t, y, _ in frames]
    assert np.allclose(starts, 10.0 * np.arange(n)) and starts[-1] + 10 <= r.t.max()
    assert (actions == 1.0).all()
    again = frame_grid(r, cfg, start=0, fill=True)
    assert np.array_equal(again[1], actions)
    for fa, fb in zip(again[0], frames):
        assert all(np.array_equal(x, z) for x, z in zip(fa, fb))
    frames, _ = frame_grid(r, cfg, start=0, n_frames=2, advance=1.0, fill=True)
    assert len(frames) == 10  # fill overrides n_frames


def test_dataset_items_and_seeding(records):
    cfg = frame_cfg()
    ds = FrameDataset(records, cfg, seed=0)
    assert len(ds) == 12 and ds.indices == list(range(12))
    item = ds[3]
    assert set(item) == {"frames", "actions", "label", "index"}
    assert len(item["frames"]) == 4 and item["actions"].shape == (4, 1)
    assert item["label"] == records[3].label and item["index"] == 3
    assert not np.array_equal(item["actions"], ds[3]["actions"])
    fixed = FrameDataset(records, cfg, seed=0, epoch_seed=False)
    a, b = fixed[3], fixed[3]
    assert np.array_equal(a["actions"], b["actions"])
    for fa, fb in zip(a["frames"], b["frames"]):
        assert all(np.array_equal(x, z) for x, z in zip(fa, fb))
    short = simulate(1, SURVEYS, SimConfig(baseline_days=100.0), seed=1)[0]
    assert short.t.max() - short.t.min() < 160  # < window * (1 + 3 * advance[0])
    mixed = records[:5] + [short] + records[5:]
    with pytest.warns(UserWarning, match="1 of 13 records shorter than 160.0 d"):
        ds = FrameDataset(mixed, cfg, seed=0, epoch_seed=False)
    assert len(ds) == 12 and ds.indices == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
    assert all(r is mixed[j] for r, j in zip(ds.records, ds.indices))
    assert ds[5]["index"] == 6 and ds[5]["label"] == records[5].label
    assert np.array_equal(ds[4]["actions"], fixed[4]["actions"])  # seeded by j
    assert not np.array_equal(ds[5]["actions"], fixed[5]["actions"])  # j = 6
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            FrameDataset([short, short], cfg)


def test_collate_shapes_and_padding(records):
    cfg = frame_cfg()
    ds = FrameDataset(records, cfg, seed=0, epoch_seed=False)
    items = [ds[i] for i in range(4)]
    sizes = np.array([[len(f[0]) for f in item["frames"]] for item in items])
    assert len({tuple(row) for row in sizes}) > 1  # windows differ in size
    batch = collate_frames(items, band_wavelengths=WL, time_scale=0.5)
    assert set(batch) == {"frames", "actions", "n_tokens", "label", "index"}
    assert len(batch["frames"]) == 4
    for t, tok in enumerate(batch["frames"]):
        assert isinstance(tok, Tokens)
        n = sizes[:, t].max()
        assert tok.values.shape == (4, n, 1)
        assert tok.positions.shape == (4, 2, n) and tok.pad_mask.shape == (4, n)
        assert tok.n_real.tolist() == sizes[:, t].tolist()
        assert torch.equal(batch["n_tokens"][:, t], tok.n_real)
        real = tok.positions[:, 0][~tok.pad_mask]
        assert (real >= 1).all() and (real <= cfg.window / 0.5 + 1).all()
        assert (tok.positions[:, 0][tok.pad_mask] == 0).all()
    assert batch["actions"].shape == (4, 4, 1)
    assert batch["actions"].dtype == torch.float32
    assert np.allclose(batch["actions"].numpy(), [it["actions"] for it in items])
    assert batch["n_tokens"].shape == (4, 4) and batch["n_tokens"].dtype == torch.long
    assert batch["label"].dtype == batch["index"].dtype == torch.long
    assert batch["index"].tolist() == [0, 1, 2, 3]
    assert batch["label"].tolist() == [r.label for r in records[:4]]
    loader = DataLoader(
        ds,
        batch_size=4,
        num_workers=0,
        collate_fn=partial(collate_frames, band_wavelengths=WL, time_scale=0.5),
    )
    loaded = next(iter(loader))
    assert loaded["index"].tolist() == [0, 1, 2, 3]
    assert torch.equal(loaded["n_tokens"], batch["n_tokens"])
    for a, b in zip(loaded["frames"], batch["frames"]):
        assert torch.equal(a.values, b.values) and torch.equal(a.positions, b.positions)


def test_loader_epoch_refresh(records):
    cfg = frame_cfg()
    collate_fn = partial(collate_frames, band_wavelengths=WL)
    loader = DataLoader(FrameDataset(records, cfg), batch_size=4, collate_fn=collate_fn)
    first, second = next(iter(loader)), next(iter(loader))
    assert not torch.equal(first["actions"], second["actions"])
    fixed = FrameDataset(records, cfg, epoch_seed=False)
    loader = DataLoader(fixed, batch_size=4, collate_fn=collate_fn)
    first, second = next(iter(loader)), next(iter(loader))
    assert torch.equal(first["actions"], second["actions"])
    for a, b in zip(first["frames"], second["frames"]):
        assert torch.equal(a.values, b.values) and torch.equal(a.pad_mask, b.pad_mask)


def test_with_err_frames_carry_errors_as_extras():
    from romae_lc import RoMAE

    records = [normalize(r) for r in simulate(4, SURVEYS, CFG, seed=3)]
    cfg = FrameConfig(n_frames=3, window=30.0, with_err=True)
    rng = np.random.default_rng(0)
    frames, actions = sample_frames(records[0], cfg, rng)
    assert actions.shape == (3, 1)
    for t, y, band, err in frames:
        assert t.shape == y.shape == band.shape == err.shape
        assert err.dtype == np.float32 and (err > 0).all()
    # the error of every point is the record's error at that point
    t0, y0, _, e0 = frames[0]
    for ti, yi, ei in zip(t0, y0, e0):
        j = np.flatnonzero(np.isclose(records[0].y, yi))
        assert np.any(np.isclose(records[0].err[j], ei))
    grid, _ = frame_grid(records[0], cfg, fill=True)
    assert all(len(f) == 4 for f in grid)
    ds = FrameDataset(records, cfg, epoch_seed=False)
    batch = collate_frames([ds[i] for i in range(len(ds))], band_wavelengths=WL)
    for f in batch["frames"]:
        assert f.extras is not None and f.extras.shape == f.pad_mask.shape
        assert (f.extras[f.pad_mask] == 0).all() and (f.extras[~f.pad_mask] > 0).all()
    # triples without the flag are unchanged and carry no extras
    plain = FrameDataset(
        records, FrameConfig(n_frames=3, window=30.0), epoch_seed=False
    )
    batch = collate_frames([plain[0], plain[1]], band_wavelengths=WL)
    assert all(f.extras is None for f in batch["frames"])
    z = RoMAE(encoder=dict(d_model=24, nhead=2, depth=1))(*batch["frames"][0])
    assert z.shape == (2, 24)
