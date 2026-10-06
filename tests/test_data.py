"""Records: simulator, PHOEBE rows, normalisation, views, dataset and collate."""

from __future__ import annotations

import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from romae_lc.data import (
    CLASSES,
    PHOEBE_BANDS,
    PHOEBE_MORPHOLOGIES,
    LightCurveDataset,
    Record,
    SimConfig,
    SurveyConfig,
    ViewConfig,
    collate,
    load_phoebe,
    make_views,
    normalize,
    sample_epochs,
    sample_view,
    simulate,
    time_series,
)
from romae_lc.tokenize import Tokens

SURVEYS = (SurveyConfig("g", 480.7, 40, 0.02), SurveyConfig("r", 622.1, 25, 0.03))
CFG = SimConfig(baseline_days=100.0)
WL = {i: s.wavelength_nm for i, s in enumerate(SURVEYS)}


@pytest.fixture(scope="module")
def records():
    return simulate(12, SURVEYS, CFG, seed=0)


def same(a: Record, b: Record) -> bool:
    keys = ("t", "y", "err", "band")
    arrays = all(np.array_equal(getattr(a, k), getattr(b, k)) for k in keys)
    return arrays and (a.label, a.period, a.meta) == (b.label, b.period, b.meta)


def test_simulate_is_deterministic(records):
    again = simulate(12, SURVEYS, CFG, seed=0)
    assert all(same(a, b) for a, b in zip(records, again))
    assert not same(records[0], simulate(1, SURVEYS, CFG, seed=1)[0])
    drw = simulate(4, SURVEYS, SimConfig(100.0, class_probs=(0, 0, 0, 0, 1)))
    assert [r.label for r in drw] == [CLASSES.index("drw")] * 4


def test_record_invariants(records):
    assert {r.label for r in records} <= set(range(len(CLASSES)))
    for r in records:
        assert r.n == 65 and r.bands.tolist() == [0, 1]
        assert all(a.dtype == np.float32 for a in (r.t, r.y, r.err))
        assert r.band.dtype == np.int64 and r.band.tolist() == [0] * 40 + [1] * 25
        assert 0 <= r.t.min() and r.t.max() <= 100
        assert np.isclose(r.period, 10 ** r.meta["logP"])
        assert set(r.meta) == {"logP", "amp", "color", "fine", "shape"}
        for b, s in zip(r.bands, SURVEYS):
            m = r.band == b
            assert (np.diff(r.t[m]) >= 0).all()
            assert 0.7 * s.sigma <= r.err[m].min() and r.err[m].max() <= 1.3 * s.sigma
            assert np.isfinite(r.y[m]).all()


def test_sample_epochs_stay_inside_seasons():
    rng = np.random.default_rng(0)
    t = sample_epochs(rng, 60, 1000.0, season_len=240.0, year=365.25)
    assert t.shape == (60,) and (np.diff(t) > 0).all() and 0 <= t[0] and t[-1] <= 1000
    assert np.diff(t).max() >= 365.25 - 240.0  # at least one inter-season gap
    topped = sample_epochs(rng, 60, 1000.0, season_len=10.0, year=365.25)
    assert topped.shape == (60,) and (np.diff(topped) > 0).all()


def test_normalize_per_band_and_per_object(records):
    r = records[0]
    y_before = r.y.copy()
    n = normalize(r)
    assert n is not r and np.array_equal(r.y, y_before) and np.array_equal(n.t, r.t)
    for b in n.bands:
        m = n.band == b
        assert abs(np.median(n.y[m])) < 1e-6
        assert np.isclose(1.4826 * np.median(np.abs(n.y[m])), 1.0, atol=1e-3)
        ratio = n.err[m] / r.err[m]  # the shared 1 / scale
        assert np.allclose(ratio, ratio[0])
        assert np.allclose((r.y[m] - np.median(r.y[m])) * ratio, n.y[m], atol=1e-5)
    whole = normalize(r, mode="object")
    assert abs(np.median(whole.y)) < 1e-6 and whole.y.dtype == np.float32
    with pytest.raises(ValueError):
        normalize(r, mode="star")


def test_normalize_zero_mad_fallbacks():
    def rec(y, band):
        y = np.asarray(y, dtype=np.float32)
        t = np.arange(len(y), dtype=np.float32)
        return Record(t, y, np.full_like(y, 0.1), np.asarray(band), 0, 1.0)

    quantised = normalize(rec([1, 1, 1, 1, 2, 1, 1, 1, 3], [0] * 9))  # MAD 0
    assert np.abs(quantised.y).max() < 10 and quantised.err.max() < 10
    y = np.array([1, 1, 1, 1, 2, 1, 1, 1, 3], dtype=np.float32)
    assert np.allclose(quantised.y, (y - 1) / y.std(), atol=1e-5)
    single = normalize(rec([5.0, 1.0, 2.0, 4.0], [0, 1, 1, 1]))  # one-point band
    assert single.y[0] == 0 and np.isclose(single.err[0], 0.1)
    assert np.isclose(1.4826 * np.median(np.abs(single.y[1:])), 1.0, atol=1e-3)
    flat = normalize(rec([2.0] * 5, [0] * 5), mode="object")  # constant band
    assert (flat.y == 0).all() and np.allclose(flat.err, 0.1)


def window_indices(record: Record, y: np.ndarray) -> np.ndarray:
    """Indices of the record points an unaugmented view was cut from."""
    return np.array([np.flatnonzero(record.y == v)[0] for v in y])


def dense_record() -> Record:
    """Two bands sampled daily / every other day over 100 d, no gaps."""
    t = np.concatenate([np.arange(101.0), np.arange(0.5, 100.0, 2.0)])
    y = np.random.default_rng(0).normal(size=t.size).astype(np.float32)
    band = np.repeat([0, 1], [101, 50])
    return Record(t.astype(np.float32), y, np.full_like(y, 0.1), band, 0, 1.0)


def test_views_stay_inside_the_window():
    r, rng = dense_record(), np.random.default_rng(0)
    cfg = ViewConfig(max_tokens=1000, min_tokens=4)
    for _ in range(10):
        t, y, band = sample_view(r, rng, (0.2, 0.4), cfg, augment=False)
        assert (t.dtype, y.dtype, band.dtype) == ("f4", "f4", "i8")
        assert 30 <= len(t) == len(y) == len(band) <= 62
        assert 0 <= t.min() and t.max() <= 40.0 + 1e-4
        src = window_indices(r, y)
        assert np.array_equal(r.band[src], band)
        assert np.allclose(r.t[src] - r.t[src[0]], t - t[0], atol=1e-3)
    t, y, band = sample_view(r, rng, (1.0, 1.0), cfg, augment=False)
    assert np.array_equal(t, r.t) and np.array_equal(y, r.y)
    assert np.array_equal(band, r.band)
    t, y, band = sample_view(r, rng, (0.3, 0.3), cfg)  # augmented: resampled y
    assert 0 <= t.min() and t.max() <= 30.0 + 1e-4
    assert not np.isin(y, r.y).all() and np.isin(band, r.bands).all()
    t, _, _ = sample_view(r, rng, (0.2, 0.2), ViewConfig(min_tokens=r.n + 1))
    assert np.array_equal(t, r.t)  # too few points: whole-curve fallback
    t, y, _ = sample_view(r, rng, (1.0, 1.0), ViewConfig(max_tokens=20), augment=False)
    assert len(t) == 20 and (np.diff(window_indices(r, y)) > 0).all()


def test_make_views_counts_and_lengths(records):
    r, rng = records[2], np.random.default_rng(1)
    cfg = ViewConfig(n_global=2, n_local=3, max_tokens=30, min_tokens=2)
    glob, loc = make_views(r, cfg, rng)
    assert len(glob) == 2 and len(loc) == 3
    span = r.t.max() - r.t.min()
    for t, y, band in glob + loc:
        assert len(t) == len(y) == len(band) <= 30 and t.min() >= 0
    assert all(t.max() <= 0.4 * span + 1e-4 for t, _, _ in loc)


def test_dataset_items_and_seeding(records):
    cfg = ViewConfig(max_tokens=30)
    ds = LightCurveDataset(records, view_cfg=cfg, seed=0)
    item = ds[3]
    assert len(ds) == 12 and set(item) == {"views", "full", "label", "index"}
    t, y, band = item["full"]  # capped, times relative to the first epoch
    src = window_indices(records[3], y)
    assert len(t) == 30 and np.allclose(records[3].t[src] - records[3].t.min(), t)
    assert item["label"] == records[3].label and item["index"] == 3
    assert len(item["views"][0]) == 2 and len(item["views"][1]) == 4
    assert not np.array_equal(item["views"][0][0][0], ds[3]["views"][0][0][0])
    fixed = LightCurveDataset(records, view_cfg=cfg, seed=0, epoch_seed=False)
    assert np.array_equal(fixed[3]["views"][0][0][0], fixed[3]["views"][0][0][0])
    plain = LightCurveDataset(records)[0]
    assert plain["views"] is None and len(plain["full"][0]) == records[0].n


def test_full_curve_is_whole_without_a_cap():
    recs = simulate(300, seed=0)  # 600 points each, more than the default cap
    assert all(r.n > ViewConfig().max_tokens for r in recs)
    ds = LightCurveDataset(recs, epoch_seed=False)
    for i in (0, 26, 79, 150):  # 26 and 79 lost their last epoch in float32
        t, y, band = ds[i]["full"]
        assert len(t) == recs[i].n and np.array_equal(y, recs[i].y)
        assert np.array_equal(band, recs[i].band)
        assert np.allclose(t, recs[i].t - recs[i].t.min(), atol=1e-4)
        assert np.isclose(t.max(), recs[i].t.max() - recs[i].t.min(), atol=1e-4)
    huge = ViewConfig(max_tokens=10**6)
    assert all(len(ds[i]["full"][0]) == r.n for i, r in enumerate(recs))
    with_cfg = LightCurveDataset(recs, huge, epoch_seed=False)
    assert all(len(with_cfg[i]["full"][0]) == r.n for i, r in enumerate(recs))
    capped = LightCurveDataset(recs, ViewConfig(), epoch_seed=False)[0]["full"]
    assert len(capped[0]) == 512  # a view_cfg caps full at its max_tokens
    explicit = LightCurveDataset(recs, ViewConfig(), max_tokens=100)[0]["full"]
    assert len(explicit[0]) == 100
    explicit = LightCurveDataset(recs, max_tokens=100)[0]["full"]
    assert len(explicit[0]) == 100 and np.isin(explicit[1], recs[0].y).all()


def test_collate_shapes_and_padding(records):
    cfg = ViewConfig(max_tokens=30)
    ds = LightCurveDataset(records, view_cfg=cfg)
    batch = collate([ds[i] for i in range(4)], band_wavelengths=WL, time_scale=0.5)
    assert set(batch) == {"global", "local", "full", "label", "index"}
    assert len(batch["global"]) == 2 and len(batch["local"]) == 4
    for tok in batch["global"] + batch["local"] + [batch["full"]]:
        assert isinstance(tok, Tokens)
        b, n, c = tok.values.shape
        assert (b, c) == (4, 1) and n <= 30
        assert tok.positions.shape == (4, 2, n) and tok.pad_mask.shape == (4, n)
    assert batch["full"].values.shape == (4, 30, 1)
    assert (batch["full"].positions[:, 0] >= 1).all()  # t / 0.5 + 1, no padding
    assert batch["label"].dtype == batch["index"].dtype == torch.long
    assert batch["index"].tolist() == [0, 1, 2, 3]
    assert batch["label"].tolist() == [r.label for r in records[:4]]
    mixed = records[:2] + simulate(2, SURVEYS[:1], CFG, seed=2)
    loader = DataLoader(
        LightCurveDataset(mixed),
        batch_size=4,
        collate_fn=partial(collate, band_wavelengths=WL),
    )
    batch = next(iter(loader))
    assert batch["global"] == [] and batch["local"] == []
    assert batch["full"].pad_mask.shape == (4, 65)
    assert batch["full"].n_real.tolist() == [65, 65, 40, 40]
    assert (batch["full"].positions[2:, :, 40:] == 0).all()


def test_time_series_lists(records):
    times, values, bands = time_series(records[:3])
    assert len(times) == len(values) == len(bands) == 3
    assert times[1] is records[1].t and bands[2] is records[2].band


class FakeDataset(list):
    """The slice of ``datasets.Dataset`` that :func:`load_phoebe` touches."""

    def select(self, idx):
        return FakeDataset(self[i] for i in idx)

    def select_columns(self, cols):
        return FakeDataset({k: row[k] for k in cols} for row in self)

    def with_format(self, fmt):
        return self


def phoebe_rows(n: int) -> FakeDataset:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = dict(id=i, period=1.0 + i, t0=0.5, morphology=PHOEBE_MORPHOLOGIES[i % 3])
        for b in PHOEBE_BANDS:
            k = 5 + i
            row[f"{b}_time"] = np.sort(rng.uniform(0, 10, k))
            row[f"{b}_flux"] = rng.normal(size=k)
            row[f"{b}_flux_err"] = np.full(k, 0.1)
        rows.append(row)
    return FakeDataset(rows)


def test_load_phoebe_parses_rows(monkeypatch, tmp_path):
    rows = phoebe_rows(3)
    stub = SimpleNamespace(
        load_from_disk=lambda path: {"train": rows},
        load_dataset=lambda source, split: rows,
    )
    monkeypatch.setitem(sys.modules, "datasets", stub)
    records = load_phoebe(str(tmp_path), max_rows=2)
    assert len(records) == 2
    r = records[1]
    assert r.n == 6 * len(PHOEBE_BANDS) and r.label == 1 and r.period == 2.0
    assert r.meta == dict(id=1, t0=0.5, morphology="detached")
    assert r.band.tolist() == np.repeat(np.arange(5), 6).tolist()
    assert np.allclose(r.t[r.band == 2], rows[1]["LSST_r_time"])
    assert np.allclose(r.y[r.band == 4], rows[1]["TESS_T_flux"])
    assert (r.t.dtype, r.y.dtype, r.err.dtype) == ("f4", "f4", "f4")
    remote = load_phoebe("someone/phoebe", split="test", bands=PHOEBE_BANDS[:2])
    assert len(remote) == 3 and remote[0].n == 10 and remote[0].bands.tolist() == [0, 1]


def pc_dataset_dir(tmp_path):
    """A two-split PC_matches-style DatasetDict written to disk."""
    datasets = pytest.importorskip("datasets")
    rng = np.random.default_rng(0)

    def rows(n, offset):
        out = dict(
            gaia_dr3_source_id=[str(1000 + offset + i) for i in range(n)],
            period=[0.3 + 0.1 * i for i in range(n)],
            class_str=["EW/EB", "RRAB", "DSCT", "LPV"][:n],
            superclass_str=["ECL", "RR", "DSCT", "LPV"][:n],
            lightcurve=[],
        )
        for i in range(n):
            lc = {}
            for b, k in (("g_ZTF", 6), ("r_ZTF", 4), ("i_ZTF", 0)):
                t = np.sort(rng.uniform(58000, 58100, k))
                lc[b] = dict(
                    mjd=t.tolist(),
                    mag=(15 + rng.normal(size=k)).tolist(),
                    mag_unc=np.full(k, 0.05).tolist(),
                    clean=([True] * k if k else []),
                    mag_sys="AB",
                )
            if i == 1:  # one bad point and one unclean point in r
                lc["r_ZTF"]["mag"][0] = float("nan")
                lc["r_ZTF"]["clean"][1] = False
            out["lightcurve"].append(lc)
        return datasets.Dataset.from_dict(out)

    dd = datasets.DatasetDict(train=rows(4, 0), validation=rows(2, 10))
    dd.save_to_disk(str(tmp_path / "FAKExPC"))
    return str(tmp_path / "FAKExPC"), dd


def test_load_pc_reads_arrow_columns(tmp_path):
    from romae_lc.data import PC_BANDS, PC_SUPERCLASSES, load_pc
    from romae_lc.tokenize import wavelengths_for

    path, dd = pc_dataset_dir(tmp_path)
    records = load_pc(path, "train")
    assert len(records) == 4
    r0, r1 = records[0], records[1]
    assert r0.n == 10 and r0.bands.tolist() == [0, 1]  # i_ZTF is empty
    assert r0.label == PC_SUPERCLASSES.index("ECL") and r0.period == 0.3
    assert r0.meta["class_str"] == "EW/EB" and r0.meta["id"] == "1000"
    row = dd["train"][0]["lightcurve"]
    t_all = np.concatenate([row["g_ZTF"]["mjd"], row["r_ZTF"]["mjd"]])
    assert r0.meta["t0"] == t_all.min() and r0.t.min() == 0.0
    assert np.allclose(r0.t[r0.band == 1] + r0.meta["t0"], row["r_ZTF"]["mjd"])
    assert np.allclose(r0.y[r0.band == 0], -np.array(row["g_ZTF"]["mag"]))
    assert (r0.t.dtype, r0.y.dtype, r0.err.dtype) == ("f4", "f4", "f4")
    assert r1.n == 8  # the NaN and the unclean r points are gone
    assert load_pc(path, "train", clean_only=False)[1].n == 9  # NaN still gone
    # every band id indexes PC_BANDS, so the tokenizer table is the same
    # for every sub-dataset
    assert all(PC_BANDS[b] in ("g_ZTF", "r_ZTF") for b in r0.bands)
    assert set(wavelengths_for(PC_BANDS)) >= set(r0.bands.tolist())
    val = load_pc(path, "validation", max_rows=1, min_points=11)
    assert val == []
    fine = load_pc(path, "train", label_field="class_str", classes=("RRAB", "LPV"))
    assert [r.label for r in fine] == [0, 1]
    assert [r.period for r in fine] == pytest.approx([0.4, 0.6])
    with pytest.raises(ValueError, match="none of"):
        load_pc(path, "train", bands=("TESS",))


def test_normalize_past_only_reference():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 3000, 400)).astype(np.float32)
    y = (np.sin(t) + np.where(t > 1500, 5.0, 0.0)).astype(np.float32)  # a jump after day 1500
    r = Record(t, y, np.full_like(y, 0.1), np.zeros(400, dtype=np.int64), 0, 1.0)
    whole = normalize(r)
    past = normalize(r, ref_days=1000.0)
    early = t <= 1000
    # the past-only statistics centre the early part, not the whole curve
    assert abs(np.median(past.y[early])) < 0.05 and abs(np.median(whole.y[early])) > 0.5
    assert np.isclose(1.4826 * np.median(np.abs(past.y[early] - np.median(past.y[early]))), 1.0, atol=0.05)
    # later points are transformed with the same early statistics, so the jump stays visible
    assert np.median(past.y[t > 1500]) > 3.0
    # too few early points: the reference grows to the first ref_min points in time
    r2 = Record(t, y, np.full_like(y, 0.1), np.zeros(400, dtype=np.int64), 0, 1.0)
    tiny = normalize(r2, ref_days=1.0, ref_min=8)
    first8 = np.argsort(t)[:8]
    assert abs(np.median(tiny.y[first8])) < 1e-5
