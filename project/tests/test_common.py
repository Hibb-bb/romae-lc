"""CPU tests of the shared modules (data, windows, backbone, checkpoint, probes) on the toy simulator (seconds each): the error
channel and the collate, the window grid, the data args, the dense rotary ladder,
the backbone layout and the probe metrics."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from romae_lc import FrameConfig
from romae_lc.data import SimConfig, SurveyConfig, normalize, simulate

from project.data import data_args_from, subset
from project.windows import err_channel, err_stats, grid_starts, TokenSpec

SURVEYS = (SurveyConfig("g", 480.7, 160, 0.05), SurveyConfig("r", 622.1, 120, 0.06))
SIM = SimConfig(baseline_days=400.0, logP_range=(-0.3, 1.0))
WL = {i: s.wavelength_nm for i, s in enumerate(SURVEYS)}


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def records():
    recs = [normalize(r) for r in simulate(24, SURVEYS, SIM, seed=1)]
    for r in recs:
        r.meta["class_str"] = r.meta["superclass_str"] = f"c{r.label}"
    return recs


@pytest.fixture(scope="module")
def spec(records):
    return TokenSpec(dict(band_wavelengths=WL, time_scale=0.05), err_stats(records))


@pytest.fixture(scope="module")
def cfg():
    return FrameConfig(
        n_frames=3, window=30.0, min_tokens=4, max_tokens=64, with_err=True
    )


def test_err_channel_and_collate(records, spec, cfg):
    from romae_lc import FrameDataset

    ds = FrameDataset(records, cfg, epoch_seed=False)
    batch = spec.collate()([ds[0], ds[1]])
    for f in batch["frames"]:
        assert f.values.shape[-1] == 2 and f.extras is not None
        assert (f.values[f.pad_mask] == 0).all()
        real = ~f.pad_mask
        assert torch.isfinite(f.values[real]).all() and (f.extras[real] > 0).all()
    tok = spec.tokens([ds[0]["frames"][0], ds[1]["frames"][0]])
    assert tok.values.shape[-1] == 2
    single = TokenSpec(spec.tokenize, None)
    assert (
        single.n_channels == 1
        and single.collate()([ds[0]])["frames"][0].values.shape[-1] == 1
    )
    assert TokenSpec.from_dict(spec.to_dict()).err_stats == spec.err_stats
    with pytest.raises(ValueError):
        err_channel(
            type(tok)(tok.values, tok.positions, tok.pad_mask, None), spec.err_stats
        )


def test_grid_starts_and_subset(records, cfg):
    from romae_lc import frame_grid

    r = records[0]
    frames, actions = frame_grid(r, cfg, fill=True)
    starts = grid_starts(r, cfg)
    assert len(frames) == len(starts) and actions.shape == (len(starts), 1)
    for f in frames:
        assert len(f) == 4
    assert (
        subset(records, 5, 0) == subset(records, 5, 0) and len(subset(records, 5)) == 5
    )


def test_data_args_from():
    saved = dict(data="x", classes=None, max_rows=3, min_points=8, n_sim=4, seed=1)
    ns = data_args_from(saved, argparse.Namespace(data=None, max_rows=7))
    assert ns.data == "x" and ns.max_rows == 7 and ns.seed == 1


def test_dense_ladder_dealing_and_geometry():
    from project.backbone import check_ladder, deal_ladder, Ladder, _nest, resolve_model_args, rope_geometry, RopeGeometry

    w = np.geomspace(0.1, 100.0, 12)
    bands = np.array(deal_ladder(w, 2, 2, 3, "bands"))
    comb = np.array(deal_ladder(w, 2, 2, 3, "comb"))
    for dealt in (bands, comb):
        assert dealt.shape == (2, 2, 3)
        assert np.allclose(np.sort(dealt.ravel()), w)  # every rung exactly once
        for layer in dealt:  # every layer spans the whole range
            assert layer.min() <= w[1] and layer.max() >= w[-2]
    assert bands[0, 0].max() < bands[0, 1].min()  # bands: contiguous per head
    assert comb[0, 0].max() > comb[0, 1].min()  # comb: heads interleave
    assert np.array(deal_ladder(w[:3], 1, 1, 3)).shape == (1, 1, 3)
    with pytest.raises(ValueError):
        deal_ladder(w, 2, 2, 2, "bands")
    with pytest.raises(ValueError):
        deal_ladder(w, 2, 2, 3, "spiral")

    def ns(size):
        return resolve_model_args(
            argparse.Namespace(
                size=size,
                width=None,
                heads=None,
                depth=None,
                time_frac=0.875,
                p_rope=0.75,
            )
        )

    geo = rope_geometry(ns("light"))
    assert (geo.layers, geo.heads, geo.head_dim, geo.time_dim, geo.n_angles) == (
        6,
        3,
        64,
        56,
        21,
    )
    assert geo.n_rungs == 378 and rope_geometry(ns("wide")).n_rungs == 756
    assert (
        rope_geometry(
            argparse.Namespace(width=192, heads=3, depth=6, time_frac=None, p_rope=0.75)
        ).n_angles
        == 12
    )  # the old even split
    ladder = Ladder(
        0.01,
        _nest(bands / 0.0628, 2, 2),
        _nest(bands, 2, 2),
        0.1,
        100.0,
        "log",
        "",
        2,
        2,
        "bands",
    )
    assert ladder.n_rungs == 12 and ladder.per_head == 3
    assert ladder.flat == pytest.approx(w.tolist())
    check_ladder(ladder, RopeGeometry(2, 2, 12, 8, 3))
    with pytest.raises(ValueError, match="layers"):
        check_ladder(ladder, RopeGeometry(3, 2, 12, 8, 3))
    with pytest.raises(ValueError, match="heads"):
        check_ladder(ladder, RopeGeometry(2, 3, 12, 8, 3))
    with pytest.raises(ValueError, match="do not fit"):
        check_ladder(ladder, RopeGeometry(2, 2, 4, 4, 1))
    old = Ladder.from_dict(
        dict(time_scale=0.01, timescales=[1.0, 2.0], wavelengths=[0.06, 0.13], lam_min=0.06, lam_max=0.13, spacing="log")
    )  # fmt: skip
    assert old.layers == old.heads == 1 and old.deal == "shared" and old.n_rungs == 2
    assert _nest(np.ones((1, 1, 3)), 1, 1) == [1.0, 1.0, 1.0]
    assert np.array(_nest(np.ones((1, 2, 3)), 1, 2)).shape == (2, 3)


def test_dense_backbone_layout(records, spec):
    from romae_lc import RoMAE

    from project.backbone import dense_ladder, encoder_config, rope_geometry, rope_layouts

    def build_backbone(args, ladder, n_channels):
        return RoMAE(encoder=encoder_config(args), n_channels=n_channels, n_axes=2, rope=rope_layouts(args, ladder))

    args = argparse.Namespace(
        width=24, heads=2, depth=2, mlp_ratio=4.0, time_frac=0.875, p_rope=0.75, attention="softmax"
    )  # fmt: skip
    geo = rope_geometry(args)
    assert (geo.time_dim, geo.n_angles, geo.n_rungs) == (8, 3, 12)
    ladder = dense_ladder(
        records, geo, 30.0, "bands", "log", 0.75, 50.0, 8, 0, None, None
    )
    assert ladder.layers == 2 and ladder.heads == 2 and ladder.n_rungs == 12
    assert ladder.lam_max == 60.0 and ladder.deal == "bands"
    backbone = build_backbone(args, ladder, 2)
    assert backbone.per_layer_rope and len(backbone.rope_layers) == 2
    assert backbone.rope.blocks[0].per_head and backbone.rope.dims == [8, 4]
    got = sorted(
        float(v)
        for b in backbone.rope_layers
        for v in b.blocks[0].timescale[torch.isfinite(b.blocks[0].timescale)]
    )
    assert got == pytest.approx(sorted(np.asarray(ladder.timescales).ravel().tolist()))
    shared = build_backbone(
        args, dense_ladder(records, rope_geometry(argparse.Namespace(**dict(vars(args), depth=1))), 30.0, "comb", "log", 0.75, 50.0, 8, 0, None, None), 2
    )  # fmt: skip
    assert shared.per_layer_rope is False and shared.rope.blocks[0].per_head
    with pytest.raises(ValueError, match="layers"):
        build_backbone(argparse.Namespace(**dict(vars(args), depth=3)), ladder, 2)


def test_probe_metrics_and_baseline(records):
    from project.probes import balanced_accuracy, baseline_probe, describe_by_class, describe_probe, hand_features, macro_f1, probe_metrics

    y, pred = np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 1, 0])
    assert macro_f1(y, pred) == pytest.approx((4 / 6 + 4 / 5 + 0.0) / 3)
    assert balanced_accuracy(y, pred) == pytest.approx((2 / 3 + 1.0 + 0.0) / 3)
    tr, va = records[:16], records[16:]
    bands = sorted({int(b) for r in records for b in r.bands})
    x = hand_features(tr[0], bands)
    assert x.shape == (4 + 10 * len(bands),) and np.isfinite(x).all()
    x_tr = np.stack([hand_features(r, bands) for r in tr])
    x_va = np.stack([hand_features(r, bands) for r in va])
    m = probe_metrics(x_tr, tr, x_va, va, 5, min_train=1, min_val=1)
    for k in (
        "acc",
        "train_acc",
        "macro_f1",
        "balanced_acc",
        "majority_acc",
        "r2",
        "r2_within",
    ):
        assert k in m and np.isfinite(m[k])
    assert 0 <= m["acc"] <= 1 and 0 <= m["macro_f1"] <= 1 and m["n_val"] == len(va)
    assert set(m["n_by_superclass"]) == {f"c{r.label}" for r in va}
    assert sum(m["n_by_superclass"].values()) == len(va)
    assert baseline_probe(tr, va, bands, 5).keys() == m.keys()
    line = describe_by_class(
        {"ECL": 0.5, "ROT": 0.1, "X": 0.2}, {"ECL": 10, "ROT": 5, "X": 1}
    )
    assert line == "ROT 0.10/5 ECL 0.50/10 X 0.20/1"
    assert describe_probe(m).startswith("probe acc")
