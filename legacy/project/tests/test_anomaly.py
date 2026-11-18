"""CPU tests of the anomaly test on the frozen pipeline (toy simulator): a
tiny stage-1 autoencoder, its cached window grid, a tiny sequence predictor,
then :mod:`project.anomaly` end to end, plus unit tests of its pieces."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from project import anomaly, cache_latents, train_predictor
from project.train_predictor import load_predictor

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip

STRIDE = 0.5


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def pred(tmp_path_factory):
    from project import pretrain_mae

    out = tmp_path_factory.mktemp("anom")
    pretrain_mae.main(MAE_ARGS + ["--steps", "4", "--out", str(out / "mae")])
    cache_latents.main(
        [
            "--ckpt", str(out / "mae" / "mae.pt"), "--out", str(out / "latents.pt"),
            "--stride", str(STRIDE), "--device", "cpu", "--workers", "0", "--batch-size", "16",
        ]  # fmt: skip
    )
    train_predictor.main(
        [
            "--latents", str(out / "latents.pt"), "--out", str(out / "pred"), "--arch", "seq",
            "--kind", "flow", "--steps", "10", "--batch-size", "4", "--max-len", "8",
            "--seq-hidden", "16", "--seq-depth", "1", "--seq-heads", "2", "--seq-dim-head", "8",
            "--seq-mlp", "32", "--hidden", "16", "--depth", "1", "--eval-every", "10",
            "--ckpt-every", "10", "--log-every", "5", "--val-objects", "8", "--eval-samples", "2",
            "--n-euler", "4", "--device", "cpu",
        ]  # fmt: skip
    )
    return out / "pred" / "pred.pt"


def test_anomaly_end_to_end(pred, tmp_path):
    out = tmp_path / "anomaly"
    anomaly.main(
        [
            "--pred", str(pred), "--out", str(out), "--n-objects", "6", "--kinds", "phase", "bump",
            "--nll-steps", "4", "--eval-samples", "2", "--device", "cpu",
            "--data", "sim", "--n-sim", "48",
        ]  # fmt: skip
    )
    for name in ("summary.json", "tables.md", "profiles.png", "objects.npz"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    res = json.load(open(out / "summary.json"))
    assert res["kinds"] == ["phase", "bump"] and res["grid_steps_per_window"] == 2
    for kind in res["kinds"]:
        pooled = res["results"][kind]["pooled"]
        assert set(pooled) == set(anomaly.SCORES)
        for k, s in pooled.items():
            assert s["n"] > 0, (kind, k)
            for key in ("auroc_object", "auroc_window", "auroc_delta", "hit", "hit_chance"):
                assert 0.0 <= s[key] <= 1.0, (kind, k, key)
    text = open(out / "tables.md").read()
    assert "## phase" in text and "## bump" in text


def test_states_long_matches_short(pred):
    model, _ = load_predictor(pred)
    torch.manual_seed(0)
    z = torch.randn(6, model.dim)
    gaps = torch.tensor([0.0, 0.5, 0.5, 1.0, 0.5, 0.5])
    h = anomaly.states_long(model, z, gaps)
    mask = torch.ones(1, 6, dtype=torch.bool)
    assert torch.allclose(h, model.states(z[None], gaps[None], mask)[0], atol=1e-6)
    # a sequence longer than max_len is read in pieces: every position gets a state
    v = int(model.max_len) + 5
    z = torch.randn(v, model.dim)
    gaps = torch.full((v,), 0.5)
    gaps[0] = 0.0
    h = anomaly.states_long(model, z, gaps)
    assert h.shape[0] == v and torch.isfinite(h).all() and (h.abs().sum(1) > 0).all()
    # the first piece is the plain read of the first max_len windows
    m = int(model.max_len)
    mask = torch.ones(1, m, dtype=torch.bool)
    assert torch.allclose(h[: m // 2], model.states(z[None, :m], gaps[None, :m], mask)[0][: m // 2], atol=1e-6)


def test_summarize_finds_a_spike():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        n = 30
        clean = rng.normal(0, 1, n)
        event = clean + rng.normal(0, 0.1, n)
        ks = int(rng.integers(6, 24))
        event[ks - 1 : ks + 3] += 5.0
        bad = np.full(n, np.nan)
        rows.append(dict(clean=dict(a=clean, b=clean, c=bad), event=dict(a=event, b=clean.copy(), c=bad), k_star=ks))
    s = anomaly.summarize(rows, "a", w=2)
    assert s["n"] == 40 and s["hit"] > 0.9 and s["auroc_delta"] > 0.95 and s["auroc_object"] > 0.9
    assert 0 < s["hit_chance"] < 0.5
    flat = anomaly.summarize(rows, "b", w=2)  # no event in the score: chance
    assert abs(flat["auroc_object"] - 0.5) < 1e-9
    none = anomaly.summarize(rows, "c", w=2)  # nothing scored
    assert none["n"] == 0 and math.isnan(none["hit"])


def test_feature_scale():
    f = np.array([[0.0, 1.0], [1.0, 1.0], [3.0, np.nan], [4.0, 1.0]])
    s = anomaly.feature_scale(f)
    assert s.shape == (2,) and abs(s[0] - 1.001) < 1e-9 and s[1] > 0
    assert (anomaly.feature_scale(f[:1]) == 1.0).all()
