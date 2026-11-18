"""CPU tests of the latent cache (project/cache_latents.py) on the toy simulator: a
tiny autoencoder (``pretrain_mae``, 4 steps) is trained once per module and its window
grid is cached. The tests check the layout of the cache, the window starts, the extra
realisations and ``--max-objects`` / ``--seed``. Taken from the old test_predictor.py."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from project import cache_latents
from project.cache_latents import object_windows
from project.common import data_args_from, grid_starts, load_data, load_encoder
from project.tests._args import MAE_ARGS

STRIDE = 0.5


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def mae_ckpt(tmp_path_factory):
    from project import pretrain_mae

    out = tmp_path_factory.mktemp("mae")
    pretrain_mae.main(MAE_ARGS + ["--steps", "4", "--out", str(out)])
    assert (out / "mae.pt").is_file()
    return out / "mae.pt"


@pytest.fixture(scope="module")
def latents(mae_ckpt):
    out = mae_ckpt.parent / "latents.pt"
    cache_latents.main(
        [
            "--ckpt", str(mae_ckpt), "--out", str(out), "--stride", str(STRIDE),
            "--device", "cpu", "--workers", "0", "--batch-size", "16",
        ]  # fmt: skip
    )
    assert out.is_file() and out.with_suffix(".json").is_file()
    return out


def test_cache_layout_and_starts(latents, mae_ckpt):
    c = torch.load(latents, map_location="cpu", weights_only=False)
    z, n = c["z"], c["z"].shape[0]
    assert z.dtype == torch.float16 and z.shape[1] == 24 and n > 0
    assert torch.isfinite(z.float()).all()
    for key, dtype in dict(obj=torch.int32, win=torch.int16, start=torch.float32, n_tokens=torch.int16).items():
        assert c[key].dtype == dtype and c[key].shape == (n,), key
    assert set(c["splits"]) == {"train", "validation"}
    assert sum(cnt for _, cnt in c["splits"].values()) == n
    meta = c["meta"]
    assert meta["stride"] == STRIDE and meta["window"] == 30.0 and meta["dim"] == 24
    assert meta["min_tokens"] == 4 and meta["cap"] == 32 and meta["encoder_kind"] == "mae"
    assert (c["n_tokens"] >= 0).all() and (c["n_tokens"] >= meta["min_tokens"]).sum() > n // 2
    info = json.load(open(latents.with_suffix(".json")))
    assert info["counts"]["windows_total"] == n and info["stride"] == STRIDE

    enc, emeta = load_encoder(mae_ckpt)
    data = load_data(data_args_from(emeta.args))
    for split, (off, cnt) in c["splits"].items():
        ptr, objs = c["ptr"][split], c["objects"][split]
        n_obj = len(ptr) - 1
        assert ptr.dtype == torch.int64 and ptr[0] == off and ptr[-1] == off + cnt
        assert (ptr[1:] > ptr[:-1]).all()
        assert len(objs["label"]) == len(objs["period"]) == len(objs["superclass"]) == len(objs["index"]) == n_obj
        assert n_obj == len(data[split])  # no --max-objects: every record
        for i in range(n_obj):
            rows = slice(int(ptr[i]), int(ptr[i + 1]))
            rec = data[split][int(objs["index"][i])]
            starts = grid_starts(rec, emeta.cfg, advance=STRIDE)
            assert len(starts) == ptr[i + 1] - ptr[i]
            assert np.allclose(c["start"][rows].numpy(), starts, atol=1e-3)
            assert (c["win"][rows].long() == torch.arange(len(starts))).all()
            assert (c["obj"][rows] == objs["index"][i]).all()
            assert objs["label"][i] == rec.label

    # a stored row is what the frozen encoder gives the same window
    split, i, w = "validation", 1, 3
    ptr, objs = c["ptr"][split], c["objects"][split]
    rec = data[split][int(objs["index"][i])]
    starts, frames, n_tokens = object_windows(
        rec, emeta.cfg, STRIDE, emeta.cfg.max_tokens, meta["created"]["seed"], int(objs["index"][i])
    )
    row = int(ptr[i]) + w
    assert c["n_tokens"][row] == n_tokens[w] and len(frames[w][0]) <= 32
    ref = enc.encode([emeta.spec.tokens([frames[w]])])[0, 0]
    assert torch.allclose(z[row].float(), ref, atol=1e-2, rtol=1e-2)


def test_cache_realisations(latents, mae_ckpt):
    from project.train_predictor import LatentStore

    out = mae_ckpt.parent / "latents_real.pt"
    cache_latents.main(
        [
            "--ckpt", str(mae_ckpt), "--out", str(out), "--stride", str(STRIDE),
            "--device", "cpu", "--workers", "0", "--batch-size", "7", "--realisations", "3",
        ]  # fmt: skip
    )
    c, base = torch.load(out, map_location="cpu", weights_only=False), torch.load(latents, map_location="cpu", weights_only=False)
    n = c["z"].shape[0]
    # realisation 0 is the plain cache (different batch padding: float16 rounding only)
    assert torch.allclose(c["z"].float(), base["z"].float(), atol=2e-3, rtol=1e-2)
    assert c["z_alt"].shape == (2, n, 24) and torch.isfinite(c["z_alt"].float()).all()
    assert c["meta"]["realisations"] == 3 and c["meta"]["real_drop"] == 0.5
    d = (c["z_alt"].float() - c["z"].float()[None]).norm(dim=-1)
    assert (d > 0).float().mean() > 0.9  # a realisation moves the latent
    assert (c["z_alt"][0] != c["z_alt"][1]).any()  # and the two differ
    store = LatentStore(c, "validation", "cpu", int(c["meta"]["min_tokens"]))
    assert store.n_real == 3 and store.z_all.shape == (3, store.n, 24)
    rows = torch.arange(min(5, store.n))
    g = torch.Generator().manual_seed(0)
    got = store.gather(rows, g)
    assert got.shape == (len(rows), 24)
    ok = torch.stack([(got == store.z_all[k, rows]).all(-1) for k in range(3)]).any(0)
    assert ok.all()  # every gathered row is one of its realisations
    assert torch.isfinite(store.replicate_std(rows)).all() and store.replicate_std(rows).shape == (len(rows), 24)
    plain = LatentStore(base, "validation", "cpu", int(base["meta"]["min_tokens"]))
    assert plain.n_real == 1 and torch.equal(plain.gather(rows), plain.z[rows])
    assert torch.isnan(plain.replicate_std(rows)).all()


def test_cache_max_objects_and_seed(latents, mae_ckpt):
    # With --max-objects the obj column must still be the data[split]
    # position (what objects[split]["index"] holds), and --seed picks the
    # subset and the cap only: the records stay the checkpoint's.
    out = mae_ckpt.parent / "latents_sub.pt"
    cache_latents.main(
        [
            "--ckpt", str(mae_ckpt), "--out", str(out), "--stride", str(STRIDE),
            "--splits", "train", "--max-objects", "5", "--seed", "3",
            "--device", "cpu", "--workers", "0", "--batch-size", "16",
        ]  # fmt: skip
    )
    c = torch.load(out, map_location="cpu", weights_only=False)
    _, emeta = load_encoder(mae_ckpt)
    data = load_data(data_args_from(emeta.args))  # the checkpoint's seed
    ptr, objs = c["ptr"]["train"], c["objects"]["train"]
    idx = objs["index"]
    assert len(ptr) - 1 == 5 and c["meta"]["created"]["seed"] == 3
    assert (idx[1:] > idx[:-1]).all() and not torch.equal(idx, torch.arange(5))
    for i in range(5):
        rows = slice(int(ptr[i]), int(ptr[i + 1]))
        assert (c["obj"][rows] == idx[i]).all()
        rec = data["train"][int(idx[i])]
        assert objs["label"][i] == rec.label
        assert objs["period"][i] == pytest.approx(float(rec.period), rel=1e-6)
        assert len(grid_starts(rec, emeta.cfg, advance=STRIDE)) == ptr[i + 1] - ptr[i]
