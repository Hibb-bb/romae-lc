"""CPU tests of the latent cache (project/eval/encode.py) on the toy simulator: a
tiny autoencoder (``project.train``, 4 steps) is trained once per module and its window
grid is cached. The tests check the layout of the cache, the window starts and
``--max-objects`` / ``--seed``."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from project.eval import encode
from project.windows import object_windows
from project.checkpoint import load_encoder
from project.data import data_args_from, load_data
from project.windows import grid_starts
from project.tests._args import MAE_ARGS

STRIDE = 0.5


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def mae_ckpt(tmp_path_factory):
    from project import train

    out = tmp_path_factory.mktemp("mae")
    train.main(MAE_ARGS + ["--steps", "4", "--out", str(out)])
    assert (out / "mae.pt").is_file()
    return out / "mae.pt"


@pytest.fixture(scope="module")
def latents(mae_ckpt):
    out = mae_ckpt.parent / "latents.pt"
    encode.main(
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


def test_cache_max_objects_and_seed(latents, mae_ckpt):
    # With --max-objects the obj column must still be the data[split]
    # position (what objects[split]["index"] holds), and --seed picks the
    # subset and the cap only: the records stay the checkpoint's.
    out = mae_ckpt.parent / "latents_sub.pt"
    encode.main(
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
