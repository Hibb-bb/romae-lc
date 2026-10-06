"""CPU test of the end-to-end brightness forecast (toy simulator): a tiny
encoder, its cache, a tiny sequence predictor and a tiny decoder, then
:mod:`project.forecast` on a few stars."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from project import cache_latents, forecast, lookback, train_decoder, train_predictor

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    from project import pretrain_mae

    out = tmp_path_factory.mktemp("fc")
    pretrain_mae.main(MAE_ARGS + ["--steps", "4", "--out", str(out / "mae")])
    mae = out / "mae" / "mae.pt"
    cache_latents.main(["--ckpt", str(mae), "--out", str(out / "latents.pt"), "--stride", "0.5",
                        "--device", "cpu", "--workers", "0", "--batch-size", "16"])  # fmt: skip
    train_predictor.main(
        ["--latents", str(out / "latents.pt"), "--out", str(out / "pred"), "--arch", "seq", "--kind", "flow",
         "--steps", "10", "--batch-size", "4", "--max-len", "8", "--seq-hidden", "16", "--seq-depth", "1",
         "--seq-heads", "2", "--seq-dim-head", "8", "--seq-mlp", "32", "--hidden", "16", "--depth", "1",
         "--eval-every", "10", "--ckpt-every", "10", "--log-every", "5", "--val-objects", "8",
         "--eval-samples", "2", "--n-euler", "4", "--device", "cpu"]  # fmt: skip
    )
    train_decoder.main(
        ["--ckpt", str(mae), "--kind", "mse", "--out", str(out / "dec"), "--steps", "2", "--batch-size", "4",
         "--width", "24", "--heads", "2", "--depth", "1", "--eval-every", "2", "--ckpt-every", "2",
         "--val-objects", "6", "--workers", "0", "--device", "cpu"]  # fmt: skip
    )
    lookback.main(
        ["--ckpt", str(mae), "--out", str(out / "lb"), "--steps", "2", "--batch-size", "4", "--width", "24",
         "--heads", "2", "--depth", "1", "--eval-every", "2", "--ckpt-every", "2", "--log-every", "1",
         "--val-objects", "6", "--workers", "0", "--device", "cpu", "--stride", "0.5", "--n-ctx", "2"]  # fmt: skip
    )
    return out


def test_forecast_with_lookback(chain, tmp_path):
    out = tmp_path / "fc_lb"
    forecast.main(
        ["--pred", str(chain / "pred" / "pred.pt"), "--dec", str(chain / "lb" / "dec.pt"), "--out", str(out),
         "--n-objects", "8", "--per-star", "1", "--horizons", "1", "--samples", "2", "--min-hist", "2",
         "--ls-cap", "2000", "--device", "cpu", "--data", "sim", "--n-sim", "48"]  # fmt: skip
    )
    res = json.load(open(out / "summary.json"))
    by = res["tables"]["by horizon"]["1 window ahead"]
    assert "no_latent" in by and np.isfinite(by["no_latent"]["nll"]) and 0.0 <= by["beats_no_latent"] <= 1.0
    assert "no_latent" in open(out / "tables.md").read() and (out / "examples_folded.png").is_file()


def test_forecast_end_to_end(chain, tmp_path):
    out = tmp_path / "fc"
    forecast.main(
        ["--pred", str(chain / "pred" / "pred.pt"), "--dec", str(chain / "dec" / "dec.pt"), "--out", str(out),
         "--n-objects", "8", "--per-star", "2", "--horizons", "1", "2", "--samples", "3", "--min-hist", "2",
         "--ls-cap", "2000", "--device", "cpu", "--data", "sim", "--n-sim", "48"]  # fmt: skip
    )
    for name in ("summary.json", "tables.md", "forecasts.npz", "examples.png", "examples_folded.png"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    res = json.load(open(out / "summary.json"))
    assert res["n_forecasts"] > 0
    by = res["tables"]["by horizon"]
    for hz in ("1 window ahead", "2 window ahead"):
        assert hz in by and by[hz]["n"] > 0
        for m in forecast.METHODS:
            assert np.isfinite(by[hz][m]["nll"]) and np.isfinite(by[hz][m]["rmse"]), (hz, m)
        for m in forecast.METHODS[1:]:
            assert 0.0 <= by[hz][f"beats_{m}"] <= 1.0
    for hz in ("1 window ahead", "2 window ahead"):
        for k in forecast.LATENT_KEYS:
            assert np.isfinite(by[hz]["latent"][k]) and by[hz]["latent"][k] >= 0, (hz, k)
    d = np.load(out / "forecasts.npz")
    assert "latent_spread" in d.files
    assert len(d["horizon"]) == res["n_forecasts"] and set(d["horizon"].tolist()) <= {1, 2}
    text = open(out / "tables.md").read()
    assert "### skill" in text and "oracle_fold" in text


def test_mixture_and_fold():
    y = np.array([0.0, 1.0, 2.0])
    mu = np.array([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]])
    var = np.ones_like(mu)
    one = forecast.mixture_nll(y, mu[:1], var[:1])
    two = forecast.mixture_nll(y, mu, var)
    assert abs(one - 0.5 * np.log(2 * np.pi)) < 1e-9 and two > one and two < one + np.log(2) + 1e-9
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 300, 400))
    band = rng.integers(0, 2, 400)
    period = 1.3
    y = np.sin(2 * np.pi * t / period) + 0.5 * band
    tq = np.linspace(300, 320, 50)
    bq = np.zeros(50, dtype=int)
    mu, var = forecast.fold_forecast(t, y, band, period, tq, bq)
    assert np.abs(mu - np.sin(2 * np.pi * tq / period)).max() < 0.25 and (var > 0).all()
    mu_bad, _ = forecast.fold_forecast(t, y, band, float("nan"), tq, bq)
    assert np.allclose(mu_bad, np.median(y))
