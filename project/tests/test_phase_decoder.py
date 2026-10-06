"""CPU tests of the phase decoder on the toy simulator: the template fit,
the phase coordinate, the model's latent switch, a short training run
with and without context and both targets, and the final period ladder."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from project import phase_decoder as pdm

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip


@pytest.fixture(scope="module")
def mae(tmp_path_factory):
    from project import pretrain_mae

    out = tmp_path_factory.mktemp("pd")
    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(out / "mae")])
    return out / "mae" / "mae.pt"


def test_template_and_phases():
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 400, 600))
    band = rng.integers(0, 2, 600)
    period = 0.7
    y = 0.8 * np.sin(2 * np.pi * t / period) + 0.3 * band
    rec = SimpleNamespace(t=t, y=y, err=np.full(600, 0.05), band=band, period=period)
    coefs = pdm.fourier_fit(rec, period, 3)
    tq = np.linspace(400, 405, 50)
    assert np.abs(pdm.template_at(coefs, tq, np.zeros(50, int), period, 3) - 0.8 * np.sin(2 * np.pi * tq / period)).max() < 0.05
    t_ref = pdm.reference_epoch(rec, period)
    assert 0 <= t_ref < period
    pos = torch.tensor([[[1.0, 1.0 + 0.35 / 0.01], [0.0, 0.0]]])  # time axis: t / time_scale + 1, time_scale 0.01
    ph = pdm.phases(pos, 0.01, torch.tensor([100.0], dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
                    torch.tensor([0.7], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64))  # fmt: skip
    assert torch.allclose(ph, torch.tensor([[(100.0 / 0.7) % 1, (100.35 / 0.7) % 1]]).float(), atol=1e-4)


def test_model_and_training(mae, tmp_path):
    torch.manual_seed(0)
    dec = pdm.PhaseDecoder(24, 24, d_model=24, nhead=2, depth=1, n_harm=4, ctx_features="encoder")
    b, n, m = 2, 6, 9
    z = torch.randn(b, 24)
    q_phase, q_band, q_pad = torch.rand(b, n), torch.randint(0, 2, (b, n)), torch.zeros(b, n, dtype=torch.bool)
    ctx = (torch.randn(b, m, 26), torch.rand(b, m), torch.randint(0, 2, (b, m)), torch.zeros(b, m, dtype=torch.bool))
    # a query one full cycle away attends like one at the same phase: the rotary rungs are harmonics of a cycle
    mu_a, _ = dec(z, q_phase, q_band, q_pad, ctx)
    mu_b, _ = dec(z, q_phase + 1.0, q_band, q_pad, ctx)
    assert torch.allclose(mu_a, mu_b, atol=1e-4)
    small = pdm.PhaseDecoder(24, 24, d_model=24, nhead=2, depth=1, n_harm=4)
    assert small.ctx_dim == 2 and pdm.PhaseDecoder(**small.hparams).ctx_features == "values"
    mu, lv = dec(z, q_phase, q_band, q_pad, ctx)
    assert mu.shape == (b, n) and lv.shape == (b, n)
    mu0, _ = dec(z, q_phase, q_band, q_pad, None)  # no context
    assert mu0.shape == (b, n) and not torch.allclose(mu0, mu)
    mu1, _ = dec(z, q_phase, q_band, q_pad, ctx, torch.tensor([True, False]))
    mu2, _ = dec(torch.randn(b, 24), q_phase, q_band, q_pad, ctx, torch.tensor([False, False]))
    assert torch.allclose(mu1[0], mu[0]) and torch.allclose(mu1[1], mu2[1])
    assert pdm.PhaseDecoder(**dec.hparams).kind == "phase"
    common = ["--ckpt", str(mae), "--steps", "3", "--batch-size", "4", "--width", "24", "--heads", "2", "--depth", "1",
              "--eval-every", "3", "--ckpt-every", "3", "--log-every", "1", "--val-objects", "8", "--workers", "0",
              "--device", "cpu", "--stride", "0.5", "--data", "sim", "--n-sim", "48", "--ls-cap", "2000", "--n-harm", "4"]  # fmt: skip
    out = tmp_path / "ceiling"
    pdm.main(common + ["--out", str(out), "--n-ctx", "0", "--phase", "oracle", "--target", "template", "--eval-periods", "oracle"])
    final = json.load(open(out / "final.json"))["final"]
    assert final["oracle"]["n"] > 0 and np.isfinite(final["oracle"]["rmse_template_model"])
    assert (out / "tables.md").is_file() and pdm.load_phase_decoder(out / "dec.pt", "cpu")[1]["n_ctx"] == 0
    out = tmp_path / "ctx"
    pdm.main(common + ["--out", str(out), "--n-ctx", "2", "--phase", "window", "--target", "obs", "--eval-periods", "oracle", "ls",
                       "--ctx-features", "encoder"])  # fmt: skip
    final = json.load(open(out / "final.json"))["final"]
    for src in ("oracle", "ls"):
        assert final[src]["n"] > 0 and np.isfinite(final[src]["nll_model"]) and np.isfinite(final[src]["skill_no_latent"]), src
    text = open(out / "tables.md").read()
    assert "ls" in text and "RMSE vs template" in text
    # evaluation only, with the Lomb-Scargle peak sharpened by the fine search
    pdm.main(common + ["--out", str(out), "--n-ctx", "2", "--phase", "window", "--eval-only", "--eval-periods", "ls_refined"])
    final = json.load(open(out / "final.json"))["final"]
    assert "ls_refined" in final and final["ls_refined"]["n"] > 0
    # the whole chain: a tiny sequence predictor on the toy cache, its latent fed to the decoder
    from project import cache_latents, train_predictor

    cache_latents.main(["--ckpt", str(mae), "--out", str(tmp_path / "latents.pt"), "--stride", "0.5", "--device", "cpu",
                        "--workers", "0", "--batch-size", "16"])  # fmt: skip
    train_predictor.main(
        ["--latents", str(tmp_path / "latents.pt"), "--out", str(tmp_path / "pred"), "--arch", "seq", "--kind", "flow",
         "--steps", "4", "--batch-size", "4", "--max-len", "8", "--seq-hidden", "16", "--seq-depth", "1", "--seq-heads", "2",
         "--seq-dim-head", "8", "--seq-mlp", "32", "--hidden", "16", "--depth", "1", "--eval-every", "4", "--ckpt-every", "4",
         "--log-every", "2", "--val-objects", "8", "--eval-samples", "2", "--n-euler", "4", "--device", "cpu"]  # fmt: skip
    )
    pdm.main(common + ["--out", str(out), "--n-ctx", "2", "--phase", "window", "--eval-only", "--eval-periods", "oracle",
                       "--pred", str(tmp_path / "pred" / "pred.pt")])  # fmt: skip
    final = json.load(open(out / "final.json"))["final"]["oracle"]
    assert np.isfinite(final["skill_predicted"]) and "predicted latent" in open(out / "tables.md").read()
