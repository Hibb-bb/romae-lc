"""CPU test of the phase probe on a toy cache: targets are well defined,
the shuffle control exists, and a latent that carries the phase is read."""

from __future__ import annotations

import json

import numpy as np
import torch

from project import cache_latents, phase_probe

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip


def test_phase_error_and_probe(tmp_path):
    e = phase_probe.phase_error(np.array([0.1, 0.95, 0.5]), np.array([0.2, 0.05, 0.0]))
    assert np.allclose(e, [0.1, 0.1, 0.5])
    from project import pretrain_mae

    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(tmp_path / "mae")])
    cache_latents.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--out", str(tmp_path / "latents.pt"), "--stride", "0.5",
                        "--device", "cpu", "--workers", "0", "--batch-size", "16"])  # fmt: skip
    out = tmp_path / "phase"
    phase_probe.main(["--latents", str(tmp_path / "latents.pt"), "--out", str(out), "--steps", "20", "--batch-size", "64",
                      "--hidden", "16", "--device", "cpu", "--data", "sim", "--n-sim", "48"])  # fmt: skip
    res = json.load(open(out / "results.json"))
    assert set(res["models"]) == {"ridge / phase", "mlp / phase", "ridge / shuffled control", "mlp / shuffled control"}
    for v in res["models"].values():
        assert 0.0 <= v["median"] <= 0.5 and -1.0 <= v["cos"] <= 1.0 and v["n"] > 0
    d = np.load(out / "predictions.npz")
    assert len(d["psi"]) == res["n_val"] and (d["psi"] >= 0).all() and (d["psi"] < 1).all()
    # a latent that holds the phase is read by the ridge
    import argparse

    from project.common import data_args_from, load_data, load_encoder

    cache = torch.load(tmp_path / "latents.pt", map_location="cpu", weights_only=False)
    _, meta = load_encoder(str(tmp_path / "mae" / "mae.pt"), "cpu")
    recs = load_data(data_args_from(meta.args, argparse.Namespace(data="sim", max_rows=None, n_sim=48)), splits=("validation",))["validation"]
    rows, psi, _, _ = phase_probe.window_targets(cache, "validation", recs, 4, log=lambda *a: None)
    x = torch.stack([torch.cos(2 * np.pi * torch.as_tensor(psi)), torch.sin(2 * np.pi * torch.as_tensor(psi))], 1).float()
    y = x.clone()
    pred = phase_probe.to_phase(phase_probe.fit_ridge(x, y, x, 1e-3))
    assert np.median(phase_probe.phase_error(pred, psi)) < 0.02
