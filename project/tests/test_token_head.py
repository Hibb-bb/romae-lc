"""CPU test of the token read-out on the toy simulator."""

from __future__ import annotations

import json

import numpy as np
import torch

from project import token_head as th

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip


def test_token_head(tmp_path):
    net = th.TokenPeakNet(8, 50, [1.0, 3.0], d_model=16, n_queries=2, nhead=2, hidden=32)
    toks, t_pos, pad = torch.randn(3, 7, 8), torch.rand(3, 7) * 5 + 1, torch.zeros(3, 7, dtype=torch.bool)
    pad[0, 5:] = True
    lg, off = net(toks, t_pos, pad)
    assert lg.shape == (3, 50) and off.shape == (3, 50) and th.TokenPeakNet(**net.hparams).n_bins == 50
    from project import pretrain_mae

    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(tmp_path / "mae")])
    out = tmp_path / "th"
    th.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--out", str(out), "--steps", "3", "--batch-size", "2", "--n-frames", "2",
             "--d-model", "16", "--hidden", "32", "--grid-rel", "2e-2", "--p-min", "0.1", "--p-max", "50", "--eval-every", "3",
             "--ckpt-every", "3", "--log-every", "1", "--val-objects", "6", "--workers", "0", "--device", "cpu",
             "--data", "sim", "--n-sim", "48"])  # fmt: skip
    assert (out / "token_head.pt").is_file() and (out / "predictions.npz").is_file()
    res = json.load(open(out / "results.json"))
    assert 0.0 <= res["object"]["within_0.1"] <= 1.0 and res["n_objects"] > 0
    d = np.load(out / "predictions.npz", allow_pickle=True)
    assert d["p_top"].shape[1] == 5 and d["index"].max() < 48
