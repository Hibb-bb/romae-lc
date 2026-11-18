"""CPU tests of the period head: the grid, the batched spectrum against the
probe's Lomb-Scargle, the target and loss, and a short training run on the
toy simulator."""

from __future__ import annotations

import json

import numpy as np
import torch

from project import period_head as ph

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip


def test_grid_and_spectrum():
    g = ph.FrequencyGrid(0.1, 50.0, 1e-3)
    p = np.array([0.3, 7.0])
    assert np.allclose(g.period_of(g.bin_of(p)), p)
    assert g.shift == round(np.log(2) / np.log1p(1e-3))
    assert ph.FrequencyGrid(**g.to_dict()).n == g.n
    # the batched spectrum matches the probe's generalised Lomb-Scargle on one window
    from project.period_probe import gls_power

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 30, 40))
    band = rng.integers(0, 2, 40)
    y = np.sin(2 * np.pi * t / 0.7) + 0.4 * band + 0.05 * rng.standard_normal(40)
    err = np.full(40, 0.05)
    freqs = g.freqs
    yc = y.copy()
    for b in (0, 1):
        m = band == b
        yc[m] -= np.average(yc[m], weights=1 / err[m] ** 2)
    ref = gls_power(t, yc, 1 / err**2, freqs)
    tt = torch.tensor(t, dtype=torch.float32)[None]
    got = ph.window_spectrum(tt, torch.tensor(y, dtype=torch.float32)[None], torch.tensor(1 / err**2, dtype=torch.float32)[None],
                             torch.tensor(band)[None], torch.tensor(freqs, dtype=torch.float32), 2)[0].numpy()  # fmt: skip
    assert np.abs(got - ref).max() < 2e-3 and abs(1 / freqs[got.argmax()] - 0.7) < 0.01
    chans = ph.spectrum_channels(torch.tensor(got)[None], g.shift)
    assert chans.shape == (1, 6, len(freqs))


def test_target_loss_and_training(tmp_path):
    coord = torch.tensor([10.0, 20.4, float("nan")])
    tg = ph.soft_target(coord[:2], 40)
    assert tg.shape == (2, 40) and torch.allclose(tg.sum(1), torch.ones(2)) and tg[0].argmax() == 10
    logits = torch.zeros(3, 40)
    logits[1, 20] = 5.0
    loss = ph.head_loss(logits, torch.zeros(3, 40), coord)
    assert torch.isfinite(loss) and loss > 0
    net = ph.PeakNet(8, ch=16, depth=2, kernel=5)
    lg, off = net(torch.randn(2, 6, 100), torch.randn(2, 8))
    assert lg.shape == (2, 100) and off.shape == (2, 100) and ph.PeakNet(**net.hparams).depth == 2
    from project import pretrain_mae

    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(tmp_path / "mae")])
    out = tmp_path / "head"
    ph.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--out", str(out), "--steps", "3", "--batch-size", "2", "--n-frames", "2",
             "--ch", "16", "--depth", "2", "--kernel", "5", "--grid-rel", "5e-3", "--p-min", "0.1", "--p-max", "50", "--eval-every", "3",
             "--ckpt-every", "3", "--log-every", "1", "--val-objects", "6", "--workers", "0", "--device", "cpu",
             "--data", "sim", "--n-sim", "48"])  # fmt: skip
    assert (out / "period_head.pt").is_file() and (out / "predictions.npz").is_file()
    res = json.load(open(out / "results.json"))
    for who in ("head", "ls"):
        assert 0.0 <= res["object"][who]["within_0.1"] <= 1.0
    head, grid, meta = ph.load_period_head(out / "period_head.pt", "cpu")
    assert grid.n > 100 and meta["step"] == 3
    d = np.load(out / "predictions.npz", allow_pickle=True)
    assert d["p_top"].shape[1] == 5 and len(d["index"]) == res["n_objects"]
