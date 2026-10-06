"""CPU test of the period-sensitivity test on the toy simulator."""

from __future__ import annotations

import json

import numpy as np

from project.eval import period_sensitivity as ps

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip


def test_threshold():
    shifts = np.logspace(-4, -1, 7)
    assert np.isinf(ps.threshold(shifts, np.ones(7), 2.0))
    ratio = np.array([0.5, 0.8, 1.0, 1.5, 3.0, 5.0, 8.0])
    t = ps.threshold(shifts, ratio, 2.0)
    assert shifts[3] < t < shifts[4]
    assert ps.threshold(shifts, np.full(7, 9.0), 2.0) == shifts[0]


def test_sensitivity_end_to_end(tmp_path):
    from project import pretrain_mae

    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(tmp_path / "mae")])
    out = tmp_path / "sens"
    ps.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--out", str(out), "--n-objects", "6", "--n-shifts", "5",
             "--device", "cpu", "--data", "sim", "--n-sim", "48"])  # fmt: skip
    res = json.load(open(out / "results.json"))
    assert res["n_stars"] > 0 and len(res["all"]["curve"]) == 5 and 0.0 <= res["all"]["never"] <= 1.0
    d = np.load(out / "per_star.npz", allow_pickle=True)
    assert d["curves"].shape == (res["n_stars"], 5) and (out / "tables.md").is_file()
