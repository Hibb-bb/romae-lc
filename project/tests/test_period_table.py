"""CPU test of the all-split period table on the toy simulator."""

from __future__ import annotations

import csv
import json

from project import cache_latents, pretrain_mae
from project.eval import period_table as pt
from project.tests.test_ls_benchmark import MAE_ARGS


def test_period_table(tmp_path):
    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(tmp_path / "mae")])
    cache_latents.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--out", str(tmp_path / "latents.pt"), "--stride", "0.5", "--device", "cpu", "--workers", "0", "--batch-size", "16"])
    cache_latents.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--out", str(tmp_path / "latents_test.pt"), "--splits", "test", "--stride", "0.5", "--device", "cpu", "--workers", "0", "--batch-size", "16"])
    out = tmp_path / "table"
    res = pt.main(["--latents", str(tmp_path / "latents.pt"), "--test-latents", str(tmp_path / "latents_test.pt"), "--out", str(out), "--mlp-steps", "20",
                   "--bins", "20", "--n-objects", "4", "--device", "cpu", "--data", "sim", "--n-sim", "48"])  # fmt: skip
    assert set(res["splits"]) == {"train", "validation", "test"} and res["splits"]["train"]["in_sample_readout"]
    rows = list(csv.DictReader(open(out / "periods.csv")))
    assert len(rows) == 12 and {r["split"] for r in rows} == {"train", "validation", "test"}
    assert all(float(r["p_model_cands"]) > 0 for r in rows) and "verdict" in rows[0]
    json.load(open(out / "summary.json"))
