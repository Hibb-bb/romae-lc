"""CPU test of the random-forest classification benchmark on the toy simulator."""

from __future__ import annotations

import json

from project import cache_latents, pretrain_mae
from project.eval import classify_rf as cr
from project.tests.test_ls_benchmark import MAE_ARGS


def test_classify_rf(tmp_path):
    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(tmp_path / "mae")])
    cache_latents.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--out", str(tmp_path / "latents.pt"), "--stride", "0.5", "--device", "cpu", "--workers", "0", "--batch-size", "16"])
    cache_latents.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--out", str(tmp_path / "latents_test.pt"), "--splits", "test", "--stride", "0.5", "--device", "cpu", "--workers", "0", "--batch-size", "16"])
    out = tmp_path / "rf"
    res = cr.main(["--latents", str(tmp_path / "latents.pt"), "--test-latents", str(tmp_path / "latents_test.pt"), "--features", "mean", "hand", "--out", str(out),
                   "--grid", '{"n_estimators": [10], "max_depth": [3, null]}', "--n-jobs", "2", "--data", "sim", "--n-sim", "48"])  # fmt: skip
    assert set(res["results"]) == {"mean", "hand"} and len(res["results"]["mean"]["seeds"]) == 3
    assert 0.0 <= res["results"]["mean"]["f1_mean"] <= 1.0 and res["results"]["mean"]["validation_macro_f1"] is not None
    for name in ("results.json", "tables.md", "confusion_mean.png", "report_hand.csv"):
        assert (out / name).is_file(), name
    json.load(open(out / "results.json"))
