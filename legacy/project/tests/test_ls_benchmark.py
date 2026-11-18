"""CPU test of the Lomb-Scargle efficiency benchmark on the toy simulator."""

from __future__ import annotations

import json

import numpy as np

from project.eval import ls_benchmark as lb

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip


def test_hit_table():
    p_true = np.array([1.0, 1.0, 1.0, 1.0])
    p_hat = np.array([1.00005, 2.0, 1.05, np.nan])
    h = lb.hit_table(p_hat, p_true)
    assert h["within_0.0001"] == 0.25 and h["within_0.1"] == 0.5 and h["alias_tolerant_0.0001"] == 0.5


def test_benchmark_end_to_end(tmp_path):
    import argparse

    from project import pretrain_mae
    from project.common import data_args_from, load_data, load_encoder

    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(tmp_path / "mae")])
    _, meta = load_encoder(str(tmp_path / "mae" / "mae.pt"), "cpu")
    val = load_data(data_args_from(meta.args, argparse.Namespace(data="sim", max_rows=None, n_sim=48)), splits=("validation",))["validation"]
    idx = np.array([i for i, r in enumerate(val) if r.period and r.period > 0][:6])
    rng = np.random.default_rng(0)
    p = np.array([float(val[i].period) for i in idx])
    rough = p * rng.uniform(0.95, 1.05, len(idx))
    np.savez(tmp_path / "pred.npz", index=idx, p_model=rough, p_top=np.stack([rough * f for f in (1.0, 2.0, 0.5, 1.5, 0.75)], 1),
             n_windows=np.full(len(idx), 12), points=np.full(len(idx), 20.0))  # fmt: skip
    np.savez(tmp_path / "head.npz", index=idx, p_model=p * 1.00002)
    out = tmp_path / "bench"
    lb.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--predictions", str(tmp_path / "pred.npz"), "--head", str(tmp_path / "head.npz"),
             "--out", str(out), "--n-objects", "6", "--budgets", "500", "2000", "--two-stage-budgets", "300", "--device", "cpu",
             "--data", "sim", "--n-sim", "48", "--astropy"])  # fmt: skip
    res = json.load(open(out / "results.json"))
    assert set(res["methods"]) == {"ls_full@500", "ls_full@2000", "ls_two_stage@300", "model_refined", "model_top5", "head",
                                   "astropy_1b@500", "astropy_1b@2000", "astropy_mb@500", "astropy_mb@2000"}
    assert any(k.startswith("[within 0.01%] astropy, multiband") for k in res["crossing"])
    assert res["methods"]["head"]["within_0.0001"] == 1.0  # the head's periods were set within 0.01 %
    assert res["methods"]["ls_full@2000"]["trials_median"] > res["methods"]["ls_full@500"]["trials_median"]
    for name in ("tables.md", "crossing.png", "heat_points_windows.png", "heat_cadence_span.png", "scatter_points_windows.png", "per_star.npz"):
        assert (out / name).is_file(), name
    d = np.load(out / "per_star.npz", allow_pickle=True)
    assert len(d["p_true"]) == res["n_stars"] and "p_model_top5" in d.files
