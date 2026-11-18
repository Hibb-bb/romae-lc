"""CPU test of the all-star period report on the toy simulator."""

from __future__ import annotations

import csv
import json

import numpy as np

from project.eval import period_report as pr
from project.tests.test_ls_benchmark import MAE_ARGS


def test_hit_table_and_kinds():
    p_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    p_hat = np.array([1.00005, 2.0, 1.05, 1.15, np.nan])
    h = pr.hit_table(p_hat, p_true)
    assert h["within_0.0001"] == 0.2 and h["within_0.1"] == 0.4 and h["within_0.2"] == 0.6 and h["alias_0.0001"] == 0.4
    assert [pr.alias_kind(x) for x in (1.001, 2.0, 0.5, 3.0, 1.3)] == ["same period, sharper", "double", "half", "3 x", "different"]


def test_report_end_to_end(tmp_path):
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
    rough[0] = 2.0 * p[0]  # one star the model calls at the double: the candidate search should bring it back
    np.savez(tmp_path / "pred.npz", index=idx, p_model=rough, p_top=np.stack([rough * f for f in (1.0, 2.0, 0.5)], 1))
    np.savez(tmp_path / "relu.npz", index=idx, p_model=p * 1.02)
    out = tmp_path / "report"
    pr.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--predictions", str(tmp_path / "pred.npz"), "--compare", f"relu={tmp_path / 'relu.npz'}",
             "--out", str(out), "--ls-budgets", "500", "--astropy-budget", "0", "--device", "cpu", "--data", "sim", "--n-sim", "48"])  # fmt: skip
    res = json.load(open(out / "results.json"))
    assert set(res["methods"]) == {"model", "model_refined", "model_cands", "ls_full@500", "relu"}
    assert res["methods"]["relu"]["within_0.1"] == 1.0 and res["methods"]["relu"]["within_0.01"] == 0.0
    assert res["methods"]["model_cands"]["trials_median"] > res["methods"]["model_refined"]["trials_median"]
    assert "verdict" in open(out / "better_than_catalogue.csv").readline()
    d = np.load(out / "per_star.npz", allow_pickle=True)
    assert len(d["p_true"]) == 6 and np.isfinite(d["r2_catalogue"]).all()
    for name in ("tables.md", "hit_vs_period.png", "scatter_bend.png", "better_than_catalogue.csv"):
        assert (out / name).is_file(), name
    rows = list(csv.DictReader(open(out / "better_than_catalogue.csv")))
    assert all(float(r["gain"]) > 0.1 for r in rows)
