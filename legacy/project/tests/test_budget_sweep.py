"""CPU test of the model budget sweep and its figure on the toy simulator."""

from __future__ import annotations

import json

import numpy as np

from project.eval import budget_sweep as bs
from project.tests.test_ls_benchmark import MAE_ARGS


def test_sweep_and_plot(tmp_path):
    import argparse

    from project import pretrain_mae
    from project.common import data_args_from, load_data, load_encoder

    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(tmp_path / "mae")])
    _, meta = load_encoder(str(tmp_path / "mae" / "mae.pt"), "cpu")
    val = load_data(data_args_from(meta.args, argparse.Namespace(data="sim", max_rows=None, n_sim=48)), splits=("validation",))["validation"]
    idx = np.array([i for i, r in enumerate(val) if r.period and r.period > 0][:5])
    p = np.array([float(val[i].period) for i in idx])
    np.savez(tmp_path / "pred.npz", index=idx, p_model=p * 1.02, p_top=np.stack([p * 1.02, p * 2.04, p * 0.51], 1))
    out = tmp_path / "sweep"
    res = bs.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--predictions", str(tmp_path / "pred.npz"), "--out", str(out), "--n-objects", "5",
                   "--rels", "0.03", "0.1", "--tops", "1", "3", "--device", "cpu", "--data", "sim", "--n-sim", "48"])  # fmt: skip
    assert set(res["settings"]) == {"top1_rel0.03", "top1_rel0.1", "top3_rel0.03", "top3_rel0.1"}
    assert res["settings"]["top3_rel0.1"]["trials_median"] > res["settings"]["top1_rel0.1"]["trials_median"]
    assert res["settings"]["top1_rel0.1"]["within_0.1"] == 1.0
    bench = dict(methods={f"astropy_mb@{b}": {"trials_median": b, "within_0.1": 0.1, "within_0.01": 0.05, "within_0.0001": 0.01} for b in (500, 2000)})
    bs.plot_budget(tmp_path / "budget.png", json.load(open(out / "results.json")), bench)
    assert (tmp_path / "budget.png").is_file()
