"""CPU test of the runtime benchmark on the toy simulator."""

from __future__ import annotations

import json

import numpy as np

from project.eval import runtime as rt
from project.tests.test_ls_benchmark import MAE_ARGS


def test_runtime_end_to_end(tmp_path):
    import argparse

    from project import pretrain_mae
    from project.common import data_args_from, load_data, load_encoder

    pretrain_mae.main(MAE_ARGS + ["--steps", "2", "--out", str(tmp_path / "mae")])
    _, meta = load_encoder(str(tmp_path / "mae" / "mae.pt"), "cpu")
    val = load_data(data_args_from(meta.args, argparse.Namespace(data="sim", max_rows=None, n_sim=48)), splits=("validation",))["validation"]
    idx = np.array([i for i, r in enumerate(val) if r.period and r.period > 0][:5])
    np.savez(tmp_path / "pred.npz", index=idx, p_model=np.array([float(val[i].period) for i in idx]) * 1.02)
    out = tmp_path / "rt"
    rt.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--predictions", str(tmp_path / "pred.npz"), "--out", str(out), "--n-objects", "5",
             "--ls-budgets", "500", "--astropy-budgets", "--batch-sweep", "8", "--workers", "0", "--device", "cpu", "--data", "sim", "--n-sim", "48"])  # fmt: skip
    res = json.load(open(out / "results.json"))
    assert set(res["steps"]) == {"cut", "encode", "read", "search", "ls_full@500"}
    res2 = rt.main(["--ckpt", str(tmp_path / "mae" / "mae.pt"), "--predictions", str(tmp_path / "pred.npz"), "--out", str(tmp_path / "rt2"), "--n-objects", "5",
                    "--ls-budgets", "--astropy-budgets", "--static-pad", "--batch-multiple", "4", "--batch-sweep", "8", "16", "--workers", "0", "--device", "cpu", "--data", "sim", "--n-sim", "48"])  # fmt: skip
    assert set(res2["batch_sweep"]) == {"8", "16"} or set(res2["batch_sweep"]) == {8, 16}
    assert res2["static_pad"] and res2["steps"]["encode"]["n"] >= 1
    assert res["steps"]["encode"]["n"] >= 1 and res["model_total"]["median"] > 0 and res["encode_batched_per_star"] > 0
    assert (out / "tables.md").is_file() and (out / "runtime.png").is_file()
