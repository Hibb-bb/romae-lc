"""CPU tests of the gate on frozen latents (``project.gate``) on the toy
simulator: a tiny stage-1 autoencoder is trained for four steps once per
module, then the gate runs on it. A minute or so in all."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from project import gate

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip

GATE_ARGS = [
    "--n-objects", "8", "--probe-train", "16", "--probe-val", "8",
    "--advances", "0.5", "1.0", "--batch-size", "4", "--device", "cpu",
    "--shuffle-objects", "6",
]  # fmt: skip

PROBE_KEYS = ("acc", "macro_f1", "balanced_acc", "majority_acc", "r2", "r2_within", "r2_by_superclass")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def mae_ckpt(tmp_path_factory):
    from project import pretrain_mae

    out = tmp_path_factory.mktemp("mae")
    pretrain_mae.main(MAE_ARGS + ["--steps", "4", "--out", str(out)])
    assert (out / "mae.pt").is_file()
    return out / "mae.pt"


def test_gate_run(mae_ckpt, tmp_path):
    out = tmp_path / "gate.json"
    args = gate.parse_args(["--ckpt", str(mae_ckpt), "--out", str(out)] + GATE_ARGS)
    res = gate.run(args)
    saved = json.load(open(out))
    assert saved["passed"] == res["passed"] and saved["kind"] == "mae"
    assert res["verdict"].startswith("GATE PASSED" if res["passed"] else "GATE FAILED")
    # part 1: probe and baseline on the same records
    p = res["probe"]
    assert p["status"] == "run" and isinstance(p["passed"], bool)
    for key in PROBE_KEYS:
        assert key in p["probe"] and key in p["baseline"]
    assert p["probe"]["n_val"] == p["baseline"]["n_val"] == p["n_val"] <= 8
    assert 0.0 <= p["shuffle_score"] <= 1.0 + 1e-6
    # part 2: the effect curve
    e = res["effect"]
    assert e["status"] == "run" and isinstance(e["passed"], bool)
    assert 0 < e["n_used"] <= 8 and e["n_used"] + e["n_no_window"] == e["n_candidates"]
    assert np.isfinite(e["floor_median"]) and e["floor_median"] > 0
    assert [r["advance"] for r in e["per_advance"]] == [0.5, 1.0]
    for r in e["per_advance"]:
        assert 0 < r["n"] <= 8 and r["n"] + r["n_skipped"] == e["n_used"]
        assert np.isfinite(r["ratio_median"]) and np.isfinite(r["ratio_mean"])
        assert r["ratio_median"] >= 0 and -1 - 1e-6 <= r["cosine_median"] <= 1 + 1e-6
        assert sum(r["n_by_superclass"].values()) == r["n"]
        assert set(r["ratio_by_superclass"]) <= set(e["n_by_superclass"])
    gated = next(r for r in e["per_advance"] if r["advance"] == 1.0)
    assert e["ratio_at_train_advance"] == gated["ratio_median"]
    assert e["passed"] == (gated["ratio_median"] >= 1.5)
    # part 3 not run without --decoder-results
    assert res["decoder"]["status"] == "not run" and res["decoder"]["passed"] is None
    # a train-advance outside the list is added to it
    args = gate.parse_args(
        ["--ckpt", str(mae_ckpt), "--out", str(out), "--train-advance", "0.75", "--skip-probe"]
        + GATE_ARGS
    )
    res = gate.run(args)
    assert res["probe"]["status"] == "skipped"
    assert [r["advance"] for r in res["effect"]["per_advance"]] == [0.5, 0.75, 1.0]


def test_gate_main_exits_on_failure(mae_ckpt, tmp_path):
    out = tmp_path / "gate.json"
    argv = ["--ckpt", str(mae_ckpt), "--out", str(out), "--min-effect", "1e9"] + GATE_ARGS
    with pytest.raises(SystemExit) as exc:
        gate.main(argv + ["--skip-probe"])
    assert exc.value.code == 1
    saved = json.load(open(out))
    assert saved["passed"] is False and saved["verdict"].startswith("GATE FAILED: part 2")


def test_decoder_part(tmp_path):
    def metrics(gp):
        return dict(
            kind="eval",
            step=5,
            n_windows=10,
            holdout="random",
            frac=0.2,
            decoder=dict(nll=1.0, rmse=0.5, nll_pred=None, per_superclass={}),
            linear=dict(nll=1.5, rmse=0.6, nll_pred=1.4, per_superclass={}),
            **({"gp_rbf": dict(nll=gp, rmse=0.55, nll_pred=1.1, per_superclass={})} if gp else {}),
        )

    log = tmp_path / "log.jsonl"
    with open(log, "w") as f:
        f.write(json.dumps(dict(kind="train", step=1, loss=2.0)) + "\n")
        f.write(json.dumps(metrics(1.2)) + "\n")
    d = gate.decoder_part(log)
    assert d["passed"] is True and d["nll"]["gp_rbf"] == 1.2 and d["step"] == 5
    torch.save(dict(metrics=metrics(0.9), step=5), tmp_path / "dec.pt")
    d = gate.decoder_part(tmp_path / "dec.pt")
    assert d["passed"] is False and d["note"] is None
    json.dump(metrics(None), open(tmp_path / "m.json", "w"))
    d = gate.decoder_part(tmp_path / "m.json")
    assert d["passed"] is False and "GP" in d["note"]
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        gate.decoder_part(empty)
