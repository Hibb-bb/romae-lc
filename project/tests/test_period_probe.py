"""CPU tests of :mod:`project.period_probe` on the toy simulator: a tiny
stage-1 autoencoder is trained once (4 steps), its window grid cached, and
the period probe runs end to end on the cache with a handful of MLP steps
and Lomb-Scargle objects. Unit tests check the Lomb-Scargle on clean
sinusoids and the recovery bookkeeping. About two minutes in all."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch

from project import cache_latents, period_probe
from romae_lc.data import CLASSES, normalize, simulate

MAE_ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2",
    "--heads", "2", "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "4", "--ckpt-every", "2", "--log-every", "2",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu",
]  # fmt: skip

STRIDE = 0.5
PNGS = (
    "pred_vs_true_model.png",
    "pred_vs_true_ls_window.png",
    "resid_vs_cycles.png",
    "resid_vs_points.png",
    "ratio_hist.png",
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def mae_ckpt(tmp_path_factory):
    from project import pretrain_mae

    out = tmp_path_factory.mktemp("mae")
    pretrain_mae.main(MAE_ARGS + ["--steps", "4", "--out", str(out)])
    assert (out / "mae.pt").is_file()
    return out / "mae.pt"


@pytest.fixture(scope="module")
def latents(mae_ckpt):
    out = mae_ckpt.parent / "latents.pt"
    cache_latents.main(
        [
            "--ckpt", str(mae_ckpt), "--out", str(out), "--stride", str(STRIDE),
            "--device", "cpu", "--workers", "0", "--batch-size", "16",
        ]  # fmt: skip
    )
    assert out.is_file()
    return out


def _finite_or_nan(v) -> bool:
    return isinstance(v, (int, float)) and (math.isfinite(v) or math.isnan(v))


def test_fold_r2():
    from project import fold

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 400, 300))
    band = rng.integers(0, 2, 300)
    period = 1.7
    y = np.sin(2 * np.pi * t / period) + 0.3 * np.sin(4 * np.pi * t / period) + 0.5 * band
    err = np.full(300, 0.1)
    y = y + rng.normal(0, 0.1, 300)
    right = fold.fold_r2(t, y, err, band, period)
    wrong = fold.fold_r2(t, y, err, band, period * fold.WRONG_FACTOR)
    near = fold.fold_r2(t, y, err, band, period * 1.01)  # 1 % off: 2 cycles of drift
    assert right > 0.95 and abs(wrong) < 0.1 and near < 0.3
    assert np.isnan(fold.fold_r2(t, y, err, band, float("nan")))
    assert np.isnan(fold.fold_r2(t[:5], y[:5], err[:5], band[:5], period))

    class R:
        pass

    r = R()
    r.t, r.y, r.err, r.band = t, y, err, band
    r2 = fold.fold_compare([r, r], np.array([period, period * 1.01]), np.array([period, period]))
    assert r2["catalogue"][0] == r2["model"][0] and r2["model"][1] < r2["catalogue"][1]
    tabs = fold.fold_tables(r2, np.array(["A", "A"]), np.array([period, period * 1.01]), np.array([period, period]), ["A"])
    assert tabs["superclass"][0]["n"] == 2 and tabs["superclass"][0]["as_good"] == 0.5
    assert [row["n"] for row in tabs["period_error"]] == [1, 0, 0, 1, 0, 0]

    # many trial periods at once give the same numbers as one at a time
    trial = np.array([period, period * 1.01, period * fold.WRONG_FACTOR])
    grid = fold.fold_r2_grid(t, y, err, band, trial)
    one = np.array([fold.fold_r2(t, y, err, band, p) for p in trial])
    assert np.allclose(grid, one, atol=1e-6)
    # a rough period 3 % off is sharpened to the true one
    p_ref, r2_ref, n_trials = fold.refine_period(t, y, err, band, period * 1.03)
    assert abs(p_ref / period - 1) < 1e-3 and r2_ref > 0.95 and n_trials > 10
    # a rough period 40 % off is out of reach of the narrow search
    p_far, r2_far, _ = fold.refine_period(t, y, err, band, period * 1.4)
    assert abs(p_far / period - 1) > 0.2 and r2_far < 0.3
    rows = fold.precision_table(np.array([period, period * 1.0005, period * 1.5]), np.full(3, period))
    assert [round(r["share"], 3) for r in rows] == [0.667, 0.667, 0.667, 0.333, 0.333]


def test_period_probe_end_to_end(latents, tmp_path):
    out = tmp_path / "probe"
    period_probe.main(
        [
            "--latents", str(latents), "--out", str(out), "--n-ls", "6",
            "--mlp-steps", "50", "--bins", "20", "--device", "cpu",
            "--data", "sim", "--n-sim", "48", "--worst", "5",
        ]  # fmt: skip
    )
    for name in ("results.json", "tables.md", "worst.csv", "log.jsonl") + PNGS:
        assert (out / name).is_file(), name
        assert (out / name).stat().st_size > 0, name
    res = json.load(open(out / "results.json"))
    assert (out / "fold_r2.png").is_file() and (out / "fold_r2.npz").is_file()
    assert res["fold"]["harmonics"] == 3
    assert res["fold"]["tables"]["superclass"][0]["bin"] == "all"
    assert "Phase-fold test" in open(out / "tables.md").read()
    assert res["window"] == 30.0 and res["n_ls"] == 6
    assert res["n_train"] > 0 and res["n_val"] > 0
    # only the cache features without --pred
    assert set(res["models"]) == {
        f"{f}/{r}" for f in ("hand", "mean", "meanmax") for r in ("ridge", "mlp", "bins")
    }
    assert res["best"] in res["models"] and not res["best"].startswith("hand/")
    assert res["models"]["meanmax/ridge"]["dim"] == 48
    for name, s in list(res["models"].items()) + list(res["ls"].items()):
        assert s["n"] > 0, name
        for k in ("rec1", "rec10", "alias"):
            assert 0.0 <= s[k] <= 1.0, (name, k)
        assert s["rec1"] <= s["rec10"], name
        assert s["alias"] + s["rec10"] <= 1.0 + 1e-9, name
        assert math.isfinite(s["med_abs"]) and s["med_abs"] >= 0, name
        assert _finite_or_nan(s["r2"]) and _finite_or_nan(s["r2_within"]), name
        assert sum(b["n"] for b in s["by_superclass"].values()) == s["n"], name
    for m in ("ls_window", "ls_full", "ls_top5"):
        assert res["ls"][m]["n"] == 6 and math.isfinite(res["ls"][m]["r2"]), m
    # the oracle over the top peaks can only do better than the top peak
    assert res["ls"]["ls_top5"]["rec10"] >= res["ls"]["ls_full"]["rec10"]
    assert res["ls"]["ls_top5"]["med_abs"] <= res["ls"]["ls_full"]["med_abs"] + 1e-9
    b = res["models"]["mean/bins"]
    assert b["best_of_two_rec10"] >= b["rec10"]
    # the failure tables cover every validation object once
    for key, n in (("failures_model", res["n_val"]), ("failures_ls_window", 6)):
        tables = res[key]
        assert set(tables) == {
            "superclass", "fine_class", "true_period_d", "cycles_per_window",
            "points_per_window", "n_valid_windows", "amplitude",
        }  # fmt: skip
        for name, rows in tables.items():
            assert sum(r["n"] for r in rows) == n, (key, name)
            for r in rows:
                assert 0.0 <= r["rec10"] <= 1.0 and math.isfinite(r["med_abs"])
    assert len(res["worst_bins_model"]) <= 2
    rows = list(csv.DictReader(open(out / "worst.csv")))
    assert len(rows) == 5 and set(rows[0]) >= {"id", "class", "true_period", "pred_period", "ratio", "n_windows", "mean_points", "amplitude"}
    ratios = [abs(math.log10(float(r["ratio"]))) for r in rows]
    assert ratios == sorted(ratios, reverse=True)
    md = open(out / "tables.md").read()
    assert "Lomb-Scargle" in md and "P / 2" in md and res["best"] in md
    lines = [json.loads(l) for l in open(out / "log.jsonl")]
    assert {l["kind"] for l in lines} == {"model", "ls"}


def _sinusoid_records(n_max=3):
    """Clean simulated sinusoids with periods between 0.5 and 5 d."""
    recs = []
    for r in simulate(60, seed=1):
        if CLASSES[r.label] == "sinusoid" and 0.5 < r.period < 5.0:
            recs.append(normalize(r))
        if len(recs) == n_max:
            break
    assert recs
    return recs


def test_lomb_scargle_recovers_sinusoid():
    # a pure sinusoid at 1.3 d in one 30 d window: the top peak within 1 %
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 30, 120))
    y = np.sin(2 * np.pi * t / 1.3) + 0.05 * rng.standard_normal(120)
    err, band = np.full(120, 0.05), np.zeros(120, dtype=np.int64)
    got = period_probe.ls_periods(t, y, err, band, 0.2, 10.0, 200_000, 5, "cpu")
    assert abs(got["best"] / 1.3 - 1) < 0.01 and len(got["top"]) == 5
    assert got["top"][0] == got["best"]
    # simulated sinusoid records: the whole curve and the first 30 d window
    # with at least 8 points; the top-5 oracle always has the period
    for r in _sinusoid_records():
        full = period_probe.ls_periods(r.t, r.y, r.err, r.band, 0.2, 10.0, 200_000, 5, "cpu")
        ratio, _, hit10, alias = period_probe.recovery(np.array([full["best"]]), np.array([r.period]))
        assert hit10[0] or alias[0], (r.period, full["best"])
        top = np.asarray(full["top"])
        assert np.abs(top / r.period - 1).min() < 0.10
        # the densest 30 d window (starts every 5 d), a clean single window
        t = r.t.astype(np.float64)
        starts = np.arange(t.min(), t.max() - 30, 5.0)
        s = starts[np.argmax([((t >= s) & (t < s + 30)).sum() for s in starts])]
        m = (t >= s) & (t < s + 30)
        assert m.sum() >= 15
        win = period_probe.ls_periods(t[m], r.y[m], r.err[m], r.band[m], 0.2, 10.0, 200_000, 5, "cpu")
        top = np.asarray(win["top"])
        assert np.abs(top / r.period - 1).min() < 0.10 or np.abs(top / r.period - 0.5).min() < 0.05 or np.abs(top / r.period - 2).min() < 0.2


def test_gls_power_matches_sine_fit():
    # the power at the true frequency of a noiseless sinusoid is 1, and the
    # floating mean removes an offset exactly
    t = np.sort(np.random.default_rng(1).uniform(0, 50, 200))
    y = 3.0 + 0.7 * np.sin(2 * np.pi * t / 2.5 + 0.4)
    p = period_probe.gls_power(t, y, np.ones_like(t), np.array([1 / 2.5, 1 / 3.1]), "cpu")
    assert p[0] == pytest.approx(1.0, abs=1e-6) and p[1] < 0.5
    f = period_probe.freq_grid(100.0, 0.5, 10.0, 200_000)
    assert f[0] == pytest.approx(1 / 200) and f[-1] == pytest.approx(4.0)
    assert len(f) == int((4.0 - 1 / 200) * 1000) + 1
    assert len(period_probe.freq_grid(100.0, 0.5, 10.0, 50)) == 50


def test_recovery_and_score():
    p_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    p_pred = np.array([1.005, 1.05, 2.05, 0.51, 3.0])
    ratio, h1, h10, alias = period_probe.recovery(p_pred, p_true)
    assert h1.tolist() == [True, False, False, False, False]
    assert h10.tolist() == [True, True, False, False, False]
    assert alias.tolist() == [False, False, True, True, False]
    groups = np.array(["RR", "RR", "ECL", "ECL", "ECL"])
    # shifted truths: only the first object stays a hit, only the third an alias
    y_true = np.log10(p_true) + np.array([0, 0.1, 0, 0.2, 0])
    s = period_probe.score(np.log10(p_pred), y_true, groups, {"RR": 0.0, "ECL": 0.1}, ["RR", "ECL"], min_n=2)
    assert s["n"] == 5 and s["rec10"] == pytest.approx(0.2) and s["alias"] == pytest.approx(0.2)
    assert s["rec1"] == pytest.approx(0.2) and s["med_abs"] == pytest.approx(math.log10(2.05))
    assert set(s["by_superclass"]) == {"RR", "ECL"} and s["by_superclass"]["ECL"]["n"] == 3
    assert math.isfinite(s["r2"]) and math.isfinite(s["r2_within"])
    peaks = period_probe.top_peaks(np.array([0.1, 0.5, 0.2, 0.9, 0.3, 0.4, 0.0]), np.arange(7.0) + 1, 3)
    assert peaks.tolist() == [4.0, 2.0, 6.0]
