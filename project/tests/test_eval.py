"""CPU tests of the evaluation helpers (project/eval): the shared scores, the forecast modes, the
summary table. The cards themselves are not run here: an end-to-end run
on the toy simulator takes minutes (``python -m project.eval.test_card --ckpt ... --data sim``)."""

import numpy as np

from project.eval import summary
from project.eval.card_forecast import CARD_MODES, TEST_CARD, Mode
from project.eval.common_eval import fourier_forecast, period_hits, run_name, zstats


def test_period_hits_strict_and_alias():
    p = np.array([1.0, 2.0, 4.0, 5.0, np.nan])
    p_hat = np.array([1.05, 4.0, 2.0, np.nan, 3.0])  # close, double, half, no estimate, no truth
    n, strict, alias = period_hits(p_hat, p)
    assert n == 3
    assert strict == [1 / 3, 0.0, 0.0, 0.0]
    assert alias == [1.0, 2 / 3, 2 / 3, 2 / 3]


def test_zstats():
    s = zstats(np.array([-3.0, -0.5, 0.5, 1.5]))
    assert s == dict(n=4, median_abs_z=1.0, within_1=0.5, within_2=0.75, mean_z=-0.375)
    assert zstats(np.array([]))["n"] == 0 and np.isnan(zstats(np.array([]))["within_2"])


def test_fourier_forecast_recovers_a_sine():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 30, 80))
    f = 0.31
    curve = lambda x: 0.3 + 0.8 * np.sin(2 * np.pi * f * x + 0.4) + 0.2 * np.cos(4 * np.pi * f * x)
    t_new = np.linspace(30, 40, 20)
    y = fourier_forecast(t, curve(t), np.full(len(t), 0.1), f, t_new, 4)
    assert np.allclose(y, curve(t_new), atol=1e-6)


def test_forecast_modes():
    tail, gap, pgap = (Mode(**CARD_MODES[k]) for k in ("tail10", "gap0.3", "pgap0.2"))
    assert (tail.gap, gap.gap, pgap.gap) == (False, True, True) and (tail.pgap, gap.pgap, pgap.pgap) == (False, False, True)
    assert (tail.suffix, gap.suffix, pgap.suffix) == ("", "_gap0.3", "_pgap0.2")
    assert tail.dt_bins == [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0)]
    assert Mode(tail=60.0, dt=(0.5, 60.0)).dt_bins[-1] == (40.0, 60.0)
    assert gap.dt_bins == pgap.dt_bins == [(0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 60.0)]
    assert TEST_CARD == dict(split="test", fit_split="validation", n_fit=1000, n_stars=0, stride=1.0)
    assert run_name("project/runs/enc_x/mae.pt") == "enc_x"


def test_summary_table(tmp_path):
    import json

    row = lambda v: {"all": dict(within_2=v)}
    cards = {
        "class": {"class/attn": dict(mean_std=dict(macro_f1=[0.6, 0.01], acc=[0.8, 0.0], val_best=[0.65, 0.0])),
                  "class/mean": dict(mean_std=dict(macro_f1=[0.58, 0.02], acc=[0.8, 0.0], val_best=[0.63, 0.0]))},
        "period": {"logp": {"logp/attn": dict(mean_std=dict(r2=[0.9, 0.0], rec1=[0.1, 0.0], rec10=[0.7, 0.0], med_dex=[0.0, 0.0]))},
                   "peak": dict(strict=[0.5, 0.4, 0.3, 0.2], alias=[0.8, 0.7, 0.6, 0.5])},
        "phase": {"h+cls | mlp": [0.4, 0.7], "raw | mlp": [0.0, 0.5], "h+cls shuffled | mlp (control)": [0.0, 0.0]},
        "forecast": {"tail10": {"predicted latent [mlp]": row(0.7), "recon head": row(0.72), summary.FOURIER: row(0.5), "window mean": row(0.2)}},
    }  # fmt: skip
    h = summary.headline(cards)
    assert h["class_f1_attn"] == 0.6 and h["class_f1_attn_val"] == 0.65 and h["logp_r2"] == 0.9
    assert h["peak_10_strict"] == 0.5 and h["peak_0.01_alias"] == 0.5 and h["phase_cos_r2"] == 0.4
    assert h["tail10_pred_within2"] == 0.7 and h["tail10_fourier_within2"] == 0.5 and "gap0.3_pred_within2" not in h
    assert summary.headline({}) == {}
    (tmp_path / "enc_a.json").write_text(json.dumps(dict(run="enc_a", step=20, cards=cards)))
    (tmp_path / "enc_b.json").write_text(json.dumps(dict(run="enc_b", step=5, cards={})))
    rs = summary.rows(tmp_path)
    text = summary.table(rs)
    assert [r["run"] for r in rs] == ["enc_a", "enc_b"]
    assert "| enc_a | 20 | 0.600 ± 0.010 | 0.580 ± 0.020 | 0.900 | 0.700 | 0.500 / 0.800 | 0.400 / 0.700 | 0.400 | 0.700 |" in text
    assert "gap0.3" not in text and "RF F1" not in text  # empty columns are left out
    assert summary.rows(tmp_path, ["enc_b", "missing"])[0]["run"] == "enc_b"
