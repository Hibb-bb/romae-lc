"""The test card of one run: the four cards, one JSON.

    python -m project.eval.test_card --run enc_o3_ph      # project/runs/enc_o3_ph/mae.pt
                                                          # -> project/results/test_card/enc_o3_ph.json

Steps, each with the protocol the old job scripts used (nothing is tuned here):
  latents   ``project.eval.encode``: latents.pt (train + validation) and latents_test.pt next to the
            checkpoint, made only when missing (window stride 0.25).
  class     ``card_class``: star probe on the 13 fine classes, 40 epochs, seeds 42 / 100 / 200.
            Fit on train, epoch picked on validation, scored on TEST.
  period    ``card_period``: the same probe on log10 period (TEST), and the spectral-peak hit rate on
            all VALIDATION stars with a period.
  phase     ``card_phase``: per-token phase R2 on 600 VALIDATION stars, split 70 / 30 by star.
  forecast  ``card_forecast``, three modes (last 10 days hidden, 30 % random gaps, a 0.2-cycle phase
            gap): every TEST star is scored, non-overlapping windows; the latent decoder is fitted on
            1,000 validation stars.
  rf        only with ``--rf``: the random-forest reference (``classify_rf``, 16 classes, fixed
            parameters, fit on train + validation, scored on TEST); it also writes its own folder
            project/results/classify_rf_NAME.

The JSON holds the full result of every card under ``cards``, the few table numbers under ``headline``
(``project.eval.summary`` prints them for many runs), and the checkpoint step, its training flags and
the git commit. The file is written after every card. A card that is already in the file (for the same
checkpoint step) is not run again, so a job that ran out of time can be started again; ``--force``
runs everything again.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch

from project.eval import card_class, card_forecast, card_period, card_phase, classify_rf, encode
from project.eval.common_eval import ZTFXPC, git_commit, run_name, write_json
from project.eval.summary import headline

CARDS = ("class", "period", "phase", "forecast")
#: The fixed random-forest settings of the old job (eval_encoder.sh).
RF_PARAMS = '{"max_depth": null, "min_samples_split": 5, "n_estimators": 100}'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run name: the checkpoint is project/runs/NAME/mae.pt")
    g.add_argument("--ckpt", help="or the checkpoint itself (the run name is its folder's name)")
    p.add_argument("--out", default=None, help="the JSON (default: <results>/test_card/NAME.json)")
    p.add_argument("--results", default="project/results", help="results folder")
    p.add_argument("--cards", nargs="+", default=list(CARDS), choices=CARDS)
    p.add_argument("--rf", action="store_true", help="also the random-forest reference")
    p.add_argument("--force", action="store_true", help="run cards again that are already in the JSON")
    p.add_argument("--data", default=ZTFXPC, help="data root of the peak, phase and forecast cards; the class and "
                   "period probes read the checkpoint's own data. 'sim' for a toy run")
    p.add_argument("--epochs", type=int, default=40, help="epochs of the star probe")
    p.add_argument("--min-context", type=int, default=16, help="forecast: fewest visible tokens of a scored window")
    p.add_argument("--workers", type=int, default=8, help="data workers of the latent cache")
    return p.parse_args(argv)


def make_latents(ckpt: Path, workers: int) -> tuple[str, str]:
    """The two caches next to the checkpoint, made when missing."""
    lat, lat_test = ckpt.parent / "latents.pt", ckpt.parent / "latents_test.pt"
    common = ["--ckpt", str(ckpt), "--workers", str(workers), "--batch-size", "64"]
    if not lat.exists():
        encode.main(common + ["--out", str(lat)])
    if not lat_test.exists():
        encode.main(common + ["--out", str(lat_test), "--splits", "test"])
    return str(lat), str(lat_test)


def main(argv=None):
    a = parse_args(argv)
    ckpt = Path(a.ckpt) if a.ckpt else Path("project/runs") / a.run / "mae.pt"
    name = a.run or run_name(ckpt)
    out = Path(a.out) if a.out else Path(a.results) / "test_card" / f"{name}.json"
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    step, train_args = saved.get("step"), saved.get("args")
    del saved

    card = dict(run=name, ckpt=str(ckpt), step=step, cards={}, seconds={})
    if out.exists() and not a.force:
        old = json.load(open(out))
        if old.get("step") == step and old.get("ckpt") == str(ckpt):
            card["cards"], card["seconds"] = old.get("cards", {}), old.get("seconds", {})
        else:
            print(f"{out} is from another checkpoint (step {old.get('step')}): starting again", flush=True)
    card.update(commit=git_commit(), data=a.data, args=train_args)

    def done(key, result, t0):
        card["cards"][key] = result
        card["seconds"][key] = round(time.time() - t0)
        card["headline"] = headline(card["cards"])
        write_json(out, card)
        print(f"[test card] {name}: {key} done in {card['seconds'][key]} s -> {out}", flush=True)

    todo = [c for c in a.cards if c not in card["cards"]] + (["rf"] if a.rf and "rf" not in card["cards"] else [])
    print(f"[test card] {name}: step {step}, cards to run: {todo or 'none (all in ' + str(out) + ')'}", flush=True)
    inputs = None
    if {"class", "period", "rf"} & set(todo):
        lat, lat_test = make_latents(ckpt, a.workers)
        hp = argparse.ArgumentParser()
        card_class.add_probe_args(hp)
        hp = hp.parse_args(["--epochs", str(a.epochs)])
        hp.latents, hp.test_latents = lat, lat_test
        if {"class", "period"} & set(todo):
            inputs = card_class.probe_inputs(lat, lat_test)

    if "class" in todo:
        t0 = time.time()
        done("class", dict(split="test", **card_class.run(hp, ("class",), inputs)), t0)
    if "period" in todo:
        t0 = time.time()
        logp = dict(split="test", **card_class.run(hp, ("logp",), inputs))
        peak = card_period.peak_hits([str(ckpt)], a.data, 100000, "validation", label=name)[str(ckpt)]
        done("period", dict(logp=logp, peak=peak), t0)
    inputs = None
    if "phase" in todo:
        t0 = time.time()
        res = card_phase.run([str(ckpt)], a.data, n_stars=600, label=name, split="validation")[str(ckpt)]
        done("phase", dict(split="validation", **res), t0)
    if "forecast" in todo:
        t0 = time.time()
        res = {}
        for mode, kw in card_forecast.CARD_MODES.items():
            r = card_forecast.run(str(ckpt), card_forecast.Mode(min_context=a.min_context, **kw), a.data, **card_forecast.TEST_CARD)
            if r is None:  # not a JEPA: no predictor to test
                res = None
                break
            res[mode] = dict(protocol=card_forecast.TEST_CARD, **r[1])
        done("forecast", res, t0)
    if "rf" in todo:
        t0 = time.time()
        rf_out = Path(a.results) / f"classify_rf_{name}"
        s = classify_rf.main(["--latents", lat, "--test-latents", lat_test, "--features", "mean", "--n-jobs", "16",
                              "--skip-hpo", "--best-params", RF_PARAMS, "--out", str(rf_out)])
        keep = ("dim", "best_params", "seeds", "f1_mean", "f1_std", "accuracy_mean", "accuracy_std", "precision_mean", "recall_mean")
        done("rf", dict(split="test", folder=str(rf_out), labels=s["labels"], **{f: {k: r[k] for k in keep} for f, r in s["results"].items()}), t0)
    card["headline"] = headline(card["cards"])
    write_json(out, card)
    print(f"[test card] {name}: " + json.dumps({k: round(v, 4) for k, v in card["headline"].items()}), flush=True)
    print("wrote", out)
    return card


if __name__ == "__main__":
    main()
