"""One table over runs, from the JSON files of ``project.eval.test_card``.

    python -m project.eval.summary                          # every run in project/results/test_card
    python -m project.eval.summary --runs enc_o3_ph enc_le1 --csv table.csv

Columns (the split each number is scored on is in the legend under the table):
  F1 attn / F1 mean   class macro F1 of the star probe, attention pooling and the mean control (3 seeds)
  logP R2 / P 10 %    log-period probe: R2, and the share of stars within 10 % of the period
  peak 10 % / 1 %     spectral-peak hit rate, strict / alias-tolerant
  phase cos           per-token phase, cosine R2 of the MLP on token + CLS
  tail10, gap0.3, pgap0.2   forecast: share of hidden points within 2 sigma, decoder of the predicted latent
  RF F1               the random-forest reference, when the card has it
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

FOURIER = "Fourier fit (4 harmonics, f of the window)"
LEGEND = ("Splits: F1, logP R2 and P 10 % on TEST (probe fit on train, epoch picked on validation). "
          "Peak and phase on VALIDATION stars. Forecast on TEST stars (decoder fit on validation stars). "
          "RF F1 on TEST (fit on train + validation).")


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def headline(cards: dict) -> dict:
    """The few numbers of a card that go into the table. A missing card gives no keys."""
    h = {}
    for pool in ("attn", "mean"):
        ms = _get(cards, "class", f"class/{pool}", "mean_std")
        if ms:
            h[f"class_f1_{pool}"], h[f"class_f1_{pool}_std"] = ms["macro_f1"]
            h[f"class_f1_{pool}_val"] = ms["val_best"][0]
    ms = _get(cards, "period", "logp", "logp/attn", "mean_std")
    if ms:
        h["logp_r2"], h["logp_rec10"], h["logp_rec1"] = ms["r2"][0], ms["rec10"][0], ms["rec1"][0]
    pk = _get(cards, "period", "peak")
    if pk:
        for i, t in enumerate(("10", "1", "0.1", "0.01")):
            h[f"peak_{t}_strict"], h[f"peak_{t}_alias"] = pk["strict"][i], pk["alias"][i]
    ph = _get(cards, "phase")
    if ph:
        h["phase_cos_r2"], h["phase_sin_r2"] = ph["h+cls | mlp"]
        h["phase_cos_r2_raw"] = ph["raw | mlp"][0]
        h["phase_cos_r2_shuffled"] = ph["h+cls shuffled | mlp (control)"][0]
    for mode, res in (_get(cards, "forecast") or {}).items():
        for key, row in (("pred", "predicted latent [mlp]"), ("recon", "recon head"), ("fourier", FOURIER), ("mean", "window mean")):
            v = _get(res, row, "all", "within_2")
            if v is not None:
                h[f"{mode}_{key}_within2"] = v
    rf = _get(cards, "rf", "mean")
    if rf:
        h["rf_f1"], h["rf_f1_std"] = rf["f1_mean"], rf["f1_std"]
    return h


COLUMNS = [
    ("F1 attn", lambda h: _pm(h, "class_f1_attn")), ("F1 mean", lambda h: _pm(h, "class_f1_mean")),
    ("logP R2", lambda h: _f(h, "logp_r2")), ("P 10 %", lambda h: _f(h, "logp_rec10")),
    ("peak 10 %", lambda h: _pair(h, "peak_10")), ("peak 1 %", lambda h: _pair(h, "peak_1")),
    ("phase cos", lambda h: _f(h, "phase_cos_r2")),
    ("tail10", lambda h: _f(h, "tail10_pred_within2")), ("gap0.3", lambda h: _f(h, "gap0.3_pred_within2")),
    ("pgap0.2", lambda h: _f(h, "pgap0.2_pred_within2")), ("RF F1", lambda h: _pm(h, "rf_f1")),
]  # fmt: skip


def _f(h, k):
    return "" if h.get(k) is None else f"{h[k]:.3f}"


def _pm(h, k):
    return "" if h.get(k) is None else f"{h[k]:.3f} ± {h[k + '_std']:.3f}"


def _pair(h, k):
    return "" if h.get(k + "_strict") is None else f"{h[k + '_strict']:.3f} / {h[k + '_alias']:.3f}"


def rows(folder, runs=None) -> list[dict]:
    """One row per card file: run, step and the table cells."""
    files = [Path(folder) / f"{r}.json" for r in runs] if runs else sorted(Path(folder).glob("*.json"))
    out = []
    for f in files:
        if not f.exists():
            print(f"no card for {f.stem} ({f})")
            continue
        card = json.load(open(f))
        h = headline(card.get("cards", {}))
        out.append({"run": card.get("run", f.stem), "step": str(card.get("step", "")), **{name: cell(h) for name, cell in COLUMNS}})
    return out


def table(rs: list[dict]) -> str:
    """A markdown table; columns that are empty for every run are left out."""
    names = ["run", "step"] + [n for n, _ in COLUMNS if any(r[n] for r in rs)]
    lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    lines += ["| " + " | ".join(r[n] for n in names) + " |" for r in rs]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="project/results/test_card", help="folder with the card files")
    p.add_argument("--runs", nargs="*", default=None, help="run names, in table order (default: every file)")
    p.add_argument("--csv", default=None, help="also write the table as CSV")
    a = p.parse_args(argv)
    rs = rows(a.dir, a.runs)
    if not rs:
        print(f"no card files in {a.dir}")
        return rs
    print(table(rs))
    print()
    print(LEGEND)
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
        print("wrote", a.csv)
    return rs


if __name__ == "__main__":
    main()
