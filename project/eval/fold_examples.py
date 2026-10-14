"""Side-by-side folds for the talk: validation stars where the model's
period folds the light curve much more cleanly than the catalogue's. Picks
``--n`` stars from the report's ``model_fine_search.csv`` with the largest
R2 gain, at most two per class and a mix of the two verdicts (another
period; the same period made sharper), and draws each as one row:
catalogue period on the left, model period on the right, points only. One
figure with every row and one PNG per star, at slide-sized text.

    python -m project.eval.fold_examples --report project/results/period_report_spec4 --out project/results/slides/folds
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.common import data_args_from, load_data, load_encoder
from project.eval.period_report import fold_panel
from project.plot_period_examples import INK, INK2, SURFACE


def artifact_ids(rows):
    """The stars the artifact page already shows (its first six 'different period' rows and first three others)."""
    diff = [r for r in rows if r["verdict"] == "different period"][:6]
    imp = [r for r in rows if r["verdict"] != "different period"][:3]
    return {r["ztf_id"] for r in diff + imp}


def pick(rows, n, per_class=2, min_points=400, seed=0, min_gain=0.5):
    skip = artifact_ids(rows)
    rows = [r for r in rows if int(r["n_points"]) >= min_points and float(r["r2_best"]) >= 0.8 and float(r["gain"]) >= min_gain and r["ztf_id"] not in skip]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)  # a random draw from the clear cases, not the top of the list
    out, seen = [], {}
    want = {"different period": (n + 1) // 2, "catalogue imprecise": n // 2}
    for r in rows:
        c, v = r["class"], r["verdict"]
        if seen.get(c, 0) >= per_class or want.get(v, 0) <= 0:
            continue
        out.append(r); seen[c] = seen.get(c, 0) + 1; want[v] -= 1
        if len(out) == n:
            break
    return out


def draw_row(axes, r, row, fs, cat_name="catalogue", extra=()):
    """One row: the catalogue period, any ``extra`` (period, title, r2) panels, the model's period."""
    cat = (float(row["p_catalogue_d"]), f"{cat_name} period", float(row["r2_catalogue"]))
    mod = (float(row["p_best_d"]), "model period" + (f" ({float(row['best_over_catalogue']):.3g} × the {cat_name})" if row["verdict"] == "different period" else " (same, sharpened)"), float(row["r2_best"]))
    for ax, (p, title, r2) in zip(axes, (cat, *extra, mod)):
        fold_panel(ax, r, p, title, r2, template=False)
        ax.set_title(f"{title}\nP = {p:.6g} d, fold R2 {r2:.2f}", fontsize=fs["title"], color=INK, loc="left")
        ax.tick_params(labelsize=fs["tick"])
        for line in ax.get_lines():
            line.set_markersize(max(line.get_markersize(), 3.5))
    axes[0].set_ylabel(f"{row['class']}\n{row['ztf_id']}\n\nbrightness", fontsize=fs["label"], color=INK)
    for ax in axes:
        ax.set_xlabel("phase (shown twice)", fontsize=fs["label"], color=INK2)


def chen_rows(chen_dir: str, n: int, seed: int, per_class: int = 2, min_points: int = 400, exclude=(), min_r2: float = 0.8):
    """Rows in the same shape as the report's list, from the finished Chen shards: the model's candidate
    period against Chen's listed period, where the model's fold is clean and Chen's is not."""
    rows = []
    for f in sorted(Path(chen_dir).glob("shard_*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                r2m, r2c, pc, pm = float(r["r2_model_cands"]), float(r["r2_chen"]), float(r["p_chen"]), float(r["p_model_cands"])
            except ValueError:
                continue
            if not (np.isfinite(r2m) and np.isfinite(r2c) and np.isfinite(pc) and pc > 0) or int(r["n_points"]) < min_points:
                continue
            if r["sourceid"] in exclude:
                continue
            if min_points and (r2m < min_r2 or r2m - r2c < 0.5 or pm > 100.0):  # beyond ~100 d a 6-harmonic wave fits almost any slow curve: no evidence
                continue
            rows.append(dict(ztf_id=r["sourceid"], shard=r["shard"], **{"class": r["class_chen"]}, verdict="catalogue imprecise" if r["kind"] == "same period, sharper" else "different period",
                             p_catalogue_d=pc, p_best_d=pm, best_over_catalogue=pm / pc, r2_catalogue=r2c, r2_best=r2m, gain=r2m - r2c, n_points=r["n_points"]))
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    out, seen = [], {}
    want = {"different period": (n + 1) // 2, "catalogue imprecise": n // 2}
    for r in rows:
        c, v = r["class"], r["verdict"]
        if seen.get(c, 0) >= per_class or want.get(v, 0) <= 0:
            continue
        out.append(r); seen[c] = seen.get(c, 0) + 1; want[v] -= 1
        if len(out) == n:
            break
    return out


def chen_records(rows, root: str):
    from project.eval.period_chen import load_shard

    recs = {}
    for shard in sorted({r["shard"] for r in rows}):
        want = {r["ztf_id"] for r in rows if r["shard"] == shard}
        for rec in load_shard(str(Path(root) / shard)):
            if rec.meta["id"] in want:
                recs[rec.meta["id"]] = rec
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="project/results/period_report_spec4")
    ap.add_argument("--ckpt", default="project/runs/maew_spec/mae.pt", help="for the data arguments")
    ap.add_argument("--out", default="project/results/slides/folds")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chen", default=None, help="a period_chen results folder: draw Chen-set stars instead (model against Chen's listed period)")
    ap.add_argument("--chen-root", default="/work/hdd/bfrf/hibb/chen20_ztf_dr24")
    ap.add_argument("--ls-budget", type=int, default=0, help="also run the collaborator's astropy multiband Lomb-Scargle at this many trials on each star and draw its fold as a middle column")
    ap.add_argument("--ls-p-min", type=float, default=0.02)
    ap.add_argument("--ids", nargs="*", default=None, help="draw exactly these star ids (Chen sourceids or ZTF ids) instead of a random draw")
    ap.add_argument("--exclude", nargs="*", default=(), help="ids already shown elsewhere")
    ap.add_argument("--min-r2", type=float, default=0.8, help="the model's fold must reach this")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fs = dict(title=14, label=13, tick=12)
    if a.chen:
        rows = chen_rows(a.chen, a.n, a.seed, exclude=set(a.exclude), min_r2=a.min_r2) if not a.ids else chen_rows(a.chen, 10**6, a.seed, per_class=10**6, min_points=0)
        if a.ids:
            by = {r["ztf_id"]: r for r in rows}
            rows = [by[i] for i in a.ids if i in by]
        recs = chen_records(rows, a.chen_root)
        get, cat_name = (lambda row: recs[row["ztf_id"]]), "Chen"
    else:
        rows = pick(list(csv.DictReader(open(Path(a.report) / "model_fine_search.csv"))), a.n, seed=a.seed)
        _, meta = load_encoder(a.ckpt, "cpu")
        val = load_data(data_args_from(meta.args), splits=("validation",))["validation"]
        get, cat_name = (lambda row: val[int(row["val_index"])]), "catalogue"
    ncol = 3 if a.ls_budget else 2
    fig, axes = plt.subplots(len(rows), ncol, figsize=(6 * ncol, 3.6 * len(rows)), dpi=140, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax_row, row in zip(axes, rows):
        r = get(row)
        extra = ()
        if a.ls_budget:
            from project import fold
            from project.eval.ls_benchmark import astropy_search, star_inputs

            p_ls, _ = astropy_search(r, a.ls_budget, True, a.ls_p_min)
            t, y, err, band, _ = star_inputs(r)
            r2_ls = fold.fold_r2(t, y, err, band, p_ls, 6) if np.isfinite(p_ls) else float("nan")
            extra = ((p_ls, f"Lomb-Scargle today, {a.ls_budget // 1000}k trials ({p_ls / float(row['p_catalogue_d']):.3g} ×)", r2_ls),)
            row["p_ls_d"], row["r2_ls"] = p_ls, r2_ls
        draw_row(ax_row, r, row, fs, cat_name, extra)
        f2, ax2 = plt.subplots(1, ncol, figsize=(6 * ncol, 4.2), dpi=140)
        f2.patch.set_facecolor(SURFACE)
        draw_row(ax2, r, row, fs, cat_name, extra)
        f2.tight_layout(); f2.savefig(out / f"fold_{row['class'].replace('/', '-')}_{row['ztf_id']}.png", facecolor=SURFACE); plt.close(f2)
        ls_txt = f" | LS {row['p_ls_d']:.5g} d (R2 {row['r2_ls']:.2f})" if "p_ls_d" in row else ""
        print(f"{row['class']:10s} {row['ztf_id']}  verdict {row['verdict']:20s} catalogue {float(row['p_catalogue_d']):.5g} d (R2 {float(row['r2_catalogue']):.2f}){ls_txt} -> model {float(row['p_best_d']):.5g} d (R2 {float(row['r2_best']):.2f}), {row['n_points']} points")
    fig.suptitle(f"Stars where the model's period folds the light curve more cleanly than the {cat_name}{'' if cat_name == 'Chen' else chr(39) + 's'} period", fontsize=fs["title"] + 2, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "folds_all.png", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
