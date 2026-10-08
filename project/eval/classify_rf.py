"""Fine-class classification with a random forest on frozen features, the
collaborators' protocol: a grid search on the training split with the
validation split as the one held-out fold (macro F1), then three seeds
trained on train + validation and scored on the TEST split (macro F1,
accuracy, precision, recall, a per-class report, a confusion matrix).

Feature sets: ``mean`` (the pooled latent averaged over a star's valid
windows), ``hand`` (the hand-crafted statistics of ``period_probe``, the
no-model baseline), ``meanhand`` (both). Labels: the fine ``class_str``
labels of the training split (``--exclude-anomaly`` drops the three anomaly
classes everywhere).

    python -m project.eval.classify_rf --latents latents.pt --test-latents latents_test.pt --features mean hand --out DIR
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from project.common import ANOMALY_CLASSES, data_args_from, dump_json, load_data
from project.period_probe import object_table

PARAM_GRID = {"n_estimators": [100, 200, 500], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]}
DEFAULT_PARAMS = {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2}


def plot_confusion(cm, labels, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmn = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.55 * len(labels) + 2, 0.5 * len(labels) + 2), dpi=150)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cmn[i, j] > 0.005:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", required=True, help="cache with the train and validation splits")
    p.add_argument("--test-latents", required=True, help="cache with the test split")
    p.add_argument("--features", nargs="+", default=["mean", "hand"], choices=["mean", "hand", "meanhand"])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--exclude-anomaly", action="store_true", help="drop the three anomaly classes from every split")
    p.add_argument("--skip-hpo", action="store_true")
    p.add_argument("--best-params", default=None, help="JSON, with --skip-hpo")
    p.add_argument("--grid", default=None, help="JSON parameter grid (default: the collaborators' 36-point grid)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--n-objects", type=int, default=0, help="per split, for a quick run")
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    return p.parse_args(argv)


def run(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    tcache = torch.load(args.test_latents, map_location="cpu", weights_only=False)
    meta = cache["meta"]
    ckpt = torch.load(meta["ckpt"], map_location="cpu", weights_only=False)
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(ckpt["args"], over), splits=("train", "validation", "test"))
    bands = sorted({int(b) for r in data["train"][:200] for b in np.unique(r.band)})
    caches = {"train": cache, "validation": cache, "test": tcache}
    tables = {s: object_table(caches[s], s, data[s], bands, int(meta["min_tokens"])) for s in ("train", "validation", "test")}
    rng = np.random.default_rng(args.seed)
    for s, tab in tables.items():
        keep = np.ones(len(tab["fine"]), dtype=bool)
        if args.exclude_anomaly:
            keep &= ~np.isin(tab["fine"], ANOMALY_CLASSES)
        if args.n_objects and keep.sum() > args.n_objects:
            idx = np.flatnonzero(keep)
            keep[:] = False
            keep[rng.choice(idx, args.n_objects, replace=False)] = True
        tab["sel"] = keep
    labels = sorted(set(tables["train"]["fine"][tables["train"]["sel"]].tolist()))
    label2idx = {l: i for i, l in enumerate(labels)}
    print(f"splits {[(s, int(t['sel'].sum())) for s, t in tables.items()]}; {len(labels)} classes: {labels}; loaded in {time.time() - t0:.0f}s", flush=True)
    seeds = [args.seed, args.seed + 58, args.seed + 158]
    grid = json.loads(args.grid) if args.grid else PARAM_GRID
    results = {}
    for feat in args.features:
        def xy(split):
            tab = tables[split]
            sel = tab["sel"] & np.isin(tab["fine"], labels)
            f = tab["features"]
            x = np.concatenate([f["mean"], f["hand"]], 1) if feat == "meanhand" else f[feat]
            return x[sel].astype(np.float32), np.array([label2idx[c] for c in tab["fine"][sel]])
        x_tr, y_tr = xy("train")
        x_va, y_va = xy("validation")
        x_te, y_te = xy("test")
        if args.standardize:
            sc = StandardScaler().fit(x_tr)
            x_tr, x_va, x_te = sc.transform(x_tr), sc.transform(x_va), sc.transform(x_te)
        t1 = time.time()
        if args.skip_hpo:
            best = json.loads(args.best_params.replace("None", "null")) if args.best_params else dict(DEFAULT_PARAMS)
            cv_score = None
        else:
            ps = PredefinedSplit(np.concatenate([-np.ones(len(x_tr), dtype=int), np.zeros(len(x_va), dtype=int)]))
            gs = GridSearchCV(RandomForestClassifier(random_state=args.seed, n_jobs=1), grid, cv=ps, scoring="f1_macro", n_jobs=args.n_jobs, verbose=0)
            gs.fit(np.concatenate([x_tr, x_va]), np.concatenate([y_tr, y_va]))
            best, cv_score = dict(gs.best_params_), float(gs.best_score_)
        print(f"[{feat}] dim {x_tr.shape[1]}; best params {best} (validation macro F1 {cv_score}); {time.time() - t1:.0f}s", flush=True)
        x_full, y_full = np.concatenate([x_tr, x_va]), np.concatenate([y_tr, y_va])
        per_seed, reports, cms = [], [], []
        for s in seeds:
            model = RandomForestClassifier(random_state=s, n_jobs=args.n_jobs, **best).fit(x_full, y_full)
            y_pred = model.predict(x_te)
            pr, rc, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
            per_seed.append(dict(seed=s, f1=float(f1), accuracy=float(accuracy_score(y_te, y_pred)), precision=float(pr), recall=float(rc)))
            reports.append(classification_report(y_te, y_pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0))
            cms.append(confusion_matrix(y_te, y_pred, labels=list(range(len(labels)))))
            print(f"  [{feat}] seed {s}: macro F1 {f1:.4f}  acc {per_seed[-1]['accuracy']:.4f}  precision {pr:.4f}  recall {rc:.4f}", flush=True)
        mean_report = {c: {m: float(np.mean([r[c][m] for r in reports])) for m in ("precision", "recall", "f1-score", "support")} for c in labels}
        cm_mean = np.mean(cms, 0)
        plot_confusion(cm_mean, labels, out / f"confusion_{feat}.png", f"{feat}: test confusion matrix, mean of {len(seeds)} seeds (rows normalised)")
        with open(out / f"report_{feat}.csv", "w") as f:
            f.write("class,precision,recall,f1,support\n")
            for c in labels:
                r = mean_report[c]
                f.write(f"{c},{r['precision']:.4f},{r['recall']:.4f},{r['f1-score']:.4f},{r['support']:.0f}\n")
        results[feat] = dict(dim=int(x_tr.shape[1]), best_params=best, validation_macro_f1=cv_score, seeds=per_seed,
                             f1_mean=float(np.mean([p["f1"] for p in per_seed])), f1_std=float(np.std([p["f1"] for p in per_seed])),
                             accuracy_mean=float(np.mean([p["accuracy"] for p in per_seed])), accuracy_std=float(np.std([p["accuracy"] for p in per_seed])),
                             precision_mean=float(np.mean([p["precision"] for p in per_seed])), recall_mean=float(np.mean([p["recall"] for p in per_seed])),
                             per_class=mean_report, confusion_mean=cm_mean.tolist())
        print(f"[{feat}] TEST macro F1 {results[feat]['f1_mean']:.4f} ± {results[feat]['f1_std']:.4f}, accuracy {results[feat]['accuracy_mean']:.4f} ± {results[feat]['accuracy_std']:.4f}", flush=True)
    summary = dict(latents=args.latents, test_latents=args.test_latents, labels=labels, n={s: int(t["sel"].sum()) for s, t in tables.items()},
                   exclude_anomaly=args.exclude_anomaly, standardize=args.standardize, grid=grid, seeds=seeds, results=results, seconds=time.time() - t0)
    dump_json(summary, out / "results.json")
    md = [f"# Fine-class classification, random forest on frozen features ({len(labels)} classes, test split)\n",
          "| features | dim | validation macro F1 (grid search) | test macro F1 | test accuracy | precision | recall |", "|---|---|---|---|---|---|---|"]
    for feat, r in results.items():
        md.append(f"| {feat} | {r['dim']} | {'' if r['validation_macro_f1'] is None else round(r['validation_macro_f1'], 3)} | {r['f1_mean']:.3f} ± {r['f1_std']:.3f} | {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} | {r['precision_mean']:.3f} | {r['recall_mean']:.3f} |")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
