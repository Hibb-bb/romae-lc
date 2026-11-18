"""Labels, superclasses and catalogue periods of every ZTFxPC record, per split,
as one small file for the dataset figures (project.eval.dataset_figs):
project/results/figures/ztfxpc_summary.npz. Prints the class counts.

    python -m project.eval.dataset_summary
"""
import numpy as np
from project.common import ANOMALY_CLASSES, _read_ckpt, data_args_from, load_data

meta_args = _read_ckpt("project/runs/maew_spec/mae.pt")["args"]
d = load_data(data_args_from(meta_args), splits=("train", "validation", "test"))
out = {}
for split in ("train", "validation", "test"):
    recs = d[split]
    out[f"{split}_label"] = np.array([str(r.meta.get("class_str", "?")) for r in recs])
    out[f"{split}_period"] = np.array([float(r.period) if r.period is not None else np.nan for r in recs])
    out[f"{split}_super"] = np.array([str(r.meta.get("superclass_str", "?")) for r in recs])
    print(split, len(recs), "records", flush=True)
np.savez("project/results/figures/ztfxpc_summary.npz", **out)
lab = np.concatenate([out[f"{s}_label"] for s in ("train", "validation", "test")])
per = np.concatenate([out[f"{s}_period"] for s in ("train", "validation", "test")])
u, c = np.unique(lab, return_counts=True)
print("records", len(lab), "with period", int(np.isfinite(per).sum()),
      "period min/median/max", float(np.nanmin(per)), float(np.nanmedian(per)), float(np.nanmax(per)))
for a, b in sorted(zip(u, c), key=lambda x: -x[1]):
    print(f"{a:14s} {b:6d}{'  (anomaly, left out of the 13)' if a in ANOMALY_CLASSES else ''}")
