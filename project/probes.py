"""The small probes that run DURING training, on frozen features.

- the linear class probe (:func:`linear_probe`),
- the ridge regression on log10 period (:func:`ridge_r2`),
- the time-shuffle score (:func:`shuffle_score`),
- the hand-feature baseline that an encoder has to beat (:func:`baseline_probe`).

:func:`probe` runs the first two on the features of a model. They are fitted
on train records and scored on validation records. The catalogue period is
only a target here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from romae_lc import FrameDataset, Record, time_shuffle_score

from project.data import ANOMALY_CLASSES, fine_class, subset, superclass


def _standardize(z_tr, z_va):
    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-6
    return (z_tr - mu) / sd, (z_va - mu) / sd


def linear_probe(z_tr, y_tr, z_va, y_va, n_classes, steps=300, wd=1e-3) -> dict:
    """Multinomial logistic regression on standardised features (L-BFGS)."""
    x_tr, x_va = (torch.from_numpy(z).float() for z in _standardize(z_tr, z_va))
    y_tr, y_va = (torch.as_tensor(y, dtype=torch.long) for y in (y_tr, y_va))
    head = nn.Linear(x_tr.shape[1], n_classes)
    opt = torch.optim.LBFGS(
        head.parameters(), max_iter=steps, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(head(x_tr), y_tr) + wd * head.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pred_va, pred_tr = head(x_va).argmax(1).numpy(), head(x_tr).argmax(1).numpy()
    y_tr, y_va = y_tr.numpy(), y_va.numpy()
    majority = np.bincount(y_tr).argmax()
    return dict(
        acc=float((pred_va == y_va).mean()),
        train_acc=float((pred_tr == y_tr).mean()),
        macro_f1=macro_f1(y_va, pred_va),
        balanced_acc=balanced_accuracy(y_va, pred_va),
        majority_acc=float((y_va == majority).mean()),
        n_val=int(y_va.size),
    )


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean F1 over the classes present in ``y_true`` or
    ``y_pred`` (a class never predicted and never true does not count)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = []
    for c in np.unique(np.concatenate([y_true, y_pred])):
        tp = np.sum((y_pred == c) & (y_true == c))
        denom = (
            2 * tp
            + np.sum((y_pred == c) & (y_true != c))
            + np.sum((y_pred != c) & (y_true == c))
        )
        f1.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1))


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean recall over the classes present in ``y_true``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(
        np.mean([np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)])
    )


def ridge_r2(z_tr, t_tr, z_va, t_va, alpha=0.1) -> float:
    x_tr, x_va = (x.astype(np.float64) for x in _standardize(z_tr, z_va))
    t_tr, t_va = np.asarray(t_tr, dtype=np.float64), np.asarray(t_va, dtype=np.float64)
    n, d = x_tr.shape
    a = x_tr.T @ x_tr / n + alpha * np.eye(d)
    w = np.linalg.solve(a, x_tr.T @ (t_tr - t_tr.mean()) / n)
    resid = t_va - t_tr.mean() - x_va @ w
    return float(1.0 - (resid**2).sum() / ((t_va - t_va.mean()) ** 2).sum())


@torch.no_grad()
def embed_records(model, records, cfg, spec, batch_size, device, project=False, seed=0):
    """``(z [n, D], kept indices)``: mean over the windows of one deterministic
    sequence per record (records too short for a sequence are dropped)."""
    ds = FrameDataset(records, cfg, seed=seed, epoch_seed=False)
    loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=spec.collate())
    was = model.training
    model.eval()
    zs = []
    for batch in loader:
        frames = [f.to(device) for f in batch["frames"]]
        zs.append(model.encode(frames, project=project).mean(1).float().cpu())
    model.train(was)
    return torch.cat(zs).numpy(), list(ds.indices)


@torch.no_grad()
def shuffle_score(model, records, cfg, spec, device, n=512, batch_size=64, seed=0) -> float:
    """The package's :func:`~romae_lc.time_shuffle_score` on the first window
    of one deterministic sequence per record, standardised with shared
    statistics so the batches are comparable (near 1 = the encoder is blind
    to time)."""
    recs = subset(records, n, seed)
    ds = FrameDataset(recs, cfg, seed=seed, epoch_seed=False)
    loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=spec.collate())
    backbone = model.backbone
    was = backbone.training
    backbone.eval()
    toks = [b["frames"][0].to(device) for b in loader]
    z = torch.cat([backbone(*t).float() for t in toks])
    stats = (z.mean(0), z.std(0, unbiased=False) + 1e-6)
    gen = torch.Generator().manual_seed(seed)
    scores = [
        time_shuffle_score(backbone, t, generator=gen, stats=stats).cpu() for t in toks
    ]
    backbone.train(was)
    return float(torch.cat(scores).mean())


def probe_metrics(
    z_tr, r_tr, z_va, r_va, n_classes, min_train: int = 30, min_val: int = 10
) -> dict:
    """The probe metrics of features ``z`` of records ``r``: the linear
    class probe (accuracy, macro F1, balanced accuracy, the majority-class
    accuracy it must beat) and ridge R2 on log10 period pooled (``r2``),
    within superclasses (``r2_within``: the target is log10 period minus its
    superclass mean, so class-level period differences do not count) and
    per superclass with at least ``min_train`` / ``min_val`` records
    (``r2_by_superclass``, with every superclass's validation count in
    ``n_by_superclass``). Records of the :data:`ANOMALY_CLASSES` are left out
    of the class probe (fit and score; ``n_class_excluded`` counts them) and
    kept in every period metric."""
    y_tr, y_va = (np.array([r.label for r in rs]) for rs in (r_tr, r_va))
    # records without a period (the Chen set: no clean fold) stay out of the period metrics; unlabelled ones out of the class probe
    p_tr, p_va = (np.log10(np.array([float(r.period) if r.period and r.period > 0 else np.nan for r in rs])) for rs in (r_tr, r_va))
    k_tr, k_va = (
        np.array([fine_class(r) not in ANOMALY_CLASSES and r.label >= 0 for r in rs], dtype=bool)
        for rs in (r_tr, r_va)
    )
    out = linear_probe(z_tr[k_tr], y_tr[k_tr], z_va[k_va], y_va[k_va], n_classes)
    out["n_class_excluded"] = int((~k_tr).sum() + (~k_va).sum())
    f_tr, f_va = np.isfinite(p_tr), np.isfinite(p_va)
    z_tr, p_tr, r_tr = z_tr[f_tr], p_tr[f_tr], [r for r, f in zip(r_tr, f_tr) if f]
    z_va, p_va, r_va = z_va[f_va], p_va[f_va], [r for r, f in zip(r_va, f_va) if f]
    out["n_period"] = int(f_va.sum())
    out["r2"] = ridge_r2(z_tr, p_tr, z_va, p_va)
    g_tr, g_va = (np.array([superclass(r) for r in rs]) for rs in (r_tr, r_va))
    means = {g: float(p_tr[g_tr == g].mean()) for g in set(g_tr.tolist())}
    glob = float(p_tr.mean())
    c_tr = p_tr - np.array([means[g] for g in g_tr])
    c_va = p_va - np.array([means.get(g, glob) for g in g_va])
    out["r2_within"] = (
        ridge_r2(z_tr, c_tr, z_va, c_va) if c_va.std() > 0 else float("nan")
    )
    by, n_by = {}, {}
    for g in sorted(set(g_va.tolist())):
        m_tr, m_va = g_tr == g, g_va == g
        n_by[g] = int(m_va.sum())
        if m_tr.sum() >= min_train and m_va.sum() >= min_val and p_va[m_va].std() > 0:
            by[g] = ridge_r2(z_tr[m_tr], p_tr[m_tr], z_va[m_va], p_va[m_va])
    out["r2_by_superclass"], out["n_by_superclass"] = by, n_by
    return out


def probe(
    model, train, val, cfg, spec, batch_size, device, n_classes, project=False, seed=0
):
    """:func:`probe_metrics` on the frozen features of ``model`` (mean over
    the windows of one deterministic sequence per record)."""
    z_tr, i_tr = embed_records(
        model, train, cfg, spec, batch_size, device, project, seed
    )
    z_va, i_va = embed_records(model, val, cfg, spec, batch_size, device, project, seed)
    return probe_metrics(
        z_tr, [train[i] for i in i_tr], z_va, [val[i] for i in i_va], n_classes
    )


HEADLINE = ("ROT", "RR", "ECL", "CEP", "DSCT", "LPV")


def describe_by_class(by: dict, n_by: dict | None = None) -> str:
    """``ROT 0.12/112 RR 0.25/422 ...``: the per-superclass period R2 with the
    validation count, headline classes first, then by count."""
    n_by = n_by or {}
    order = [g for g in HEADLINE if g in by]
    order += sorted((g for g in by if g not in HEADLINE), key=lambda g: -n_by.get(g, 0))
    return " ".join(
        f"{g} {by[g]:.2f}" + (f"/{n_by[g]}" if g in n_by else "") for g in order
    )


def describe_probe(pr: dict) -> str:
    return (
        f"probe acc {pr['acc']:.3f} f1 {pr['macro_f1']:.3f} bal {pr['balanced_acc']:.3f} "
        f"(majority {pr['majority_acc']:.3f}, train acc {pr['train_acc']:.3f}) | logP R2 "
        f"within {pr['r2_within']:.3f} pooled {pr['r2']:.3f} by class "
        f"{describe_by_class(pr['r2_by_superclass'], pr['n_by_superclass'])}"
    )


def hand_features(record: Record, bands: Sequence[int]) -> np.ndarray:
    """Statistics a linear probe can read off a standardised record without
    any model: log points, span, median cadence, band count, and per band
    (zeros with a 0 flag for a band with under three points) a flag, log
    points, scatter, skewness, excess kurtosis, MAD over std, the fractions
    of points beyond +1 and -1, log median error and range over std. The
    per-band standardisation of :func:`~romae_lc.normalize` has already
    removed mean brightness and colour, so these are shape, scatter and
    noise statistics only, which is what the encoder sees too."""
    t = record.t.astype(np.float64)
    span = float(t.max() - t.min()) if record.n > 1 else 0.0
    gaps = np.diff(np.sort(t))
    gaps = gaps[gaps > 0]
    f = [
        np.log10(max(record.n, 1)),
        span / 1000.0,
        np.log10(np.median(gaps)) if gaps.size else 0.0,
        float(len(record.bands)),
    ]
    for b in bands:
        m = record.band == b
        n = int(m.sum())
        if n < 3:
            f += [0.0] * 10
            continue
        y, e = record.y[m].astype(np.float64), record.err[m].astype(np.float64)
        sd = float(y.std()) + 1e-6
        z = (y - y.mean()) / sd
        f += [
            1.0,
            np.log10(n),
            sd,
            float(np.mean(z**3)),
            float(np.mean(z**4) - 3.0),
            float(np.median(np.abs(y - np.median(y)))) / sd,
            float(np.mean(y > 1.0)),
            float(np.mean(y < -1.0)),
            np.log10(float(np.median(e)) + 1e-6),
            float(np.ptp(y)) / sd,
        ]
    return np.array(f, dtype=np.float64)


def baseline_probe(train, val, bands: Sequence[int], n_classes: int) -> dict:
    """:func:`probe_metrics` on :func:`hand_features`: the bar an encoder's
    probe has to clear."""
    bands = sorted(bands)
    x_tr = np.stack([hand_features(r, bands) for r in train])
    x_va = np.stack([hand_features(r, bands) for r in val])
    return probe_metrics(x_tr, train, x_va, val, n_classes)
