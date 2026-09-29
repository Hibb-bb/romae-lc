"""The phase-fold test of a period, the way astronomers check one.

Take a period. Fold the light curve on it, so every cycle lies on top of the
others. Fit a smooth wave to the folded points: a mean plus a few harmonics,
one fit per band, every point weighted by its error. R2 is the share of the
star's brightness change that the wave explains. A right period lines the
points up and R2 is high. A wrong period scatters them and R2 is near zero.

The test needs no catalogue. Here it is run twice per star: once on the
period the model predicts and once on the catalogue period. A third fold on
a period that is wrong on purpose gives the floor.

R2 is adjusted for the number of fitted values, so a wave with many
harmonics does not look good by chance.

The test is strict. Over a baseline of ``T`` days a period error of ``dP``
shifts the last cycle by ``T dP / P^2`` cycles. The fold stays clean only
when that is well under one cycle: ``dP / P < 0.1 P / T``. For a 0.3 day
star over 2700 days that is one part in 90,000.
"""

from __future__ import annotations

import math

import numpy as np

#: The wrong-on-purpose period is the catalogue period times this.
WRONG_FACTOR = 1.2347


def design(t: np.ndarray, period: float, harmonics: int) -> np.ndarray:
    """The columns of the wave: 1, then cos and sin of every harmonic."""
    ph = 2.0 * np.pi * t / period
    cols = [np.ones_like(t)]
    for k in range(1, harmonics + 1):
        cols += [np.cos(k * ph), np.sin(k * ph)]
    return np.stack(cols, 1)


def fold_r2(t, y, err, band, period: float, harmonics: int = 3, err_floor: float = 1e-3) -> float:
    """Adjusted R2 of the wave fitted to the light curve folded on
    ``period`` (days). One fit per band, pooled. NaN when no band has enough
    points or the period is not a positive number."""
    if not np.isfinite(period) or period <= 0:
        return float("nan")
    t, y = np.asarray(t, dtype=np.float64), np.asarray(y, dtype=np.float64)
    err, band = np.asarray(err, dtype=np.float64), np.asarray(band)
    n_par = 2 * harmonics + 1
    ss_res = ss_tot = 0.0
    n_tot = p_tot = n_bands = 0
    for b in np.unique(band):
        m = band == b
        n = int(m.sum())
        if n < n_par + 3:
            continue
        w = 1.0 / (np.maximum(err[m], 0.0) ** 2 + err_floor**2)
        sw = np.sqrt(w)
        x = design(t[m], period, harmonics)
        beta, *_ = np.linalg.lstsq(x * sw[:, None], y[m] * sw, rcond=None)
        res = y[m] - x @ beta
        mean = (w * y[m]).sum() / w.sum()
        ss_res += float((w * res**2).sum())
        ss_tot += float((w * (y[m] - mean) ** 2).sum())
        n_tot, p_tot, n_bands = n_tot + n, p_tot + n_par, n_bands + 1
    if n_tot - p_tot <= 0 or ss_tot <= 0:
        return float("nan")
    return float(1.0 - (ss_res / (n_tot - p_tot)) / (ss_tot / (n_tot - n_bands)))


def fold_compare(records, p_model, p_cat, harmonics: int = 3) -> dict:
    """The fold R2 of every record on the model's period, on the catalogue
    period and on a wrong period (the floor)."""
    n = len(records)
    out = dict(model=np.full(n, np.nan), catalogue=np.full(n, np.nan), wrong=np.full(n, np.nan))
    for i, r in enumerate(records):
        args = (r.t, r.y, r.err, r.band)
        out["model"][i] = fold_r2(*args, float(p_model[i]), harmonics)
        out["catalogue"][i] = fold_r2(*args, float(p_cat[i]), harmonics)
        out["wrong"][i] = fold_r2(*args, float(p_cat[i]) * WRONG_FACTOR, harmonics)
    return out


def fold_r2_grid(t, y, err, band, periods, harmonics: int = 3, device="cpu",
                 chunk_elems: int = 40_000_000, err_floor: float = 1e-3) -> np.ndarray:  # fmt: skip
    """:func:`fold_r2` for many trial periods of one star at once (torch, in
    chunks): the adjusted R2 of every period in ``periods``."""
    import torch

    periods = torch.as_tensor(np.asarray(periods, dtype=np.float64), device=device)
    f = periods.numel()
    band = np.asarray(band)
    n_par = 2 * harmonics + 1
    ss_res = torch.zeros(f, dtype=torch.float64, device=device)
    ss_tot, n_tot, p_tot, n_bands = 0.0, 0, 0, 0
    eye = 1e-9 * torch.eye(n_par, dtype=torch.float64, device=device)
    k = torch.arange(1, harmonics + 1, dtype=torch.float64, device=device)
    for b in np.unique(band):
        m = band == b
        n = int(m.sum())
        if n < n_par + 3:
            continue
        tb = torch.as_tensor(np.asarray(t, dtype=np.float64)[m], device=device)
        yb = torch.as_tensor(np.asarray(y, dtype=np.float64)[m], device=device)
        eb = torch.as_tensor(np.asarray(err, dtype=np.float64)[m], device=device)
        w = 1.0 / (eb.clamp_min(0.0) ** 2 + err_floor**2)
        mean = (w * yb).sum() / w.sum()
        ss_tot += float((w * (yb - mean) ** 2).sum())
        step = max(1, chunk_elems // (n * n_par))
        for lo in range(0, f, step):
            p = periods[lo : lo + step]
            ph = 2.0 * math.pi * tb[None, :] / p[:, None]  # [F, N]
            arg = ph[..., None] * k  # [F, N, K]
            x = torch.cat([torch.ones_like(ph)[..., None], arg.cos(), arg.sin()], -1)
            xw = x * w[None, :, None]
            a = xw.transpose(1, 2) @ x + eye
            rhs = (xw * yb[None, :, None]).sum(1)
            beta = torch.linalg.solve(a, rhs)
            res = yb[None] - (x @ beta[..., None]).squeeze(-1)
            ss_res[lo : lo + step] += (w[None] * res**2).sum(1)
        n_tot, p_tot, n_bands = n_tot + n, p_tot + n_par, n_bands + 1
    if n_tot - p_tot <= 0 or ss_tot <= 0:
        return np.full(f, np.nan)
    r2 = 1.0 - (ss_res / (n_tot - p_tot)) / (ss_tot / (n_tot - n_bands))
    return r2.cpu().numpy()


def refine_period(t, y, err, band, p0: float, rel: float = 0.1, oversample: float = 5.0,
                  harmonics: int = 3, device="cpu", max_trials: int = 200_000):  # fmt: skip
    """A fine search near a rough period ``p0``: every period within ``rel``
    of it, on a grid fine enough for the star's baseline, and the one whose
    fold has the best R2. This is how astronomers sharpen a rough period.
    The range is narrow on purpose, so the search cannot jump to half or
    twice the period: which of those is right stays the rough guess's call.
    Returns ``(period, r2, n_trials)``."""
    t = np.asarray(t, dtype=np.float64)
    if not np.isfinite(p0) or p0 <= 0 or t.size < 4:
        return float("nan"), float("nan"), 0
    span = float(t.max() - t.min())
    if span <= 0:
        return float("nan"), float("nan"), 0
    f0 = 1.0 / p0
    n = int(np.ceil(2.0 * rel * f0 * oversample * span)) + 1
    n = int(min(max(n, 3), max_trials))
    freqs = np.linspace(f0 * (1.0 - rel), f0 * (1.0 + rel), n)
    r2 = fold_r2_grid(t, y, err, band, 1.0 / freqs, harmonics, device)
    if not np.isfinite(r2).any():
        return float("nan"), float("nan"), n
    j = int(np.nanargmax(r2))
    return float(1.0 / freqs[j]), float(r2[j]), n


def refine_all(records, p_model, rel=0.1, oversample=5.0, harmonics=3, device="cpu", log=print):
    """:func:`refine_period` for every record: the refined periods, their
    fold R2 and the number of trial periods."""
    import time

    n = len(records)
    p, r2, trials = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i, r in enumerate(records):
        p[i], r2[i], trials[i] = refine_period(
            r.t, r.y, r.err, r.band, float(p_model[i]), rel, oversample, harmonics, device
        )
        if (i + 1) % 500 == 0:
            log(f"  fine search {i + 1} / {n} stars, {time.time() - t0:.0f}s")
    return dict(period=p, r2=r2, trials=trials, seconds=time.time() - t0)


def precision_table(p_pred, p_cat) -> list[dict]:
    """The share of stars whose period is within each tolerance of the
    catalogue period."""
    off = np.abs(np.asarray(p_pred, dtype=np.float64) / np.asarray(p_cat, dtype=np.float64) - 1.0)
    off = np.where(np.isfinite(off), off, np.inf)
    return [
        dict(bin=name, n=int(len(off)), share=float((off < tol).mean()))
        for name, tol in (("within 10 %", 0.1), ("within 1 %", 1e-2), ("within 0.1 %", 1e-3),
                          ("within 0.01 %", 1e-4), ("within 0.001 %", 1e-5))  # fmt: skip
    ]


def _row(name: str, r2: dict, m: np.ndarray, keep: float) -> dict:
    ok = m & np.isfinite(r2["model"]) & np.isfinite(r2["catalogue"])
    n = int(ok.sum())
    if n == 0:
        return dict(bin=name, n=0, r2_catalogue=float("nan"), r2_model=float("nan"),
                    r2_wrong=float("nan"), as_good=float("nan"))  # fmt: skip
    cat, mod = r2["catalogue"][ok], r2["model"][ok]
    return dict(
        bin=name,
        n=n,
        r2_catalogue=float(np.median(cat)),
        r2_model=float(np.median(mod)),
        r2_wrong=float(np.nanmedian(r2["wrong"][ok])),
        # the model's fold counts as good when it keeps most of what the
        # catalogue's fold explains
        as_good=float((mod >= keep * cat).mean()),
    )


def fold_tables(r2: dict, groups, p_model, p_cat, order, keep: float = 0.9) -> dict:
    """Median fold R2 per superclass and per kind of period error."""
    groups = np.asarray(groups)
    ratio = np.asarray(p_model, dtype=np.float64) / np.asarray(p_cat, dtype=np.float64)
    off = np.abs(ratio - 1.0)
    alias = ((np.abs(ratio - 2.0) < 0.2) | (np.abs(ratio - 0.5) < 0.05)) & (off >= 0.1)
    everyone = np.ones(len(groups), dtype=bool)
    by_class = [_row("all", r2, everyone, keep)] + [_row(g, r2, groups == g, keep) for g in order]
    by_error = [
        _row("within 0.01 %", r2, off < 1e-4, keep),
        _row("0.01 % to 0.1 %", r2, (off >= 1e-4) & (off < 1e-3), keep),
        _row("0.1 % to 1 %", r2, (off >= 1e-3) & (off < 1e-2), keep),
        _row("1 % to 10 %", r2, (off >= 1e-2) & (off < 0.1), keep),
        _row("alias (2 P or P / 2)", r2, alias, keep),
        _row("other miss", r2, (off >= 0.1) & ~alias, keep),
    ]
    return dict(superclass=by_class, period_error=by_error)


def plot_fold(path, r2: dict, groups, order, colour, title: str) -> None:
    """The model's fold R2 against the catalogue's, one dot per star."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = np.asarray(groups)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=140)
    for g in order:
        m = (groups == g) & np.isfinite(r2["model"]) & np.isfinite(r2["catalogue"])
        if m.any():
            ax.scatter(r2["catalogue"][m], r2["model"][m], s=5, alpha=0.5, color=colour(g), label=g, linewidths=0)
    ax.plot([0, 1], [0, 1], color="0.4", lw=0.8)
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("fold R2 on the catalogue period")
    ax.set_ylabel("fold R2 on the model's period")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, markerscale=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
