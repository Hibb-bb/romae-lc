"""The golden check of the clean-up (docs/CLEANUP-PLAN.md).

    python -m project.tests.golden record     # once, from the unchanged code
    python -m project.tests.golden check      # after every clean-up step

``record`` writes the golden files to ``.golden/`` at the repo root (untracked). ``check`` runs the same
things again and compares. Three parts:

- ``flags``: each ICML run's flags resolve to the values in its ``args.json``.
- ``ckpt``: each trained ICML checkpoint loads and gives the same features, loss and prediction on one
  fixed token batch.
- ``train``: short seeded toy runs of the live training paths give the same logged numbers, the same
  final weights and the same checkpoint layout.

Everything runs on CPU with 4 threads. A value passes as ``exact`` (bit for bit) or as ``close`` (inside
a tight tolerance, see ``TOL``); ``--strict`` turns ``close`` into a failure. Run it from the repo root.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, set before torch is imported

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# ======================================================================================================
# THE SEAM. Later clean-up steps change only this block: module paths, entry-point names, flag names.
# ======================================================================================================

#: Trained runs live in the main checkout. They are only read.
RUNS = Path("/projects/bfrf/hibb/romae-lc/project/runs")
#: The 9 trained ICML runs (checkpoint part).
CKPT_RUNS = ("enc_j1", "enc_o2_noadd", "enc_o2_fc", "enc_o2_fcbig", "enc_o2_fcbig_ph", "enc_o3_fcph", "enc_o3_ph",
             "enc_o3_fcph80", "enc_o3_fcph_pband")  # fmt: skip
#: The 10 ICML runs (flags part). enc_le1 has flags but no checkpoint yet.
FLAG_RUNS = CKPT_RUNS + ("enc_le1",)
#: Flags that the flags part reports but does not compare. Empty on purpose; add a name only with a reason.
FLAGS_NOT_COMPARED: tuple[str, ...] = ()


def train(argv: list[str]) -> None:
    """Run one pretraining job."""
    from project import pretrain_mae

    pretrain_mae.main(argv)


def load_encoder(ckpt):
    """``(frozen encoder, meta)`` from a checkpoint path or dict, on CPU."""
    from project.common import load_encoder as f

    return f(ckpt, "cpu")


def flag_parser() -> argparse.ArgumentParser:
    """The training parser. ``record`` uses it to turn an ``args.json`` into a command line."""
    from project import pretrain_mae
    from project.common import add_data_args, add_frame_args, add_ladder_args, add_model_args
    from project.tracking import add_wandb_args

    p = argparse.ArgumentParser()
    add_data_args(p)
    add_frame_args(p)
    add_ladder_args(p)
    add_model_args(p)
    pretrain_mae.add_train_args(p)
    add_wandb_args(p)
    return p


def resolve_flags(run: str, argv: list[str]) -> dict:
    """The resolved flags of one ICML run, as ``main`` would see them. Today: parse the recorded command
    line ``argv``. From step 3 on: load the run's recipe file instead (``run`` names it)."""
    from project.common import resolve_model_args

    return vars(resolve_model_args(flag_parser().parse_args(argv)))


#: Toy sizes shared by every training case (as in project/tests/test_jepa.py).
TOY = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2", "--heads", "2",
    "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "3", "--ckpt-every", "3", "--log-every", "1",
    "--probe-train", "16", "--probe-val", "8", "--shuffle-objects", "6",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu", "--seed", "0",
]  # fmt: skip
#: What every ICML run has on: the spectral layer and its aux loss, the JEPA.
JEPA = ["--jepa", "--spectral", "--spectral-aux", "0.5"]
EMA = JEPA + ["--jepa-proj", "8", "--jepa-proj-hidden", "16", "--jepa-recon-weight", "0.5"]
PHASE = ["--phase-rope", "4", "--phase-freq", "local", "--dec-width", "48"]
#: The training cases: name -> list of stages. A stage is ``(extra flags, steps)``. A later stage of a case
#: runs in the same folder, so it resumes ``last.pt``. ``{name}`` in a flag is the folder of another case.
TRAIN_CASES: dict[str, list[tuple[list[str], int]]] = {
    # EMA JEPA + reconstruction + blockplus (enc_j1, enc_o2_noadd); the second stage tests resume
    "ema_blockplus": [(EMA + ["--mask-mode", "blockplus"], 4), (EMA + ["--mask-mode", "blockplus"], 6)],
    # + forecast tail + phase rope, warm start from a JEPA checkpoint (enc_o2_fc*, enc_o3_fcph, enc_o3_ph)
    "ema_tail_phase": [(EMA + PHASE + ["--mask-mode", "blockplus", "--forecast-tail", "3", "10",
                                       "--jepa-init", "{ema_blockplus}/mae.pt"], 6)],  # fmt: skip
    # phase-band mask (enc_o3_fcph_pband)
    "ema_phaseband": [(EMA + PHASE + ["--mask-mode", "phaseband", "--forecast-tail", "3", "10"], 6)],
    # random mask at 0.8 (the mask ratio of enc_o3_fcph80, on the default mask mode)
    "ema_random80": [(EMA + ["--mask-ratio", "0.8"], 6)],
    # no EMA, views, SIGReg (enc_le1)
    "noema_views": [(JEPA + PHASE + ["--jepa-no-ema", "--jepa-proj", "16", "--jepa-proj-hidden", "16", "--sig-slices", "64",
                                     "--n-frames", "1", "--mask-ratio", "0.85", "--view-keep", "0.3",
                                     "--local-span", "1", "3"], 6)],  # fmt: skip
}

# ======================================================================================================
# End of the seam.
# ======================================================================================================

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / ".golden"
THREADS = 4
#: ``close`` means inside these bounds. Used only when a value is not bit-identical.
TOL = dict(rtol=1e-4, atol=1e-6)
#: Keys of log.jsonl rows that hold wall-clock or machine numbers.
LOG_SKIP = ("elapsed", "eval_seconds", "s_per_step", "data_s", "compute_s", "data_frac", "seconds", "time", "gpu_", "mem_", "seq_per_s")
#: Top-level checkpoint keys compared in full (plain dicts and lists).
CKPT_DICT_KEYS = ("kind", "backbone", "mae", "jepa", "spec", "frames", "classes", "step")


def setup() -> None:
    """One fixed CPU set-up, and a guard that this checkout's code is the code that runs."""
    torch.set_num_threads(THREADS)
    import project
    import romae_lc

    for m in (project, romae_lc):
        if not str(Path(m.__file__).resolve()).startswith(str(ROOT) + os.sep):
            sys.exit(f"{m.__name__} was imported from {m.__file__}, not from {ROOT}. Run from the repo root with python -m.")


def plain(x):
    """JSON form (tuples become lists), so values compare the same before and after a save."""
    return json.loads(json.dumps(x))


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(sd: dict) -> str:
    """One hash over names, shapes, dtypes and bytes of a state dict, in its own order."""
    h = hashlib.sha256()
    for k, v in sd.items():
        h.update(f"{k}|{tuple(v.shape)}|{v.dtype}|".encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------- comparing


class Report:
    """Collects one line per compared item. Status: exact, close or FAIL."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, note: str = "") -> None:
        self.rows.append((name, status, note))
        if status != "exact":
            print(f"  [{status}] {name} {note}", flush=True)

    def count(self, status: str) -> int:
        return sum(r[1] == status for r in self.rows)

    def tensor(self, name: str, new, old) -> None:
        new, old = torch.as_tensor(new), torch.as_tensor(old)
        if new.shape != old.shape or new.dtype != old.dtype:
            return self.add(name, "FAIL", f"shape/dtype {tuple(new.shape)} {new.dtype} vs {tuple(old.shape)} {old.dtype}")
        if torch.equal(new, old) or (new.is_floating_point() and sha(new) == sha(old)):
            return self.add(name, "exact")
        if not new.is_floating_point():
            return self.add(name, "FAIL", "values differ")
        diff = float((new.double() - old.double()).abs().max())
        ok = torch.allclose(new.double(), old.double(), equal_nan=True, **TOL)
        self.add(name, "close" if ok else "FAIL", f"max abs diff {diff:.3g}")

    def value(self, name: str, new, old) -> None:
        """Numbers, strings, None, and nested dicts and lists of them."""
        if isinstance(old, dict) and isinstance(new, dict):
            for k in sorted(set(old) | set(new)):
                if k not in new or k not in old:
                    self.add(f"{name}.{k}", "FAIL", "missing now" if k not in new else "new key")
                else:
                    self.value(f"{name}.{k}", new[k], old[k])
        elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
            for i, (a, b) in enumerate(zip(new, old)):
                self.value(f"{name}[{i}]", a, b)
        elif isinstance(old, float) and isinstance(new, (float, int)) and not isinstance(new, bool):
            if new == old or (np.isnan(new) and np.isnan(old)):
                self.add(name, "exact")
            else:
                ok = bool(np.isclose(new, old, **TOL))
                self.add(name, "close" if ok else "FAIL", f"{new!r} vs {old!r}")
        else:
            self.add(name, "exact" if new == old and type(new) is type(old) else "FAIL", "" if new == old else f"{new!r} vs {old!r}")


# ----------------------------------------------------------------------------------------- flags part


def argv_for(target: dict) -> list[str]:
    """The shortest command line that makes the parser give ``target`` (an ``args.json``). Flags are added
    one at a time, in the file's order, until nothing differs. So a value that another flag already sets
    (the width that ``--size`` gives) is not passed twice."""
    actions = {a.dest: a for a in flag_parser()._actions}
    argv: list[str] = []
    for _ in range(len(target) + 1):
        now = plain(resolve_flags("", argv))
        diff = [k for k in target if now.get(k, "<missing>") != target[k]]
        if not diff:
            return argv
        k, v, a = diff[0], target[diff[0]], actions[diff[0]]
        if a.option_strings[0] in argv:
            raise ValueError(f"flag {a.option_strings[0]} does not resolve to {v!r}")
        if a.nargs == 0:  # store_true / store_false
            argv += [a.option_strings[0]]
        elif isinstance(v, list):
            argv += [a.option_strings[0], *[str(x) for x in v]]
        elif v is None:
            raise ValueError(f"{k}: None cannot be passed on a command line")
        else:
            argv += [a.option_strings[0], str(v)]
    raise ValueError("no command line reproduces the args")


def record_flags() -> dict:
    defaults = plain(resolve_flags("", []))
    out = {}
    for run in FLAG_RUNS:
        target = json.load(open(RUNS / run / "args.json"))
        argv = argv_for(target)
        now = plain(resolve_flags(run, argv))
        assert all(now[k] == v for k, v in target.items()), run
        out[run] = dict(argv=argv, set={k: v for k, v in target.items() if defaults.get(k, "<missing>") != v}, all=now)
        print(f"  {run}: {len(out[run]['set'])} flags set, {len(now)} in all: {' '.join(argv)}", flush=True)
    return out


def check_flags(rep: Report) -> None:
    gold = json.load(open(GOLDEN / "flags.json"))
    for run in FLAG_RUNS:
        g = gold[run]
        now = plain(resolve_flags(run, g["argv"]))
        dropped = []
        for k, v in g["all"].items():
            name = f"flags/{run}/{k}"
            if k in FLAGS_NOT_COMPARED:
                continue
            if k in now:
                rep.value(name, now[k], v)
            elif k in g["set"]:
                rep.add(name, "FAIL", f"the run set this flag to {v!r} and it is gone")
            else:
                dropped.append(k)  # a flag the run left at its default: it may be removed
        new = sorted(set(now) - set(g["all"]))
        print(f"  {run}: {len(g['set'])} set flags checked; {len(dropped)} default flags gone, {len(new)} new" + (f" ({', '.join(new)})" if new else ""), flush=True)


# ------------------------------------------------------------------------------------ checkpoint part


def token_batch(meta, seed: int = 0):
    """One fixed synthetic token batch for a checkpoint: 6 one-window light curves of 40 to 120 points
    (a sine plus noise, random bands), tokenised by the checkpoint's own token spec, and one mask that
    hides the same number of real tokens in every row."""
    rng = np.random.default_rng(seed)
    bands = sorted(meta.spec.tokenize["band_wavelengths"])
    frames = []
    for _ in range(6):
        n = int(rng.integers(40, 121))
        t = np.sort(rng.uniform(0.0, float(meta.cfg.window), n))
        y = np.sin(2 * np.pi * t / float(rng.uniform(0.3, 20.0)) + rng.uniform(0, 6.28)) + 0.1 * rng.standard_normal(n)
        band = rng.choice(bands, n)
        err = rng.uniform(0.02, 0.1, n)
        frames.append((t.astype(np.float32), y.astype(np.float32), band.astype(np.int64), err.astype(np.float32))[: 4 if meta.spec.err_stats else 3])
    tok = meta.spec.tokens(frames)
    pad = tok.pad_mask
    k = int(0.4 * int((~pad).sum(1).min()))
    mask = torch.zeros_like(pad)
    for i in range(pad.shape[0]):
        idx = rng.choice(int((~pad[i]).sum()), k, replace=False)
        mask[i, torch.from_numpy(idx)] = True
    return dict(values=tok.values, positions=tok.positions, pad=pad, mask=mask)


def run_ckpt(run: str, inputs: dict | None = None) -> dict:
    path = RUNS / run / "mae.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    enc, meta = load_encoder(ckpt)
    built = token_batch(meta)
    x = inputs or built
    model = enc.model
    with torch.no_grad():
        pooled = enc.backbone(x["values"], x["positions"], x["pad"])
        tokens, _ = model.encode(x["values"], x["positions"], x["pad"])
        o = model(x["values"], x["positions"], x["pad"], x["mask"])
    out = dict(pooled=pooled.float(), tokens=tokens.float(), loss=o.loss.float(), pred=o.pred.float(), target=o.target.float())
    if getattr(model, "recon_head", None) is not None:
        out["recon"] = model.last_recon.float()
    info = dict(
        keys=sorted(ckpt), state_keys=list(ckpt["state_dict"]), state_hash=state_hash(ckpt["state_dict"]),
        model_state_hash=state_hash(model.state_dict()), model_class=type(model).__name__, encoder_kind=enc.kind, dim=enc.dim,
        meta=plain(dict(spec=meta.spec.to_dict(), frames=meta.cfg.__dict__, classes=list(meta.classes), step=meta.step,
                        ladder=meta.ladder.to_dict(), args=meta.args)),
        **{k: plain(ckpt[k]) for k in CKPT_DICT_KEYS if k in ckpt},
    )  # fmt: skip
    return dict(inputs=built, used=x, out=out, info=info)


def record_ckpt() -> None:
    for run in CKPT_RUNS:
        t0 = time.time()
        r = run_ckpt(run)
        torch.save(dict(inputs=r["inputs"], out=r["out"], info=r["info"]), GOLDEN / f"ckpt_{run}.pt")
        print(f"  {run}: {r['info']['model_class']}, loss {float(r['out']['loss']):.6f}, tokens {tuple(r['out']['tokens'].shape)}, {time.time() - t0:.0f}s", flush=True)


def check_ckpt(rep: Report) -> None:
    for run in CKPT_RUNS:
        name = f"ckpt/{run}"
        gpath = GOLDEN / f"ckpt_{run}.pt"
        if not gpath.is_file():
            rep.add(name, "FAIL", "no golden file")
            continue
        g = torch.load(gpath, map_location="cpu", weights_only=False)
        try:
            r = run_ckpt(run, g["inputs"])
        except Exception as e:  # a checkpoint that no longer loads is the failure this part looks for
            rep.add(name, "FAIL", f"{type(e).__name__}: {e}")
            continue
        for k, v in g["inputs"].items():  # the token spec still makes the same tokens
            rep.tensor(f"{name}/inputs/{k}", r["inputs"][k], v)
        for k, v in g["out"].items():
            if k in r["out"]:
                rep.tensor(f"{name}/{k}", r["out"][k], v)
            else:
                rep.add(f"{name}/{k}", "FAIL", "missing now")
        rep.value(f"{name}/info", r["info"], g["info"])
        print(f"  {run}: loss {float(r['out']['loss']):.6f} (golden {float(g['out']['loss']):.6f})", flush=True)


# -------------------------------------------------------------------------------------- training part


def log_rows(path: Path) -> list[dict]:
    rows = []
    for line in open(path):
        row = json.loads(line)
        rows.append({k: v for k, v in row.items() if not any(k == s or k.startswith(s) for s in LOG_SKIP)})
    return rows


def run_train(tmp: Path) -> dict:
    """Run every case of ``TRAIN_CASES`` under ``tmp``. Returns name -> the log rows, the final weights and
    the plain parts of the final checkpoint."""
    out = {}
    for name, stages in TRAIN_CASES.items():
        t0 = time.time()
        folder = tmp / name
        for extra, steps in stages:
            flags = [f.format(**{n: str(tmp / n) for n in TRAIN_CASES}) for f in extra]
            train(TOY + flags + ["--steps", str(steps), "--out", str(folder)])
        ckpt = torch.load(folder / "mae.pt", map_location="cpu", weights_only=False)
        out[name] = dict(
            log=log_rows(folder / "log.jsonl"), weights=dict(ckpt["state_dict"]), state_hash=state_hash(ckpt["state_dict"]),
            keys=sorted(ckpt), parts={k: plain(ckpt[k]) for k in CKPT_DICT_KEYS + ("ladder", "metrics") if k in ckpt},
        )  # fmt: skip
        out[name]["parts"]["metrics"] = {k: v for k, v in (out[name]["parts"].get("metrics") or {}).items() if not any(k == s or k.startswith(s) for s in LOG_SKIP)}
        print(f"  {name}: {len(out[name]['log'])} log rows, weights {out[name]['state_hash'][:12]}, {time.time() - t0:.0f}s", flush=True)
    return out


def check_train(rep: Report) -> None:
    gold = torch.load(GOLDEN / "train.pt", map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory(prefix="golden_") as tmp:
        try:
            now = run_train(Path(tmp))
        except (Exception, SystemExit) as e:
            rep.add("train", "FAIL", f"a training case stopped: {type(e).__name__}: {e}")
            return
    for name, g in gold.items():
        n = now.get(name)
        if n is None:
            rep.add(f"train/{name}", "FAIL", "case is gone")
            continue
        rep.add(f"train/{name}/state_hash", "exact" if n["state_hash"] == g["state_hash"] else "close", "" if n["state_hash"] == g["state_hash"] else "weights not bit-identical, see the tensors")
        if list(n["weights"]) != list(g["weights"]):
            rep.add(f"train/{name}/state_names", "FAIL", "state-dict names or order changed")
        for k, v in g["weights"].items():
            if k in n["weights"]:
                rep.tensor(f"train/{name}/weights/{k}", n["weights"][k], v)
        missing = sorted(set(g["keys"]) - set(n["keys"]))
        rep.add(f"train/{name}/ckpt_keys", "FAIL" if missing else "exact", f"missing {missing}" if missing else "")
        extra = sorted(set(n["keys"]) - set(g["keys"]))
        if extra:
            print(f"  {name}: new checkpoint keys {extra}", flush=True)
        rep.value(f"train/{name}/ckpt", n["parts"], g["parts"])
        if len(n["log"]) != len(g["log"]):
            rep.add(f"train/{name}/log", "FAIL", f"{len(n['log'])} rows vs {len(g['log'])}")
        for i, (a, b) in enumerate(zip(n["log"], g["log"])):
            rep.value(f"train/{name}/log[{i}:{b.get('kind')}@{b.get('step')}]", a, b)


# ------------------------------------------------------------------------------------------------ main


def machine() -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "?", False
    return dict(host=platform.node(), arch=platform.machine(), python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, threads=torch.get_num_threads(), commit=commit, dirty=dirty,
                time=time.strftime("%Y-%m-%d %H:%M:%S"))  # fmt: skip


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("record", "check"))
    p.add_argument("--part", nargs="+", choices=("flags", "ckpt", "train"), default=("flags", "ckpt", "train"), help="parts to run (default: all)")
    p.add_argument("--strict", action="store_true", help="check: a value that is close but not bit-identical fails")
    p.add_argument("--force", action="store_true", help="record: overwrite existing golden files")
    args = p.parse_args(argv)
    setup()
    t0 = time.time()
    if args.command == "record":
        files = dict(flags=["flags.json"], ckpt=[f"ckpt_{r}.pt" for r in CKPT_RUNS], train=["train.pt"])
        have = [f for part in args.part for f in files[part] if (GOLDEN / f).exists()]
        if have and not args.force:
            sys.exit(f"golden files exist ({', '.join(have)}). They are recorded once, from the unchanged code. Use --force only if you mean it.")
        GOLDEN.mkdir(exist_ok=True)
        info = json.load(open(GOLDEN / "info.json")) if (GOLDEN / "info.json").is_file() else {}
        for part in args.part:
            print(f"record {part}", flush=True)
            if part == "flags":
                json.dump(record_flags(), open(GOLDEN / "flags.json", "w"), indent=1)
            elif part == "ckpt":
                record_ckpt()
            else:
                with tempfile.TemporaryDirectory(prefix="golden_") as tmp:
                    torch.save(run_train(Path(tmp)), GOLDEN / "train.pt")
            info[part] = machine()
        json.dump(info, open(GOLDEN / "info.json", "w"), indent=1)
        print(f"recorded {', '.join(args.part)} in {GOLDEN} ({time.time() - t0:.0f}s)")
        return 0
    rep = Report()
    info, here = json.load(open(GOLDEN / "info.json")), machine()
    for part in args.part:
        rec = info.get(part)
        if rec is None:
            rep.add(part, "FAIL", "this part was never recorded")
            continue
        differs = [f"{k} {rec[k]} -> {here[k]}" for k in ("host", "arch", "torch", "numpy", "threads") if rec[k] != here[k]]
        print(f"check {part} (recorded {rec['time']} at {rec['commit'][:7]}{'; now: ' + ', '.join(differs) if differs else ''})", flush=True)
        dict(flags=check_flags, ckpt=check_ckpt, train=check_train)[part](rep)
    n_exact, n_close, n_fail = rep.count("exact"), rep.count("close"), rep.count("FAIL")
    bad = n_fail + (n_close if args.strict else 0)
    print(f"golden check: {'FAIL' if bad else 'PASS'}: {n_exact} exact, {n_close} close, {n_fail} failed ({time.time() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
