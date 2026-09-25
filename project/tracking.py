"""Weights & Biases tracking for the training scripts.

``Tracker`` wraps ``wandb`` behind a flag so every script runs without it.
The run id is stored in the checkpoints, so a resumed run (the 2 h job
chains) continues the same wandb run. Besides the metrics the scripts log,
``gpu_stats`` reads the utilisation and memory of the current device through
NVML (``torch.cuda.utilization`` when ``pynvml`` is present, ``nvidia-smi``
otherwise) so it can be logged on the same step axis as the losses; wandb's
own system monitor adds the usual time-based system panel. The API key is
read from ``~/.netrc`` (``wandb login``), never from the repository.
"""

from __future__ import annotations

import os
import subprocess
import time

import torch


def add_wandb_args(parser, project: str = "lc-world-model") -> None:
    g = parser.add_argument_group("tracking")
    g.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    g.add_argument("--wandb-project", default=project)
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-name", default=None, help="run name (default: --out)")
    g.add_argument("--wandb-group", default=None)
    g.add_argument("--wandb-tags", nargs="*", default=None)


def gpu_stats(device=None) -> dict:
    """``{'gpu_util': %, 'gpu_mem_used_gb', 'gpu_mem_alloc_gb', 'gpu_mem_reserved_gb'}``
    or ``{}`` without CUDA."""
    if not torch.cuda.is_available():
        return {}
    dev = torch.device(device or "cuda")
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    out = dict(
        gpu_mem_alloc_gb=torch.cuda.memory_allocated(idx) / 2**30,
        gpu_mem_reserved_gb=torch.cuda.memory_reserved(idx) / 2**30,
    )
    try:
        out["gpu_util"] = float(torch.cuda.utilization(idx))
        free, total = torch.cuda.mem_get_info(idx)
        out["gpu_mem_used_gb"] = (total - free) / 2**30
    except Exception:
        try:
            q = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(idx),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                .stdout.strip()
                .split(",")
            )
            out["gpu_util"], out["gpu_mem_used_gb"] = float(q[0]), float(q[1]) / 1024
        except Exception:
            pass
    return out


class Tracker:
    """A no-op unless ``args.wandb``; ``log(metrics, step)`` prefixes nothing,
    callers pass ``train/...`` and ``val/...`` keys."""

    def __init__(
        self, args, config: dict, run_id: str | None = None, job_type: str = "train"
    ):
        self.enabled = bool(getattr(args, "wandb", False))
        self.run = None
        self.id = run_id
        if not self.enabled:
            return
        import wandb

        name = args.wandb_name or os.path.basename(
            str(getattr(args, "out", "run")).rstrip("/")
        )
        self.run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            job_type=job_type,
            config=config,
            id=run_id,
            resume="allow",
            dir=str(getattr(args, "out", ".")),
        )
        self.id = self.run.id
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    def log(self, metrics: dict, step: int) -> None:
        if self.run is None:
            return
        flat = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        flat[f"{k}/{kk}"] = vv
            elif isinstance(v, (int, float)):
                flat[k] = v
        flat["step"] = step
        self.run.log(flat, step=step)

    def summary(self, **kw) -> None:
        if self.run is not None:
            for k, v in kw.items():
                self.run.summary[k] = v

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


class StepTimer:
    """Splits wall time per step into data wait (blocking on the loader) and
    compute (forward, backward, optimiser; synchronised on CUDA)."""

    def __init__(self, device):
        self.sync = (
            torch.cuda.synchronize
            if torch.device(device).type == "cuda"
            else (lambda: None)
        )
        self.reset()

    def reset(self):
        self.data, self.compute, self.n = 0.0, 0.0, 0
        self._t = time.perf_counter()

    def got_batch(self):
        now = time.perf_counter()
        self.data += now - self._t
        self._t = now

    def done_step(self):
        self.sync()
        now = time.perf_counter()
        self.compute += now - self._t
        self._t = now
        self.n += 1

    def report(self) -> dict:
        n = max(self.n, 1)
        return dict(
            data_s=self.data / n,
            compute_s=self.compute / n,
            data_frac=self.data / max(self.data + self.compute, 1e-9),
        )
