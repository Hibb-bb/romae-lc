"""Small helpers: learning-rate schedule, seeding, the device flag, JSON logs."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch


def cosine_schedule(total: int, warmup_frac: float = 0.01):
    warmup = max(1, int(warmup_frac * total))

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return multiplier


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_device_arg(parser) -> None:
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )


def get_device(args) -> torch.device:
    return torch.device(args.device)


class JsonlLog:
    """Append-only JSON lines log with a timestamp per line."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **fields) -> None:
        with open(self.path, "a") as f:
            f.write(
                json.dumps(dict(time=time.time(), **fields), default=_json_default)
                + "\n"
            )


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
