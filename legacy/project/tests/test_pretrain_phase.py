"""CPU test of the phase-bottleneck stage 1 on the toy simulator: it trains,
saves a stage-1 checkpoint every later stage can read, and the phase
decoder next to it."""

from __future__ import annotations

import torch

from project import cache_latents, pretrain_phase
from project.common import load_encoder
from project.phase_decoder import PhaseDecoder

ARGS = [
    "--data", "sim", "--n-sim", "48", "--width", "24", "--depth", "2", "--heads", "2",
    "--dec-width", "24", "--dec-heads", "2", "--dec-depth", "1", "--n-harm", "4",
    "--window", "30", "--min-tokens", "4", "--max-tokens", "32", "--n-frames", "4",
    "--batch-size", "4", "--eval-every", "3", "--ckpt-every", "3", "--log-every", "1",
    "--ladder-curves", "8", "--workers", "0", "--device", "cpu", "--val-objects", "8",
]  # fmt: skip


def test_phase_bottleneck_trains_and_saves(tmp_path):
    out = tmp_path / "pb"
    pretrain_phase.main(ARGS + ["--steps", "3", "--out", str(out)])
    assert (out / "mae.pt").is_file() and (out / "DONE").is_file() and (out / "phase_decoder.pt").is_file()
    enc, meta = load_encoder(str(out / "mae.pt"), "cpu")
    assert enc.dim == 24 and meta.step == 3
    ck = torch.load(out / "phase_decoder.pt", map_location="cpu", weights_only=False)
    dec = PhaseDecoder(**ck["hparams"])
    dec.load_state_dict(ck["state_dict"])
    assert dec.n_harm == 4
    # the cache reads the checkpoint like any stage-1 one
    cache_latents.main(["--ckpt", str(out / "mae.pt"), "--out", str(out / "latents.pt"), "--stride", "0.5",
                        "--device", "cpu", "--workers", "0", "--batch-size", "16"])  # fmt: skip
    assert (out / "latents.pt").is_file()
    # resume: picks up and stops at once
    pretrain_phase.main(ARGS + ["--steps", "3", "--out", str(out)])
    # a second run started from the first's encoder weights
    out2 = tmp_path / "pb2"
    pretrain_phase.main(ARGS + ["--steps", "2", "--out", str(out2), "--init", str(out / "mae.pt")])
    enc2, meta2 = load_encoder(str(out2 / "mae.pt"), "cpu")
    assert enc2.dim == 24 and meta2.step == 2
