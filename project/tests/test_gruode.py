"""CPU test of the GRU-ODE encoder in the masked-pretraining pipeline (needs the TFM code at TFM_ROOT)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from project import cache_latents, pretrain_mae
from project.common import load_encoder
from project.gruode import TFM_ROOT, GRUODEForPreTraining
from project.tests.test_ls_benchmark import MAE_ARGS

pytestmark = pytest.mark.skipif(not os.path.isdir(TFM_ROOT), reason="the collaborators' code is not on this machine")


def test_gruode_trains_caches_and_reloads(tmp_path):
    out = tmp_path / "gruode"
    pretrain_mae.main(MAE_ARGS + ["--encoder", "gruode", "--rnn-ode-hidden", "16", "--mask-mode", "blockplus", "--steps", "2", "--out", str(out)])
    assert (out / "mae.pt").is_file()
    enc, meta = load_encoder(str(out / "mae.pt"), "cpu")
    assert isinstance(enc.model, GRUODEForPreTraining) and enc.model.n_band_feats == 1
    cache_latents.main(["--ckpt", str(out / "mae.pt"), "--out", str(tmp_path / "latents.pt"), "--stride", "0.5", "--device", "cpu", "--workers", "0", "--batch-size", "8"])
    z = torch.load(tmp_path / "latents.pt", map_location="cpu", weights_only=False)["z"]
    assert z.shape[1] == enc.model.cfg.d_model and np.isfinite(z.float().numpy()).all()
