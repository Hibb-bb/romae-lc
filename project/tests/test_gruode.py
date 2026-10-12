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


def test_token_features_are_in_token_order():
    """With the reversed walk, the LAST token is consumed first from the zero state, so its feature must be
    the GRU cell applied to it alone; the feature of a padded slot must stay the zero state."""
    import torch
    from project.gruode import build_gru_ode

    torch.manual_seed(0)
    rnn = build_gru_ode(8, 3, 8).eval()
    b, n = 2, 6
    feats = torch.randn(b, n, 3)
    t = torch.sort(torch.rand(b, n), 1).values
    real = torch.ones(b, n, dtype=torch.bool)
    real[0, 4:] = False  # row 0: two padded slots at the end
    outputs, h = rnn(feats, t, real, reverse=True)
    outputs = outputs.flip(1)  # what GRUODEForPreTraining.run_transformer does
    cell = rnn.rnn_ode_cell.rnn_cell
    last = cell(feats[1, -1][None], torch.zeros(1, 8))
    assert torch.allclose(outputs[1, -1], last[0], atol=1e-5)
    assert torch.allclose(outputs[0, 4:], torch.zeros(2, 8), atol=1e-6)
    assert not torch.allclose(outputs[0, 3], torch.zeros(8))
