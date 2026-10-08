"""The collaborators' GRU-ODE encoder (``RNNODEEncoderV2`` of the Transient
Foundation Model / SELDON code) in our stage-1 pipeline, everything else
unchanged: the same windows, the same masked-reconstruction objective and
light decoder, the same latent cache and probes.

The encoder reads the window's visible tokens as one sequence sorted by
time, all bands together; each token is ``[brightness, standardised log
error, band wavelength]`` (``band="centre"``: the band's central wavelength,
the number our rotary encoding rotates by, in microns and standardised; or
``band="percentiles"``: the 10 / 50 / 90 % wavelengths of the band's filter
curve) and ``t`` is the token's time in days over ``time_unit``. A GRU cell
updates the hidden state at every observation and a small ODE evolves it
across the gap to the next one (their default: the sequence is read in
reverse). The per-token hidden states play the role of the encoder's token
outputs for the decoder, and the final hidden state (``hid`` in their code)
is the pooled latent in the CLS slot, so the probes read it unchanged.

The checkpoint has ``kind == "gruode"``; :func:`project.common.load_mae`
rebuilds it. Their code is imported from ``TFM_ROOT``.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from romae_lc.model import RoMAEForPreTraining

TFM_ROOT = os.environ.get("TFM_ROOT", "/projects/bfrf/hibb/Transient-Foundation-Model-1.1.1")

#: approximate 10 / 50 / 90 % cumulative-throughput wavelengths of the ZTF filters, microns (published filter curves)
ZTF_PERCENTILES = {472.0: (0.410, 0.478, 0.540), 634.0: (0.575, 0.640, 0.705), 782.0: (0.730, 0.790, 0.850)}


def build_gru_ode(hidden: int, input_size: int, ode_hidden: int, steps: int = 1, dropout: float = 0.0):
    """Their ``RNNODEEncoderV2`` as the SELDON config builds it (GRU cell, a
    one-layer residual ODE net, a fixed-step solver), constructed from their
    classes directly so their training stack (Lightning etc.) is not
    imported."""
    if TFM_ROOT not in sys.path:
        sys.path.insert(0, TFM_ROOT)
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    from TAE.models.ModelNN import ResidualNetwork
    from TAE.models.neuralode import ODEModelWrapper, ODESolver, RNNODECell, RNNODEEncoderV2

    ode_net = ODEModelWrapper(ResidualNetwork(n_features=hidden, n_output=hidden, n_hidden=ode_hidden, n_layers=1, dropout=dropout,
                                              activation="GELU", n_heads=1, norm=True, residual_input=False), autonomous=True)
    solver = ODESolver(ode_net, use_fixed_solver=True, fixed_solver_steps=steps)
    cell = RNNODECell(nn.GRUCell(input_size, hidden), solver)
    return RNNODEEncoderV2(cell, use_class_token=False, return_sequence=True)


class GRUODEForPreTraining(RoMAEForPreTraining):
    """``RoMAEForPreTraining`` whose encoder is the GRU-ODE: ``embed`` builds
    the token vectors, ``add_cls`` reserves the CLS slot, ``run_transformer``
    runs the recurrence and puts the final hidden state in that slot. The
    parent's transformer blocks are built but never run."""

    def __init__(self, *args, rnn: dict | None = None, **kwargs):
        kwargs["spectral"] = None
        super().__init__(*args, **kwargs)
        rnn = dict(rnn or {})
        self.rnn_cfg = dict(band=rnn.get("band", "centre"), ode_hidden=int(rnn.get("ode_hidden", 128)), steps=int(rnn.get("steps", 1)),
                            time_unit=float(rnn.get("time_unit", 100.0)), time_scale=float(rnn.get("time_scale", 1.0)),
                            band_wavelengths=[float(w) for w in rnn.get("band_wavelengths", [472.0, 634.0, 782.0])], dropout=float(rnn.get("dropout", 0.0)))
        wl = torch.tensor(self.rnn_cfg["band_wavelengths"])
        self.register_buffer("band_wl", wl)
        if self.rnn_cfg["band"] == "percentiles":
            feats = torch.tensor([ZTF_PERCENTILES.get(float(w), (w / 1e3 * 0.86, w / 1e3, w / 1e3 * 1.13)) for w in wl.tolist()])
        else:
            feats = (wl / 1e3)[:, None]  # microns
        self.register_buffer("band_feats", (feats - 0.62) / 0.15)  # roughly centred over the optical bands
        self.n_band_feats = int(feats.shape[1])
        self.rnn = build_gru_ode(self.cfg.d_model, self.n_channels + self.n_band_feats, self.rnn_cfg["ode_hidden"], self.rnn_cfg["steps"], self.rnn_cfg["dropout"])

    @property
    def hparams(self) -> dict:
        h = dict(super().hparams)
        h.pop("spectral", None)
        return h

    def embed(self, values, positions):
        band = (positions[:, 1][..., None] - self.band_wl[None, None]).abs().argmin(-1)  # [B, N]
        return torch.cat([values.float(), self.band_feats[band].to(values.device)], -1)

    def add_cls(self, x, positions, pad_mask):
        b = x.shape[0]
        x = torch.cat([x.new_zeros(b, 1, x.shape[-1]), x], 1)
        positions = torch.cat([positions.new_zeros(b, positions.shape[1], 1), positions], 2)
        pad_mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), pad_mask], 1)
        return x, positions, pad_mask

    def run_transformer(self, x, positions, pad_mask, values=None):
        feats, real = x[:, 1:], ~pad_mask[:, 1:]
        t = (positions[:, 0, 1:].float() - 1.0) * self.rnn_cfg["time_scale"] / self.rnn_cfg["time_unit"]
        with torch.autocast(device_type=x.device.type, enabled=False):
            outputs, h = self.rnn(feats.float(), t.float(), real, reverse=True)
        return torch.cat([h[:, None], outputs], 1).to(x.dtype if x.is_floating_point() else torch.float32)

    @property
    def rnn_params(self) -> int:
        return sum(p.numel() for p in self.rnn.parameters())

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "GRUODEForPreTraining":
        model = cls(**ckpt["mae"], **ckpt["backbone"], rnn=ckpt["gruode"])
        model.load_state_dict(ckpt["state_dict"])
        return model


def gruode_state(model, spec, cfg, ladder, classes, args, step, metrics=None) -> dict:
    from project.common import mae_state

    state = mae_state(model, spec, cfg, ladder, classes, args, step, metrics)
    state["kind"] = "gruode"
    state["gruode"] = dict(model.rnn_cfg)
    return state
