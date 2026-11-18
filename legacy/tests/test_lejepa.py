"""LeJEPA: training and eval paths, predictor routing, SIGReg."""

from __future__ import annotations

import importlib
import sys

import pytest
import torch
import torch.nn as nn

from romae_lc.lejepa import LeJEPA, SlicedEppsPulley, mlp
from romae_lc.model import RoMAE
from romae_lc.tokenize import tokenize

D = 32


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def backbone():
    return RoMAE(encoder=dict(d_model=D, nhead=2, depth=1))


def views(seed=0, n=4):
    """A token batch of ``n`` objects with 3-8 observations each."""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(3, 9, (n,), generator=g).tolist()
    times = [torch.rand(m, generator=g) * 10 for m in lengths]
    values = [torch.randn(m, generator=g) for m in lengths]
    bands = [torch.randint(0, 2, (m,), generator=g) for m in lengths]
    return tokenize(times, values, bands, band_wavelengths={0: 480.0, 1: 620.0})


def test_train_forward_and_backward():
    model = LeJEPA(backbone(), proj=mlp([D, 16, 8]), n_slices=32)
    out = model(global_views=[views(0), views(1)], local_views=[views(2)])
    assert torch.isfinite(out.loss) and out.loss.requires_grad
    assert torch.allclose(out.loss, out.inv_loss + model.lamb * out.sigreg_loss)
    assert out.embedding.shape == (8, D) and not out.embedding.requires_grad
    assert out.projection.shape == (8, 8) and not out.projection.requires_grad
    assert out.features.shape == (12, D) and out.features.requires_grad
    out.loss.backward()
    assert model.backbone.projection.weight.grad is not None
    assert int(model.sigreg.step) == 1
    assert LeJEPA(backbone()).proj[-1].out_features == 512


def test_eval_forward_embeds():
    model = LeJEPA(backbone(), proj=mlp([D, 16, 8]), n_slices=32).eval()
    v = views()
    out = model(values=v.values, positions=v.positions, pad_mask=v.pad_mask)
    assert out.loss == 0 and out.features is None
    assert out.embedding.shape == (4, D) and out.projection.shape == (4, 8)
    assert not out.embedding.requires_grad and not out.projection.requires_grad
    assert torch.equal(out.embedding, model.embed(*v))
    assert model.embed(*v).requires_grad


def test_mode_errors():
    model = LeJEPA(backbone(), proj=mlp([D, 16, 8]), n_slices=32)
    with pytest.raises(ValueError):
        model()
    with pytest.raises(ValueError):
        model.eval()(global_views=[views()])


def test_predictor_changes_only_the_routed_rows():
    model = LeJEPA(
        backbone(),
        proj=nn.Identity(),
        predictor=mlp([D, 16, D], batch_norm=False),
        predictor_inst=1,
        n_slices=32,
    )
    view_inst = torch.tensor([[0, 1], [1, 1], [0, 0], [1, 0]])
    *_, feats, proj = model.loss_on_views([views(0), views(1)], view_inst=view_inst)
    for j, f in enumerate(feats):
        routed = view_inst[:, j] == 1
        assert torch.equal(proj[j][~routed], f[~routed])
        assert not torch.allclose(proj[j][routed], f[routed])
    with pytest.raises(ValueError):
        model.loss_on_views([views(0), views(1)])


def test_single_row_predictor_batch_does_not_crash():
    model = LeJEPA(
        backbone(),
        proj=mlp([D, 16, 8]),
        predictor=mlp([D, 16, D]),
        predictor_inst=1,
        n_slices=32,
    )
    view_inst = torch.tensor([[1, 0], [0, 0], [0, 0], [0, 0]])
    out = model(global_views=[views(0), views(1)], view_inst=view_inst)
    out.loss.backward()
    assert torch.isfinite(out.loss) and model.predictor.training


def test_sigreg_prefers_gaussian_samples():
    sigreg = SlicedEppsPulley(n_slices=64)
    g = torch.Generator().manual_seed(0)
    gauss = torch.randn(512, 8, generator=g)
    bimodal = 2 * torch.sign(gauss) + 0.1 * torch.randn(512, 8, generator=g)
    assert sigreg(gauss) < sigreg(bimodal)
    assert int(sigreg.step) == 2
    assert sigreg(torch.stack([gauss, gauss])).shape == ()


def test_mlp_layers():
    net = mlp([4, 8, 2])
    assert [type(m) for m in net] == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Linear]
    plain = mlp([4, 8, 2], batch_norm=False)
    assert [type(m) for m in plain] == [nn.Linear, nn.ReLU, nn.Linear]
    assert net(torch.randn(3, 4)).shape == (3, 2)


def test_import_without_torch_distributed(monkeypatch):
    """The package imports and SIGReg runs on a torch built without
    distributed support (``torch.distributed.nn`` is not importable)."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    for name in ("group", "ReduceOp"):
        monkeypatch.delattr(torch.distributed, name)
    for name in [m for m in sys.modules if m.startswith("torch.distributed.nn")]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ImportError):
        importlib.import_module("torch.distributed.nn")
    import romae_lc.lejepa as lejepa

    importlib.reload(lejepa)
    stat = lejepa.SlicedEppsPulley(n_slices=8)(torch.randn(64, 4))
    assert torch.isfinite(stat)
