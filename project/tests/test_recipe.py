"""Tests of the recipe loader (project/recipe.py) and of the recipe files in project/recipes/."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from project import train
from project.checkpoint import load_mae
from project.recipe import apply_recipe, read_recipe, recipe_text

from ._args import MAE_ARGS

RECIPES = Path(train.__file__).parent / "recipes"
RUNS = ("enc_j1", "enc_o2_noadd", "enc_o2_fc", "enc_o2_fcbig", "enc_o2_fcbig_ph", "enc_o3_ph", "enc_o3_fcph", "enc_o3_fcph80",
        "enc_o3_fcph_pband", "enc_le1")  # fmt: skip


def parse(argv):
    return apply_recipe(train.build_parser(), argv)


def test_every_run_has_a_recipe_that_parses():
    for run in RUNS:
        a = parse(["--recipe", str(RECIPES / f"{run}.toml")])
        assert a.jepa and a.spectral and a.spectral_aux == 0.5 and a.size == "wide" and a.window == 250.0
        assert a.recipe.endswith(f"{run}.toml")


def test_extends_and_command_line_wins():
    f = str(RECIPES / "enc_o3_fcph80.toml")
    a = parse(["--recipe", f])
    # from the file, from enc_o3_fcph, from enc_o3_ph, from from_j1, from enc_j1, from base
    assert (a.mask_ratio, a.forecast_tail, a.phase_rope, a.steps, a.jepa_proj, a.max_tokens) == (0.8, [10.0, 60.0], 4, 20000, 32, 256)
    assert a.jepa_init == "project/runs/enc_j1/mae.pt" and a.wandb_group == "option3"
    b = parse(["--recipe", f, "--mask-ratio", "0.6", "--steps", "7"])
    assert (b.mask_ratio, b.steps, b.phase_rope) == (0.6, 7, 4)
    text = recipe_text(f)
    assert text.index("# ---- base.toml") < text.index("# ---- enc_j1.toml") < text.index("# ---- enc_o3_fcph80.toml")
    assert parse([]).recipe is None  # no recipe: the parser's own defaults


def test_bad_recipes_stop(tmp_path, capsys):
    cases = {
        "typo": "mask_raito = 0.8\n",
        "choice": 'mask_mode = "mix"\n',
        "count": "forecast_tail = [10.0]\n",
        "bool": 'jepa = "yes"\n',
        "list": "mask_ratio = [0.5, 0.8]\n",
    }
    for name, text in cases.items():
        f = tmp_path / f"{name}.toml"
        f.write_text(text)
        with pytest.raises(SystemExit):
            parse(["--recipe", str(f)])
    assert "unknown keys ['mask_raito']" in capsys.readouterr().err
    (tmp_path / "ok.toml").write_text("[mask]\nmask_ratio = 1\n")  # a table header; an int for a float flag
    assert read_recipe(tmp_path / "ok.toml") == {"mask_ratio": 1}
    assert parse(["--recipe", str(tmp_path / "ok.toml")]).mask_ratio == 1.0


def test_recipe_text_is_stored_and_kept_on_resume(tmp_path):
    rec = tmp_path / "toy.toml"
    rec.write_text('# a toy run\n[jepa]\njepa = true\njepa_proj = 8\njepa_proj_hidden = 16\n[mask]\nmask_mode = "blockplus"\n')
    out = tmp_path / "run"
    train.main(MAE_ARGS + ["--recipe", str(rec), "--steps", "2", "--out", str(out)])
    ckpt = torch.load(out / "mae.pt", map_location="cpu", weights_only=False)
    assert ckpt["recipe"] == recipe_text(rec) and "# a toy run" in ckpt["recipe"]
    assert ckpt["args"]["recipe"] == str(rec) and ckpt["args"]["jepa_proj"] == 8 and ckpt["kind"] == "jepa"
    model, meta = load_mae(out / "mae.pt")
    assert model.proj_dim == 8 and meta.args["mask_mode"] == "blockplus"
    # a resume keeps the recipe the run started with, whatever is passed now
    rec.write_text("jepa = true\njepa_proj = 4\n")
    train.main(MAE_ARGS + ["--recipe", str(rec), "--steps", "3", "--out", str(out)])
    again = torch.load(out / "mae.pt", map_location="cpu", weights_only=False)
    assert again["recipe"] == ckpt["recipe"] and again["args"]["jepa_proj"] == 8 and again["step"] == 3
