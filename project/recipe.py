"""Run recipes: a TOML file that sets training flags.

The defaults stay in the argparse parser. A recipe sets some of them. Any
``--flag`` on the command line beats the recipe. A key that is not a flag of
the parser is an error, so a typo stops the run.

    # project/recipes/enc_o3_fcph.toml
    extends = "enc_o3.toml"       # read first, from the same folder
    forecast_tail = [10, 60]      # the key is the flag name with "_" for "-"

``[table]`` headers are only for reading: the keys are flat, so a key may
appear once. Run settings (``--out``, ``--workers``, ``--time-budget``,
``--wandb``, ...) stay on the command line.

    python -m project.train --recipe project/recipes/enc_o3_ph.toml --out project/runs/NAME
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def recipe_files(path) -> list[Path]:
    """The recipe file and the files it extends, the base first."""
    path = Path(path)
    base = tomllib.loads(path.read_text()).get("extends")
    return (recipe_files(path.parent / base) if base else []) + [path]


def read_recipe(path) -> dict:
    """The recipe as a flat dict ``{flag name: value}``. A later file beats the file it extends."""
    out: dict = {}
    for f in recipe_files(path):
        raw = tomllib.loads(f.read_text())
        raw.pop("extends", None)
        for k, v in raw.items():
            out.update(v if isinstance(v, dict) else {k: v})
    return out


def recipe_text(path) -> str:
    """The text of the recipe and of every file it extends, the base first. This is what the checkpoint stores."""
    return "".join(f"# ---- {f.name}\n{f.read_text().rstrip()}\n\n" for f in recipe_files(path))


def _convert(action: argparse.Action, value, fail):
    """``value`` as the flag itself would give it (same type, same choices)."""
    name = action.dest
    if action.nargs == 0:  # a store_true flag
        if not isinstance(value, bool):
            fail(f"{name} must be true or false, got {value!r}")
        return value
    many = action.nargs is not None
    if many != isinstance(value, list):
        fail(f"{name} must be {'a list' if many else 'one value'}, got {value!r}")
    if isinstance(action.nargs, int) and len(value) != action.nargs:
        fail(f"{name} takes {action.nargs} values, got {value!r}")
    items = value if many else [value]
    if action.type is not None:
        items = [action.type(x) for x in items]
    for x in items:
        if action.choices is not None and x not in action.choices:
            fail(f"{name} must be one of {list(action.choices)}, got {x!r}")
    return items if many else items[0]


def apply_recipe(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Parse the command line with ``--recipe FILE`` as one more flag: the
    recipe's values become the parser's defaults, then the command line is
    parsed, so its flags win. ``args.recipe`` is the path (None without one)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--recipe")
    known, _ = pre.parse_known_args(argv)
    parser.add_argument("--recipe", default=None, help="TOML run recipe (project/recipes/*.toml); flags on the command line beat it")
    if known.recipe:
        rec = read_recipe(known.recipe)
        actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "recipe")}
        bad = sorted(set(rec) - set(actions))
        if bad:
            parser.error(f"recipe {known.recipe}: unknown keys {bad}")
        fail = lambda msg: parser.error(f"recipe {known.recipe}: {msg}")  # noqa: E731
        parser.set_defaults(**{k: _convert(actions[k], v, fail) for k, v in rec.items()})
    return parser.parse_args(argv)
