"""Run recipes: a TOML file of argparse defaults. CLI flags still win."""
import argparse, sys, tomllib
from pathlib import Path


def read_recipe(path) -> dict:
    """The recipe as a flat dict; `extends = "other.toml"` is read first."""
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    base = raw.pop("extends", None)
    out = read_recipe(path.parent / base) if base else {}
    for k, v in raw.items():  # [table] headers are only for reading; keys are flat
        out.update(v if isinstance(v, dict) else {k: v})
    return out


def apply_recipe(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Parse --recipe first, load it as defaults, then parse the full command line."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--recipe")
    known, _ = pre.parse_known_args(argv)
    parser.add_argument("--recipe", default=None, help="TOML run recipe; flags override it")
    if known.recipe:
        rec = read_recipe(known.recipe)
        acts = {a.dest: a for a in parser._actions}
        bad = sorted(set(rec) - set(acts))
        if bad:
            parser.error(f"recipe {known.recipe}: unknown keys {bad}")
        for k, v in rec.items():  # same types as the flag would give
            t = acts[k].type
            if t is not None and v is not None:
                v = [t(x) for x in v] if isinstance(v, list) else t(v)
            rec[k] = v
        parser.set_defaults(**rec)
    return parser.parse_args(argv)
