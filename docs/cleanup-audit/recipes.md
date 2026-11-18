## Recipe formats for romae-lc: hyrax vs Hydra vs a small TOML loader

**Recommendation:** use option 3. Each run gets a plain TOML recipe file. A loader of about 35 lines reads it into the argparse parser we already have. I built a working version and tested it against the real parser on the cleanup branch (details below).

### Comparison

| Need | hyrax (lincc-frameworks, 0.13.1, Oct 2026) | Hydra (hydra-core 1.3.8 + OmegaConf) | Option 3: TOML into argparse |
|---|---|---|---|
| File format | TOML. One big default file (`src/hyrax/hyrax_default_config.toml`, 785 lines). Your `hyrax_config.toml` is merged on top (`ConfigManager.merge_configs`, `config_utils.py:472`). | YAML. Configs are built from groups (`defaults:` list) and can refer to other values (`${...}`). | TOML. Keys are the argparse names (`mask_ratio`). An `extends = "base.toml"` line covers ablations. |
| Defaults | Must live in hyrax's default file. A key with no default only gets a warning (`_validate_runtime_config`, `config_utils.py:510`). | In YAML, or in dataclasses ("structured configs"). | Stay where they are now: the `add_argument(..., default=)` calls. |
| Command-line overrides | None. The CLI only has `-c/--runtime-config` (`src/hyrax_cli/main.py:66`). Overrides happen in Python through `Hyrax.set_config("a.b", v)`. | `key=value`, `+key=value` (adds a new key), `-m` for sweeps. Our current `--flag` style no longer works. | Unchanged. Any `--flag` beats the recipe, so the sbatch scripts stay as they are. |
| Saving the resolved config | Writes `runtime_config.toml` into a new timestamped folder `<results_dir>/<YYYYmmdd-HHMMSS>-<verb>-<rand>` (`create_results_dir`, `config_utils.py:629`; `log_runtime_config`, `:749`). | Writes `.hydra/config.yaml`, `overrides.yaml` and `hydra.yaml` under `outputs/<date>/<time>/` (`hydra.run.dir`). The checkpoint needs `OmegaConf.to_container(cfg, resolve=True)`. Pickling a DictConfig would make every later load need omegaconf. | `ckpt["args"] = vars(args)` stays as it is. Add `ckpt["recipe"]` (the file text) and `args.recipe` (its path). `args.json` stays (`pretrain_mae.py:475`). |
| Old checkpoints | Not covered. Hyrax has its own `config_version` migrations, but only for hyrax configs. | Not covered. The checkpoint format changes unless we flatten the config to a dict. | Not affected. The format is the same. Readers use `ckpt.get("recipe")`. |
| Use for configs only | Hard. `import hyrax` loads the whole package (`__init__.py` imports `Hyrax`). Install pulls in about 40 dependencies: pytorch-ignite<=0.5.5, chromadb, mlflow, lancedb, qdrant-client, confluent-kafka, lsdb, holoviews, bokeh, onnx, and more. Needs Python >=3.11. Built around verbs (train, infer) and its own dataset and model registry. | Possible with the Compose API, but its docs say this gives up features. `@hydra.main` takes over argv, logging and the output folder. | Yes. Uses the standard-library `tomllib` (our venv is Python 3.12.11). No new dependency. |
| Slurm | No launcher. You would run `hyrax train -c file` in your own sbatch script. | `hydra-submitit-launcher` 1.2.0 (last release May 2022) writes its own sbatch. That clashes with our self-requeuing chains (`queue()` in `jobs/pretrain_mae.sh`, `MAX_LINKS`, `BUDGET`, `DONE` files). | Our own sbatch scripts, unchanged. |
| Upkeep | Very active (commit 13 Nov 2026), but the config schema still changes (migrations, Pydantic). | Releases still come out (1.3.8, Oct 2026), but main had 1 commit in the last year. hydra-core 1.3.8 requires `omegaconf<2.4`, and omegaconf 2.4.0 is already out, so it needs a pin. | Our code, about 35 lines. |
| Lock-in | High. | Medium to high: the entry point, the override syntax and the output folders all change. | None. Deleting `--recipe` gives back today's behaviour. |

### Why option 3
1. Our needs are a flat set of about 25 changed flags per run, small ablations, overrides from sbatch, and the config saved in the checkpoint. Option 3 covers all of these and changes nothing else.
2. Hyrax only accepts its own way of working: its default file, its verbs, its results folders and about 40 dependencies. It has no command-line overrides at all. Its useful idea is "default TOML + user TOML + save the resolved TOML next to the results". Option 3 copies exactly that idea without the package.
3. Hydra would change how every one of the ~50 job scripts passes flags (`--mask-ratio 0.8` becomes `mask_ratio=0.8`). It takes over the output folders and needs a pin against omegaconf 2.4. Its two main strengths, deep config composition and `-m` sweeps over Slurm, do not fit our chained 2-hour jobs or our flat flags.
4. Old checkpoints keep loading as they are. All 15 places that read `ckpt["args"]` stay correct, for example `project/common.py:1060` (`_meta_from`), `data_args_from` (`common.py:182`), `eval/attn_probe.py:108` and `eval/classify_rf.py:95`.

### Prototype (tested, read-only, scratchpad)
- Loader: `/tmp/claude-91745/-projects-bfrf-hibb-romae-lc/f846f30c-9ae3-4667-b5f6-cb47d35ed7de/scratchpad/proto/recipe.py` (35 lines). `read_recipe(path)` follows `extends` and flattens `[table]` headers. `apply_recipe(parser, argv)` reads `--recipe` first and stops on unknown keys. It converts each value with the flag's own `type`, then calls `parser.set_defaults(**rec)` and then `parser.parse_args(argv)`, so CLI flags win.
- Test: `/tmp/claude-91745/-projects-bfrf-hibb-romae-lc/f846f30c-9ae3-4667-b5f6-cb47d35ed7de/scratchpad/proto/test_proto.py`. It builds the real parser (`add_data_args`, `add_frame_args`, `add_ladder_args`, `add_model_args`, `add_train_args`, `add_wandb_args`).
- Result: recipe `enc_o3_ph.toml` plus the launcher flags from `jobs/pretrain_mae.sh:73-77` match `project/runs/enc_o3_ph/args.json` exactly, apart from two things:
  - The 7 `--jepa-no-ema` flags added after that run (`views`, `view_keep`, `local_span`, `band_drop`, `sig_lambda`, `sig_slices`, `jepa_no_ema`). They are simply missing from the old `args.json`.
  - `device`, which is detected automatically ("cpu" on the login node).
- `--mask-ratio 0.8 --forecast-tail 10 60` correctly beat the recipe. The typo `mask_raito` stops the run with "unknown keys".
- Where it plugs in:
  - Replace `parser.parse_args(argv)` at `project/pretrain_mae.py:437` with `apply_recipe(parser, argv)`.
  - Keep the resume block at `:443-447` as it is: on resume, the checkpoint args beat both the recipe and the CLI, except `RUN_CONTROL` (`:96`).
  - Add `recipe=Path(args.recipe).read_text() if args.recipe else None` to the saved checkpoint dict.
  - The evaluation scripts need no change. They read `ckpt["args"]`.

### Recipe sketch for enc_o3_ph (`project/recipes/`)
The values come from `B=` in `project/jobs/option3_one.sh` (lane 1) and the fixed flags in `jobs/pretrain_mae.sh:73-75`.

`project/recipes/enc_o3.toml` (the shared base of the option-3 lanes):
```toml
# Option 3 (docs/PLAN.md, end of Phase 1): token JEPA warm-started from enc_j1,
# small predictor (2 x 192), phase rope at the encoder's own frequency. 20k steps.
[model]
size = "wide"
spectral = true
spectral_aux = 0.5
phase_rope = 4
phase_freq = "local"

[jepa]
jepa = true
jepa_proj = 32
jepa_recon_weight = 0.5
jepa_init = "project/runs/enc_j1/mae.pt"

[mask]
mask_mode = "blockplus"
mask_ratio = 0.5
mask_block_share = 0.5

[data]
window = 250.0
min_tokens = 16
max_tokens = 256

[train]
steps = 20000
batch_size = 64
```

`project/recipes/enc_o3_ph.toml` (lane 1: phase rope only, no tail):
```toml
extends = "enc_o3.toml"
wandb_group = "option3"
```

The other lanes are just as short:
- `enc_o3_fcph.toml` adds `forecast_tail = [10, 60]`.
- `enc_o3_fcph80` extends `enc_o3_fcph.toml` and adds `mask_ratio = 0.8`.
- `enc_o3_fcph_pband` extends `enc_o3_fcph.toml` and adds `mask_mode = "phaseband"` and `phase_band = [0.1, 0.3]`.

Run settings stay as flags in the sbatch script, not in the recipe: `--out`, `--workers`, `--pin-memory`, `--eval-every`, `--ckpt-every`, `--time-budget`, `--wandb`. These are the `RUN_CONTROL` keys.

In the job script:
```
train enc_o3_ph --recipe project/recipes/enc_o3_ph.toml
```

### Small points to settle
- The `[table]` headers are only for reading. The keys must be unique across tables, because the loader flattens them.
- Relative paths such as `jepa_init` are resolved from the repo root (`#SBATCH --chdir`), as they are today.
- `enc_le1` would get `project/recipes/enc_le1.toml` with the `ARGS=` of `project/jobs/lejepa.sh`.