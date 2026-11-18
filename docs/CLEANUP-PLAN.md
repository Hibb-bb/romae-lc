# Clean-up plan (branch `cleanup`, started 2026-11-18)

Goal: a codebase where it is easy to see which file holds what, and how training, validation and
testing work. The audits behind this plan are in `docs/cleanup-audit/` (flags, common, splits,
evalsurface, tests, recipes, critic). Read the ones your step needs.

`main` is tagged `pre-cleanup`. Work happens only in this checkout
(`/projects/bfrf/hibb/romae-lc-cleanup`). The main checkout (`/projects/bfrf/hibb/romae-lc`) must
not be touched: a queued training job (`enc_le1`) runs its code. Run outputs live only there
(`project/runs`, `project/results`, gitignored); read them, never write them.

## Hard rules

1. **Behaviour does not change.** The clean-up moves, deletes and renames. It does not change what
   any kept training recipe or evaluation computes. The golden check (below) must pass after every step.
2. **Every ICML checkpoint keeps loading**: enc_j1, enc_o2_noadd, enc_o2_fc, enc_o2_fcbig,
   enc_o2_fcbig_ph, enc_o3_fcph, enc_o3_ph, enc_o3_fcph80, enc_o3_fcph_pband (and enc_le1 once trained).
   So: `TokenJEPA` and `RoMAEForPreTraining` keep every constructor argument a checkpoint stores
   (`abs_timescales`, `use_cls`, `rope`, `n_axes`, `n_channels`, `spectral`, `attn_pool`, `clean_target`,
   `ema`, `ema_end`, `loss`, `use_target`, `recon_weight`, `proj_dim`, `proj_hidden`, `phase_rope`,
   `phase_freq`, `no_ema`, `sig_slices`), state-dict names stay the same, and `TokenSpec.from_dict`,
   `FrameConfig`, `Ladder.from_dict` keep their formats. The checkpoint format (`kind`, `args`, `backbone`,
   `mae`, `jepa`, `spec`, `frames`, `ladder`, `classes`, `wandb_id`) does not change.
3. **`project/` and `project/runs/` keep their paths.** Checkpoints store relative paths
   (`out="project/runs/<run>"`, `jepa_init="project/runs/enc_j1/mae.pt"`).
4. **Legacy code moves to `legacy/`, it is not deleted.** `legacy/` mirrors the old paths
   (`legacy/project/...`, `legacy/romae_lc/...`, `legacy/examples/...`, `legacy/jobs/...`, `legacy/tests/...`).
   Nothing outside `legacy/` imports it. Legacy code is frozen and is not expected to run from this
   branch; `legacy/README.md` says to check out the tag `pre-cleanup` to run it. Use `git mv`.
   Watch `.gitignore`: the unanchored `runs/` and `results/` patterns also hide folders under `legacy/`.
5. **Untracked files at the repo root of the main checkout are not ours.** Leave them.
6. **No Slurm jobs, no pushes, no changes to the main checkout.** Small CPU checks on the login node
   are fine. Always use `/projects/bfrf/hibb/romae-lc/.venv/bin/python`, run from the root of this
   checkout with `python -m ...` or `python -m pytest ...` so that this checkout's code is imported
   (the venv's editable install points at the MAIN checkout; assert `project.__file__` and
   `romae_lc.__file__` start with this checkout's path when in doubt).
7. One commit per step on branch `cleanup`, message ending with
   `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
8. Docs and comments in plain, short sentences.

## The golden check (step 0 builds it; every later step runs it)

`python -m project.tests.golden check` compares against files recorded BEFORE any change
(stored in `.golden/`, untracked):

- **Checkpoints**: for each of the 9 trained ICML runs, `load_encoder` on
  `/projects/bfrf/hibb/romae-lc/project/runs/<run>/mae.pt` (CPU), encode one fixed synthetic token batch,
  and one masked `model(values, positions, pad, mask)` forward: features, loss, pred must match
  (`allclose`, tight tolerance).
- **Training**: short seeded CPU runs on `--data sim` (a few steps) for the live code paths:
  EMA JEPA with reconstruction + blockplus; + forecast tail + phase rope; phaseband mask; random mask at
  0.8; `--jepa-no-ema` views (le1 path). The logged losses and a hash of the final weights must match.
- **Flags**: for each of the 10 ICML runs, the flags that the run set (non-default values in its
  `args.json`) must resolve to the same values (from step 3 on: through its recipe file).

## Target layout

```
romae_lc/                 the library: tokens, rotary ladder, transformer, spectral layer, encoder, data,
                          frames (FrameConfig, frame_grid, FrameDataset), analysis (time-encoding helpers)
project/
  data.py                 records: load_data, splits, class names, data args
  windows.py              records -> windows -> tokens: TokenSpec, frame config, WindowDataset, collate
  masks.py                what is hidden: random / block / blockplus / phase band / forecast tail
  views.py                global and local views of a window (make_views) for the no-EMA recipe
  backbone.py             encoder size presets and the rotary ladder
  jepa.py                 the model: TokenJEPA (EMA and no-EMA), predictor + phase rope, forward_views, sigreg
  checkpoint.py           save / load: mae_state, jepa_state, load_mae, load_encoder, EncoderMeta
  probes.py               the small probes run DURING training (linear class probe, log-period ridge)
  recipe.py               TOML recipe loader (recipe file -> argparse defaults; CLI flags win)
  recipes/                one TOML file per run (enc_j1, enc_o2_*, enc_o3_*, enc_le1), `extends` for variants
  train.py                pretraining: flags, loop, periodic validation, checkpoints
  tracking.py, utils.py   wandb / timers; small helpers
  eval/
    common_eval.py        shared helpers (fourier_forecast, zstats, star picking, window loader)
    encode.py             cache window latents for train / validation / test
    card_class.py         test card 1: class macro F1 (attention probe)
    card_period.py        test card 2: log-period probe and spectral-peak hit rate
    card_phase.py         test card 3: per-token phase R2
    card_forecast.py      test card 4: forecast tail / random gap / phase gap
    card_demo.py          forecast example plots
    classify_rf.py        random-forest reference
    test_card.py          runs the card for one run, writes one JSON
    summary.py            one table over runs
    style.py              plot style
  jobs/                   train.sh (recipe -> run), test_card.sh, demo.sh, plus the chain helper
  tests/                  tests of the kept code (+ golden.py)
tests/                    library tests
legacy/                   everything else, frozen
docs/                     README-level guides, PLAN.md, CODE-MAP.md, this plan
```

Old module names that other people's commands may still use get no shims: the branch is a clean break,
`pre-cleanup` keeps the old tree. (Exception during the work: a re-export shim may live for one step.)

## Recipes

Hyrax-style TOML without the hyrax package (see `docs/cleanup-audit/recipes.md` and the working
prototype in `docs/cleanup-audit/recipe-proto/`): defaults stay in the argparse parser; a recipe file
sets some of them; `extends = "base.toml"` for variants; any `--flag` on the command line beats the
recipe; unknown keys are an error; the checkpoint stores `args` as before plus the recipe text.
`python -m project.train --recipe project/recipes/enc_o3_ph.toml --out project/runs/NAME`.

## The split rule (written in docs; evaluation protocols do NOT change in this clean-up)

- **Train**: the encoder, and the fit set of probes.
- **Validation**: monitoring during training, picking a probe's epoch or read-out, fitting the forecast
  decoder, demos, and comparing runs.
- **Test**: final numbers.
- **Catalogue period**: a training target on train stars (spectral aux loss, phase-band mask); at
  evaluation only a target, a test mask or a baseline; never an encoder input.

`docs/cleanup-audit/splits.md` lists places where today's scripts differ from this rule (test F1 used to
compare runs, the period probe never scoring test, whole-curve normalisation, unseeded forecast masks,
different fit sets per probe). These are NOT fixed here: list them in the docs as open items.

## Steps

0. Golden files and `project/tests/golden.py` (no other change).
1. Move never-imported code to `legacy/` (old world-model line, old eval pages and figures, examples,
   `romae_lc/lewm.py`, `romae_lc/lejepa.py`, their tests, about 55 job scripts); trim
   `romae_lc/__init__.py`; remove the `sys.path.insert` lines from eval scripts; remove
   `#SBATCH --chdir=...` from kept job scripts (jobs are submitted from the repo root) .
2. Cut dead options from `pretrain_mae.py`, `common.py`, `period_probe.py`, `token_phase_probe.py`,
   `cache_latents.py` (bottleneck, GRU-ODE, templates and query tokens, phase / fold losses, the dyn add-on,
   CLS terms, the flow predictor `--pred`, world-model loaders); move `sigreg` into `jepa.py`; move the
   freed modules to `legacy/`. Flags that only dead paths read are removed.
3. New layout (module split above), `project/train.py`, the recipe loader and one recipe per ICML run.
4. Evaluation: the `card_*` files, `common_eval.py`, `test_card.py`, `summary.py`, the new job scripts.
5. Docs: root `README.md` (where is what; how train / validation / test work; how to train and test a
   run), `docs/CODE-MAP.md` rewritten for the new tree, `legacy/README.md`, open items from the audits.
