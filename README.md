# romae-lc

Self-supervised encoders for light curves of variable stars (ZTF).
Every observation is one token with a real time and a real wavelength. Nothing is binned.
The encoder is trained as a token-level JEPA: it predicts the latents of hidden observations.

This page tells you four things:

1. [Where is what](#1-where-is-what): one line per file.
2. [How train / validation / test work](#2-how-train--validation--test-work): which step uses which split.
3. [How to train a run and how to test it](#3-how-to-train-a-run-and-how-to-test-it).
4. [Open items](#4-open-items): known problems that the clean-up did not fix.

More detail:

- `docs/CODE-MAP.md`: the main functions of every file, and old file name -> new file name.
- `docs/LIBRARY.md`: how to use the `romae_lc` library on its own.
- `docs/PLAN.md`: the research plan and the results so far.
- `legacy/README.md`: the old code that is no longer used.

## Set-up

```bash
uv sync --all-extras          # makes .venv with torch, datasets, matplotlib, wandb, scikit-learn
source .venv/bin/activate
python -m pytest project/tests tests -q     # CPU, a few minutes
```

Run every script from the repo root as a module: `python -m project.train ...`.
Do not run `python project/train.py`. The imports only work with `python -m`.

The data is the ZTFxPC set at `/projects/bfrf/data/PC_matches/ZTFxPC`. `--data sim` uses a small
simulator instead and needs no files.

## 1. Where is what

The order follows the data: records, windows, masks, model, training, checkpoints, evaluation.

### The library, `romae_lc/`

| file | what it holds |
|---|---|
| `romae_lc/data.py` | `Record` (one star: times, values, errors, bands, class, period), `load_pc` (reads ZTFxPC), `simulate` (toy stars), `normalize` |
| `romae_lc/frames.py` | cutting a record into time windows: `FrameConfig`, `FrameDataset`, `frame_grid` |
| `romae_lc/tokenize.py` | observations -> tokens with (time, wavelength) positions: `tokenize`, `Tokens` |
| `romae_lc/rope.py` | the rotary position encoding for real-valued positions |
| `romae_lc/transformer.py` | the transformer blocks |
| `romae_lc/spectral.py` | the spectral layer (a learned periodogram inside the encoder) and its loss against a period |
| `romae_lc/model.py` | the encoder `RoMAE` and the masked autoencoder `RoMAEForPreTraining` |
| `romae_lc/analysis.py` | helpers to choose the time scales of the rotary encoding |

### The research code, `project/`

| step | file | what it holds |
|---|---|---|
| data | `project/data.py` | `load_data`: the records of each split, the class names, the data flags |
| windows | `project/windows.py` | records -> windows -> tokens: `TokenSpec`, the error channel, the window loaders, `WindowDataset` |
| masks | `project/masks.py` | which tokens are hidden: random, block, blockplus, phase band, forecast tail |
| views | `project/views.py` | global and local views of a window (`make_views`), for the no-EMA recipe |
| model | `project/backbone.py` | the encoder sizes (`--size`) and the rotary time ladder |
| model | `project/jepa.py` | the model: `TokenJEPA` (EMA and no-EMA), the predictor, `forward_views`, `sigreg` |
| training | `project/recipe.py` | reads a recipe file (TOML) and turns it into training flags |
| training | `project/recipes/` | one recipe file per run (`enc_j1`, `enc_o2_*`, `enc_o3_*`, `enc_le1`); `base.toml` and `from_j1.toml` are shared parts |
| training | `project/train.py` | the training script: flags, loop, validation every few steps, checkpoints |
| training | `project/probes.py` | the small probes that run during training (linear class probe, log-period ridge, shuffle score, hand-feature baseline) |
| training | `project/tracking.py` | wandb logging and step timers |
| training | `project/utils.py` | small helpers: learning-rate schedule, seeds, device, JSON logs |
| checkpoints | `project/checkpoint.py` | save and load: `mae_state`, `jepa_state`, `load_mae`, `load_encoder`, `EncoderMeta` |
| evaluation | `project/eval/encode.py` | encodes every window once and saves the latents (`latents.pt`, `latents_test.pt`) |
| evaluation | `project/eval/card_class.py` | test card 1: class macro F1 with the star probe |
| evaluation | `project/eval/card_period.py` | test card 2: log-period probe, and the hit rate of the spectral peak |
| evaluation | `project/eval/card_phase.py` | test card 3: phase R2 per token |
| evaluation | `project/eval/card_forecast.py` | test card 4: fill in hidden points (last days, random gaps, a phase gap) |
| evaluation | `project/eval/test_card.py` | runs the four cards for one run and writes one JSON file |
| evaluation | `project/eval/summary.py` | one table over the JSON files of many runs |
| evaluation | `project/eval/classify_rf.py` | the random-forest reference for the class task |
| evaluation | `project/eval/card_demo.py` | forecast example plots (figures only, no score) |
| evaluation | `project/eval/common_eval.py` | helpers the cards share (no command line) |
| evaluation | `project/eval/style.py` | plot colours |
| evaluation, extra | `project/period_probe.py` | many period read-outs from the latents next to Lomb-Scargle, on validation. Not part of the test card |
| evaluation, extra | `project/phase_probe.py` | phase of a window's start from the window latent, on validation. Not part of the test card. The phase card uses its `reference_phase` |
| evaluation, extra | `project/fold.py` | the phase-fold test of a period. Only `period_probe.py` uses it |

### Jobs, tests, the rest

| path | what it holds |
|---|---|
| `project/jobs/train.sh` | Slurm job: trains one recipe, in several linked jobs if needed, then runs the test card |
| `project/jobs/test_card.sh` | Slurm job: the test card of one or more runs |
| `project/jobs/demo.sh` | Slurm job: the forecast example plots |
| `project/jobs/after_done.sh` | waits for a file, then submits a job |
| `project/jobs/in_session.sh` | runs a command inside a running Slurm session named `debug` |
| `project/tests/` | tests of `project/`. `golden.py` is the before/after check of the clean-up |
| `tests/` | tests of the library |
| `project/runs/NAME/` | one folder per run (not in git): `args.json`, `log.jsonl`, `last.pt`, `mae.pt`, `DONE`, the latent caches |
| `project/results/` | evaluation results (not in git). The test cards are in `project/results/test_card/` |
| `legacy/` | old code, frozen. Nothing imports it. See `legacy/README.md` |
| `docs/` | the plan, the notes, the audits of the clean-up |

## 2. How train / validation / test work

### Where the splits come from

- ZTFxPC has three fixed splits made upstream: 43,497 train, 5,472 validation and 5,464 test stars.
  No star is in two splits.
- `project.data.load_data` loads `train` and `validation` unless a script asks for `test` by name.
- Stars with fewer than 8 good points are dropped (`--min-points`). That leaves 43,191 train and
  5,414 validation stars.
- Every star is scaled per band with the median and the scatter of its whole light curve.
- Training needs stars long enough for a sequence of windows. With 8 windows of 250 days that leaves
  40,309 train stars. `enc_le1` uses one window per sample, so it keeps more.
- `--data sim` makes its own split: 60 % train, 20 % validation, 20 % test.

### The rule

- **Train**: the encoder learns from it. Probes are fitted on it.
- **Validation**: watching a run while it trains, picking a probe's epoch or read-out, fitting the
  forecast decoder, demo plots, and comparing runs.
- **Test**: final numbers only.
- **Catalogue period**: it is a training target on train stars (the spectral aux loss, the
  phase-band mask). At evaluation it is only a target, a mask for a test, or a baseline.
  It is never an input of the encoder.

Today's scripts do not follow this rule everywhere. Section 4 lists where.

### Which step uses which split

Training (`project/train.py`):

| step | split | for what | where in the code |
|---|---|---|---|
| gradient steps | train | the encoder loss | `build_loaders`, `train_step` |
| rotary time ladder | train (128 stars, `--ladder-curves`) | the time scales of the position encoding; no labels | `build_tokens` -> `backbone.get_ladder` |
| error-channel statistics | train | scaling the error channel of the tokens | `build_tokens` -> `windows.err_stats` |
| catalogue period as a target | train | the spectral aux loss (`--spectral-aux 0.5`, every run); the `phaseband` mask (`enc_o3_fcph_pband`); period-hit logging of the views (`enc_le1`) | `Run.train_periods`, `window_loss`, `views_loss`, `jepa.forward_views` |
| validation loss | validation: 1,000 stars (`--val-objects`), 931 of them long enough | the loss with random points hidden and with stretches hidden; no aux loss | `validation_losses` |
| probes during training | fit: 4,000 train stars (`--probe-train`). Score: the validation-loss stars, so at most 931 (`--probe-val 2000` cannot give more) | linear class probe (13 classes) and ridge on log period | `evaluate` -> `probes.probe` |
| hand-feature baseline | the same stars as the probes | the bar the encoder has to clear | `baseline` |
| shuffle score | validation (512 of the validation-loss stars) | does the encoder use the order in time | `evaluate` -> `probes.shuffle_score` |
| the final checkpoint | none | `mae.pt` is the last step. No step is picked on validation | `finish` |

Evaluation (`project/eval/`), as `test_card.py` runs it:

| step | fit on | picked on | scored on | notes |
|---|---|---|---|---|
| `encode.py` | - | - | - | `latents.pt` holds train + validation, `latents_test.pt` holds test. Every window of every star, stride 0.25 window |
| class card (`card_class.py`) | train | validation (best epoch, macro F1) | **test** | 13 classes, 40 epochs, seeds 42 / 100 / 200 |
| period card, log-period probe (`card_period.py`) | train | validation (best epoch, R2) | **test** | the same probe with one output |
| period card, spectral peak (`card_period.py`) | nothing is fitted | - | **validation**, all stars with a period | the encoder's own period estimate against the catalogue period |
| phase card (`card_phase.py`) | 70 % of 600 validation stars | - | the other 30 % of the **validation** stars | the split is by star |
| forecast card (`card_forecast.py`) | decoder on 1,000 validation stars | - | **test**, all stars with 64 points or more | windows do not overlap (stride 1.0). Three modes: last 10 days, 30 % random gaps, a 0.2-cycle phase gap |
| random forest (`classify_rf.py`, only with `--rf`) | train + validation | fixed parameters, no search | **test** | 16 classes, 3 seeds |

Outside the test card:

| step | fit on | scored on | notes |
|---|---|---|---|
| `card_forecast.py` run by hand with its defaults | 70 % of 600 validation stars | the other 30 % | stride 0.25. This is the protocol of the forecast numbers in `docs/PLAN.md` |
| `project/period_probe.py` | train | validation | 13 read-outs; the best one is picked on validation and reported on validation. It never scores test |
| `project/phase_probe.py` | train windows | validation windows | |
| `card_demo.py` | - | - | validation stars; figures only |

### Where the catalogue period is used

- As a training target, on train stars only: the spectral aux loss in every run, the `phaseband` mask
  in `enc_o3_fcph_pband`, and the period-hit numbers that `forward_views` logs.
- As an encoder input: never. The phase rope of the predictor uses the spectral layer's own estimate.
- At evaluation: it is the target of the period probes and of the spectral-peak test. The phase card
  builds its phase target from it. The forecast card uses it for the phase-gap mask and for one
  Fourier baseline. The demo uses it to pick stars and to fold the plots.

## 3. How to train a run and how to test it

### Recipes

A recipe is a TOML file in `project/recipes/`. Its keys are the flags of `project/train.py`
(`--mask-ratio` is `mask_ratio`).

- The defaults live in the parser of `project/train.py`. A recipe sets some of them.
- `extends = "other.toml"` takes another recipe first. `enc_o3_fcph.toml` extends `enc_o3_ph.toml`,
  which extends `from_j1.toml`, and so on down to `base.toml`.
- A flag on the command line beats the recipe.
- A key that is not a flag stops the run. So does a wrong value.
- Run settings are not recipe keys. Give them on the command line: `--out`, `--workers`,
  `--pin-memory`, `--eval-every`, `--ckpt-every`, `--time-budget`, `--wandb`, `--device`.
- The checkpoint stores all flags (`args`) and the text of the recipe files (`recipe`).

`python -m project.train --help` lists every flag.

### Train on the cluster

Submit from the repo root. The script trains, and then runs the test card.

```bash
sbatch --job-name=enc_o3_ph project/jobs/train.sh enc_o3_ph
# a long run as two linked jobs:
sbatch --job-name=enc_le1 --time=24:00:00 --export=ALL,MAX_LINKS=2,BUDGET=84000 project/jobs/train.sh enc_le1
```

- The first argument is a recipe name or the path of a `.toml` file. More arguments go to
  `project.train` and beat the recipe.
- The run folder is `project/runs/<recipe name>`. Set `OUT` for another folder.
- `BUDGET` is the training time of one job in seconds. `MAX_LINKS` is how many jobs the run may take.
  A later job resumes from `last.pt`.
- `EVAL=0` trains only.
- The script uses `.venv` in the repo root. Set `VENV` for another one.
- The `enc_o2_*` and `enc_o3_*` recipes start from `project/runs/enc_j1/mae.pt`. That file has to exist.
- `enc_j1` itself was trained by older code. Its recipe gives the same flags, not the same code.
  Keep its `mae.pt`. Training it again will not give the same weights.

### Train by hand

```bash
python -m project.train --recipe project/recipes/enc_o3_ph.toml --out project/runs/NAME --wandb
```

A small run on simulated stars, on the CPU (about 2 minutes with 4 threads). It uses the `enc_le1` recipe and makes
everything small on the command line:

```bash
python -m project.train --recipe project/recipes/enc_le1.toml --data sim --n-sim 300 \
    --width 24 --depth 2 --heads 2 --dec-width 48 --dec-heads 2 --dec-depth 1 \
    --window 100 --min-tokens 4 --max-tokens 64 --batch-size 8 --steps 20 --eval-every 10 \
    --probe-train 64 --probe-val 32 --shuffle-objects 16 --ladder-curves 8 \
    --local-span 10 30 --view-keep 0.3 --sig-slices 64 --workers 0 --device cpu \
    --out /tmp/toy_le1
```

On a busy login node, put `OMP_NUM_THREADS=4` in front. Without it the run can be many times slower.

What a run folder holds:

- `args.json`: every flag of the run. `ladder.json`: the time ladder.
- `log.jsonl`: one row per log step and per validation.
- `last.pt`: the newest checkpoint, with the optimizer. A run started again with the same `--out`
  resumes from it. Model and data flags then come from the checkpoint, not from the command line.
- `mae.pt` and `DONE`: written when `--steps` is reached. `mae.pt` is what the evaluation loads.

### The test card

```bash
sbatch project/jobs/test_card.sh enc_o3_ph enc_o3_fcph       # on the cluster
python -m project.eval.test_card --run enc_o3_ph             # by hand
python -m project.eval.summary                               # one table over all runs
```

- `--run NAME` reads `project/runs/NAME/mae.pt`. `--ckpt PATH` takes any checkpoint.
- It first makes `latents.pt` and `latents_test.pt` next to the checkpoint, if they are missing.
- It then runs the class, period, phase and forecast cards and writes
  `project/results/test_card/NAME.json`. The file is written after every card.
- A card that is already in the file is skipped. So a job that ran out of time can be submitted again.
  `--force` runs everything again. `--cards class period` runs only some cards.
- `--rf` (or `RF=1` for the job) adds the random-forest reference.
- `PERIOD_PROBE=1` for the job also runs `project.period_probe` (validation read-outs).
- `project.eval.summary` prints the table. The line under the table says which split each column is
  scored on.

The test card of the small run from above (5 to 10 minutes on the CPU):

```bash
python -m project.eval.test_card --ckpt /tmp/toy_le1/mae.pt --data sim --results /tmp/toy_le1/results \
    --epochs 2 --workers 0 --min-context 4
python -m project.eval.summary --dir /tmp/toy_le1/results/test_card
```

Each card also runs alone, for example `python -m project.eval.card_forecast --help`.
The forecast example plots: `sbatch project/jobs/demo.sh enc_o3_fcph_pband`.

### Tests and the golden check

```bash
python -m pytest project/tests tests -q
CUDA_VISIBLE_DEVICES= python -m project.tests.golden check --strict
```

The golden check compares today's code with files recorded before the clean-up (in `.golden/`, not
in git): the flags of the 10 runs, the outputs of the 9 trained checkpoints on a fixed batch, and
5 short training runs. It reads the checkpoints from `/projects/bfrf/hibb/romae-lc/project/runs`.

## 4. Open items

The clean-up moved and renamed code. It did not change what any evaluation computes.
These known problems are still there.

### Split protocol

1. **Test numbers are used to compare runs.** The results table in `docs/PLAN.md` ("Option 3
   results") compares runs by class F1 on test, and decisions were drawn from it. The validation
   score of the same probe is saved (`val_best` in the card, `class_f1_attn_val` in its headline).
   `summary.py` still prints the test number.
2. **The PLAN table mixes splits without saying so.** Forecast columns are validation numbers,
   class F1 is a test number, and "period within 10 %" is the best of 13 read-outs on validation.
3. **`period_probe.py` never scores test.** Its best read-out is picked on validation and reported
   on validation, so the number is too good.
4. **The peak and phase cards score validation, not test.** By the rule, final numbers belong on test.
5. **Each probe fits on a different set.** The class and period probes fit on train. The random
   forest fits on train + validation. The forecast decoder fits on validation stars. The phase probe
   fits on 70 % of 600 validation stars.
6. **The same metric has more than one protocol.** Forecast by hand: 600 validation stars, stride
   0.25. Forecast in the test card: all test stars, stride 1.0. Class probe: 13 classes. Random
   forest: 16 classes. Probe during training: linear, 13 classes.
7. **The probe during training is smaller than its flag says.** `--probe-val 2000` draws from the
   validation-loss stars (`--val-objects 1000`, 931 after the length filter). It also only sees
   stars long enough for 8 windows.
8. **Every run trains with the catalogue period as a target** (`--spectral-aux 0.5`, train stars).
   This is not a leak between splits. But the period and phase numbers are not label-free, and
   `docs/PLAN.md` only states the rule for inference.
9. **Scaling uses the whole light curve.** The median and scatter of a star include the points that
   the forecast and gap tests hide. `--norm-days` exists; no run uses it.
10. **The forecast masks are not seeded.** `card_forecast.py` draws them with `torch.rand` and nothing
    seeds it. The gap and phase-gap numbers move a little from run to run. `--seed` only fixes the stars.
11. **The random-forest parameters have no known source.** `test_card.py` uses fixed values
    (`RF_PARAMS`) taken from the old job script. Nobody knows which data they were tuned on.
12. **The peak, phase and forecast cards read `--data`, not the checkpoint's data.** The default is
    ZTFxPC. The class and period probes use the data path stored in the checkpoint.

### Other

- **No test card files exist yet for the trained runs.** Their results are in the old folders of the
  main checkout (`project/results/attn_probe_<run>`, `project/results/jepa_forecast`, and others),
  written by the old scripts. Run `project/jobs/test_card.sh` on a run to get its
  `test_card/NAME.json`.
- **`enc_le1` is not trained yet.** The golden check covers its flags but not its checkpoint.
- **`enc_j1` cannot be rebuilt as it was** (see section 3).
- **`project/period_probe.py` (1,177 lines) and `project/fold.py` were not reworked.** They were only
  trimmed.
- **Do not merge this branch into `main` while a training chain is queued there.** A queued job runs
  the code of the checkout it was submitted from, and its old job script is gone on this branch.
- **The Python environment belongs to the main checkout.** Its editable install points there. From
  another checkout, `python -m ...` from that checkout's root imports the right code. A script run
  by its path does not.
- **`pytest` without a path runs only `tests/`** (`pyproject.toml`). Give both folders, as above.
- **Older notes in `docs/` use old file names** (`EXPERIMENTS.md`, `ROPE.md`, `SPECTRAL-LAYER.md`, the
  session notes, the design doc). `docs/CODE-MAP.md` has the table old name -> new name.
- **A checkpoint trained with a removed option cannot be loaded or resumed here.** The scripts say so
  and name the git tag `pre-cleanup`, which holds the full old tree.

## Credits

RoMAE: Zivanovic et al. 2025, [arXiv:2505.20535](https://arxiv.org/abs/2505.20535).
LeJEPA (SIGReg, views): Balestriero and LeCun 2025, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544).
The full list is at the end of `docs/LIBRARY.md`. MIT license.
