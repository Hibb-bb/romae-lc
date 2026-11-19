# legacy/

Old code that the current pipeline does not use. It was moved here in the clean-up
(branch `cleanup`, see `docs/CLEANUP-PLAN.md`).

- The folders mirror the old paths: `legacy/project/anomaly.py` was `project/anomaly.py`,
  `legacy/tests/test_lewm.py` was `tests/test_lewm.py`, and so on.
- The code is frozen. Nobody maintains it and no test runs it.
- Nothing outside `legacy/` imports it.
- It is not expected to run from here. Its imports and job scripts still use the old
  paths (`project.train_wm`, `project/jobs/gate.sh`, ...).

To run any of it, check out the tag `pre-cleanup`. That tag holds the full old tree:

    git worktree add ../romae-lc-pre-cleanup pre-cleanup

## What is here

- `examples/`: the first example scripts (`train_mae.py`, `train_lewm.py`, `train_lejepa.py`,
  the time-encoding analyses) and their two job scripts.
- `project/`: the old world-model line. Stage 2 world model (`train_wm.py`), decoder
  training (`train_decoder.py`), the gate, the residual model, the energy and inference code,
  the anomaly and forecast tests, baselines, the period head, the token head, the phase
  pretraining, the sine test and the speed benchmark.
- `project/eval/`: old evaluation pages and figures. Lomb-Scargle all-stars run, budget
  sweep, runtime, period table, period sensitivity, period page and figures, slide figures,
  method page and diagrams, old demos, the add-on (DynHead) forecast probe and its figures,
  dataset figures, StarEmbed hand features.
- `project/jobs/`: the Slurm scripts of all of the above, and of old ablations.
- `project/tests/` and `tests/`: the tests of the legacy code. `test_project.py`,
  `test_predictor.py` and `test_encoder_loading.py` are the full old files; the tests in them
  that cover kept code were copied to `project/tests/test_common.py`,
  `project/tests/test_cache_latents.py` and `project/tests/test_encoder_loading.py`.

## Moved in step 2 (dead options cut from the kept scripts)

The kept scripts lost the options that no ICML run used. The modules behind those options
are here now:

- `project/bottleneck.py`, `project/gruode.py`, `project/decoder.py`, `project/flow.py`: the
  bottleneck autoencoder, the GRU-ODE encoder, the query decoder and the flow predictor.
- `project/dyn.py`: the dynamic-state add-on. Its `sigreg` function is still used; it now
  lives in `project/jepa.py`.
- `project/templates.py`, `project/phase_decoder.py`: smooth-fit targets, query tokens and
  the phase / fold losses.
- `project/train_predictor.py`, `project/lookback.py`, `project/inject.py`,
  `project/diagnostics.py`: the predictor line. `shuffle_score` is still used; it now lives
  in `project/probes.py`. The template helpers that the phase probes use now live in
  `project/phase_probe.py`.
- `project/chen_data.py`, `project/eval/period_chen.py`, `project/eval/ls_benchmark.py`,
  `project/eval/period_report.py`, `project/plot_period_examples.py`, `lomb_scargle.py`: the
  Chen 800k data path and the old period pages.
- `romae_lc/lewm.py`, `romae_lc/lejepa.py`: the old world model and LeJEPA classes.
- `project/tests/test_jepa.py`, `test_mae_data.py`, `test_cycles.py`, `test_common.py`,
  `test_cache_latents.py`: the full old test files. The tests of removed options (templates,
  query tokens, bottleneck, window ranges, the `mix` mask, error weights, noise realisations
  of the cache, the world-model loaders) are only here; the other tests stay in
  `project/tests/`.

Code that was cut from inside a kept file (for example the dead branches of
`project/pretrain_mae.py`, now `project/train.py`, and the world-model loaders of
`project/common.py`, now `project/checkpoint.py`) has no copy
here. The tag `pre-cleanup` has it.

A checkpoint trained with one of the removed options cannot be resumed or loaded from this
branch. The scripts say so and name the tag.

## Renamed in step 3 (new module layout)

Nothing moved to `legacy/` in this step. Kept files were renamed and split:

| old | now |
|---|---|
| `project/pretrain_mae.py` | `project/train.py` |
| `project/common.py` | `project/data.py`, `windows.py`, `backbone.py`, `checkpoint.py`, `probes.py`, `utils.py` |
| `project/mae_data.py` | `project/masks.py` |
| the view helpers of `project/jepa.py` | `project/views.py` |

The flags of the runs are now recipe files in `project/recipes/`. `docs/CODE-MAP.md` has the full
table of old and new names.

## Moved in step 4 (evaluation and job scripts)

The evaluation scripts were renamed and split into functions. The old files have no copy here
(the tag `pre-cleanup` has them); this table says where each one went:

| old | now |
|---|---|
| `project/cache_latents.py` | `project/eval/encode.py` |
| `project/eval/attn_probe.py` | `project/eval/card_class.py` (class) and `card_period.py` (log period) |
| `project/eval/spectral_peak_probe.py` | `project/eval/card_period.py` (`--chen-csv` is gone) |
| `project/eval/token_phase_probe.py` | `project/eval/card_phase.py` |
| `project/eval/jepa_forecast_probe.py` | `project/eval/card_forecast.py` |
| `project/eval/jepa_forecast_demo.py` | `project/eval/card_demo.py` |
| `project/eval/dataset_figs_style.py` | `project/eval/style.py` |

`project/jobs/` keeps `train.sh` (a recipe, the self-requeue chain, then the test card),
`test_card.sh`, `demo.sh`, and the helpers `after_done.sh` and `in_session.sh`. These job scripts
moved to `legacy/project/jobs/`:

- `pretrain_mae.sh`: the old training chain. `train.sh` replaces it.
- `lejepa.sh`, `option2.sh`, `option2_one.sh`, `option3_one.sh`: the run lanes. Their flags are
  now the recipe files in `project/recipes/`.
- `eval_encoder.sh`, `test_forecast.sh`: `test_card.sh` replaces both.
- `rope_demo.sh`, `rope_demo_int.sh`: `demo.sh` replaces them.
- `attn_probe.sh`, `token_phase_probe.sh`, `cache_latents.sh`, `period_probe.sh`,
  `classify_rf.sh`: wrappers of single steps. Run the python module instead.
- `submit_queue.sh`: the queue helper for the interactive partition.

## Moved in step 5 (docs)

- `project/README.md`: the old guide of the world-model line, with its dated decisions. The root
  `README.md` replaces it as the entry page.
- The old root `README.md` described the library together with the LeJEPA and LeWorldModel classes.
  Its library part is now `docs/LIBRARY.md`. The tag `pre-cleanup` has the old page.
