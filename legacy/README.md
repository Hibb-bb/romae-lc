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

More code moves here in later clean-up steps (the modules that the training script still
imports today, such as the bottleneck, GRU-ODE, templates and the old `romae_lc/lewm.py`).
