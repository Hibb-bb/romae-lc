Completeness critic: findings. I changed nothing. All line numbers refer to /projects/bfrf/hibb/romae-lc-cleanup unless another path is named.

## Spot-checks (9 claims)
1. Correct (flags, common, tests). jepa.py:370 is `from project.dyn import sigreg`, inside forward_views. sigreg is at dyn.py:43 and uses only torch.
2. Correct (recipes). `parser.parse_args(argv)` is at pretrain_mae.py:437, the resume setattr loop is at :443-447, `RUN_CONTROL` is at :96, and `dump_json(vars(args), out/"args.json")` is at :475.
3. Correct (flags). The `--phase-freq` default is "soft" (pretrain_mae.py:299). Small error: the TokenJEPA default "local" is at jepa.py:67, not :70.
4. Partly wrong (flags). The flags audit says `TokenJEPA.from_checkpoint` is at jepa.py:155. It is at jepa.py:169, as the common audit says. It does `cls(**ckpt["backbone"], **ckpt["mae"], **ckpt["jepa"])`.
5. Correct (splits, evalsurface). `sys.path.insert(0, "/projects/bfrf/hibb/romae-lc")` is at jepa_forecast_probe.py:32, jepa_forecast_demo.py:15, spectral_peak_probe.py:5, token_phase_probe.py:31 and state_decode_probe.py:39.
6. Correct (tests). romae_lc/__init__.py imports lejepa (:54) and lewm (:55). tests/test_spectral.py imports `project.period_probe.gls_power`. period_probe.py:68 has `from project import fold` at the top. phase_probe.py:35 imports `project.inject.band_templates`.
7. Correct (splits). jepa_forecast_probe.py:69, :74, :84 and :89 call `torch.rand`, and there is no `manual_seed`.
8. Correct (evalsurface, item 5). lejepa.sh runs pretrain_mae.sh inline with MAX_LINKS=2. Link 2 is queued as plain pretrain_mae.sh (pretrain_mae.sh:65-70), so it never runs eval_encoder.sh or test_forecast.sh.
9. Correct (evalsurface). No job trains enc_j1. phase1.sh:23, option2*.sh and option3_one.sh:27 only use it as `--jepa-init project/runs/enc_j1/mae.pt`.

## (1) Claims that are wrong or overstated
- The flags audit warns that TokenJEPA and RoMAEForPreTraining must keep accepting `clean_target`, `abs_timescales` and `attn_pool`. This is correct, and it is the only hard limit on checkpoint loading (see section 3). The jepa.py:155 reference is wrong (see check 4).
- The tests audit says pytest only runs tests/. That is true for a bare `pytest`. But project/tests has an `__init__.py` and tests/ does not, so the rootdir rule matters (see "Venv" below).
- The recipes audit says "the evaluation scripts need no change". This is true for reading args. But the job scripts that start them still run from the main checkout (see the next section).

## (2) Important things no audit covered
- **Every sbatch script has `#SBATCH --chdir=/projects/bfrf/hibb/romae-lc` and `source .venv/bin/activate`.** Examples: pretrain_mae.sh:11 and lejepa.sh:11. If you sbatch from the cleanup branch, Slurm runs the MAIN checkout's code. The splits audit only flagged the `sys.path` lines in the eval scripts. Change `--chdir`, or drop it and submit from the repo root. Do this before any test run from the branch.
- **Venv editable install.** `.venv/lib/python3.12/site-packages/_editable_impl_romae_lc.pth` adds `/projects/bfrf/hibb/romae-lc` (the main root) to `sys.path`. The `.venv` is in the main checkout.
  - With `python -m`, the current folder comes first, so the branch code wins. eval_encoder.sh and test_forecast.sh use `python -m`, so they are fine.
  - Any `python path/to/script.py`, or a run from another folder, silently imports the main checkout's `project` and `romae_lc`. A module you moved to legacy/ would then still seem to import, so the bug stays hidden.
  - Guard: in the smoke test, check `project.__file__` and `romae_lc.__file__`.
- **Pickled classes in checkpoints: none.** I scanned the pickle opcodes of every mae.pt, last.pt, latents.pt and latents_test.pt in the 9 finished ICML runs (and older runs). The only globals are torch storages, `torch._utils._rebuild_tensor_v2` and `collections.OrderedDict`.
  - jepa_state (jepa.py:446) and mae_state (common.py:1010) save plain dicts: `spec.to_dict()`, `asdict(cfg)`, `ladder.to_dict()` and `asdict(model.cfg)`.
  - So moving modules does not break `torch.load(..., weights_only=False)` (common.py:1042).
  - What breaks loading is the rebuild code: `TokenSpec.from_dict`, `FrameConfig(**...)`, `Ladder.from_dict` (common.py:1054-1063) and the constructor kwargs listed in section 3.
- **wandb.** `save()` stores `wandb_id=tracker.id` (pretrain_mae.py:938). On resume, `Tracker(run_id=ckpt.get("wandb_id"))` (:929) calls `wandb.init(id=..., resume="allow", dir=out)` (tracking.py:88-97). The `wandb_*` names are in RUN_CONTROL (:114-119).
  - Keep these keys and the `Tracker` signature, or resumed runs open new wandb runs.
  - Each run folder also has a `wandb/` folder. It is ignored by the `runs/` rule.
- **The job chain.**
  - Removing the THEN/PIPELINE/WM_OUT code in pretrain_mae.sh is fine.
  - Keep the `case "${1:-}" in *.pt) shift` line (:40) only if a legacy stage still chains into it. Otherwise drop it too.
  - The self-chain needs these exact env names: `OUT`, `LINK`, `MAX_LINKS`, `BUDGET` and `SLURM_JOB_NAME`. lejepa.sh, option2*.sh, option3_one.sh and phase1*.sh pass them inline.
- **enc_le1 has not trained yet.** It has args.json, ladder.json and log.jsonl, but no last.pt. If its chain runs while the cleanup merges into main, link 2 would run the new code (because of `--chdir` main) and resume from link 1's last.pt. Do not merge into main while any chain is queued. Check `squeue` first.
- **Relative paths baked into args.** `jepa_init="project/runs/enc_j1/mae.pt"` and `out="project/runs/<run>"` are stored in `ckpt["args"]`, and on resume they beat the CLI. So `project/runs/` must keep its path relative to the repo root. Do not rename `project/` or move `runs/`.
- **`.gitignore` uses the unanchored patterns `runs/` (line 8) and `results/` (line 14).** Any `legacy/**/runs/` or `legacy/**/results/` folder will also be ignored. Check `git status` after the move.
- **enc_j1 cannot be rebuilt as it was.** It was trained before 9b1ccf3 and by code that no job script holds. A recipe made from its args.json reproduces the flags, not the code. Say this in the recipe file, and keep its mae.pt as a fixed artifact.
- **token_phase_probe reads the `aux.dyn` key but does not need dyn.py.** token_phase_probe.py:205 does `torch.load(ck)` to read `.get("aux", {}).get("dyn")`. All ICML checkpoints have `aux == {}`, so dropping the branch at :204-209 and the `dyn=` parameter of `embed` (:88, :99-100) is safe.

## (3) Riskiest moves and how to guard them
1. **dyn.py goes to legacy while sigreg is still live** (jepa.py:370). If you miss it, enc_le1 training fails at the first step. Move `sigreg` into jepa.py first, then move dyn.py. test_jepa_no_ema_views covers this.
2. **Changing constructor kwargs breaks loading of all 9 trained ICML checkpoints.** `TokenJEPA.__init__` and `RoMAEForPreTraining` must keep every key that `model.hparams` (romae_lc/model.py:215-223, 806-812) and `jepa_hparams` (jepa.py:104-106) wrote: `abs_timescales`, `use_cls`, `rope`, `n_axes`, `n_channels`, `spectral`, `attn_pool`, `clean_target`, `ema`, `ema_end`, `loss`, `use_target`, `recon_weight`, `proj_dim`, `proj_hidden`, `phase_rope` and `phase_freq`. state_dict names must also stay the same, because `load_state_dict` is strict.
   - Guard: a golden test, before and after the move. Run `load_encoder` on each of the 9 mae.pt files, encode one fixed batch, and check `torch.allclose` against features saved before the move.
   - Also check `cache_latents` on one run and compare it to the existing latents.pt.
3. **Trimming `romae_lc/__init__.py`.** Every `romae_lc` import runs it. Trim it in the same commit as the lewm/lejepa move. Keep `analysis` (suggest_time_encoding, timescales_for, time_shuffle_score) and `frames` (FrameConfig, frame_grid).
4. **Splitting common.py.** `load_mae` must keep its jepa branch, and `_meta_from` must keep the TokenSpec, FrameConfig and Ladder dict formats. Keep `project/common.py` as a re-export shim for one commit.
5. **Hidden imports to fix before the move.**
   - pretrain_mae.py:55-56 imports `project.bottleneck` and `project.gruode` at the top.
   - period_probe.py:68 imports `fold` at the top.
   - phase_probe.py:35 imports `inject` at the top.
   - jepa_forecast_probe.py:35 imports `forecast_tail_mask` from `pretrain_mae`.
   - test_jepa.py and test_classify_rf.py import `MAE_ARGS` from test_ls_benchmark.py.
   - The legacy-only `sys.path` hack in gruode.py:53 (`TFM_ROOT`) moves with gruode.py.
   - Guard: after the move, from the branch root, run `python -c "import project.pretrain_mae, project.cache_latents, project.period_probe, project.eval.attn_probe, project.eval.classify_rf"`. Then run a 2-step `pretrain_mae` with the enc_o3_fcph_pband flags and with the enc_le1 flags.
   - The four eval scripts that parse args at import (spectral_peak_probe, token_phase_probe, jepa_forecast_probe, jepa_forecast_demo) cannot be imported. Check them with `--help`, or a tiny run, after removing their `sys.path` lines.
6. **Checking a moved name.** Use `git grep -n "<name>"`, then `rg "import_module|__import__|getattr\(.*project"`. I found no importlib or string imports in the core files, but rerun this after the move.