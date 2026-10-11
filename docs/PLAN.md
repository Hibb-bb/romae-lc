# Plan and to-do list

The living plan of the period / phase work. Updated 2026-10-09. Every item carries a
status: **done**, **alive** (still the plan), **parked** (not wrong, not now),
**dropped** (superseded or rejected, with the reason). Dates are absolute.

## Where we are

- The 250-day wide encoder with the spectral layer is the best encoder (`project/runs/maew_spec`):
  period within 10 % 0.79, within 1 % 0.72 on 5,346 validation stars; with the fine search
  0.77 within 0.01 %; with candidates 0.80. Artifact: https://claude.ai/artifact/JoubE7CVCj2KSB1QtGzwub
- The frozen latent holds no phase. Placing a light curve in time needs the period to
  1e-4 and the phase, which the encoder does not store (2026-09-30 and 2026-10-01 results).
- The whole-curve template run of 2026-10-09 (`enc_longspecwtpl`) learned to copy the nearest
  seen point and nothing else: it explains 0.8 of the smooth target within 0.1 cycle of a seen
  point and 0.03 beyond 2 cycles. Its queries were drawn in days (up to 30 d), which is more
  than 5 cycles for 97 % of the training stars (73 % have periods under 0.5 d), so the loss
  gave no partial credit for a rough period. The targets themselves are sound (template
  against observations 0.15 mean squared error per window, template loss lower than the
  observed-target loss).

## The roadmap (agreed 2026-10-07, revised 2026-10-09)

| step | what | status |
|---|---|---|
| 0 | a period for every star of every split (`project/results/period_table_spec/periods.csv`, 53,307 stars: sharpened catalogue, model, model + search, candidates, verdict) | done 2026-10-08 |
| 1 | teacher-forced phase learning: smooth whole-record templates on the table's periods as targets, with query tokens at any time | alive, reshaped (items A to E below); first attempt with day offsets failed, see above |
| 2 | a bottleneck so the pooled latent must carry period and phase | parked until step 1 works; the stage-1 bottleneck of 2026-09-28 never learned a period |
| 3 | a student that infers its own period, distilled on the teacher's outputs | alive, after step 1 |
| 4 | the world model conditioned on the time gap (copy the right phase point forward) | alive, after step 3 |
| 5 | scale to the 800k-star Chen set (`/projects/bfrf/data/chen`): our own period table first (label-free pretraining with model periods); the Chen labels only as a weak check | alive, plan only, no coding yet |

## Items added 2026-10-09 (step 1, the revised recipe)

- **A. Queries in cycles, not days.** `--query-cycles LO HI`: the offset of a query from a seen
  point is drawn log-uniform in cycles of the star's table period, between 0.1 and a cap that
  ramps from LO (5) to HI (a few hundred) over training. A `--query-random-share` puts a share
  of the queries at random times in the window, so the query layout does not reveal the period
  (the shortcut to guard against: the largest offset in a window would otherwise be about cap
  times the period). The day-based `--query-delta` stays as a flag but is no longer the plan.
  Status: built 2026-10-09 (`--query-cycles`, `--query-cycles-min`, `--query-random-share`; `templates.sample_queries_cycles`).
- **B. Warm start for the masked autoencoder.** `--init project/runs/maew_spec/mae.pt`: start
  from the encoder that already knows the period, keep its spectral auxiliary loss (weight 0.5)
  so the period is not forgotten while the cap is low. Status: built 2026-10-09 (`--init`, `common.warm_start`).
- **C. Phasor features at the model's own period.** The spectral layer's soft peak gives a
  period estimate from the observations alone; every hidden and query token gets sine and
  cosine of its time at that frequency and a few harmonics, so "copy the right phase point" is
  a lookup rather than modular arithmetic the decoder has to invent. Label-free at inference.
  Status: built 2026-10-09 (`--phasor H`, `romae_lc.model.PhasorFeatures`, zero-initialised map; the checkpoint carries it).
- **D. The experiment.** 250-day setting (where the encoder works), warm start + cycle
  queries, with and without the phasor features; the period probe at every eval shows whether
  the period is kept and sharpened; the explained variance of the template against the
  distance in cycles (the replay of 2026-10-09) is the learning curve to watch.
  Status: submitted 2026-10-09 evening on ghx4 as `cyc` (job 3351435, probe 3351436) and `cycph` (job 3351437, probe 3351438): random masks, cap 5 -> 300 cycles, 50k steps; results in `project/results/period_enc_cyc[ph]`, training logs in `project/runs/enc_cyc[ph]/log.jsonl` (`query_cap_cycles` per eval).
- **E. Fall back to the observations where the template is poor.** Rotators and long-period
  variables are only roughly periodic; a fixed wave is a bad target for them (the model
  explains only 0.2 of their targets even within half a cycle). Use the template's R2 as the
  gate (`--target mix --template-min-r2`), not the model's error. Status: alive, to decide.
- **H. The sine test (2026-10-11).** `project/sine_test.py` + `jobs/sine_test.sh`: clean sinusoids (random or
  regular sampling, log-uniform periods 0.05 to 50 d), the same encoder, block masking and decoder at a small
  size; variants plain / phasor at the model's own frequency / oracle phasor at the true frequency, plus regular
  sampling, a second harmonic, noise, and no label in the spectral auxiliary loss. Answers two questions:
  how far in cycles the decoder can predict, and the shortest period the model recovers (with the peak width
  in grid steps next to it). Status: queued on the interactive chain right after the cyc probe; results in
  `project/results/sine_test/<variant>/`.
- **I. Short periods and the grid.** Model alone within 10 %: 0.82 to 0.87 for 0.2 to 0.5 d, 0.11 to 0.28
  below 0.2 d (46 stars; the misses are not aliases). Peak width below one grid step under 0.125 d, and only
  200 delta Scuti in training. Fix: `--spectral-rel 2e-4` (twice the cost) and the 800k set. Status: parked.
- **F. Artifact to-do.** Push the search budget below 772 trials per star (narrower windows,
  coarser grid) to find where model + fine search breaks. Status: alive, not started.
- **G. Whole-curve windows.** The 2,000-day window loses the period even with observed targets
  (0.57 within 10 %). Hypothesis, unverified: the spectral grid step (0.05 %) was sized for
  250-day windows and undersamples the narrower peaks of a 2,000-day baseline; the test is
  `--spectral-rel 1e-4` (five times the layer's cost). Status: parked; 250-day windows for now.

## Runs in the queue (ghx4 batch, all pending on priority as of 2026-10-09 evening)

| job | what | status |
|---|---|---|
| maespec50 | 250-day wide spectral MAE at 50k with the block-plus mask, the control for the JEPA runs | alive |
| jepainit, jepahyb | token JEPA warm-started from maew_spec; JEPA plus a reconstruction head | alive (the JEPA comparison the PIs asked for) |
| gruode | the collaborators' GRU-ODE encoder in our pipeline, same objective and probes | alive |
| cyc, cycph | the cycle curriculum pair (item D) | alive, submitted 2026-10-09 |
| jepaclean, jepacleanshort, longspecwtpl2 | template runs with queries in days | cancelled 2026-10-09 (superseded by A to C) |

## Other open items

- Delta CPU job for the StarEmbed hand-crafted baseline (`project/jobs/delta_cpu_starembed.sh`,
  now installs its own x86 environment): submit from the Delta login node. Alive.
- `git push origin main` from the user's shell (the job shell has no GitHub credential). Alive.
- Rerun the phase-decoder ladder with the model's periods instead of the catalogue. Parked.
- Reconstruction demo on finished whole-curve runs; runtime, quantisation and sweep tables on
  the artifact. Optional.
- Classification: the random-forest benchmark is done (13 classes, 5 seeds: latent macro F1
  0.577 against our hand features 0.416); the StarEmbed baseline waits on the Delta job. Alive.

## Dropped on 2026-10-09, and why

- Query offsets in days as the curriculum: 30 days is 100 cycles for an RR Lyrae and a tenth
  of a cycle for a long-period variable; cycles put every class on the same curriculum.
- Continuing training only on the samples with low explained variance: those are the
  far-in-cycles tokens of short-period stars, already most of the loss and flat for 50k steps;
  selecting them adds no signal and also selects the unlearnable targets.
- Per-window template fits (degenerate fits, loss spikes): replaced by whole-record fits
  (2026-10-09, done).
- The observed-target twin of the whole-curve run: cancelled 2026-10-08.
- Token JEPA alone as the way to period precision: run 1 reached 0.63 within 10 % and 0.13
  within 1 %; the objective is satisfied by a rough period. Kept only as a comparison.
