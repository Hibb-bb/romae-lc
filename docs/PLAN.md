# Plan and to-do list

**The current plan is the ICML plan below (agreed 2026-11-03).** The sections after it are the
older period / phase plan, kept as history. Every item carries a status: **done**, **alive**
(still the plan), **parked** (not wrong, not now), **dropped** (with the reason). Dates are absolute.

## ICML plan (agreed 2026-11-03)

### Goal

A light-curve world model trained with a JEPA-based framework, with as few auxiliary losses
as possible: good views should do the work. It must support physical-parameter inference
and forecasting. New ideas that are not in this plan wait until the current phase is done.

### Decisions

- Prediction losses at the **token level only**. No JEPA, invariance or SIGReg terms on the CLS.
  Evidence: J4 is J3b plus the two CLS terms and nothing else; macro F1 fell from 0.586 to
  0.533 (seed spread about 0.01).
- The catalogue period never enters the encoder at inference. Using it to build training
  pairs or masks is allowed.
- Chen labels and periods are not used for training or evaluation. Evaluation is on ZTFxPC.

### Test card (every run)

1. Classification: macro F1, 13 classes, attention probe (`project/eval/attn_probe.py`).
2. Period: log-period R2 and the spectral-peak hit rate.
3. Per-token phase R2 (`project/eval/token_phase_probe.py`).
4. Forecasting: share within 2 sigma, 0.5 to 10 days past the window
   (`project/eval/state_decode_probe.py`, figure `project/eval/decode_figs.py`).
5. Later: cross-survey retrieval and transfer; physical parameters.

### Where we are (2026-11-03)

| run | what | macro F1 (mean head) |
|---|---|---|
| maew_spec | masked autoencoder + spectral layer + aux | 0.598 |
| J3 | J1 (JEPA + reconstruction 0.5 + aux 0.5) + cross-window next-state add-on, prediction and SIGReg at weight 1 | 0.597 |
| J3b | J3 with lambda 0.05 | 0.586 |
| J4 | J3b + CLS invariance and SIGReg | 0.533 |
| hand-crafted features | (number from the user) | 0.578 |

Forecast decoding of J3 (100 stars, 30 held out, 8,773 forecasts, MLP decoder):
within 2 sigma, predicted state 0.46, no change 0.27, Fourier fit 0.67, the target's own
state 0.97 (the ceiling). The state holds the brightness; the prediction step is the weak part.

The add-on predictor (`project/dyn.py`, `DynHead`) is a fixed rotation: state = Linear(384 -> 16)
of the token output, 8 pairs rotated by 2 pi k f dt (f = the star's median spectral estimate),
times a damping exp(-g_k dt) from the source CLS. About 31k learned weights; the step itself
is not learned. It can only rotate and shrink the state.

### Phase 1: debug the predictor (first, about 2 weeks) — alive

1a. Diagnostics on the J3 checkpoint, no training (one evaluation job):
- D1: refit the decoder on predicted states. Off the set of real states, or lost information?
- D2: the catalogue frequency in place of the encoder's (evaluation only). How much error comes from f?
- D3: do real states rotate, s(t_j) = R(dt) s(t_i)? Do a star's states form a loop or a Gaussian cloud? By class.
- D4: forecast error by class and by amplitude (the failures looked like peaky stars).

1b. Short training runs from J1, 20k steps (about 3 h each). Code written, not yet checked on a GPU
(smoke job 3521844):
- J3t: targets from the target encoder on the full window, no gradient (`--dyn-target ema`).
- J3t + a likelihood decoder on the predicted state, told the band (`--dyn-decode`, to be upgraded
  from a fixed error to a learned spread and a band input).
- SIGReg off on the states, or SIGReg across stars instead of within a star.
- Rotation off (`--dyn-no-rot`), control.
- J1 + 20k steps with no add-on, control for the extra steps.

If D3 shows that real states do not rotate, change the predictor (a step learned from the CLS,
more harmonics, or a small transformer over state, dt and f) before adding losses.

### Phase 2: find what can be removed (about weeks 2 to 4) — alive

- Top-down from J1: drop one part at a time (contrast, mean penalty, damping, reconstruction,
  aux loss). Includes LeNEPA with no contrast (never tried).
- J1's starting weights were trained with reconstruction and the aux loss, so the final minimal
  recipe needs one run from scratch (about a day) to confirm it.
- The bottom-up floor is the plain token JEPA (`enc_jepa`, 0.63 within 10 % on period).

### Phase 3: cross-survey views (about weeks 4 to 7) — alive

- Data: ZTFxPC, ATLASxPC, ASASSNxPC under /projects/bfrf/data/PC_matches. Each survey's own split.
  Checked 2026-11-03: the splits agree by Gaia DR3 id (no star in train in one survey and in
  validation or test in another).
  ZTF train/validation/test 43.5k/5.5k/5.5k; also in ATLAS 40.6k/5.1k/5.1k; also in ASAS-SN
  15.2k/1.9k/1.8k; in all three (`ZTF-ATLAS-ASASSN-isect`) 15.0k/1.9k/1.8k.
- Views: for a source token, the targets are the star's other windows in the same survey (the
  star can be its own target) plus its windows in the other surveys when matched. Stars with
  no match get the normal views.
- Across surveys the predictor advances by phase, dphi = (t_j - t_i) / P mod 1, with the
  catalogue P (training pairs only). Within a survey it keeps the encoder's own f.
- Preprocessing: fold each matched star's two surveys together at P and keep it as a
  cross-survey pair only if the joint fold is clean (drops stars whose phase does not hold
  over years: long-period variables, spotted stars, period changes). Same-phase positives
  (0.02 cycles) absorb small phase errors. The predictor is told the target's band.
- Loader work: band wavelengths for ATLAS c/o and ASAS-SN V/g, error statistics per survey,
  one time system, pairing by Gaia id.
- ZTF + ATLAS first (40k pairs), then ASAS-SN.
- Tests: retrieval (is a star's ZTF latent closest to its own ATLAS latent?), probe transfer
  (fit on ZTF, score on ATLAS), forecast z-scores per survey.

### Phase 4: physical parameters — choose the targets early

- On disk: Gaia parallax, proper motion and radial velocity (in the data); Gaia-specxPC
  (probably temperature and metallicity, not yet inspected).
- A clean physical test: distances of RR Lyrae and Cepheids through the period-luminosity relation.

### Queue decisions still open (2026-11-03)

- The interactive queue holds J7, J3c, J4c and J4Hami (about 30 h). J4c and J4Hami carry the
  CLS terms the plan drops; proposed: remove them, keep J7 and J3c.
- Submit Phase 1a and 1b once the smoke job passes. Done 11-04: job 3528784 (`project/jobs/phase1.sh`, one
  4-GPU node, four lanes): D1-D4 on J3, J3b, J4; runs enc_p1_j3t, enc_p1_j3tdec (band + learned spread),
  enc_p1_sig0, enc_p1_sigbatch (SIGReg across stars, no mean penalty), enc_p1_norot, enc_p1_noadd; each then
  eval_encoder.sh and the decoding probe. J4c and J4Hami parked in queue.txt (commented out).
- Added 11-04, job 3529039 (`project/jobs/phase1b.sh`, second 4-GPU node): enc_p1_nce0 (J3, no contrast),
  enc_p1_mask75 (J3, 70-80 % hidden), enc_p1_fc (masked JEPA in forecast mode, `--forecast-tail 5 30`: the last
  5-30 days of each window hidden, no add-on), enc_p1_fc75 (both). Forecast test of the masked predictor:
  `project/eval/jepa_forecast_probe.py`. On J3 (30 stars, CPU check) its predictor cannot forecast past the
  window: within 2 sigma, recon head 0.10, decoded predicted latent 0.20, window mean 0.12, Fourier 0.54,
  target latent 0.72 (ceiling); latent error / variance 0.99. Expected: J3's predictor only ever interpolated.

### Phase 1 results (11-06)

Warm-start bug: `--jepa-init` copied NOTHING from a JEPA checkpoint (keys carry "context."), so J3, J3b, J4 and
every Phase 1 run trained 20k steps from random weights; J1 itself (50k) also had no init. Fair among
themselves, but "J1 + 20k" was never what ran. Fixed 11-06 (`TokenJEPA.init_from_mae` loads JEPA checkpoints;
copying nothing is now an error).

Add-on forecast (state_decode_probe, 600 stars, 55,753 pairs, within 2 sigma; Fourier fit 0.61):

| run | forecast | class F1 (mean head) | phase R2 of states |
|---|---|---|---|
| J3 | 0.43 | 0.597 | 0.29 |
| J3t (EMA targets) | 0.40 | 0.573 | 0.18 |
| J3t + decoder (band, learned spread) | 0.44 (own decoder 0.45) | 0.560 | 0.24 |
| SIGReg off | 0.45 | 0.570 | 0.43 |
| SIGReg across stars, no mean penalty | 0.39 | (pending) | 0.24 |
| rotation off | 0.24 (= no change) | 0.575 | 0.05 |
| no contrast | 0.40 | 0.605 | 0.26 |
| 70-80 % masking | 0.36 | 0.584 | 0.30 |
| no add-on | - | 0.587 | - |

F1 spread about +-0.01. Diagnostics on J3: D2 the encoder's f is within 2 % of the catalogue for 88 % of stars and
the catalogue f changes nothing (f is not the problem); D1 refitting the decoder on predicted states is worse
(0.36: information lost, not just off the real states); D3 a 4-harmonic function of phase explains only 29 % of
a star's state variance (no loop to rotate); D4 failures grow with amplitude (top third 0.31 vs Fourier 0.49).

Masked predictor in forecast mode (`--forecast-tail 5 30`, jepa_forecast_probe): latent error / variance drops
from ~1.0 to 0.44, but decoded to magnitude 0.25 within 2 sigma, the same as the window mean (0.22): it
predicts a flat curve (spread 0.21 vs 1.64). The reconstruction head is weak everywhere (in-window error 2.1 vs
variance 2.7).

Decision (11-06): D3 says the states do not rotate, so change the predictor. Option 2 first (the masked
predictor forecasts, no add-on): job 3543932 (`project/jobs/option2.sh`), all from J1 for real: enc_o2_noadd
(control), enc_o2_fc (tail 10-60 d), enc_o2_fcbig (+ predictor 4 x 384), enc_o2_fcbig_ph (+ phase rope at the
encoder's own f). Option 1 (a learned step in the add-on) is the fallback.

### Option 2 results (11-12)

Ran as four single-GPU jobs (3563719-22). Masked-predictor forecast, last 10 days hidden, 0.5-10 d past the last
visible token, 600 stars (jepa_forecast_probe; within 2 sigma; window mean 0.22, Fourier at the window's f 0.55,
Fourier at the catalogue f 0.65):

| run | recon head | decoded predicted latent | ceiling | class F1 | period within 10 % | token phase cos / sin |
|---|---|---|---|---|---|---|
| J1 | 0.22 | 0.24 | 0.86 | - | - | - |
| enc_o2_noadd (J1 + 20k) | 0.22 | 0.25 | 0.86 | 0.595 | 0.717 | 0.14 / 0.02 |
| enc_o2_fc (tail 10-60 d) | 0.22 | 0.23 | 0.90 | 0.580 | 0.693 | 0.10 / 0.01 |
| enc_o2_fcbig (+ predictor 4 x 384) | 0.22 | 0.23 | 0.86 | 0.574 | 0.692 | 0.11 / 0.01 |
| **enc_o2_fcbig_ph (+ phase rope)** | **0.67** | **0.63** | 0.93 | 0.562 | 0.674 | 0.12 / 0.04 |

- The phase rope is the fix: the predictor forecasts, beats Fourier at the encoder's own f (0.55) and at the
  catalogue f (0.65). It holds at every gap (0.70 / 0.66 / 0.66 for 0.5-2 / 2-5 / 5-10 d), every density
  (0.56 on the sparsest fifth, 0.72 on the densest; Fourier 0.47 to 0.59) and every class but DSCT (0.36).
- No leak: f comes from the context encoder's spectral layer, which sees only the visible tokens (the target
  encoder is a separate copy and runs first); the rope adds f * t, no magnitude.
- The tokens still carry almost no phase (sin R2 0.04): the phase comes in through the rope, not the latents.
- Cost: class F1 0.562 vs 0.595 for the control, period 0.674 vs 0.717. To check whether this is the
  bigger predictor or the rope.

Missing points inside the window (gap mode, `--gap-share`, job 3579825, J1 / p1_noadd / p1_fc / o2_noadd): without
the phase rope every run scores about 0.5 within 2 sigma when a visible token lies within 0.5 d (same night) and
the window mean (0.22-0.26) beyond that, against Fourier 0.55-0.66: the predictor copies the nearest neighbour.
Same at 10, 30 and 50 % hidden. Gap mode on the option 2 runs: folded into job 3586877.

### Option 3 (11-12): close Phase 1, open Phase 2 with two views

Four single-GPU jobs (`project/jobs/option3_one.sh LANE`, 3586857-60), all from J1, 20k steps, small predictor
(2 x 192), phase rope at the encoder's own f, no add-on:
- Phase 1 close-out (is the class / period cost of enc_o2_fcbig_ph the bigger predictor or the rope?):
  enc_o3_fcph (tail 10-60 d + rope), enc_o3_ph (rope only).
- Phase 2 views (added 11-12 at a coworker's suggestion; views, not losses, as the goal asks):
  enc_o3_fcph80 (80 % hidden), enc_o3_fcph_pband (a phase band of 0.1-0.3 cycles hidden at the catalogue
  period, then random points to 50 %: `--mask-mode phaseband`, period at training only). Folding at the
  TRUE period matters: at a random period the band is nearly random points for the short-period stars.
  On the 800k the band would use our own period table (712,278 stars), not Chen's.
- New test (card item 4b): phase gap, `jepa_forecast_probe --phase-gap 0.2`: hide 0.2 cycles at the catalogue
  period (evaluation only) in every cycle of the window; no same-phase point is left to copy. Reference runs
  (J1, o2_noadd, o2_fc, o2_fcbig_ph): job 3586877.

# Older plan: period and phase (2026-10-07 to 2026-10-28)

### Where we are

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

### The roadmap (agreed 2026-10-07, revised 2026-10-09)

| step | what | status |
|---|---|---|
| 0 | a period for every star of every split (`project/results/period_table_spec/periods.csv`, 53,307 stars: sharpened catalogue, model, model + search, candidates, verdict) | done 2026-10-08 |
| 1 | teacher-forced phase learning: smooth whole-record templates on the table's periods as targets, with query tokens at any time | alive, reshaped (items A to E below); first attempt with day offsets failed, see above |
| 2 | a bottleneck so the pooled latent must carry period and phase | parked until step 1 works; the stage-1 bottleneck of 2026-09-28 never learned a period |
| 3 | a student that infers its own period, distilled on the teacher's outputs | alive, after step 1 |
| 4 | the world model conditioned on the time gap (copy the right phase point forward) | alive, after step 3 |
| 5 | scale to the 800k-star Chen set: our own period table (done, 712,278 stars), then pretraining with the auxiliary loss on our own periods | started 2026-10-19: cache job + 9 links of the base recipe (`chen_spec`) on the chain; the rope/cycle fine-tune follows from that checkpoint |

### Items added 2026-10-09 (step 1, the revised recipe)

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
- **C2. The phase rope (2026-10-12).** cycph came out identical to cyc: its phasor map was pulled to zero by weight
  decay before the decoder found a use for it. Two fixes: the map now has no weight decay and a settable start
  (`--phasor-init`), and the stronger mechanism `--phase-rope H` puts the phase in cycles at the encoder's own
  frequency on a rotary axis of the decoder, so same-phase tokens line up in attention with nothing to learn.
  Both are being compared on the sine test (`sine_test10k_rope`) before a real-data run.
- **C3. The estimator (2026-10-16).** cycrope on real stars: the rope works exactly as on sines wherever the
  frequency it gets is within 0.05 % (0.97 to 20 cycles, 0.88 at 20 to 100), and the soft-peak estimate reached
  that on only 3 % of windows. The local centroid around the argmax reaches it on 66 % (median error 0.011 %).
  `--phase-freq local`; cycrope2 queued. The remaining ~30 % of windows pick an alias or harmonic as the top
  peak: the next estimator step is a fold check among the top peaks inside the model.
- **D. The experiment.** 250-day setting (where the encoder works), warm start + cycle
  queries, with and without the phasor features; the period probe at every eval shows whether
  the period is kept and sharpened; the explained variance of the template against the
  distance in cycles (the replay of 2026-10-09) is the learning curve to watch.
  Status: cyc done (reach ~20 cycles on real stars, probe 0.808/0.733); cycph done but inert (see C2); next real-data run = the winner of the sine comparison. Originally submitted 2026-10-09 evening on ghx4 as `cyc` (job 3351435, probe 3351436) and `cycph` (job 3351437, probe 3351438): random masks, cap 5 -> 300 cycles, 50k steps; results in `project/results/period_enc_cyc[ph]`, training logs in `project/runs/enc_cyc[ph]/log.jsonl` (`query_cap_cycles` per eval).
- **E. Fall back to the observations where the template is poor.** Rotators and long-period
  variables are only roughly periodic; a fixed wave is a bad target for them (the model
  explains only 0.2 of their targets even within half a cycle). Use the template's R2 as the
  gate (`--target mix --template-min-r2`), not the model's error. Status: alive, to decide.
- **H. The sine test (2026-10-11).** `project/sine_test.py` + `jobs/sine_test.sh`: clean sinusoids (random or
  regular sampling, log-uniform periods 0.05 to 50 d), the same encoder, block masking and decoder at a small
  size; variants plain / phasor at the model's own frequency / oracle phasor at the true frequency, plus regular
  sampling, a second harmonic, noise, and no label in the spectral auxiliary loss. Answers two questions:
  how far in cycles the decoder can predict, and the shortest period the model recovers (with the peak width
  in grid steps next to it). Status: first pass done 2026-10-11 (plain = copier, 0 beyond 2 cycles with the frequency found; phasor at the model's
  own frequency = 0.86 at 30 to 100 cycles and 0.64 at 100 to 300, every period from 0.12 d; the limit at both ends is the
  grid step; the transition to using the feature was late or missing in 4 of 5 phasor runs with a zero-start map ->
  map now starts small, 10k-step rerun queued as `sine_test10k`).
- **I. Short periods and the grid.** Model alone within 10 %: 0.82 to 0.87 for 0.2 to 0.5 d, 0.11 to 0.28
  below 0.2 d (46 stars; the misses are not aliases). Peak width below one grid step under 0.125 d, and only
  200 delta Scuti in training. Fix: `--spectral-rel 2e-4` (twice the cost) and the 800k set. Status: parked.
- **F. Artifact to-do.** Push the search budget below 772 trials per star (narrower windows,
  coarser grid) to find where model + fine search breaks. Status: alive, not started.
- **G. Whole-curve windows.** The 2,000-day window loses the period even with observed targets
  (0.57 within 10 %). Hypothesis, unverified: the spectral grid step (0.05 %) was sized for
  250-day windows and undersamples the narrower peaks of a 2,000-day baseline; the test is
  `--spectral-rel 1e-4` (five times the layer's cost). Status: parked; 250-day windows for now.

### Runs in the queue (ghx4 batch, all pending on priority as of 2026-10-09 evening)

| job | what | status |
|---|---|---|
| maespec50 | 250-day wide spectral MAE at 50k with the block-plus mask, the control for the JEPA runs | alive |
| jepainit, jepahyb | token JEPA warm-started from maew_spec; JEPA plus a reconstruction head | alive (the JEPA comparison the PIs asked for) |
| gruode | the collaborators' GRU-ODE encoder (RNNODEEncoderV2) in our pipeline, same objective and probes | alive; moved to the head of the interactive chain 2026-10-11 (user); its throughput benchmark (`project/eval/runtime.py --ckpt project/runs/enc_gruode/mae.pt --batch-sweep`) comes after the probe |
| cyc, cycph | the cycle curriculum pair (item D) | alive, submitted 2026-10-09 |
| jepaclean, jepacleanshort, longspecwtpl2 | template runs with queries in days | cancelled 2026-10-09 (superseded by A to C) |

### Other open items

- Delta CPU job for the StarEmbed hand-crafted baseline (`project/jobs/delta_cpu_starembed.sh`,
  now installs its own x86 environment): submit from the Delta login node. Alive.
- `git push origin main` from the user's shell (the job shell has no GitHub credential). Alive.
- Rerun the phase-decoder ladder with the model's periods instead of the catalogue. Parked.
- Reconstruction demo on finished whole-curve runs; runtime, quantisation and sweep tables on
  the artifact. Optional.
- Classification: the random-forest benchmark is done (13 classes, 5 seeds: latent macro F1
  0.577 against our hand features 0.416); the StarEmbed baseline waits on the Delta job. Alive.

### Dropped on 2026-10-09, and why

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

### Clean-up of the repository (started 2026-10-19)

Order agreed with the user: record first, remove later; no code moves while
the 800k pretraining chain runs from this working tree.

1. **done 2026-10-19** — the record: `docs/EXPERIMENTS.md` (one row per run
   and result folder with recipe, headline number and verdict) and
   `docs/CODE-MAP.md` (one line per file with a proposed status); git tag
   `pre-cleanup-2026-10-19`.
2. open — secure the paper's evidence: copy the checkpoints and result
   folders listed in the ledger's section 5 to one archive under `/work`
   with a manifest; list the rest for the user before anything is removed.
3. open — move the world-model line and the rejected ideas to
   `project/archive/` (code, jobs, tests), in a separate checkout, merged
   between chain links; delete after a month if nothing missed them.
4. open — make it usable by others: one entry README with the commands that
   reproduce the main table, data and output roots behind one config, CPU
   tests in GitHub Actions.
5. open — split `project/common.py` last.

### JEPA variants with the aux loss (proposed 2026-10-26)

Facts behind the plan. The aux loss (cross-entropy of the spectral layer's
logits on the catalogue period's bin, training only) is the biggest win in
the ledger: 0.852 / 0.838 / 0.828 / 0.746 within 10 / 1 / 0.1 / 0.01 % on all
validation stars, against 0.72 / 0.19 (pooled) without it. The JEPA runs never
had it. No encoder holds the phase per token (token-phase probe 10-26: the
context part of the phase is at 0.07 R2 at best). Dennis's synthetic test bed
(lcjepa.py) shows what puts the phase into a state: a predictor told the time
gap and the clock speed, targets that change within a star, and a same-star
contrast; plain LeJEPA and masked latent prediction with SIGReg did not.

Every variant keeps the wide encoder, the spectral layer, the aux loss at 0.5
and 250 d windows, and is scored the same way: the spectral-peak hit rate on
all validation stars (period), the per-token phase probe's sine column (the
context part of the phase on real stars), and the predictor's own hit rate.

| id | name | recipe | code | asks |
|---|---|---|---|---|
| J1 | JEPA + aux | token JEPA (EMA target), reconstruction 0.5, aux 0.5, 50k, batch 64 (the `enc_jepahyb` recipe plus the aux) | none | does the aux give JEPA the autoencoder's precision? |
| J2 | NEPA on the autoencoder | `maew_spec` warm start + the dynamic-state add-on (`--dyn 8`: rotation predictor at the spectral clock, same-window contrast, SIGReg in time), 20k; frozen-encoder control first | done, queued (`enc_dynfz`, `enc_dyn`) | can the contrast put the phase into the tokens? |
| J3 | NEPA on JEPA | J1 + `--dyn 8`, warm-started from J1, 20k | none | same, on a latent-prediction encoder |
| J4 | LeNEPA | J3 + LeJEPA's static part: the CLS pulled together across a star's 8 windows and SIGReg on the CLS across the batch | about 20 lines (`--cls-inv`, `--cls-sig`) | does the static code get cleaner without losing the phase? |
| J5 | SIGReg in place of the EMA | masked latent prediction regularised by SIGReg (the test bed's `mjepa`) | some | control only; it was the worst variant in the test bed, run only if J1 fails |
| J6 | cross-window pairs | pairs across a star's windows (`--dyn-cross`); now part of J3 and J4, see below | window starts in the batch | done 10-26 |

Order: J2 (in the queue), then J1, J3, J4. Each is 2 to 3 links of 2 h;
about ten links in all.

Update 10-26 (after J2). The unfrozen add-on on the MAE (`enc_dyn`) reached a predictor hit
rate of 0.32 but the per-token phase probe hardly moved (sin R2 0.07 to 0.10). Inside one
window a token knows its time since the window start, so a state that is a clock at the
spectral frequency passes the within-window test without any light-curve content. J3 and J4
therefore use `--dyn-cross`: the source token is in one window, the target in ANOTHER window
of the same star, and the negatives are the star's tokens outside the source window. Gaps stay
short (0.5 to 30 days): over a gap the predicted phase drifts by f dt times the relative
frequency error, which is a full cycle over 300 days for a 0.3-day star at 0.1 %. Overlapping
windows (`--advance 0.5 1.0`) make such pairs common, and an observation that sits in two
windows gives a dt = 0 pair whose two states must agree. Check on hand-made states: true phase
states hit the target 97 % of the time, a window clock 0 %.
