#!/bin/bash
#SBATCH --job-name=lejepa
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Phase 2 (11-16, the user's run after talking with Randall Balestriero): LeJEPA at the TOKEN level, from scratch.
#   no EMA, no reconstruction, no contrast; SIGReg (lambda 0.05, 1,024 slices) on the 16-number projection of the
#   token latents of every view; prediction in the same projection: from each of 2 global + 4 local views to the
#   mean of the global views' latents at every observation they hold. Every view holds ~15 % of the window's
#   tokens (globals: the whole 250 d thinned; locals: a 30-80 d stretch thinned), with point dropout and a 20 %
#   chance of dropping one band. Spectral aux 0.5 on the global views (catalogue period at training only); phase
#   rope in the predictor at the encoder's own frequency (first global view).
#   200 epochs = 200 x 42,871 stars / 256 stars per step (one 250 d window each) = 33,500 steps.
# Then the test card on the TEST split: eval_encoder.sh (attention probe: class F1 and log-period on test; the period
# probe and spectral peaks), and the forecast probe on every test star (decoder fitted on 1,000 validation stars):
# tail 10 d, 30 % random gaps, 0.2-cycle phase gap.
#   sbatch project/jobs/lejepa.sh
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
source .venv/bin/activate
NAME=${NAME:-enc_le1}
ARGS="--size wide --spectral --spectral-aux 0.5 --jepa --jepa-no-ema --jepa-proj 16 --sig-lambda 0.05 --sig-slices 1024
 --views 2 4 --view-keep 0.15 --local-span 30 80 --band-drop 0.2 --phase-rope 4 --phase-freq local
 --mask-mode random --mask-ratio 0.85 --n-frames 1 --batch-size 256 --steps 33500 --wandb-group lejepa"
OUT=project/runs/$NAME MAX_LINKS=2 BUDGET=84000 SLURM_JOB_NAME=$NAME bash project/jobs/pretrain_mae.sh $ARGS
if [ -f project/runs/$NAME/DONE ]; then
    bash project/jobs/eval_encoder.sh $NAME
    bash project/jobs/test_forecast.sh $NAME
fi
