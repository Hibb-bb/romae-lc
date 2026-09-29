#!/bin/bash
#SBATCH --job-name=maew-w250
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Stage 1, the wide plain masked autoencoder: 384 wide, 6 heads, 6 deep (756
# rotary rungs at 0.55 % spacing against the light preset's 378 at 1.2 %),
# the token-level masked reconstruction of pretrain_mae.py that learned
# period at the light size (the bottleneck variants never did: dead for
# stage 1, see README), 150k steps at batch 128. Watch the within-class
# period R2 every 10k steps: the light run plateaued at 10k, and a plateau
# here means the remaining links are not worth their GPU time. This wrapper
# runs pretrain_mae.sh (a chain of 2 h links, see there) and is usable as a
# THEN / PIPELINE stage after another chain (a *.pt argument it is handed is
# dropped by pretrain_mae.sh).
#   sbatch project/jobs/pretrain_mae_wide.sh
# MAX_LINKS is forced: a THEN hand-off from a two-link chain exports its own
# limit, which ended the first wide chain at 52k of 150k steps.
export OUT=${OUT:-project/runs/maew_w250} MAX_LINKS=${WIDE_MAX_LINKS:-8}
exec bash project/jobs/pretrain_mae.sh "$@" --size wide \
    --steps 150000 --batch-size 128 --eval-every 10000
