#!/bin/bash
#SBATCH --job-name=wm-w60
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Stage 1 at window 60 with a log-spaced dense ladder over 0.15 - 120 d: the
# sharpest test of period recovery for rotation-type stars. A rotary rung
# folds a period only within about 1 / (4 x cycles per window); the median
# ROT period (4.4 d) has 57 cycles in a 250 d window and 14 in a 60 d one,
# which a ladder of 2% spacing resolves, while the census-following quantile
# ladder of the 250 d runs spends 264 of its 378 rungs below one day. A
# 60 d window still holds about 33 ZTF points. This wrapper runs
# train_wm.sh (a chain, see there) with these settings; usable as a PIPELINE
# stage after another chain (the wm.pt argument it is handed is dropped).
#   sbatch project/jobs/train_wm_w60.sh
case "${1:-}" in *.pt) shift ;; esac
export OUT=${OUT:-project/runs/wm_w60_log} WINDOW=60
exec bash project/jobs/train_wm.sh --min-tokens 8 --lam-min 0.15 --time-spacing log "$@"
