#!/bin/bash
#SBATCH --job-name=gapprobe
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Phase 1: can the masked predictor fill in missing points INSIDE a window (not a forecast)?
# jepa_forecast_probe in gap mode, 10 / 30 / 50 % of real tokens hidden at random, on J1, the no-add-on and
# forecast-tail Phase 1 runs, and the option 2 control once it is DONE.
#   sbatch project/jobs/gap_probe.sh   or   RUNS="enc_a enc_b" sbatch --export=ALL project/jobs/gap_probe.sh
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
source .venv/bin/activate
mkdir -p project/results/jepa_gap
for r in ${RUNS:-enc_j1 enc_p1_noadd enc_p1_fc enc_o2_noadd}; do
    [ -f project/runs/$r/mae.pt ] || { echo "$r: no checkpoint, skipped"; continue; }
    [ ! -f project/runs/$r/DONE ] && { echo "$r: not DONE, skipped"; continue; }
    for p in 0.1 0.3 0.5; do
        python -m project.eval.jepa_forecast_probe --ckpts project/runs/$r/mae.pt --gap-share $p \
            --out project/results/jepa_gap/${r}_gap$p.json --save-npz project/results/jepa_gap/pairs.npz
    done
done
