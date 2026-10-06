#!/bin/bash
#SBATCH --job-name=ls-bench
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
# Period search efficiency: the model's prior against Lomb-Scargle (project.eval.ls_benchmark).
#   sbatch project/jobs/ls_benchmark.sh project/runs/mae_w250/mae.pt project/results/period_mae20/predictions.npz \
#       --out project/results/ls_benchmark [--head project/runs/mae_w250/period_head/predictions.npz]
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
python -m project.eval.ls_benchmark --ckpt "${1:?encoder checkpoint}" --predictions "${2:?predictions.npz}" "${@:3}"
