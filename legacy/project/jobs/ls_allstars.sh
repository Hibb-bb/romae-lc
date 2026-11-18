#!/bin/bash
#SBATCH --job-name=ls-allstars
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=01:30:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# The collaborator's Lomb-Scargle on every validation star at more budgets, in parallel on the CPU (project.eval.ls_allstars).
#   sbatch project/jobs/ls_allstars.sh project/runs/maew_spec/mae.pt project/results/period_maew_spec_silu2/predictions.npz --out project/results/ls_allstars_spec
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
source .venv/bin/activate
python -m project.eval.ls_allstars --ckpt "${1:?encoder checkpoint}" --predictions "${2:?predictions.npz}" --workers 30 "${@:3}"
