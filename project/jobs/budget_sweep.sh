#!/bin/bash
#SBATCH --job-name=budget-sweep
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
# The model-seeded search over its own budget (project.eval.budget_sweep) on the benchmark's 1,000 stars.
#   sbatch project/jobs/budget_sweep.sh project/runs/maew_spec/mae.pt project/results/period_maew_spec_silu/predictions.npz --out project/results/budget_sweep_spec
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
python -m project.eval.budget_sweep --ckpt "${1:?encoder checkpoint}" --predictions "${2:?predictions.npz}" "${@:3}" --device cuda
