#!/bin/bash
#SBATCH --job-name=runtime
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=01:30:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Runtime per star of the model's period path against Lomb-Scargle (project.eval.runtime).
#   sbatch project/jobs/runtime.sh project/runs/maew_spec/mae.pt project/results/period_maew_spec_silu/predictions.npz --out project/results/runtime_spec
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
python -m project.eval.runtime --ckpt "${1:?encoder checkpoint}" --predictions "${2:?predictions.npz}" "${@:3}"
