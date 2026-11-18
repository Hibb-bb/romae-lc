#!/bin/bash
#SBATCH --job-name=inject
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Injected-anomaly test (M2) on a stage-1 checkpoint:
#   sbatch project/jobs/inject.sh project/runs/wm_w500/wm.pt [inject args]
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CKPT=${1:?checkpoint path}
shift
python -m project.inject --ckpt "$CKPT" --n-objects "${N_OBJECTS:-300}" "$@"
