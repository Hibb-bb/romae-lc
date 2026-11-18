#!/bin/bash
#SBATCH --job-name=period-head
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
# The period head (project.period_head) on a frozen stage-1 encoder: a fine
# period from the raw window's spectrum, the latent as the prior.
#   sbatch project/jobs/period_head.sh project/runs/mae_w250/mae.pt [extra args]
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CKPT=${1:?stage-1 checkpoint}
shift
OUT=${OUT:-$(dirname "$CKPT")/period_head}
BUDGET=${BUDGET:-6600}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m project.period_head --ckpt "$CKPT" --out "$OUT" --workers 8 --pin-memory --time-budget "$BUDGET" --wandb --wandb-group period-head "$@"
