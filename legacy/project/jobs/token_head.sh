#!/bin/bash
#SBATCH --job-name=token-head
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
# The token read-out (project.token_head): a fine period from the encoder's token features.
#   sbatch project/jobs/token_head.sh project/runs/mae_w250/mae.pt [extra args]
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CKPT=${1:?stage-1 checkpoint}
shift
OUT=${OUT:-$(dirname "$CKPT")/token_head}
python -m project.token_head --ckpt "$CKPT" --out "$OUT" --workers 8 --pin-memory --time-budget "${BUDGET:-6600}" --wandb --wandb-group token-head "$@"
