#!/bin/bash
#SBATCH --job-name=lookback
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Stage 3b: the lookback decoder trained on the forecast task
# (project.lookback) on a frozen stage-1 encoder. One 2 h link; resume by
# resubmitting the same line (last.pt is picked up). OUT defaults to
# <ckpt dir>/lookback.
#   OUT=project/runs/mae_w250/lookback sbatch project/jobs/train_lookback.sh project/runs/mae_w250/mae.pt
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CKPT=${1:?stage-1 checkpoint}
shift
OUT=${OUT:-$(dirname "$CKPT")/lookback}
BUDGET=${BUDGET:-6300}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m project.lookback --ckpt "$CKPT" --out "$OUT" --workers 8 --time-budget "$BUDGET" --wandb "$@"
