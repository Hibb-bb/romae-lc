#!/bin/bash
#SBATCH --job-name=residual
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
# Stage 2 (M4): Gaussian (+ optional flow) residual on a stage-1 checkpoint.
#   sbatch project/jobs/residual.sh project/runs/wm_w500/wm.pt project/runs/res_w500 [--flow]
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CKPT=${1:?checkpoint path}
OUT=${2:?output directory}
shift 2
python -m project.residual --ckpt "$CKPT" --out "$OUT" --passes 2 "$@"
