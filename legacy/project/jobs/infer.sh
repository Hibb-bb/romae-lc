#!/bin/bash
#SBATCH --job-name=infer
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
# Energy-based inference (M6 / M7) with the stage 1-3 checkpoints:
#   sbatch project/jobs/infer.sh calibrate --ckpt ... --decoder ... --residual ...
#   sbatch project/jobs/infer.sh anomaly --ckpt ... --decoder ... --inject phase --out ...
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
python -m project.infer "$@"
