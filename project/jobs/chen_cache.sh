#!/bin/bash
#SBATCH --job-name=chen-cache
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
# Build the compact cache of the 800k-star Chen set (project.chen_data): one npz per shard with the standardised
# points, our period from project/results/period_chen/periods.csv and the split. Resumable: a cached shard is skipped.
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
python -m project.chen_data build --root /work/hdd/bfrf/hibb/chen20_ztf_dr24 --out /work/hdd/bfrf/hibb/chen_cache
