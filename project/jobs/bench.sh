#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=00:40:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Loader / encoder throughput benchmark of stage 1 (project.bench), logged to
# wandb group "bench". Extra arguments go to the benchmark, e.g. --window 250.
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
LADDER="--rope-wavelengths 0.01 0.053 0.107 0.218 0.428 0.835 1.64 3.43 6.98 14.3 34.9 500 --time-scale 0.0015915"
python -m project.bench --out project/results/bench_w250 \
    --window 250 --min-tokens 16 --max-tokens 256 $LADDER \
    --steps 120 --warmup 30 --wandb --wandb-group bench "$@"
