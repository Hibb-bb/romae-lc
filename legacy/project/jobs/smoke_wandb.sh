#!/bin/bash
#SBATCH --job-name=smoke-wandb
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=01:30:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Smoke run of stage 1 with Weights & Biases logging (600 steps, two evals,
# the census time ladder), then the loader throughput benchmark. The wandb
# key comes from ~/.netrc (wandb login).
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
LADDER="--rope-wavelengths 0.01 0.053 0.107 0.218 0.428 0.835 1.64 3.43 6.98 14.3 34.9 6000 --time-scale 0.0015915"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m project.train_wm --out project/runs/smoke_w500 --no-resume \
    --window 500 --min-tokens 16 --max-tokens 256 $LADDER \
    --steps 600 --batch-size 64 --workers 8 \
    --eval-every 300 --ckpt-every 300 --log-every 50 \
    --probe-train 2000 --probe-val 1000 --val-objects 1000 \
    --wandb --wandb-group smoke "$@"
python -m project.bench --out project/results/bench_w500 \
    --window 500 --min-tokens 16 --max-tokens 256 $LADDER \
    --steps 120 --warmup 30 --wandb --wandb-group bench
