#!/bin/bash
#SBATCH --job-name=rope-abl
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
# Does the time ladder matter for the masked autoencoder? Four light
# encoders that differ only in the rotary time encoding, each trained for
# STEPS steps (the period probe of the light run was flat from 10k on), then
# cached and scored with the period probe:
#   standard  the plain RoPE of the RoMAE paper: base 10000, time in days,
#             the same 21 rungs in every head and layer (6.3 d to 4500 d)
#   sharedlog 21 rungs shared by every head and layer, log spaced over the
#             range of the data (0.02 d to 500 d)
#   sharedq   21 shared rungs placed on the measured periods (quantiles)
#   dense     our ladder: 378 rungs, one per angle of every head and layer
# One job, the four runs one after the other. A run that is already done is
# skipped, so the job can be submitted again after a time-out.
#   sbatch project/jobs/rope_ablation.sh
set -uo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
STEPS=${STEPS:-15000}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# base ** (2 i / dim) for the 21 active angles of a 56-channel time block,
# as wavelengths in days (2 pi times the timescale) with one position = 1 day
STANDARD=$(python -c "import math; print(' '.join(f'{2 * math.pi * 10000 ** (2 * i / 56):.6g}' for i in range(21)))")
run() {
    local name=$1; shift
    local out=project/runs/rope_$name
    echo "#### $name: $*"
    if [ ! -f "$out/DONE" ]; then
        python -m project.pretrain_mae --out "$out" --window 250 --min-tokens 16 --max-tokens 256 \
            --steps "$STEPS" --batch-size 64 --workers 8 --pin-memory \
            --eval-every 2500 --ckpt-every 500 --wandb --wandb-group rope-ablation "$@" || return
    fi
    [ -f "$out/latents.pt" ] || python -m project.cache_latents --ckpt "$out/mae.pt" --out "$out/latents.pt" --workers 8 || return
    python -m project.period_probe --latents "$out/latents.pt" --out "project/results/period_rope_$name" \
        --n-ls 50 --no-fold --device cuda
}
run standard --rope-wavelengths $STANDARD --time-scale 1.0
run sharedlog --ladder-mode shared --time-spacing log --lam-min 0.02 --lam-max 500
run sharedq --ladder-mode shared
run dense
