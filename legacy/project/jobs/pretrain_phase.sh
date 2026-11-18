#!/bin/bash
#SBATCH --job-name=phase-bn
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
# Stage 1 with the phase bottleneck (project.pretrain_phase) on the light
# encoder, then the latent cache, the period probe and the phase probe.
#   NAME=phase_bn sbatch project/jobs/pretrain_phase.sh [extra pretrain_phase args]
set -uo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
NAME=${NAME:-phase_bn}
STEPS=${STEPS:-15000}
OUT=project/runs/$NAME
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [ ! -f "$OUT/DONE" ]; then
    python -m project.pretrain_phase --out "$OUT" --window 250 --min-tokens 16 --max-tokens 256 \
        --steps "$STEPS" --batch-size 64 --workers 8 --pin-memory --eval-every 2500 --ckpt-every 500 \
        --wandb --wandb-group phase-bottleneck "$@" || exit 1
fi
[ -f "$OUT/latents.pt" ] || python -m project.cache_latents --ckpt "$OUT/mae.pt" --out "$OUT/latents.pt" --workers 8 || exit 1
python -m project.period_probe --latents "$OUT/latents.pt" --out "project/results/period_$NAME" --n-ls 50 --no-fold --device cuda
python -m project.phase_probe --latents "$OUT/latents.pt" --out "project/results/phase_$NAME" --device cuda
