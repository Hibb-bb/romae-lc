#!/bin/bash
#SBATCH --job-name=cache
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Between stage 1 and stage 2: encode the window grid of every record once
# with the frozen encoder (project.cache_latents) into latents.pt next to the
# checkpoint (or LATENTS=path), the input of train_predictor.sh. One job, no
# chain. THEN=script submits the next stage with the latents path as its
# first argument (e.g. THEN=project/jobs/train_predictor.sh).
#   sbatch project/jobs/cache_latents.sh project/runs/mae_w250/mae.pt
#   sbatch project/jobs/cache_latents.sh project/runs/mae_w250/mae.pt --stride 0.5
#   THEN=project/jobs/train_predictor.sh sbatch --export=ALL,THEN=project/jobs/train_predictor.sh \
#       project/jobs/cache_latents.sh project/runs/mae_w250/mae.pt
# Extra arguments go to project.cache_latents.
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CKPT=${1:?mae.pt}
LATENTS=${LATENTS:-$(dirname "$CKPT")/latents.pt}
THEN=${THEN:-}
echo "ckpt $CKPT, latents $LATENTS, then '${THEN}', args: ${*:2}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m project.cache_latents --ckpt "$CKPT" --out "$LATENTS" --workers 8 "${@:2}"
if [ -n "$THEN" ] && [ -f "$THEN" ]; then
    # THEN_OUT names the successor's output directory (its own default
    # otherwise); THEN_PIPELINE is the successor's own PIPELINE (the stages
    # to run after it, e.g. a wide autoencoder after the predictor)
    echo "submitting $THEN $LATENTS (OUT=${THEN_OUT:-default}, pipeline after it '${THEN_PIPELINE:-}')"
    sbatch --export=ALL,OUT="${THEN_OUT:-}",LINK=1,THEN=,PIPELINE="${THEN_PIPELINE:-}" "$THEN" "$LATENTS"
fi
