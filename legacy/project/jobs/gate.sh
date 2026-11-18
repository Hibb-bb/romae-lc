#!/bin/bash
#SBATCH --job-name=gate
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# The gate on frozen latents (project.gate): probe vs hand-feature baseline,
# advance effect vs replicate floor, optionally the stage-3 decoder vs the GP.
# One short job on a stage-1 mae.pt (or a stage-2 wm.pt), before any stage-2
# predictor GPU time is spent; the job fails (exit 1) when the gate fails.
# THEN=<script> submits that script with the checkpoint when the gate passes
# (e.g. THEN=project/jobs/train_decoder.sh).
#   sbatch project/jobs/gate.sh project/runs/mae_w250/mae.pt
#   sbatch project/jobs/gate.sh project/runs/mae_w250/mae.pt --decoder-results project/runs/mae_w250/dec_mse/log.jsonl
#   THEN=project/jobs/train_decoder.sh sbatch --export=ALL,THEN=project/jobs/train_decoder.sh project/jobs/gate.sh project/runs/mae_w250/mae.pt
# Extra arguments go to project.gate.
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CKPT=${1:?mae.pt}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if python -m project.gate --ckpt "$CKPT" "${@:2}"; then
    if [ -n "${THEN:-}" ] && [ -f "$THEN" ]; then
        echo "gate passed: submitting $THEN $CKPT"
        sbatch --export=ALL,THEN= "$THEN" "$CKPT"
    fi
else
    echo "gate failed: not submitting ${THEN:-anything}"
    exit 1
fi
