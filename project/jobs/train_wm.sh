#!/bin/bash
#SBATCH --job-name=wm-w500
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
# Stage 1 (M1) as a chain of 2-hour interactive jobs. The QOS runs one job
# per user and accepts two submitted jobs, so every link queues its successor
# at its START (dependency afterany) and then trains until its time budget;
# the successor finds $OUT/DONE when training finished, submits the THEN job
# (e.g. THEN=project/jobs/inject.sh, called with $OUT/wm.pt) and exits. Keep
# no other job queued next to a chain.
#   sbatch project/jobs/train_wm.sh                      # window 500, 50k steps
#   OUT=project/runs/wm_w250 sbatch --job-name=wm-w250 project/jobs/train_wm.sh --window 250
# Extra arguments go to project.train_wm and are remembered in the checkpoint.
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
OUT=${OUT:-project/runs/wm_w500}
LINK=${LINK:-1}
MAX_LINKS=${MAX_LINKS:-8}
BUDGET=${BUDGET:-6000}
SCRIPT=project/jobs/train_wm.sh
echo "link $LINK / $MAX_LINKS, out $OUT, budget ${BUDGET}s, args: $*"
if [ -f "$OUT/DONE" ]; then
    echo "already done: $(cat "$OUT/DONE")"
    if [ -n "${THEN:-}" ] && [ -f "$THEN" ]; then
        echo "submitting $THEN $OUT/wm.pt"
        sbatch "$THEN" "$OUT/wm.pt"
    fi
    exit 0
fi
if [ "$LINK" -lt "$MAX_LINKS" ]; then
    sbatch --job-name="$SLURM_JOB_NAME" --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,OUT="$OUT",LINK=$((LINK + 1)),MAX_LINKS="$MAX_LINKS",BUDGET="$BUDGET",THEN="${THEN:-}" \
        "$SCRIPT" "$@" || echo "could not queue the next link (queue full); resubmit by hand if needed"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m project.train_wm --out "$OUT" \
    --window 500 --min-tokens 16 --max-tokens 256 \
    --steps 50000 --batch-size 128 --workers 8 --pin-memory \
    --eval-every 2500 --ckpt-every 500 --time-budget "$BUDGET" --wandb \
    "$@"
