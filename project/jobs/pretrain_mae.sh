#!/bin/bash
#SBATCH --job-name=mae-w250
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
# Stage 1 (the autoencoder): masked pretraining of the window encoder, or the
# bottleneck autoencoder with --bottleneck (project.pretrain_mae), as a chain
# of 2-hour interactive jobs, exactly like train_wm.sh (the successor is
# queued at the START of every link; $OUT/DONE ends the chain and submits
# the next stage, THEN or the first entry of PIPELINE, with $OUT/mae.pt as
# its first argument like every other chain: train_wm.sh turns a leading
# *.pt into --init-backbone and defaults its OUT to wm_w${WINDOW}_mae (or
# WM_OUT when set); gate.sh, cache_latents.sh and train_decoder.sh take the
# checkpoint positionally). A first positional *.pt argument (the checkpoint
# a train_wm.sh chain passes to its next stage) is dropped: the pretraining
# takes none. A refused sbatch (QOS queue full) is retried for 20 min.
#   sbatch project/jobs/pretrain_mae.sh                  # window 250, 50k steps, dense ladder
#   OUT=project/runs/bn_w250 sbatch --job-name=bn-w250 project/jobs/pretrain_mae.sh --bottleneck
#   THEN=project/jobs/train_wm.sh sbatch --export=ALL,THEN=project/jobs/train_wm.sh project/jobs/pretrain_mae.sh
#   THEN=project/jobs/cache_latents.sh sbatch --export=ALL,THEN=project/jobs/cache_latents.sh project/jobs/pretrain_mae.sh
# Extra arguments go to project.pretrain_mae and are remembered in the checkpoint.
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
WINDOW=${WINDOW:-250}
OUT=${OUT:-project/runs/mae_w${WINDOW}}
LINK=${LINK:-1}
MAX_LINKS=${MAX_LINKS:-8}
BUDGET=${BUDGET:-6000}
SCRIPT=project/jobs/pretrain_mae.sh
case "${1:-}" in *.pt) shift ;; esac   # a previous stage's checkpoint: not ours
PIPELINE=${PIPELINE:-${THEN:-}}   # THEN is the one-stage form of PIPELINE
NEXT=${PIPELINE%%:*}
REST=${PIPELINE#"$NEXT"}; REST=${REST#:}
WM_OUT=${WM_OUT:-}   # the successor's OUT; empty leaves it its own default
# The QOS refuses a third queued job; retry for 20 min before giving up so a
# busy queue does not end the chain (the link still trains to its budget).
queue() {
    for i in $(seq 40); do
        sbatch "$@" && return 0
        echo "sbatch refused (queue full), retry $i / 40 in 30 s"
        sleep 30
    done
    echo "could not queue: $*; resubmit by hand if needed"
    return 1
}
echo "link $LINK / $MAX_LINKS, out $OUT, budget ${BUDGET}s, next stage '${NEXT}' then '${REST}', args: $*"
if [ -f "$OUT/DONE" ]; then
    echo "already done: $(cat "$OUT/DONE")"
    if [ -n "$NEXT" ] && [ -f "$NEXT" ]; then
        echo "submitting $NEXT $OUT/mae.pt with OUT='$WM_OUT' (pipeline after it: '${REST}')"
        queue --export=ALL,OUT="$WM_OUT",LINK=1,THEN=,PIPELINE="$REST" "$NEXT" "$OUT/mae.pt"
    fi
    exit 0
fi
if [ "$LINK" -lt "$MAX_LINKS" ]; then
    # in the background: a full queue makes this retry for up to 20 min, and
    # waiting for it can push a link past its 2 h limit
    ( queue --job-name="$SLURM_JOB_NAME" --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,OUT="$OUT",LINK=$((LINK + 1)),MAX_LINKS="$MAX_LINKS",BUDGET="$BUDGET",THEN=,PIPELINE="$PIPELINE" \
        "$SCRIPT" "$@" || true ) &
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m project.pretrain_mae --out "$OUT" \
    --window "$WINDOW" --min-tokens 16 --max-tokens 256 \
    --steps 50000 --batch-size 64 --workers 8 --pin-memory \
    --eval-every 2500 --ckpt-every 500 --time-budget "$BUDGET" --wandb \
    "$@"
