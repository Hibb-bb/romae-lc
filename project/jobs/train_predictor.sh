#!/bin/bash
#SBATCH --job-name=pred-w250
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
# Stage 2: the flow predictor on cached frozen latents (project.train_predictor)
# as a chain of 2-hour interactive jobs, exactly like train_wm.sh: every link
# queues its successor at its START (dependency afterany) and trains until
# its time budget; the successor finds $OUT/DONE when the 400k steps are
# reached, submits the next stage (THEN, or the first entry of PIPELINE) with
# $OUT/pred.pt and exits. The first positional argument is the latents.pt of
# cache_latents.sh (default LATENTS).
#   sbatch project/jobs/train_predictor.sh project/runs/mae_w250/latents.pt
#   OUT=project/runs/pred_w250_mse sbatch --job-name=pred-mse project/jobs/train_predictor.sh \
#       project/runs/mae_w250/latents.pt --kind mse
# The sequence predictor (--arch seq: the whole light curve as a sequence of
# window latents, --batch-size objects per step, see --seq-* and --max-len):
#   LATENTS=project/runs/maew_w250/latents_r4.pt OUT=project/runs/pred_maew_seq \
#       sbatch --job-name=pred-seq project/jobs/train_predictor.sh --arch seq --batch-size 64
# Extra arguments go to project.train_predictor and are remembered in the checkpoint.
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
LATENTS=${LATENTS:-project/runs/mae_w250/latents.pt}
case "${1:-}" in *.pt) LATENTS=$1; shift ;; esac
OUT=${OUT:-project/runs/pred_w250}
LINK=${LINK:-1}
MAX_LINKS=${MAX_LINKS:-8}
BUDGET=${BUDGET:-6000}
SCRIPT=project/jobs/train_predictor.sh
PIPELINE=${PIPELINE:-${THEN:-}}   # THEN is the one-stage form of PIPELINE
NEXT=${PIPELINE%%:*}
REST=${PIPELINE#"$NEXT"}; REST=${REST#:}
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
echo "link $LINK / $MAX_LINKS, latents $LATENTS, out $OUT, budget ${BUDGET}s, next stage '${NEXT}' then '${REST}', args: $*"
if [ -f "$OUT/DONE" ]; then
    echo "already done: $(cat "$OUT/DONE")"
    if [ -n "$NEXT" ] && [ -f "$NEXT" ]; then
        echo "submitting $NEXT $OUT/pred.pt (pipeline after it: '${REST}')"
        queue --export=ALL,OUT=,LINK=1,THEN=,PIPELINE="$REST" "$NEXT" "$OUT/pred.pt"
    fi
    exit 0
fi
if [ "$LINK" -lt "$MAX_LINKS" ]; then
    # in the background: a full queue makes this retry for up to 20 min, and
    # waiting for it pushed a link past its 2 h limit (it then lost its
    # successor and its last steps)
    ( queue --job-name="$SLURM_JOB_NAME" --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,LATENTS="$LATENTS",OUT="$OUT",LINK=$((LINK + 1)),MAX_LINKS="$MAX_LINKS",BUDGET="$BUDGET",THEN=,PIPELINE="$PIPELINE" \
        "$SCRIPT" "$@" || true ) &
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m project.train_predictor --latents "$LATENTS" --out "$OUT" \
    --steps 400000 --eval-every 10000 --ckpt-every 5000 --time-budget "$BUDGET" --wandb \
    "$@"
