#!/bin/bash
#SBATCH --job-name=decoder
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
# Stage 3 (M5) decoder on a stage-1 checkpoint, as a chain like train_wm.sh
# (the successor link is queued at the start of every link; a refused sbatch
# is retried for 20 min). KIND=flow|mse.
#   KIND=mse sbatch --job-name=dec-mse project/jobs/train_decoder.sh project/runs/wm_w500/wm.pt --baselines
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CKPT=${1:?stage-1 checkpoint}
shift
KIND=${KIND:-flow}
OUT=${OUT:-$(dirname "$CKPT")/dec_$KIND}
LINK=${LINK:-1}
MAX_LINKS=${MAX_LINKS:-4}
BUDGET=${BUDGET:-6000}
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
echo "link $LINK / $MAX_LINKS, kind $KIND, out $OUT, args: $*"
if [ -f "$OUT/DONE" ]; then
    echo "already done: $(cat "$OUT/DONE")"
    if [ -n "${THEN:-}" ] && [ -f "$THEN" ]; then
        # THEN=<script> runs after the decoder, handed the stage-1 checkpoint
        echo "submitting $THEN $CKPT"
        queue --export=ALL,OUT=,LINK=1,THEN=,PIPELINE= "$THEN" "$CKPT"
    fi
    exit 0
fi
if [ "$LINK" -lt "$MAX_LINKS" ]; then
    # in the background: a full queue makes this retry for up to 20 min, and
    # waiting for it can push a link past its 2 h limit
    ( queue --job-name="$SLURM_JOB_NAME" --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,KIND="$KIND",OUT="$OUT",LINK=$((LINK + 1)),MAX_LINKS="$MAX_LINKS",BUDGET="$BUDGET" \
        project/jobs/train_decoder.sh "$CKPT" "$@" || true ) &
fi
python -m project.train_decoder --ckpt "$CKPT" --kind "$KIND" --out "$OUT" \
    --workers 8 --time-budget "$BUDGET" --wandb "$@"
