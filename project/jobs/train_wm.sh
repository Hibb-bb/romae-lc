#!/bin/bash
#SBATCH --job-name=wm-w250
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
# Stage 2 (predictor) as a chain of 2-hour interactive jobs. The QOS runs one job
# per user and accepts two submitted jobs, so every link queues its successor
# at its START (dependency afterany) and then trains until its time budget;
# the successor finds $OUT/DONE when training finished, submits the next
# stage and exits. Keep no other job queued next to a chain. The next stage
# is THEN (one script, called with $OUT/wm.pt, e.g. THEN=project/jobs/inject.sh)
# or PIPELINE, a colon-separated list of stages run one after the other, each
# chain script popping the first entry when it is done, e.g.
#   PIPELINE=project/jobs/pretrain_mae.sh:project/jobs/train_wm.sh
# runs, after this stage-2 run, the masked pretraining and then a stage-2 run
# from its encoder: pretrain_mae.sh hands its successor $OUT/mae.pt, and a
# leading *.pt argument here becomes --init-backbone with OUT defaulting to
# project/runs/wm_w${WINDOW}_mae. Stage scripts get OUT and LINK reset, so
# they use their own defaults. A refused sbatch (QOS queue full) is retried
# for 20 min.
#   sbatch project/jobs/train_wm.sh                      # window 250, 50k steps, dense ladder
#   WINDOW=500 sbatch --job-name=wm-w500 project/jobs/train_wm.sh
#   OUT=project/runs/wm_w250_mae sbatch --job-name=wm-mae project/jobs/train_wm.sh \
#       --init-backbone project/runs/mae_w250/mae.pt   # from pretrain_mae.sh
#   OUT=project/runs/wm_w250_frozen sbatch --job-name=wm-frozen project/jobs/train_wm.sh --init-backbone project/runs/mae_w250/mae.pt --freeze-backbone
#   OUT=project/runs/wm_w250_wide sbatch --job-name=wm-wide project/jobs/train_wm.sh --size wide
#   sbatch --job-name=wm-dense --export=ALL,OUT=project/runs/wm_w250_dense,PIPELINE=project/jobs/pretrain_mae.sh:project/jobs/train_wm.sh project/jobs/train_wm.sh
# Extra arguments go to project.train_wm and are remembered in the checkpoint.
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
WINDOW=${WINDOW:-250}
# A leading *.pt (what pretrain_mae.sh hands its successor) is the encoder
# to start from; such a run gets its own default OUT.
case "${1:-}" in *.pt) set -- --init-backbone "$@"; OUT=${OUT:-project/runs/wm_w${WINDOW}_mae} ;; esac
OUT=${OUT:-project/runs/wm_w${WINDOW}}
LINK=${LINK:-1}
MAX_LINKS=${MAX_LINKS:-8}
BUDGET=${BUDGET:-6000}
SCRIPT=project/jobs/train_wm.sh
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
echo "link $LINK / $MAX_LINKS, out $OUT, budget ${BUDGET}s, next stage '${NEXT}' then '${REST}', args: $*"
if [ -f "$OUT/DONE" ]; then
    echo "already done: $(cat "$OUT/DONE")"
    if [ -n "$NEXT" ] && [ -f "$NEXT" ]; then
        echo "submitting $NEXT $OUT/wm.pt (pipeline after it: '${REST}')"
        queue --export=ALL,OUT=,LINK=1,THEN=,PIPELINE="$REST" "$NEXT" "$OUT/wm.pt"
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
python -m project.train_wm --out "$OUT" \
    --window "$WINDOW" --min-tokens 16 --max-tokens 256 \
    --steps 50000 --batch-size 128 --workers 8 --pin-memory \
    --eval-every 2500 --ckpt-every 500 --time-budget "$BUDGET" --wandb \
    "$@"
