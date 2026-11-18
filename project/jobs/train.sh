#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=12:00:00
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Train one run from its recipe (project.train), then run its test card. Submit from the repo root.
#   sbatch --job-name=enc_o3_ph project/jobs/train.sh enc_o3_ph
#   sbatch --job-name=enc_le1 --time=24:00:00 --export=ALL,MAX_LINKS=2,BUDGET=84000 project/jobs/train.sh enc_le1
#   bash project/jobs/train.sh project/recipes/enc_o3_ph.toml --steps 100   (inside a session)
# The first argument is a recipe: a name in project/recipes/ or a path to a .toml file.
# Extra arguments go to project.train and beat the recipe.
#
# Settings (environment):
#   OUT        the run folder (default project/runs/<recipe name>)
#   BUDGET     seconds of training in one job (default 40000). Training saves last.pt and stops at the
#              budget. Keep it below the job's time limit, with room for the evaluation in the last link.
#   MAX_LINKS  how many jobs the run may take (default 1). While LINK < MAX_LINKS, a link queues the
#              next one when it starts (it runs after this job ends, however it ends). The next link
#              resumes from last.pt.
#   LINK       the number of this link (default 1; the chain sets it).
#   EVAL       1 (default): when the run is finished ($OUT/DONE exists), run the test card
#              (project/jobs/test_card.sh). EVERY link does this, so the evaluation also runs when the
#              training finishes in a later link, and a link that finds the run already done only
#              evaluates. The card skips what is already in its JSON. EVAL=0 trains only.
#   VENV       the python environment (default .venv)
set -uo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source "${VENV:-.venv}/bin/activate"
RECIPE=${1:?a recipe: a name in project/recipes or a .toml path}
shift
[ -f "$RECIPE" ] || RECIPE=project/recipes/${RECIPE%.toml}.toml
[ -f "$RECIPE" ] || { echo "no recipe file $RECIPE"; exit 2; }
NAME=$(basename "$RECIPE" .toml)
OUT=${OUT:-project/runs/$NAME}
LINK=${LINK:-1}
MAX_LINKS=${MAX_LINKS:-1}
BUDGET=${BUDGET:-40000}
EVAL=${EVAL:-1}
SCRIPT=project/jobs/train.sh
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
evaluate() {
    [ "$EVAL" = 1 ] || return 0
    if [ ! -f "$OUT/DONE" ]; then
        echo "no $OUT/DONE: the run is not finished, no evaluation in this link"
        return 0
    fi
    bash project/jobs/test_card.sh "$OUT/mae.pt"
}
echo "recipe $RECIPE, link $LINK / $MAX_LINKS, out $OUT, budget ${BUDGET}s, eval $EVAL, extra args: $*"
if [ -f "$OUT/DONE" ]; then
    echo "already done: $(cat "$OUT/DONE")"
    evaluate
    exit $?
fi
if [ "$LINK" -lt "$MAX_LINKS" ]; then
    # in the background: a full queue makes this retry for up to 20 min, and
    # waiting for it would eat into this link's time
    ( queue --job-name="$SLURM_JOB_NAME" --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,OUT="$OUT",LINK=$((LINK + 1)),MAX_LINKS="$MAX_LINKS",BUDGET="$BUDGET",EVAL="$EVAL" \
        "$SCRIPT" "$RECIPE" "$@" || true ) &
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m project.train --recipe "$RECIPE" --out "$OUT" \
    --workers 8 --pin-memory --eval-every 2500 --ckpt-every 500 --time-budget "$BUDGET" --wandb \
    "$@"
st=$?
echo "training exit $st"
[ $st -eq 0 ] || exit $st
evaluate
