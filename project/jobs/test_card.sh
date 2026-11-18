#!/bin/bash
#SBATCH --job-name=test-card
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=10:00:00
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# The test card of one or more runs (project.eval.test_card): latent caches, then the class, period,
# phase and forecast cards -> project/results/test_card/RUN.json. Submit from the repo root.
#   sbatch project/jobs/test_card.sh RUN [RUN ...]      (project/jobs/train.sh also calls it)
# RUN is a run name under project/runs, or the path of a checkpoint (.../NAME/mae.pt).
# A card that is already in the JSON is skipped, so the job can be submitted again after a time-out.
# Afterwards, "python -m project.eval.summary" prints one table over all runs.
#
# Settings (environment):
#   RF=1            also the random-forest reference (-> project/results/classify_rf_RUN)
#   PERIOD_PROBE=1  also the period read-outs on VALIDATION (project.period_probe -> project/results/period_enc_RUN)
#   CARD_ARGS       extra flags for project.eval.test_card (e.g. "--cards class period", "--force")
#   VENV            the python environment (default .venv)
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
source "${VENV:-.venv}/bin/activate"
RF_FLAG=""
[ "${RF:-0}" = 1 ] && RF_FLAG="--rf"
st=0
for r in "$@"; do
    case "$r" in
        *.pt) ck=$r ;;
        *) ck=project/runs/$r/mae.pt ;;
    esac
    name=$(basename "$(dirname "$ck")")
    python -m project.eval.test_card --ckpt "$ck" $RF_FLAG ${CARD_ARGS:-} || st=$?
    if [ "${PERIOD_PROBE:-0}" = 1 ]; then
        python -m project.period_probe --latents "$(dirname "$ck")/latents.pt" --out "project/results/period_enc_$name" \
            --n-ls 50 --no-fold --device cuda --bins 2400 || st=$?
    fi
done
echo "exit $st"
exit $st
