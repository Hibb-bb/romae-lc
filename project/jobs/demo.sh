#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=00:45:00
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Forecast example plots (project.eval.card_demo) of one or more runs: one validation star per class,
# the last 30 and the last 60 days of a window hidden. Figures only, nothing is scored.
# Submit from the repo root.
#   sbatch project/jobs/demo.sh enc_o2_fcbig_ph enc_o3_fcph_pband
#   -> project/results/figures/jepa_forecast_demo_<run without "enc_">_h30.png and _h60.png (and .pdf)
# Settings (environment): HIDE="30 60" the days hidden; DEMO_ARGS extra flags for project.eval.card_demo;
# VENV the python environment (default .venv).
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
source "${VENV:-.venv}/bin/activate"
mkdir -p project/results/figures
st=0
for r in "$@"; do
    for h in ${HIDE:-30 60}; do
        python -m project.eval.card_demo --ckpt "project/runs/$r/mae.pt" --hide "$h" \
            --out "project/results/figures/jepa_forecast_demo_${r#enc_}_h$h" ${DEMO_ARGS:-} || st=$?
    done
done
exit $st
