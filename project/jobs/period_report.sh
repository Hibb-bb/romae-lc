#!/bin/bash
#SBATCH --job-name=period-report
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
# The period report on every validation star (project.eval.period_report):
# first the period probe twice on the same latents (SiLU and ReLU read-out
# MLPs, which also saves the joint read-out's top-k candidates), then the
# report on the SiLU read-out with the ReLU one as a comparison.
#   sbatch project/jobs/period_report.sh project/runs/maew_spec/latents100k.pt project/results/period_maew_spec_silu \
#       project/results/period_maew_spec_relu project/results/period_report_spec [report args]
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
LAT=${1:?latents.pt}; SILU=${2:?probe out (silu)}; RELU=${3:?probe out (relu)}; OUT=${4:?report out}
for pair in "silu:$SILU" "relu:$RELU"; do
    act=${pair%%:*}; dir=${pair#*:}
    if [ ! -f "$dir/predictions.npz" ]; then
        python -m project.period_probe --latents "$LAT" --out "$dir" --n-ls 50 --no-fold --device cuda --mlp-act "$act"
    fi
done
CKPT=$(python -c "import json;print(json.load(open('$SILU/results.json'))['ckpt'])")
python -m project.eval.period_report --ckpt "$CKPT" --predictions "$SILU/predictions.npz" --compare "relu=$RELU/predictions.npz" \
    --out "$OUT" --device cuda "${@:5}"
