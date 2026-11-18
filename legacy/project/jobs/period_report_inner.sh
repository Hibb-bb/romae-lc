#!/bin/bash
# The body of period_report.sh, run inside another job (eval_bundle.sh). Same arguments.
set -eo pipefail
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
