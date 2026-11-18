#!/bin/bash
#SBATCH --job-name=phase-dec
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# The phase decoder ladder (project.phase_decoder) on a frozen stage-1 encoder:
#   ceiling   no context, oracle period and oracle phase, observed target: the latent alone
#   ceilingt  the same with the clean template as the target
#   ctx       3 context windows, phase from the window's start, observed target;
#             evaluated with the oracle, fine-searched, Lomb-Scargle and model periods
#   ctxt      the same with the clean template as the target
# MODEL_PERIODS = predictions.npz of a period_probe run on the same encoder (for 'refined'/'model').
# PRED = pred.pt of a sequence predictor on the same encoder: the evaluation then also forecasts from the
# PREDICTED latent (the whole chain), next to the true latent and the past-only variant.
#   CKPT=project/runs/mae_w250/mae.pt MODEL_PERIODS=project/results/period_mae20/predictions.npz sbatch project/jobs/phase_decoder.sh ceiling ctx
set -uo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CKPT=${CKPT:-project/runs/mae_w250/mae.pt}
STEPS=${STEPS:-20000}
MODEL_PERIODS=${MODEL_PERIODS:-}
PRED=${PRED:-}
DIR=$(dirname "$CKPT")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
run() {
    local name=$1; shift
    echo "#### $name: $*"
    python -m project.phase_decoder --ckpt "$CKPT" --out "$DIR/phase_$name" --steps "$STEPS" --workers 8 --wandb \
        --wandb-group phase-decoder ${MODEL_PERIODS:+--model-periods "$MODEL_PERIODS"} ${PRED:+--pred "$PRED"} "$@"
}
for v in ${@:-ceiling ctx}; do
    case "$v" in
        ceiling)  run ceiling  --n-ctx 0 --phase oracle --target obs --eval-periods oracle ;;
        ceilingt) run ceilingt --n-ctx 0 --phase oracle --target template --eval-periods oracle ;;
        ctx)      run ctx      --n-ctx 3 --phase window --target obs --eval-periods oracle refined ls model ;;
        ctxt)     run ctxt     --n-ctx 3 --phase window --target template --eval-periods oracle refined ls model ;;
        ctxeval)  run ctx      --n-ctx 3 --phase window --target obs --eval-only --eval-periods oracle ls ls_refined refined cands model ;;
        *) echo "unknown variant $v" ;;
    esac
done
