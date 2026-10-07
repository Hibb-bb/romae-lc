#!/bin/bash
#SBATCH --job-name=eval-bundle
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
# Every pending evaluation in one job, so the queue is waited for once:
# the budget sweep, the all-star report (fixed candidate rule), and the
# runtime variants (batched, batched + compile, three quantisations, CUDA
# graphs). Every step is skipped when its results.json exists, so the same
# script can be submitted again to finish what a 2 h link left over.
set -uo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CK=project/runs/maew_spec/mae.pt
PRED=project/results/period_maew_spec_silu/predictions.npz
step() {  # step <out dir> <command...>
    local out=$1; shift
    if [ -f "$out/results.json" ]; then echo "#### skip $out (done)"; return 0; fi
    echo "#### $(date '+%T') $out"; "$@" || echo "#### FAILED $out"
}
step project/results/budget_sweep_spec python -m project.eval.budget_sweep --ckpt $CK --predictions $PRED --out project/results/budget_sweep_spec --device cuda
step project/results/period_report_spec3 bash project/jobs/period_report_inner.sh project/runs/maew_spec/latents100k.pt project/results/period_maew_spec_silu project/results/period_maew_spec_relu project/results/period_report_spec3
RT="python -m project.eval.runtime --ckpt $CK --predictions $PRED --device cuda --ls-budgets --astropy-budgets --workers 8"
step project/results/runtime_spec_batched $RT --out project/results/runtime_spec_batched
step project/results/runtime_spec_batched_compile $RT --out project/results/runtime_spec_batched_compile --compile default --dynamic --warmup 12
for q in int8wo int8dq fp8; do
    step project/results/runtime_spec_$q $RT --out project/results/runtime_spec_$q --quant $q --ref-latents project/results/runtime_spec_batched/latents.pt
done
step project/results/runtime_spec_graphpad $RT --out project/results/runtime_spec_graphpad --compile reduce-overhead --static-pad --warmup 12 --batch-sweep 512
echo "#### $(date '+%T') bundle done"
