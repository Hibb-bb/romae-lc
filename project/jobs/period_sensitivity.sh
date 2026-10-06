#!/bin/bash
#SBATCH --job-name=sens
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=00:40:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# The smallest period change an encoder can tell from noise (project.eval.period_sensitivity),
# for one or several checkpoints: each argument is "checkpoint:name" -> project/results/sens_<name>.
#   sbatch project/jobs/period_sensitivity.sh project/runs/mae_w250/mae.pt:mae project/runs/maew_w250/last.pt:maew
set -uo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
for spec in "$@"; do
    ckpt=${spec%%:*}; name=${spec##*:}
    echo "#### $name: $ckpt"
    python -m project.eval.period_sensitivity --ckpt "$ckpt" --out "project/results/sens_$name"
done
