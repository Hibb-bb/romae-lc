#!/bin/bash
#SBATCH --job-name=classify-rf
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Fine-class classification with a random forest on frozen features, the collaborators' protocol (project.eval.classify_rf).
#   sbatch project/jobs/classify_rf.sh project/runs/maew_spec/latents100k.pt project/runs/maew_spec/latents100k_test.pt --out project/results/classify_rf_spec --features mean hand meanhand
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
source .venv/bin/activate
python -m project.eval.classify_rf --latents "${1:?latents.pt}" --test-latents "${2:?latents_test.pt}" --n-jobs 30 "${@:3}"
