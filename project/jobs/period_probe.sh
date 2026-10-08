#!/bin/bash
#SBATCH --job-name=period-probe
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Period regression from cached latents next to Lomb-Scargle, with the
# failure cases mined (project.period_probe). One short job on a latents.pt
# of cache_latents; the first argument is the cache, the rest go to the
# script (--out, --pred, --n-ls, ...).
#   sbatch project/jobs/period_probe.sh project/runs/maew_w250/latents_r4.pt --out project/results/period_maew
#   sbatch project/jobs/period_probe.sh project/runs/maew_w250/latents_r4.pt --out project/results/period_maew --pred project/runs/pred_maew_seq/pred.pt --n-ls 2000
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m project.period_probe --latents "${1:?latents.pt}" "${@:2}"
