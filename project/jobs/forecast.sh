#!/bin/bash
#SBATCH --job-name=forecast
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=01:30:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# The end-to-end brightness forecast (project.forecast): encoder, sequence
# predictor and decoder together, scored on real future observations next to
# yardsticks that use no model. Arguments: the predictor and the decoder,
# the rest go to the script.
#   sbatch project/jobs/forecast.sh project/runs/pred_maew_seq2/pred.pt project/runs/maew_w250/dec_mse/best.pt --out project/results/forecast_maew
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m project.forecast --pred "${1:?pred.pt}" --dec "${2:?decoder}" "${@:3}"
