#!/bin/bash
#SBATCH --job-name=anomaly
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
# The anomaly test on the frozen pipeline (project.anomaly): fake events are
# added to real light curves and the sequence predictor's scores are read
# next to two yardsticks that need no model. The first argument is the
# pred.pt of a sequence predictor, the rest go to the script.
#   sbatch project/jobs/anomaly.sh project/runs/pred_maew_seq2/pred.pt --out project/results/anomaly_maew
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m project.anomaly --pred "${1:?pred.pt}" "${@:2}"
