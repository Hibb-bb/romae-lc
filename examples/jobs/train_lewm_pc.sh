#!/bin/bash
#SBATCH --job-name=lewm-pc
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=examples/logs/%x-%j.out
#SBATCH --error=examples/logs/%x-%j.err
# LeWorldModel on a PC_matches sub-dataset. Any extra arguments go to the
# training script, e.g.
#   sbatch jobs/train_lewm_pc.sh --data /projects/bfrf/data/PC_matches/ZTFxPC --epochs 20
# For longer runs switch to --partition=ghx4 --time=1-00:00:00.
# The time ladder is the census recommendation (results/pc_period/
# recommendation.json, 12 angles, mix 0.5); pass --auto-time --time-spacing
# quantile to re-measure it from the training curves instead (adds ~2 min).
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python examples/train_lewm.py \
    --data /projects/bfrf/data/PC_matches/ZTFxPC \
    --time-scale 0.0015915 \
    --rope-wavelengths 0.01 0.053 0.107 0.218 0.428 0.835 1.64 3.43 6.98 14.3 34.9 6000 \
    --window 60 --min-tokens 8 --max-tokens 512 \
    --batch-size 64 --workers 8 --eval-every 1 \
    --out runs/lewm_pc.pt \
    "$@"
