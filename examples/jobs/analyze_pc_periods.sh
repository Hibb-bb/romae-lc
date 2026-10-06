#!/bin/bash
#SBATCH --job-name=pc-periods
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=examples/logs/%x-%j.out
#SBATCH --error=examples/logs/%x-%j.err
# Period / cadence census of /projects/bfrf/data/PC_matches (CPU work; the GPU
# is only requested because the ghx4 partition allocates by GPU).
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/sw/user/python/miniforge3-pytorch-2.5.0/bin/python
$PY examples/analyze_pc_periods.py --out results/pc_period "$@"
