#!/bin/bash
# Dataset summary (labels, periods per split) and the three figures: label histogram, period histogram,
# classification bars (project.eval.dataset_summary, project.eval.dataset_figs). Batch partition: no GPU work,
# but the login node is too slow to load the three splits.
#SBATCH --job-name=dataset-figs
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=00:40:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
mkdir -p project/results/figures
[ -f project/results/figures/ztfxpc_summary.npz ] || python -m project.eval.dataset_summary
python -m project.eval.dataset_figs
echo "exit $?"
