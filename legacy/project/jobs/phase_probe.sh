#!/bin/bash
#SBATCH --job-name=phase-probe
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=00:45:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Is the phase of the window's start in the frozen latent? (project.phase_probe)
#   sbatch project/jobs/phase_probe.sh project/runs/mae_w250/latents.pt --out project/results/phase_mae
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
python -m project.phase_probe --latents "${1:?latents.pt}" "${@:2}"
