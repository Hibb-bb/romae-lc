#!/bin/bash
#SBATCH --job-name=starembed-feats
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# The StarEmbed hand-crafted features (FATS + light-curve, per band, raw magnitudes) on every star, then the forest
# benchmark with them next to the pooled latent (project.eval.starembed_feats, project.eval.classify_rf).
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
source .venv/bin/activate
[ -f project/results/starembed_feats/test.npz ] || python -m project.eval.starembed_feats --out project/results/starembed_feats --workers 30
python -m project.eval.classify_rf --latents project/runs/maew_spec/latents100k.pt --test-latents project/runs/maew_spec/latents100k_test.pt \
    --extra-features starembed=project/results/starembed_feats --features mean hand starembed mean+starembed \
    --exclude-anomaly --n-seeds 5 --n-jobs 30 --out project/results/classify_rf_spec5_starembed
