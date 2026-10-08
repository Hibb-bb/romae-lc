#!/bin/bash
#SBATCH --job-name=starembed-feats
#SBATCH --account=bfrf-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120g
#SBATCH --time=06:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Delta (x86 CPU cluster) version of starembed_feats.sh: the StarEmbed hand-crafted features on every star
# (FATS + light-curve, raw magnitudes), then the forest benchmark. Submit from the Delta login node, not DeltaAI:
#   ssh login.delta.ncsa.illinois.edu
#   cd /projects/bfrf/hibb/romae-lc
#   UV_PROJECT_ENVIRONMENT=.venv-x86 uv sync --extra bench          # the x86 environment (torch CPU wheel, scikit-learn, pandas)
#   uv pip install --python .venv-x86/bin/python light-curve "git+https://github.com/nabeelre/FATS"
#   sbatch project/jobs/delta_cpu_starembed.sh
# (check the account name with `accounts`; the CPU allocation is usually bfrf-delta-cpu; the GPU one here is bfrf-dtai-gh)
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
source .venv-x86/bin/activate
[ -f project/results/starembed_feats/test.npz ] || python -m project.eval.starembed_feats --out project/results/starembed_feats --workers 60
python -m project.eval.classify_rf --latents project/runs/maew_spec/latents100k.pt --test-latents project/runs/maew_spec/latents100k_test.pt \
    --extra-features starembed=project/results/starembed_feats --features mean hand starembed mean+starembed \
    --exclude-anomaly --n-seeds 5 --n-jobs 60 --out project/results/classify_rf_spec5_starembed
