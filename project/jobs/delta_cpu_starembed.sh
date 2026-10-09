#!/bin/bash
#SBATCH --job-name=starembed-feats
#SBATCH --account=bfrf-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=15:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Delta (x86 CPU cluster) version of starembed_feats.sh: the StarEmbed hand-crafted features on every star
# (FATS + light-curve, raw magnitudes), then the forest benchmark. Submit from the Delta login node, not DeltaAI:
#   ssh login.delta.ncsa.illinois.edu
#   cd /projects/bfrf/hibb/romae-lc
#   uv venv --python 3.12 .venv-x86                                      # (re)create the x86 environment; its python must be 3.12, not /usr/bin/python 3.9
#   UV_PROJECT_ENVIRONMENT=.venv-x86 uv sync --extra bench --extra hf   # the project with torch (CPU wheel), scikit-learn, pandas, datasets
#   uv pip install --python .venv-x86/bin/python light-curve "git+https://github.com/nabeelre/FATS"
#   sbatch project/jobs/delta_cpu_starembed.sh
# (check the account name with `accounts`; the CPU allocation is usually bfrf-delta-cpu; the GPU one here is bfrf-dtai-gh)
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
source .venv-x86/bin/activate
# fail here, loudly, if the environment is not the x86 venv with its packages (a dangling venv python falls through to /usr/bin/python 3.9)
echo "python: $(command -v python) ($(python -V 2>&1))"
python - <<'PY'
import sys
assert sys.version_info >= (3, 12), f"wrong interpreter {sys.executable}: recreate the env with  uv venv --python 3.12 .venv-x86  then  UV_PROJECT_ENVIRONMENT=.venv-x86 uv sync --extra bench --extra hf"
import datasets, torch, sklearn, FATS, light_curve  # noqa: F401
print("environment ok: torch", torch.__version__, "light-curve", light_curve.__version__)
PY
[ -f project/results/starembed_feats/test.npz ] || python -m project.eval.starembed_feats --out project/results/starembed_feats --workers 60
python -m project.eval.classify_rf --latents project/runs/maew_spec/latents100k.pt --test-latents project/runs/maew_spec/latents100k_test.pt \
    --extra-features starembed=project/results/starembed_feats --features mean hand starembed mean+starembed \
    --exclude-anomaly --n-seeds 5 --n-jobs 60 --out project/results/classify_rf_spec5_starembed
