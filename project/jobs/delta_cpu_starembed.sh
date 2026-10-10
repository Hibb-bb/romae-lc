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
# (FATS + light-curve, raw magnitudes), then the forest benchmark. The job sets up its own x86 environment
# first (an x86 uv, a 3.12 python, the project with the bench and hf extras, light-curve and FATS), so it
# can be submitted straight from the Delta login node:
#   ssh login.delta.ncsa.illinois.edu
#   cd /projects/bfrf/hibb/romae-lc
#   sbatch project/jobs/delta_cpu_starembed.sh
# The setup needs internet (the uv installer, PyPI, GitHub); if the compute node has none, run the setup
# on the Delta login node first and then submit:
#   SETUP_ONLY=1 bash project/jobs/delta_cpu_starembed.sh
# ~/.local/bin/uv is the ARM build for DeltaAI and does not run here; the x86 build goes to ~/.local/uv-x86.
# (check the account name with `accounts`; the CPU allocation is usually bfrf-delta-cpu; the GPU one here is bfrf-dtai-gh)
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

# ---- the x86 environment (idempotent: every step is skipped when it is already in place)
[ "$(uname -m)" = x86_64 ] || { echo "this script is for the x86 Delta nodes, not $(uname -m)"; exit 1; }
UV=$HOME/.local/uv-x86/uv
export UV_CACHE_DIR=$HOME/.cache/uv-x86            # separate from the ARM cache
export UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python
if ! "$UV" --version >/dev/null 2>&1; then
    echo "installing an x86 uv into $HOME/.local/uv-x86"
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$HOME/.local/uv-x86 UV_NO_MODIFY_PATH=1 sh
fi
"$UV" --version
if ! .venv-x86/bin/python -c 'import sys; assert sys.version_info >= (3, 12)' >/dev/null 2>&1; then
    echo "(re)creating .venv-x86 with python 3.12 (the old one is missing or points at a python that is not there)"
    "$UV" venv --python 3.12 --clear .venv-x86
fi
UV_PROJECT_ENVIRONMENT=.venv-x86 "$UV" sync --inexact --extra bench --extra hf     # keeps light-curve and FATS; torch (CPU wheel), scikit-learn, pandas, datasets, the project
.venv-x86/bin/python -c 'import FATS, light_curve' >/dev/null 2>&1 || \
    "$UV" pip install --python .venv-x86/bin/python light-curve "git+https://github.com/nabeelre/FATS"
source .venv-x86/bin/activate
echo "python: $(command -v python) ($(python -V 2>&1))"
python - <<'PY'
import sys
assert sys.version_info >= (3, 12), f"wrong interpreter {sys.executable}"
import datasets, torch, sklearn, FATS, light_curve  # noqa: F401
print("environment ok: torch", torch.__version__, "light-curve", light_curve.__version__)
PY
[ "${SETUP_ONLY:-0}" = 1 ] && { echo "setup done"; exit 0; }

# ---- the work
[ -f project/results/starembed_feats/test.npz ] || python -m project.eval.starembed_feats --out project/results/starembed_feats --workers 60
python -m project.eval.classify_rf --latents project/runs/maew_spec/latents100k.pt --test-latents project/runs/maew_spec/latents100k_test.pt \
    --extra-features starembed=project/results/starembed_feats --features mean hand starembed mean+starembed \
    --exclude-anomaly --n-seeds 5 --n-jobs 60 --out project/results/classify_rf_spec5_starembed
