#!/bin/bash
#SBATCH --job-name=period-table
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Step 0 of the phase-coordinate plan: the test split's latents (if missing), then a period for every star of
# every split (project.eval.period_table) -> periods.csv.
#   sbatch project/jobs/period_table.sh project/runs/maew_spec/latents100k.pt project/results/period_table_spec
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
LAT=${1:?latents.pt (train + validation)}; OUT=${2:?output dir}
CK=$(python -c "import torch;print(torch.load('$LAT',map_location='cpu',weights_only=False)['meta']['ckpt'])")
TEST=${LAT%.pt}_test.pt
[ -f "$TEST" ] || python -m project.cache_latents --ckpt "$CK" --out "$TEST" --splits test --workers 8 --device cuda
python -m project.eval.period_table --latents "$LAT" --test-latents "$TEST" --out "$OUT" --device cuda "${@:3}"
