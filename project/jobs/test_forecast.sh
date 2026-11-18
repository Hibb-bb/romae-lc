#!/bin/bash
#SBATCH --job-name=testfc
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=08:00:00
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Forecast tests on EVERY test star (non-overlapping windows; latent decoder fitted on 1,000 validation stars):
# the last 10 d hidden, 30 % random gaps, a 0.2-cycle phase gap (catalogue period, evaluation only).
#   sbatch project/jobs/test_forecast.sh RUN [RUN ...]      (or called from another job script)
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
source .venv/bin/activate
mkdir -p project/results/test_forecast
T="--split test --fit-split validation --n-fit 1000 --n-stars 0 --stride 1.0"
for r in "$@"; do
    ck=project/runs/$r/mae.pt
    python -m project.eval.jepa_forecast_probe --ckpts $ck $T --out project/results/test_forecast/${r}_tail10.json
    python -m project.eval.jepa_forecast_probe --ckpts $ck $T --gap-share 0.3 --out project/results/test_forecast/${r}_gap0.3.json
    python -m project.eval.jepa_forecast_probe --ckpts $ck $T --phase-gap 0.2 --out project/results/test_forecast/${r}_pgap0.2.json
done
