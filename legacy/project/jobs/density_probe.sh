#!/bin/bash
#SBATCH --job-name=density
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Forecast skill against sampling density (D5): the add-on test on J3 and SIGReg off, the masked-predictor
# test on the forecast-tail run.   sbatch project/jobs/density_probe.sh
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
source .venv/bin/activate
for r in enc_j3 enc_p1_sig0; do
    python -m project.eval.state_decode_probe --ckpts project/runs/$r/mae.pt --out project/results/state_decode/$r.json \
        --save-npz project/results/state_decode/pairs.npz
done
python -m project.eval.jepa_forecast_probe --ckpts project/runs/enc_p1_fc/mae.pt --out project/results/jepa_forecast/enc_p1_fc.json \
    --save-npz project/results/jepa_forecast/pairs.npz
