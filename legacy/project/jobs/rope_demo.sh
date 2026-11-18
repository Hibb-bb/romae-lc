#!/bin/bash
#SBATCH --job-name=ropedemo
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=01:30:00
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Forecast examples of J1 + phase rope (enc_o2_fcbig_ph) as light curves (30 and 60 days hidden), and the tail
# forecast test out to 60 days (gap bins to 60 d) for it and the option 2 control.
#   sbatch project/jobs/rope_demo.sh
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
source .venv/bin/activate
mkdir -p project/results/figures project/results/jepa_forecast
for h in 30 60; do
    python -m project.eval.jepa_forecast_demo --ckpt project/runs/enc_o2_fcbig_ph/mae.pt --hide $h \
        --out project/results/figures/jepa_forecast_demo_o2_fcbig_ph_h$h
done
for r in enc_o2_fcbig_ph enc_o2_noadd; do
    python -m project.eval.jepa_forecast_probe --ckpts project/runs/$r/mae.pt --tail 60 --dt 0.5 60 \
        --out project/results/jepa_forecast/${r}_tail60.json
done
