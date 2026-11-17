#!/bin/bash
#SBATCH --job-name=ropedemo
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=00:45:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Forecast examples (folded at the catalogue period for display) of enc_o2_fcbig_ph and enc_o3_fcph_pband, 30 and 60 days hidden.
#   sbatch project/jobs/rope_demo_int.sh
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
source .venv/bin/activate
mkdir -p project/results/figures project/results/jepa_forecast
for r in enc_o2_fcbig_ph enc_o3_fcph_pband; do for h in 30 60; do
    python -m project.eval.jepa_forecast_demo --ckpt project/runs/$r/mae.pt --hide $h \
        --out project/results/figures/jepa_forecast_demo_${r#enc_}_h$h
done; done
