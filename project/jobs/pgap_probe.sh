#!/bin/bash
#SBATCH --job-name=pgapprobe
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
# Phase-gap test (0.2 cycles hidden at the catalogue period, evaluation only) on the finished runs, as reference
# for option 3; GAP_RUNS adds the in-window gap test first.   sbatch project/jobs/pgap_probe.sh   or   RUNS="..." sbatch --export=ALL project/jobs/pgap_probe.sh
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
source .venv/bin/activate
mkdir -p project/results/jepa_gap
for r in ${GAP_RUNS:-}; do  # optional: the in-window gap test first (10 / 30 / 50 % hidden)
    for p in 0.1 0.3 0.5; do
        python -m project.eval.jepa_forecast_probe --ckpts project/runs/$r/mae.pt --gap-share $p \
            --out project/results/jepa_gap/${r}_gap$p.json --save-npz project/results/jepa_gap/pairs.npz
    done
done
for r in ${RUNS:-enc_j1 enc_o2_noadd enc_o2_fc enc_o2_fcbig_ph}; do
    [ -f project/runs/$r/DONE ] || { echo "$r: not DONE, skipped"; continue; }
    python -m project.eval.jepa_forecast_probe --ckpts project/runs/$r/mae.pt --phase-gap 0.2 \
        --out project/results/jepa_gap/${r}_pgap0.2.json --save-npz project/results/jepa_gap/pairs.npz
done
