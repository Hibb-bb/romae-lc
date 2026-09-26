#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# The milestones after the world model, sequentially on one GPU, from a
# stage-2 wm.pt of train_wm.py (its frozen encoder and MSE predictor are
# E_dyn; the flow predictor of train_predictor.py is not wired into the
# energies yet): M4 residual, M5 mse decoder (stage 3, the workhorse of
# E_obs), M6 calibrate / anomaly-with-injection / smooth / forecast, M7
# period. The flow decoder is a separate chain (train_decoder.sh with
# KIND=flow).
#   sbatch project/jobs/pipeline.sh project/runs/wm_w500/wm.pt
set -euo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
W=${1:?stage-1 checkpoint}
RUN=$(dirname "$W")
R=$RUN/res
D=$RUN/dec_mse
O=project/results/$(basename "$RUN")
LOADER="--workers ${WORKERS:-8} ${LOADER_EXTRA:-}"
DEC_STEPS=${DEC_STEPS:-15000}
echo "pipeline on $W -> $R, $D, $O"
python -m project.residual --ckpt "$W" --out "$R" --passes 2 --wandb --wandb-group pipeline
python -m project.train_decoder --ckpt "$W" --kind mse --out "$D" --steps "$DEC_STEPS" \
    --eval-every 5000 --baselines $LOADER --wandb --wandb-group pipeline
python -m project.infer calibrate --ckpt "$W" --decoder "$D/dec.pt" --residual "$R/residual.pt" --n-objects 200 --out "$O"
python -m project.infer anomaly --ckpt "$W" --decoder "$D/dec.pt" --residual "$R/residual.pt" --weights "$O/weights.json" \
    --inject phase --n-objects 150 --map-steps 100 --out "$O"
python -m project.infer smooth --ckpt "$W" --decoder "$D/dec.pt" --residual "$R/residual.pt" --weights "$O/weights.json" \
    --n-objects 60 --plot 8 --out "$O"
python -m project.infer forecast --ckpt "$W" --decoder "$D/dec.pt" --residual "$R/residual.pt" --weights "$O/weights.json" \
    --n-objects 60 --out "$O"
python -m project.infer period --ckpt "$W" --n-objects 800 --out "$O"
echo "pipeline done"
