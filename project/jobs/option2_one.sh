#!/bin/bash
#SBATCH --job-name=o2-one
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=06:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# ICML plan, Phase 1, option 2 (11-06): make the masked JEPA predictor forecast. Every run warm-starts from J1
# for real (before 11-06 --jepa-init copied nothing from a JEPA checkpoint), 20k steps, no add-on.
#   enc_o2_noadd     J1 + 20k steps, no tail: the control
#   enc_o2_fc        the last 10-60 days of every window hidden (larger tail share than phase1b's 5-30)
#   enc_o2_fcbig     + a bigger predictor: 4 layers of width 384 (J1: 2 of 192)
#   enc_o2_fcbig_ph  + the phase at the encoder's own frequency as a rotary axis of the predictor (no catalogue period)
# Each then gets eval_encoder.sh and the masked-predictor forecast test; lane 0 first tests J1 itself.
#   sbatch project/jobs/option2.sh   (4 GPUs)   or   sbatch project/jobs/option2_one.sh LANE   (one GPU, LANE 0-3)
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
source .venv/bin/activate
mkdir -p project/logs/phase1 project/results/jepa_forecast
B="--size wide --spectral --spectral-aux 0.5 --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 --steps 20000 --batch-size 64 --wandb-group option2 --jepa --jepa-proj 32 --jepa-recon-weight 0.5 --jepa-init project/runs/enc_j1/mae.pt"
FC="--forecast-tail 10 60"
BIG_DEC="--dec-width 384 --dec-heads 6 --dec-depth 4"
PH="--phase-rope 4 --phase-freq local"

fc_probe() {
    python -m project.eval.jepa_forecast_probe --ckpts project/runs/$1/mae.pt --out project/results/jepa_forecast/$1.json \
        --save-npz project/results/jepa_forecast/pairs.npz
}
train() {  # NAME ARGS...: train, then every evaluation
    local name=$1; shift
    OUT=project/runs/$name MAX_LINKS=1 BUDGET=30000 SLURM_JOB_NAME=$name bash project/jobs/pretrain_mae.sh "$@"
    if [ -f project/runs/$name/DONE ]; then
        bash project/jobs/eval_encoder.sh $name
        fc_probe $name
    fi
}
lane() {  # GPU LOGNAME COMMAND...
    local g=$1 log=$2; shift 2
    ( export CUDA_VISIBLE_DEVICES=$g; "$@" ) > project/logs/phase1/$log-$SLURM_JOB_ID.log 2>&1
}
LANE=${1:?lane 0-3}
BIG=${BIG_JOB:-3543932}  # the 4-GPU version: cancel it while it waits, step aside if it already runs
if squeue -h -j "$BIG" -t R 2>/dev/null | grep -q .; then echo "the 4-GPU job $BIG is running: nothing to do"; exit 0; fi
scancel "$BIG" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case $LANE in
    0) fc_probe enc_j1; train enc_o2_noadd $B ;;
    1) train enc_o2_fc $B $FC ;;
    2) train enc_o2_fcbig $B $FC $BIG_DEC ;;
    3) train enc_o2_fcbig_ph $B $FC $BIG_DEC $PH ;;
esac
echo "lane $LANE done"
