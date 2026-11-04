#!/bin/bash
#SBATCH --job-name=phase1b
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400g
#SBATCH --time=08:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# ICML plan, Phase 1-2, second node (added 11-04): four single changes, one per GPU.
#   enc_p1_nce0    J3 with no contrast (--dyn-nce 0)
#   enc_p1_mask75  J3 with 70-80 % of the tokens hidden (J3: 50 %)
#   enc_p1_fc      the masked JEPA in forecast mode, no add-on: every window also hides its last 5-30 days
#                  (compare with enc_p1_noadd of phase1.sh, the same run without the tail)
#   enc_p1_fc75    enc_p1_fc with 70-80 % hidden
# Each then gets eval_encoder.sh, the add-on decoding probe (runs with --dyn) and the masked-predictor
# forecast probe (project/eval/jepa_forecast_probe.py, when it exists at run time).
#   sbatch project/jobs/phase1b.sh
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
source .venv/bin/activate
mkdir -p project/logs/phase1 project/results/state_decode project/results/jepa_forecast
J3="--size wide --spectral --spectral-aux 0.5 --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 --steps 20000 --batch-size 64 --wandb-group phase1 --jepa --jepa-proj 32 --jepa-recon-weight 0.5 --jepa-init project/runs/enc_j1/mae.pt"
ADD="--dyn 8 --dyn-cross --advance 0.5 1.0 --dyn-dt 0.5 30"
M75="--mask-ratio 0.75 --mask-ratio-range 0.7 0.8"
FC="--forecast-tail 5 30"

train() {  # NAME ARGS...: train, then every evaluation
    local name=$1; shift
    OUT=project/runs/$name MAX_LINKS=1 BUDGET=30000 SLURM_JOB_NAME=$name bash project/jobs/pretrain_mae.sh "$@"
    if [ -f project/runs/$name/DONE ]; then
        bash project/jobs/eval_encoder.sh $name
        python -m project.eval.state_decode_probe --ckpts project/runs/$name/mae.pt --context \
            --out project/results/state_decode/$name.json --save-npz project/results/state_decode/pairs.npz
        [ -f project/eval/jepa_forecast_probe.py ] && python -m project.eval.jepa_forecast_probe --ckpts project/runs/$name/mae.pt \
            --out project/results/jepa_forecast/$name.json --save-npz project/results/jepa_forecast/pairs.npz
    fi
}
lane() {  # GPU LOGNAME COMMAND...
    local g=$1 log=$2; shift 2
    ( export CUDA_VISIBLE_DEVICES=$g; "$@" ) > project/logs/phase1/$log-$SLURM_JOB_ID.log 2>&1
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
lane 0 b_lane0 bash -c "$(declare -f train); train enc_p1_nce0 $J3 $ADD --dyn-nce 0" &
lane 1 b_lane1 bash -c "$(declare -f train); train enc_p1_mask75 $J3 $ADD $M75" &
lane 2 b_lane2 bash -c "$(declare -f train); train enc_p1_fc $J3 $FC" &
lane 3 b_lane3 bash -c "$(declare -f train); train enc_p1_fc75 $J3 $M75 $FC" &
wait
echo "all lanes done"
