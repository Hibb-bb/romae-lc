#!/bin/bash
#SBATCH --job-name=o3-one
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
# ICML plan, end of Phase 1 + first Phase 2 views (11-12). All from J1, 20k steps, small predictor (2 x 192),
# phase rope at the encoder's own f, no add-on.
#   lane 0  enc_o3_fcph        tail 10-60 d + phase rope: is the class / period cost of enc_o2_fcbig_ph the big predictor?
#   lane 1  enc_o3_ph          phase rope only, no tail
#   lane 2  enc_o3_fcph80      lane 0 + 80 % hidden                                 (Phase 2 view)
#   lane 3  enc_o3_fcph_pband  lane 0 + a phase band of 0.1-0.3 cycles hidden at the catalogue period, then
#                              random points to 50 % (--mask-mode phaseband; period at training only)  (Phase 2 view)
# Each then gets eval_encoder.sh, the tail forecast test, the in-window gap test (30 %) and the phase-gap test
# (0.2 cycles at the catalogue period, evaluation only).
#   for l in 0 1 2 3; do sbatch --job-name=o3-l$l project/jobs/option3_one.sh $l; done
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
source .venv/bin/activate
mkdir -p project/results/jepa_forecast project/results/jepa_gap
B="--size wide --spectral --spectral-aux 0.5 --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 --steps 20000 --batch-size 64 --wandb-group option3 --jepa --jepa-proj 32 --jepa-recon-weight 0.5 --jepa-init project/runs/enc_j1/mae.pt --phase-rope 4 --phase-freq local"
FC="--forecast-tail 10 60"

probes() {
    local r=$1 ck=project/runs/$1/mae.pt
    python -m project.eval.jepa_forecast_probe --ckpts $ck --out project/results/jepa_forecast/$r.json --save-npz project/results/jepa_forecast/pairs.npz
    python -m project.eval.jepa_forecast_probe --ckpts $ck --gap-share 0.3 --out project/results/jepa_gap/${r}_gap0.3.json --save-npz project/results/jepa_gap/pairs.npz
    python -m project.eval.jepa_forecast_probe --ckpts $ck --phase-gap 0.2 --out project/results/jepa_gap/${r}_pgap0.2.json --save-npz project/results/jepa_gap/pairs.npz
}
train() {  # NAME ARGS...: train, then every evaluation
    local name=$1; shift
    OUT=project/runs/$name MAX_LINKS=1 BUDGET=30000 SLURM_JOB_NAME=$name bash project/jobs/pretrain_mae.sh "$@"
    if [ -f project/runs/$name/DONE ]; then
        bash project/jobs/eval_encoder.sh $name
        probes $name
    fi
}
LANE=${1:?lane 0-3}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case $LANE in
    0) train enc_o3_fcph $B $FC ;;
    1) train enc_o3_ph $B ;;
    2) train enc_o3_fcph80 $B $FC --mask-ratio 0.8 ;;
    3) train enc_o3_fcph_pband $B $FC --mask-mode phaseband --phase-band 0.1 0.3 ;;
esac
echo "lane $LANE done"
