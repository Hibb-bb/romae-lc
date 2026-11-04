#!/bin/bash
#SBATCH --job-name=phase1
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400g
#SBATCH --time=10:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# ICML plan, Phase 1 (debug the predictor), on one 4-GPU node: four lanes, one per GPU.
#   1a: the decoding probe with diagnostics D1-D4 on J3, J3b and J4 (no training)
#   1b: single changes to J3, each 20k steps from J1 (as J3), then its evaluations
#       (eval_encoder.sh: latents, period, phase and class probes; and the decoding probe)
# A run that is already DONE is skipped, and a killed run resumes from its last.pt on resubmission.
#   sbatch project/jobs/phase1.sh
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
source .venv/bin/activate
mkdir -p project/logs/phase1 project/results/state_decode
J3="--size wide --spectral --spectral-aux 0.5 --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 --steps 20000 --batch-size 64 --wandb-group phase1 --jepa --jepa-proj 32 --jepa-recon-weight 0.5 --jepa-init project/runs/enc_j1/mae.pt"
ADD="--dyn 8 --dyn-cross --advance 0.5 1.0 --dyn-dt 0.5 30"

decode() {  # the decoding probe with D1-D4
    python -m project.eval.state_decode_probe --ckpts project/runs/$1/mae.pt --context \
        --out project/results/state_decode/$1.json --save-npz project/results/state_decode/pairs.npz
}
train() {  # NAME ARGS...: train, then every evaluation
    local name=$1; shift
    OUT=project/runs/$name MAX_LINKS=1 BUDGET=30000 SLURM_JOB_NAME=$name bash project/jobs/pretrain_mae.sh "$@"
    if [ -f project/runs/$name/DONE ]; then
        bash project/jobs/eval_encoder.sh $name
        decode $name
    fi
}
lane() {  # GPU LOGNAME COMMAND...
    local g=$1 log=$2; shift 2
    ( export CUDA_VISIBLE_DEVICES=$g; "$@" ) > project/logs/phase1/$log-$SLURM_JOB_ID.log 2>&1
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
lane 0 lane0 bash -c "$(declare -f decode train); decode enc_j3; decode enc_j3b; decode enc_j4; train enc_p1_j3t $J3 $ADD --dyn-target ema" &
lane 1 lane1 bash -c "$(declare -f decode train); train enc_p1_j3tdec $J3 $ADD --dyn-target ema --dyn-decode 1.0 --dyn-dec-band --dyn-dec-nll; train enc_p1_noadd $J3" &
lane 2 lane2 bash -c "$(declare -f decode train); train enc_p1_sig0 $J3 $ADD --dyn-sig 0; train enc_p1_sigbatch $J3 $ADD --dyn-sig-mode batch --dyn-mean 0" &
lane 3 lane3 bash -c "$(declare -f decode train); train enc_p1_norot $J3 $ADD --dyn-no-rot" &
wait
echo "all lanes done"
