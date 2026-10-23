#!/bin/bash
# Per-token phase probe of frozen encoders (project/eval/token_phase_probe.py): does the
# token output know the observation's phase? CK = checkpoints (the first also gets a
# random-init control); N = stars (default 600).
#SBATCH --job-name=token-phase
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
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
source .venv/bin/activate
CK="${CK:-project/runs/maew_spec/mae.pt project/runs/enc_chenft/mae.pt project/runs/enc_cycrope2/mae.pt project/runs/chen_spec/mae.pt}"
N="${N:-600}"
OUT="${OUT:-project/results/token_phase/ztfxpc_val.json}"
python -m project.eval.token_phase_probe --ckpts $CK --data /projects/bfrf/data/PC_matches/ZTFxPC --n-stars $N --random-init --label ztfxpc-val --out "$OUT"
echo "exit $?"
