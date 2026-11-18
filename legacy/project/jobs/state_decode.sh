#!/bin/bash
#SBATCH --job-name=state-decode
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Forecast decoding of the dynamic state (project.eval.state_decode_probe), z-scores against the
# observation errors:  sbatch project/jobs/state_decode.sh NAME [NAME ...]
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
for NAME in "$@"; do
    python -m project.eval.state_decode_probe --ckpts project/runs/$NAME/mae.pt --context --out project/results/state_decode/$NAME.json --save-npz project/results/state_decode/pairs.npz ${EXTRA:-}
done
echo "exit $?"
