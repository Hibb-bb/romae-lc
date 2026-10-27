#!/bin/bash
#SBATCH --job-name=attn-probe
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Star-level attentive probe (project.eval.attn_probe) of one encoder's cached window latents.
#   sbatch project/jobs/attn_probe.sh NAME [LATENTS TEST_LATENTS]   (default: project/runs/NAME/latents.pt and latents_test.pt)
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
NAME=$1
L=${2:-project/runs/$NAME/latents.pt}
T=${3:-project/runs/$NAME/latents_test.pt}
python -m project.eval.attn_probe --latents $L --test-latents $T --out project/results/attn_probe_$NAME --epochs 40
echo "exit $?"
