#!/bin/bash
#SBATCH --job-name=period-chen
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
# Model periods for the 800k-star Chen set (project.eval.period_chen), one 2 h link at a time: every link
# resumes where the last stopped (rows are appended per shard, finished shards are marked), and stops cleanly
# at BUDGET seconds. Queue as many links as needed (about 8 for the whole set):
#   for i in $(seq 8); do echo "sbatch project/jobs/period_chen.sh" >> project/jobs/queue.txt; done
set -uo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
OUT=${OUT:-project/results/period_chen}
[ -f "$OUT/summary.json" ] && { echo "already done: $OUT"; exit 0; }
python -m project.eval.period_chen --root /work/hdd/bfrf/hibb/chen20_ztf_dr24 --latents project/runs/maew_spec/latents100k.pt \
    --out "$OUT" --workers 8 --time-budget "${BUDGET:-6300}" --device cuda
