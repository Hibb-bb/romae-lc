#!/bin/bash
# Spectral-peak period hit rates (project.eval.spectral_peak_probe) of the 800k-star encoder
# against the current one, on the same held-out stars: the Chen validation split (truth = our
# table period and Chen's sharpened period) and the ZTFxPC validation split (catalogue period).
#SBATCH --job-name=peak-chen
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120g
#SBATCH --time=00:50:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
CK="${CK:-project/runs/chen_spec/mae.pt project/runs/maew_spec/mae.pt}"   # the data root is given explicitly on both lines: the default would be the FIRST checkpoint's
N="${N:-1000}"          # stars per set; a number above the set size scores every validation star
CHEN="${CHEN:-1}"       # CHEN=0 skips the Chen set (its periods are not accurate; Dennis 10-23)
[ "$CHEN" = 1 ] && python -m project.eval.spectral_peak_probe --ckpts $CK --data /work/hdd/bfrf/hibb/chen_cache --chen-csv project/results/period_chen/periods.csv --n $N --label chen-val
python -m project.eval.spectral_peak_probe --ckpts $CK --data /projects/bfrf/data/PC_matches/ZTFxPC --n $N --label ztfxpc-val
