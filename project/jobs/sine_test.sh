#!/bin/bash
#SBATCH --job-name=sine-test
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# The sine test (project.sine_test) over its variants, on one GPU: can the decoder recover a clean
# sinusoid far from the seen points (plain / phasor at the model's own frequency / oracle phasor at the
# true frequency), down to which period, and does it change with regular sampling, a non-sinusoidal
# shape, noise, or no label in the spectral auxiliary loss. Results: project/results/sine_test/<name>/
set -uo pipefail
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
STEPS=${STEPS:-3000}
ROOT=${ROOT:-project/results/sine_test}   # e.g. ROOT=project/results/sine_test10k STEPS=10000
ONLY=${ONLY:-}   # space-separated variant names to run (default: all)
run() { local name=$1; shift; [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && return 0; echo "#### $name: $*"; [ -f "$ROOT/$name/results.json" ] || python -m project.sine_test --device cuda --steps "$STEPS" --out "$ROOT/$name" "$@"; }
run plain          --variant plain
run phasor         --variant phasor
run oracle         --variant oracle
run plain_regular  --variant plain --sampling regular
run phasor_regular --variant phasor --sampling regular
run phasor_shape   --variant phasor --shape 0.3
run phasor_noise   --variant phasor --noise 0.2
run plain_nolabel  --variant plain --spectral-aux 0
run phasor_nolabel --variant phasor --spectral-aux 0
# the phase as a rotary axis of the decoder (2026-10-12), against the phasor map without weight decay
run phaserope          --variant phaserope
run phaserope_oracle   --variant phaserope_oracle
run phaserope_shape    --variant phaserope --shape 0.3
run phaserope_noise    --variant phaserope --noise 0.2
run phaserope_nolabel  --variant phaserope --spectral-aux 0
run phasor_nowd        --variant phasor --phasor-init 0.1
echo "all done"
