#!/bin/bash
# Every evaluation of one encoder run, in one job:  sbatch project/jobs/eval_encoder.sh NAME
#   1. latent caches (train + validation, and test) of project/runs/NAME/mae.pt
#   2. period read-outs (project.period_probe -> project/results/period_enc_NAME)
#   3. spectral-peak hit rates on ALL ZTFxPC validation stars (-> the log)
#   4. per-token phase probe (-> project/results/token_phase/NAME.json)
#   5. star-level attentive probe (project.eval.attn_probe, the default probe since 10-28): fine class
#      (13 classes) and log period from a star's window latents, attention pooling and the mean as its
#      control; fit on train, epoch picked on validation, scored on test -> project/results/attn_probe_NAME
#   6. only with RF=1: the fine-class random forest on the pooled latent (the collaborators' protocol, grid
#      search skipped, 16 classes) -> project/results/classify_rf_NAME
#SBATCH --job-name=eval-enc
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
NAME=${1:?run name under project/runs}
out=project/runs/$NAME
set -x
[ -f "$out/latents.pt" ] || python -m project.cache_latents --ckpt "$out/mae.pt" --out "$out/latents.pt" --workers 8 --batch-size 64
[ -f "$out/latents_test.pt" ] || python -m project.cache_latents --ckpt "$out/mae.pt" --out "$out/latents_test.pt" --workers 8 --batch-size 64 --splits test
python -m project.period_probe --latents "$out/latents.pt" --out "project/results/period_enc_$NAME" --n-ls 50 --no-fold --device cuda --bins 2400
python -m project.eval.spectral_peak_probe --ckpts "$out/mae.pt" --data /projects/bfrf/data/PC_matches/ZTFxPC --n 100000 --label ztfxpc-val-all
python -m project.eval.token_phase_probe --ckpts "$out/mae.pt" --data /projects/bfrf/data/PC_matches/ZTFxPC --n-stars 600 --label ztfxpc-val --out "project/results/token_phase/$NAME.json"
python -m project.eval.attn_probe --latents "$out/latents.pt" --test-latents "$out/latents_test.pt" --out "project/results/attn_probe_$NAME" --epochs 40
st=$?
if [ "${RF:-0}" = 1 ]; then
    python -m project.eval.classify_rf --latents "$out/latents.pt" --test-latents "$out/latents_test.pt" --features mean --n-jobs 16 \
        --skip-hpo --best-params '{"max_depth": null, "min_samples_split": 5, "n_estimators": 100}' --out "project/results/classify_rf_$NAME"
    st=$?
fi
echo "exit $st"
