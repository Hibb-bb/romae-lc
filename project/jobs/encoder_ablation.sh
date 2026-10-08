#!/bin/bash
#SBATCH --job-name=enc-abl
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
# Two changes to what the stage-1 autoencoder is asked to do (project.mae_data),
# tested on the light encoder with the dense ladder, STEPS steps each, then
# cached and scored with the period probe. The control is rope_dense of
# rope_ablation.sh (random points hidden, one window length).
#   block   whole stretches of time hidden in every window
#   mix     half of the windows with stretches hidden, half with random points
#   range   training windows of 60 to 1000 d, random points hidden; scored
#           with the latents of four window lengths joined (125, 250, 500, 1000 d)
#   both    mix and range together
#   errw    random points hidden, every hidden point's squared error divided by its sigma^2
#   abs     absolute time features in the tokens (--abs-time, NeRF-style sinusoids on the ladder)
#   absbn   the same with the bottleneck objective (--bottleneck --bottleneck-var unit): the
#           pooled latent alone must reconstruct, so the phase has to be stored
#   phase   + the per-token phase objective (--phase-loss 1): every visible token predicts its phase
#   fold    + the phase-folded reconstruction objective (--fold-loss 1): the CLS alone predicts the
#           window's points at their phase, i.e. the model is asked to phase-fold
#   pfold   both objectives
#   long    the whole light curve as one window (2000 d, 1024 tokens), plain MAE
#   spec    + the spectral layer (romae_lc.spectral: a learned periodogram over the tokens, 20k-bin grid,
#           its summary on the CLS) after encoder block 2, plain reconstruction loss
#   specaux the same plus the auxiliary cross-entropy on the catalogue period's bin (weight 0.5)
#   hidden  the whole-curve input with one hidden stretch (an eighth of the points, ~ a 250 d window) plus
#           random points (a quarter hidden in all): the hidden-window objective, plain encoder
#   hiddensp the same with the spectral layer (+ aux 0.5): mechanism + pressure
# Every variant also gets the phase probe (project/results/phase_enc_<name>).
# The variants to run are the arguments (default: all four). A run that is
# already done is skipped.
#   sbatch project/jobs/encoder_ablation.sh block mix
#   sbatch project/jobs/encoder_ablation.sh range both
set -uo pipefail
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate
STEPS=${STEPS:-15000}
LENGTHS=${LENGTHS:-"125 500 1000"}   # the extra window lengths of the joined features
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
run() {
    local name=$1 multi=$2; shift 2
    local out=project/runs/enc_$name
    echo "#### $name: $*"
    if [ ! -f "$out/DONE" ]; then
        python -m project.pretrain_mae --out "$out" --window 250 --min-tokens 16 --max-tokens 256 \
            --steps "$STEPS" --batch-size 64 --workers 8 --pin-memory \
            --eval-every 2500 --ckpt-every 500 --wandb --wandb-group encoder-ablation "$@" || return
    fi
    [ -f "$out/latents.pt" ] || python -m project.cache_latents --ckpt "$out/mae.pt" --out "$out/latents.pt" --workers 8 || return
    local extra=()
    if [ "$multi" = multi ]; then
        for w in $LENGTHS; do
            [ -f "$out/latents_w$w.pt" ] || python -m project.cache_latents --ckpt "$out/mae.pt" \
                --out "$out/latents_w$w.pt" --window "$w" --workers 8 || return
            extra+=("$out/latents_w$w.pt")
        done
    fi
    python -m project.period_probe --latents "$out/latents.pt" --out "project/results/period_enc_$name" \
        --n-ls 50 --no-fold --device cuda ${extra:+--extra-latents "${extra[@]}"}
    python -m project.phase_probe --latents "$out/latents.pt" --out "project/results/phase_enc_$name" --device cuda
}
run_long() {
    # the whole light curve as one window: 2000 d, up to 1024 points, one window per object per batch;
    # extra arguments go to pretrain_mae (the hidden-window objectives: --mask-mode blockplus ...)
    local name=$1; shift; local out=project/runs/enc_$name
    echo "#### $name: window 2000 d, 1024 tokens, $*"
    if [ ! -f "$out/DONE" ]; then
        python -m project.pretrain_mae --out "$out" --window 2000 --n-frames 1 --min-tokens 64 --max-tokens 1024 --lam-max 4000 \
            --steps "$STEPS" --batch-size 32 --workers 8 --pin-memory \
            --eval-every 2500 --ckpt-every 500 --wandb --wandb-group encoder-ablation "$@" || return
    fi
    [ -f "$out/latents.pt" ] || python -m project.cache_latents --ckpt "$out/mae.pt" --out "$out/latents.pt" --workers 8 --batch-size 64 || return
    python -m project.period_probe --latents "$out/latents.pt" --out "project/results/period_enc_$name" --n-ls 50 --no-fold --device cuda --bins 2400
    python -m project.phase_probe --latents "$out/latents.pt" --out "project/results/phase_enc_$name" --device cuda
}
for v in ${@:-block mix range both}; do
    case "$v" in
        block) run block single --mask-mode block ;;
        mix)   run mix single --mask-mode mix ;;
        range) run range multi --window-range 60 1000 ;;
        both)  run both multi --mask-mode mix --window-range 60 1000 ;;
        errw)  run errw single --loss-weight err ;;
        abs)   run abs single --abs-time ;;
        phase) run phase single --phase-loss 1.0 ;;
        fold)  run fold single --fold-loss 1.0 ;;
        pfold) run pfold single --phase-loss 1.0 --fold-loss 1.0 ;;
        spec)  run spec single --spectral ;;
        specaux) run specaux single --spectral --spectral-aux 0.5 ;;
        specaux50) run specaux50 single --spectral --spectral-aux 0.5 ;;   # the same at STEPS=50000
        # token-level JEPA against the masked autoencoder, same wide spectral encoder, same masks (a stretch
        # plus random points), 50k steps each (STEPS=50000; the training itself runs as a pretrain_mae.sh chain
        # into project/runs/enc_<name>, this script then finds DONE and only caches and probes)
        jepa)      run jepa single --size wide --spectral --jepa --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 ;;
        maespec50) run maespec50 single --size wide --spectral --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 ;;
        longspecw) run_long longspecw --size wide --spectral --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 ;;   # the whole curve as one window, wide spectral (STEPS=50000; trained by pretrain_mae.sh lines)
        longspecwtpl) run_long longspecwtpl --size wide --spectral --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 --target mix ;;   # the same with the smooth fit as the target
        jepainit)  run jepainit single --size wide --spectral --jepa --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 --jepa-init project/runs/maew_spec/mae.pt ;;
        jepahyb)   run jepahyb single --size wide --spectral --jepa --mask-mode blockplus --mask-ratio 0.5 --mask-block-share 0.5 --jepa-recon-weight 0.5 ;;
        long)  run_long long ;;
        hidden)   run_long hidden --mask-mode blockplus --mask-ratio 0.25 --mask-block-share 0.5 ;;
        hiddensp) run_long hiddensp --mask-mode blockplus --mask-ratio 0.25 --mask-block-share 0.5 --spectral --spectral-aux 0.5 ;;
        absbn) run absbn single --abs-time --bottleneck --bottleneck-var unit ;;
        *) echo "unknown variant $v" ;;
    esac
done
