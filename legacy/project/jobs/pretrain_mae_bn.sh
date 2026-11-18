#!/bin/bash
#SBATCH --job-name=bn-w250
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
# Stage 1, the light bottleneck autoencoder: the light preset (192 wide, 3
# heads, 6 deep) with the query decoder reading the pooled latent only
# (--bottleneck) and the encoder input redrawn from the per-point errors
# (--denoise), 50k steps: the control for pretrain_mae_wide.sh that
# separates the bottleneck from the width. Two links (the second finds DONE
# and submits THEN, by default the wide run). Usable as a THEN / PIPELINE
# stage after another chain (a *.pt argument is dropped by pretrain_mae.sh).
#   sbatch --job-name=bn project/jobs/pretrain_mae_bn.sh
#   THEN=project/jobs/pretrain_mae_bn.sh sbatch --export=ALL,THEN=project/jobs/pretrain_mae_bn.sh ... project/jobs/train_decoder.sh mae.pt
export OUT=${OUT:-project/runs/bn_w250} MAX_LINKS=${MAX_LINKS:-2}
export THEN=${THEN:-project/jobs/pretrain_mae_wide.sh}
exec bash project/jobs/pretrain_mae.sh "$@" --bottleneck --denoise
