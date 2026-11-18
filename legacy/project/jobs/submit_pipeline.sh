#!/bin/bash
# Submit a training run and its probe job to the batch partition, the probe
# depending on the training job (afterok). Several pipelines can be in the
# queue at once there (no one-job limit, 2-day wall time).
#
#   project/jobs/submit_pipeline.sh NAME [pretrain_mae args...]
#
# NAME is the encoder_ablation.sh variant (enc_NAME is the run directory);
# the training job is pretrain_mae.sh with OUT=project/runs/enc_NAME and
# the given args; the probe job is encoder_ablation.sh NAME at STEPS=50000.
set -euo pipefail
cd /projects/bfrf/hibb/romae-lc
NAME=${1:?variant name}; shift
OUT=project/runs/enc_$NAME
train=$(sbatch --parsable --job-name="$NAME" --export=ALL,OUT="$OUT",MAX_LINKS=1,BUDGET="${BUDGET:-40000}" project/jobs/pretrain_mae.sh "$@")
probe=$(sbatch --parsable --job-name="$NAME-probe" --dependency=afterok:"$train" --export=ALL,STEPS="${STEPS:-50000}" project/jobs/encoder_ablation.sh "$NAME")
echo "$NAME: training job $train, probe job $probe (after it)"
