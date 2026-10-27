#!/bin/bash
# Wait for a file, then submit a job, retrying while the QOS refuses (too many queued jobs).
#   nohup project/jobs/after_done.sh FILE sbatch ARGS... >> project/logs/after.log 2>&1 &
f=$1; shift
until [ -e "$f" ]; do sleep 120; done
until out=$("$@" 2>&1); do echo "$(date '+%F %T') refused: $out"; sleep 300; done
echo "$(date '+%F %T') $* -> $out"
