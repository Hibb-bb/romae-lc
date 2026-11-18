#!/bin/bash
# Submit jobs one after another from a list.
#
# The interactive partition allows one job per user at a time (running or
# waiting), so a second job cannot be queued behind the first and a chain
# script cannot queue its own next link. This helper runs on the login node.
# It reads a file with one sbatch command per line. When the user has no job
# in the queue, it submits the first line and removes it from the file. Lines
# can be added to the file while it runs. It stops when the file is empty.
# It only sleeps and polls, so it costs the login node nothing.
#
#   echo "sbatch project/jobs/gate.sh project/runs/mae_w250/mae.pt" >> project/jobs/queue.txt
#   nohup project/jobs/submit_queue.sh project/jobs/queue.txt >> project/logs/queue.log 2>&1 &
#
# A training that needs several 2 h links: write its line several times with
# MAX_LINKS=1. Every link resumes from last.pt, and a link that finds the run
# done exits at once.
Q=${1:?a file with one sbatch command per line}
cd /projects/bfrf/hibb/romae-lc || exit 1
echo "$(date '+%F %T') started on $(hostname), list $Q"
while true; do
    line=$(grep -v '^[[:space:]]*#' "$Q" 2>/dev/null | grep -v '^[[:space:]]*$' | head -1)
    if [ -z "$line" ]; then
        # a read error (a file-system hiccup) also gives an empty line: stop only after three empty reads a minute apart
        # (10-29: the helper quit on one bad read with 22 lines still in the list)
        empty=$(( ${empty:-0} + 1 ))
        if [ "$empty" -ge 3 ]; then
            echo "$(date '+%F %T') the list is empty: done"
            exit 0
        fi
        sleep 60
        continue
    fi
    empty=0
    if [ "$(squeue -u "$USER" -p "${QUEUE_PARTITION:-ghx4-interactive}" -h 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "$(date '+%F %T') submitting: $line"
        if out=$(eval "$line" 2>&1); then
            echo "$(date '+%F %T') $out"
            # drop the first command line of the file, keep the rest
            awk 'done || /^[[:space:]]*#/ || /^[[:space:]]*$/ {print; next} {done = 1}' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
        else
            echo "$(date '+%F %T') refused, will try again: $out"
        fi
    fi
    sleep 60
done
