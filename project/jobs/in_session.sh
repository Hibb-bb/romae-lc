#!/bin/bash
# Run a command inside the running debug session (Slurm job named "debug"), from this checkout.
# Falls back to running it here when no session is up.   project/jobs/in_session.sh python -m pytest tests -q
J=$(squeue -u "$USER" -n debug -h -t R -o %i | head -1)
if [ -n "$J" ]; then
    exec srun --jobid="$J" --overlap --chdir="$PWD" "$@"
fi
echo "in_session: no debug session running, running here" >&2
exec "$@"
