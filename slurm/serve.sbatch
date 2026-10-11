#!/bin/bash
# Serve a model and run a client against it in one job (mode 1):
#   sbatch slurm/serve.sbatch qwen-3.8-27b-fp8 uv run --project ~/classifier python classify.py
# Without a client command it serves until the time limit. Resources can be overridden on the command line:
#   sbatch -t 1-00:00:00 -p h200 slurm/serve.sbatch ...
# Submit from the plyserve clone, or set PLYSERVE_REPO to it. The client runs in the submit directory.
#SBATCH --job-name=plyserve
#SBATCH --partition=h200fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=plyserve-%j.out
set -euo pipefail

repo=${PLYSERVE_REPO:-$SLURM_SUBMIT_DIR}
if [[ ! -x $repo/.venv/bin/plyserve ]]; then
    echo "no $repo/.venv/bin/plyserve: submit from the plyserve clone (or set PLYSERVE_REPO) after running \`uv sync\` there" >&2
    exit 2
fi
if [[ $# -lt 1 ]]; then
    echo "usage: sbatch slurm/serve.sbatch <model> [client command...]" >&2
    exit 2
fi
model=$1
shift

# the installed entry point rather than `uv run`, so the job never syncs the environment
exec "$repo/.venv/bin/plyserve" serve "$model" -- "$@"
