#!/bin/bash
# `plyserve serve` as a batch job; the arguments are those of `plyserve serve`.
#   server and client in one job:  sbatch slurm/serve.sbatch qwen-3.8-27b-fp8 -- uv run --project ~/classifier python classify.py
#   server for clients elsewhere:  sbatch slurm/serve.sbatch --expose qwen-3.8-27b-fp8
# The resources below fit one ~30B FP8 model on one GPU. Choose the partition (and override anything else) on
# the command line or through Slurm's SBATCH_* variables: sbatch -p <partition> -t 1-00:00:00 slurm/serve.sbatch ...
# Submit from the plyserve clone, or set PLYSERVE_REPO to it. A client command runs in the submit directory.
#SBATCH --job-name=plyserve
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
    echo "usage: sbatch slurm/serve.sbatch [--expose] <model> [-- client command...]" >&2
    exit 2
fi

# the installed entry point rather than `uv run`, so the job never syncs the environment
exec "$repo/.venv/bin/plyserve" serve "$@"
