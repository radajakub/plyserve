# plyserve

Serve open-weight models with [vLLM](https://docs.vllm.ai) in Slurm jobs, for OpenAI-compatible clients in
the same job, in another job on the cluster, or (planned) on another machine through an SSH tunnel.

plyserve starts and watches vLLM; it never imports it. The models, their pinned revisions and their vLLM flags
live in `models.toml`; the vLLM environment is pinned in `runtime/vllm/`. Cluster-specific values live in
`.env` and `local.mk`, never in the code. Setup on a known cluster: [docs/rci.md](docs/rci.md) (CTU RCI).

## Requirements

- Linux x86_64 with an NVIDIA GPU whose driver supports CUDA 12.9 (driver >= 575) and glibc >= 2.28.
- [uv](https://docs.astral.sh/uv/) on the login node; it provides Python.
- A folder on fast storage that every node mounts, for weights, run records and endpoint files.

## Setup

```bash
git clone <this repository> && cd plyserve
cp .env.example .env && chmod 600 .env   # set PLYSERVE_HOME, see the comments in the file
make runtime                             # the pinned vLLM environment, ~10 GB in runtime/vllm/.venv
uv run plyserve models                   # the models in models.toml
make download MODEL=qwen-3.8-27b-fp8     # weights at the pinned revision into $PLYSERVE_HOME/hf
```

Optional: a `local.mk` with your cluster's partitions for the interactive `make cpu`, `make gpu` and
`make gpu-serve` shells:

```make
CPU_PARTITION = ...
GPU_PARTITION = ...
SERVE_PARTITION = ...
```

## Configuration (the contract)

| Variable | Read by | Meaning |
|---|---|---|
| `PLYSERVE_HOME` | plyserve | Required. Shared folder: `hf/` (weights), `runs/` (one `server.json` per run), `endpoints/` |
| `PLYSERVE_API_KEY` | plyserve | Optional fixed API key; unset, every run makes a new one |
| `PLYSERVE_EXPOSE_BIND` | `serve --expose` | Address vLLM listens on; default `0.0.0.0` |
| `PLYSERVE_EXPOSE_HOST` | `serve --expose` | Host name written for clients; default the node's host name |
| `PLYSERVE_REPO` | `slurm/serve.sbatch` | The clone, if the job is not submitted from it |
| `HF_TOKEN`, `HF_XET_*` | downloads | Passed to the Hugging Face CLI |

They come from `.env` in the clone or from the environment; the environment wins. A client sees only:

| Variable | Meaning |
|---|---|
| `VLLM_BASE_URL` | OpenAI-compatible base URL, ending in `/v1` |
| `VLLM_API_KEY` | Bearer token for every request |
| `VLLM_MODEL` | The served name to send as `model`, the key in `models.toml` |

## Modes

**1. Server and client in one job.** The client gets the variables; vLLM stops when the client exits, and the
job exits with the client's exit code.

```bash
sbatch -p <partition> slurm/serve.sbatch qwen-3.8-27b-fp8 -- python classify.py
# or, in an interactive GPU shell:
uv run plyserve serve qwen-3.8-27b-fp8 -- python classify.py
```

**2. Server and client in separate jobs.** `--expose` makes vLLM listen for other nodes and, once it answers,
writes `$PLYSERVE_HOME/endpoints/<model>.json` (owner-only, as it holds the key). The file is removed when the
server stops; a second `--expose` server for the same model is refused while the first one answers.

```bash
sbatch -p <partition> slurm/serve.sbatch --expose qwen-3.8-27b-fp8
# in the client job, which may start before the server is ready:
eval "$(/path/to/plyserve/.venv/bin/plyserve endpoint --wait qwen-3.8-27b-fp8)"
python classify.py
```

`plyserve endpoint` checks that the server answers `/health` before printing the exports; `--wait` waits for
that (`--timeout` limits the wait). In mode 2 other users of the cluster can reach the port, so the API key is
what protects the server.

**3. Server on the cluster, client elsewhere (planned).** `serve --tunnel` will open an SSH reverse tunnel to
the client machine.

Without a client command, `serve` runs until vLLM exits or the job ends. Every run writes `server.json` to
`$PLYSERVE_HOME/runs/<job id>-<model>/`: model, revision, vLLM arguments and environment, package versions,
plyserve commit, host and GPUs. It never contains the key.

## Notes

- vLLM applies each model's `generation_config.json` sampling defaults (for Qwen3.8: temperature 1.0,
  top_k 20, top_p 0.95). Clients that need reproducible outputs should send `temperature` and `seed`.
- The runtime has no CUDA compiler, so plyserve and `models.toml` steer vLLM to prebuilt kernels
  (CUTLASS, Triton); see the comments in `models.toml` and `RUNTIME_ENV` in `src/plyserve/serve.py`.
- Development: `make test`, `make lint`, `make format`.
