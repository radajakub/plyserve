# plyserve

Serve open-weight models with [vLLM](https://docs.vllm.ai) in Slurm jobs, for OpenAI-compatible clients in
the same job, in another job on the cluster, or on another machine through an SSH tunnel.

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
| `PLYSERVE_API_KEY` | plyserve | Optional fixed API key; unset, every run makes a new one. Required by `--tunnel` |
| `PLYSERVE_EXPOSE_BIND` | `serve --expose` | Address vLLM listens on; default `0.0.0.0` |
| `PLYSERVE_EXPOSE_HOST` | `serve --expose` | Host name written for clients; default the node's host name |
| `PLYSERVE_TUNNEL_HOST` | `serve --tunnel` | Required for it. SSH destination: `user@host` or a `Host` alias |
| `PLYSERVE_TUNNEL_KEY` | `serve --tunnel` | Optional key for the tunnel; default: ssh's own configuration |
| `PLYSERVE_TUNNEL_BIND` | `serve --tunnel` | Address the remote machine listens on; default `127.0.0.1` |
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

**3. Server on the cluster, client elsewhere.** `--tunnel` opens an SSH reverse tunnel from the job to the
client machine, once vLLM answers. There, `127.0.0.1:<tunnel_port>` (from `models.toml`) forwards to the server,
and the client uses the fixed key:

```bash
sbatch -p <partition> slurm/serve.sbatch --tunnel qwen-3.8-27b-fp8
# on the client machine:
export VLLM_BASE_URL=http://127.0.0.1:18001/v1 VLLM_API_KEY=<PLYSERVE_API_KEY> VLLM_MODEL=qwen-3.8-27b-fp8
```

If ssh fails in the first 10 s (host, key, unknown host key, port taken), `serve` stops with ssh's message above.
If the tunnel drops later, plyserve reconnects after 1 s, 2 s, 4 s, ... up to 60 s and vLLM keeps serving.
`--tunnel` combines with `--expose` and with a client command.

Setup on the client machine, once:

1. A key only for the tunnel, without a passphrase, readable on the cluster (`ssh-keygen -t ed25519 -N "" -f
   <path>`; set `PLYSERVE_TUNNEL_KEY=<path>`).
2. In the client machine's `~/.ssh/authorized_keys`, the public key limited to forwarding the models' ports
   (OpenSSH >= 7.8):

   ```
   restrict,port-forwarding,permitlisten="127.0.0.1:18001",permitlisten="127.0.0.1:18002",command="/bin/false" ssh-ed25519 AAAA... plyserve
   ```

3. Recommended in its `sshd_config`: `ClientAliveInterval 30` and `ClientAliveCountMax 3`. A dropped tunnel
   otherwise keeps its port until sshd notices, and reconnects fail until then.
4. From the cluster, accept the host key once: `ssh -i <path> <PLYSERVE_TUNNEL_HOST> true` (it prints nothing
   and fails, because of `/bin/false`; the host key is saved).

Anyone on the client machine can reach `127.0.0.1:<tunnel_port>`, so the API key protects the server there too.

Without a client command, `serve` runs until vLLM exits or the job ends. Every run writes `server.json` to
`$PLYSERVE_HOME/runs/<job id>-<model>/`: model, revision, vLLM arguments and environment, package versions,
plyserve commit, host and GPUs. It never contains the key.

## Notes

- vLLM applies each model's `generation_config.json` sampling defaults (for Qwen3.8: temperature 1.0,
  top_k 20, top_p 0.95). Clients that need reproducible outputs should send `temperature` and `seed`.
- The runtime has no CUDA compiler, so plyserve and `models.toml` steer vLLM to prebuilt kernels
  (CUTLASS, Triton); see the comments in `models.toml` and `RUNTIME_ENV` in `src/plyserve/serve.py`.
- Development: `make test`, `make lint`, `make format`.
