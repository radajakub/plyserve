# plyserve

Serve open-weight models with [vLLM](https://docs.vllm.ai) or [Ollama](https://ollama.com) in Slurm jobs, for OpenAI-compatible clients in
the same job, in another job on the cluster, or on another machine through an SSH tunnel.

plyserve starts and watches vLLM or Ollama; it never imports them. The models, their pinned revisions and their
flags live in `models.toml`; the vLLM environment is pinned in `runtime/vllm/`, the Ollama release in the `Makefile`. Cluster-specific values live in
`.env`, `local.mk` and `slurm/site.sh`, never in the code. Tested on CTU RCI; it should work on any Slurm cluster that meets the
requirements below. Full setup guide: [docs/setup.md](docs/setup.md); values for RCI: [docs/rci.md](docs/rci.md).

## Requirements

- Linux x86_64 with an NVIDIA GPU whose driver supports CUDA 12.9 (driver >= 575) and glibc >= 2.28.
- [uv](https://docs.astral.sh/uv/) on the login node; it provides Python.
- A folder on fast storage that every node mounts, for weights, run records and endpoint files.
- Internet (PyPI, GitHub, huggingface.co) on the node that runs the setup. Compute nodes need none: servers start offline.
- Slurm with GPUs as a GRES, for `slurm/serve_model.sbatch` and the `make` shells.
- Mode 2: compute nodes reach each other over TCP. Mode 3: compute nodes may open SSH connections to the client machine.
- For vLLM: a CUDA toolkit matching the runtime (12.9) and a C++20 compiler (GCC >= 10) on the serving nodes. vLLM
  compiles its fastest kernels (DeepGEMM, FlashInfer) with them on first use.

No root, containers or internet on the compute nodes are needed. [docs/setup.md](docs/setup.md) shows how to check
each requirement.

## Setup

```bash
git clone <this repository> && cd plyserve
cp .env.example .env && chmod 600 .env   # set PLYSERVE_HOME, see the comments in the file
make runtime                             # the pinned vLLM environment, ~10 GB in runtime/vllm/.venv
uv run plyserve models                   # the models in models.toml
make download MODEL=qwen-3.8-27b-fp8     # weights at the pinned revision into $PLYSERVE_HOME/hf
make runtime-ollama                      # optional, for Ollama models: the pinned release (1.4 GB download) in runtime/ollama
```

A `slurm/site.sh` (gitignored) that puts the CUDA toolkit and the compiler on `PATH`; `slurm/serve_model.sbatch`
sources it, and an interactive shell needs `source slurm/site.sh` before `plyserve serve`. On a cluster with
environment modules, for example:

```bash
module load CUDA/12.9.1 GCC/13.3.0
```

Optional: a `local.mk` with your cluster's partitions for the interactive `make cpu`, `make gpu` and
`make gpu-serve` shells, and for batch jobs through `make serve-job`:

```make
CPU_PARTITION = ...
GPU_PARTITION = ...
SERVE_PARTITION = ...
SERVE_LONG_PARTITION = ...
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
| `PLYSERVE_MANIFEST` | plyserve | Optional `models.toml` to read instead of the clone's, e.g. a variant for one experiment; `--manifest` wins |
| `PLYSERVE_REPO` | `slurm/serve_model.sbatch` | The clone, if the job is not submitted from it |
| `HF_TOKEN`, `HF_XET_*` | downloads | Passed to the Hugging Face CLI |

They come from `.env` in the clone or from the environment; the environment wins. A client sees only:

| Variable | Meaning |
|---|---|
| `VLLM_BASE_URL` | OpenAI-compatible base URL, ending in `/v1` |
| `VLLM_API_KEY` | Bearer token for every request |
| `VLLM_MODEL` | The served name to send as `model`, the key in `models.toml` |
| `OLLAMA_BASE_URL`, `OLLAMA_API_KEY` | Ollama models only: the same URL and key, for plyllm's `ollama` provider |

In modes 1 and 2 plyserve hands these over at run time (with a new key each run unless `PLYSERVE_API_KEY` is set),
so a client's `.env` needs none of them. Only a client in mode 3 sets them itself, from the tunnel port and the
fixed key.

## Modes

**1. Server and client in one job.** The client gets the variables; vLLM stops when the client exits, and the
job exits with the client's exit code.

```bash
sbatch -p <partition> slurm/serve_model.sbatch qwen-3.8-27b-fp8 -- python classify.py
# or, in an interactive GPU shell:
uv run plyserve serve qwen-3.8-27b-fp8 -- python classify.py
```

**2. Server and client in separate jobs.** `--expose` makes vLLM listen for other nodes and, once it answers,
writes `$PLYSERVE_HOME/endpoints/<model>.json` (owner-only, as it holds the key). The file is removed when the
server stops; a second `--expose` server for the same model is refused while the first one answers.

```bash
sbatch -p <partition> slurm/serve_model.sbatch --expose qwen-3.8-27b-fp8
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
sbatch -p <partition> slurm/serve_model.sbatch --tunnel qwen-3.8-27b-fp8
# on the client machine:
export VLLM_BASE_URL=http://127.0.0.1:18001/v1 VLLM_API_KEY=<PLYSERVE_API_KEY> VLLM_MODEL=qwen-3.8-27b-fp8
# for an Ollama model (plyllm's ollama provider), on its own tunnel_port:
export OLLAMA_BASE_URL=http://127.0.0.1:18003/v1 OLLAMA_API_KEY=<PLYSERVE_API_KEY>
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

`serve --monitor` also samples the GPU and CPU into the same folder while the server runs: `gpu.csv` (nvidia-smi
every 2 s), `top.txt` (the user's processes every 5 s) and, where DCGM is installed, `dcgm.txt` (SM, tensor and DRAM
activity every 2 s, with timestamps) and `dcgm-gpus.txt` (DCGM's GPU ids and bus ids; DCGM lists every GPU of the
node, `gpu.csv` gives the job's bus ids). nvidia-smi's `utilization.gpu` reads 100% whenever any kernel runs, so
DCGM is what says how much of the GPU the server uses. `server.json` lists the files and what was skipped.

## Ollama

A model with `backend = "ollama"` in `models.toml` runs on Ollama instead of vLLM. Everything above works the
same: `serve`, the three modes, `slurm/serve_model.sbatch`, `endpoint` and the `VLLM_*` variables of the client, which
also gets `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` for plyllm's `ollama` provider.

```bash
make download MODEL=qwen-3.8-27b-q8      # only the GGUF file named in models.toml
sbatch -p <partition> slurm/serve_model.sbatch qwen-3.8-27b-q8 -- python classify.py
```

- **Weights.** One GGUF file from a Hugging Face repository at a pinned revision. At start, `ollama create`
  registers it under the served name in `$PLYSERVE_HOME/ollama`, with the `parameters` from `models.toml` as
  the Modelfile. The first start copies the file there (the disk use doubles); later starts only hash it.
- **API key.** Ollama has none. It listens on 127.0.0.1 only, and plyserve puts a small proxy in front that
  checks the key and forwards every request (`/v1` and Ollama's own `/api`). The proxy's `/health` answers
  for clients and endpoint files, as vLLM's does.
- **GPU only.** Ollama puts on the CPU what does not fit on the GPU. `serve` then stops with a message, so a
  run never goes on many times slower. Lower `num_ctx` or take a smaller GGUF.
- **Throughput.** Ollama serves one request at a time unless `OLLAMA_NUM_PARALLEL` says otherwise (see
  `.env.example`); vLLM batches many.
- **Run record.** `server.json` holds the Ollama version, its `OLLAMA_*` variables, the Modelfile as Ollama
  resolved it (with the chat template, or the built-in renderer and parser) and the bytes on the GPU.

Comparing Ollama with vLLM on "the same" model, these differ unless set on purpose:

- **Quantization.** A GGUF (e.g. Q8_0) is not the FP8 checkpoint vLLM serves; for the closest match use
  the BF16 checkpoint and GGUF on both.
- **Sampling.** vLLM applies the model's `generation_config.json`; Ollama does not read it. Ollama's `/v1`
  sets `temperature` and `top_p` to 1.0 when a request leaves them out, whatever the Modelfile says. Send
  `temperature`, `top_p` and `seed` in every request; `top_k` comes from `parameters` in `models.toml`.
- **Chat template.** vLLM renders the Jinja template of the Hugging Face repository with Python's Jinja;
  Ollama 0.40.3 hands the GGUF's own template to llama.cpp's Jinja engine (log: `selected=gguf_chat_template`).
  For `qwen-3.8-27b-q8` the GGUF template is byte-identical to Qwen's (checked 2026-10-13). Its variables
  (`enable_thinking`, `reasoning_effort`, default `xhigh`) still depend on what each server passes; the
  `usage.prompt_tokens` of one request sent to both servers shows whether the prompts match.
- **Context length.** Ollama's default context depends on the VRAM; set `num_ctx` to vLLM's `--max-model-len`.

## Notes

- vLLM applies each model's `generation_config.json` sampling defaults (for Qwen3.8: temperature 1.0,
  top_k 20, top_p 0.95). Clients that need reproducible outputs should send `temperature` and `seed`.
- vLLM picks its kernels itself and compiles some on first use, so a first start takes a few minutes longer.
  `serve` stops before loading the model if `nvcc` or a C++ compiler is missing, and records both versions in
  `server.json`: the kernels decide the numerics of the answers.
- Development: `make test`, `make lint`, `make format`.
