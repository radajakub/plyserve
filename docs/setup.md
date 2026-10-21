# plyserve on a Slurm cluster

How to set up and run plyserve on a Slurm cluster. It was tested on CTU RCI only ([rci.md](rci.md) has the
values and results there); it should work on other clusters that meet the requirements below.

## What the cluster needs

Check each row before the setup. "Needed for" says which part fails without it.

| Requirement | Needed for | How to check |
|---|---|---|
| Linux x86_64, glibc >= 2.28 | everything | `uname -m` (`x86_64`), `ldd --version` |
| NVIDIA GPU on the serving nodes, driver >= 575 (CUDA 12.9) | serving | on a GPU node: `nvidia-smi` (driver version, "CUDA Version: 12.9" or higher) |
| GPU memory for the model | serving | the weights plus KV cache; the models in `models.toml` need ~30-60 GB of weights |
| A folder on fast storage that every node mounts | everything | create a file on the login node, `ls` it from a job on another node |
| Internet (PyPI, GitHub, huggingface.co) on the node that runs the setup | setup | `curl -sI https://huggingface.co` on the login node |
| [uv](https://docs.astral.sh/uv/) | everything | `uv --version`; it installs without root and provides Python |
| Slurm with GPUs as a GRES (`--gres=gpu:1`) | `slurm/serve_model.sbatch`, `make gpu` | `sinfo -o "%P %G"` lists the GPU partitions and their GRES |
| Compute nodes reach each other over TCP on any unprivileged port | mode 2 (`--expose`) | `nc -l 18080` in one job, `nc -z <its host> 18080` from another |
| Outbound SSH from compute nodes to the client machine | mode 3 (`--tunnel`) | from a job: `ssh <client machine> true` |
| A CUDA toolkit matching the runtime (12.9) and a C++20 compiler (GCC >= 10) on the serving nodes | vLLM | after loading them: `nvcc --version` (12.9), `c++ --version`; e.g. `module avail CUDA GCC` |

Not needed: root, containers (Apptainer, Docker), internet on the compute nodes (servers start offline from the
shared weights), or Slurm for `plyserve serve` itself (outside a job, run records are named by the start time).

Some sites need more on every job: an account (`-A`), a QOS (`-q`), or `--gpus=1` instead of `--gres=gpu:1`.
Pass them to `sbatch` and `srun` on the command line or through `SBATCH_*` variables (e.g. `SBATCH_ACCOUNT`).

Before you leave a server running for hours (mode 2) or open a tunnel to outside the cluster (mode 3), check the
cluster's rules on long-running listeners and outbound connections.

## One-time setup

Run these on the login node, unless the cluster asks for heavy work elsewhere. In that case, use a CPU job with
internet access. Below, `<shared>` is your folder on the shared storage.

1. **uv and its cache.** Keep uv's cache on the shared storage. The runtime is ~10 GB, and a cache on the same
   file system lets uv hardlink instead of copying:

   ```bash
   echo 'export UV_CACHE_DIR=<shared>/uv-cache' >> ~/.bashrc
   ```

   An SSH login reads `~/.bash_profile`, not `~/.bashrc`. If there is no `~/.bash_profile` (and no
   `~/.profile`), create one that reads `~/.bashrc`, then log in again and check `uv cache dir`:

   ```bash
   echo '[ -f ~/.bashrc ] && . ~/.bashrc' > ~/.bash_profile
   ```

2. **No conda base.** An active conda environment puts its own libraries first and breaks the runtime:

   ```bash
   conda config --set auto_activate_base false
   ```

3. **Clone and configure.**

   ```bash
   cd <shared> && git clone <plyserve remote> && cd plyserve
   cp .env.example .env && chmod 600 .env
   ```

   In `.env` (see the comments in `.env.example` for the rest):

   ```bash
   PLYSERVE_HOME=<shared>/plyserve-home
   HF_XET_HIGH_PERFORMANCE=1
   ```

   If the nodes reach each other only on one network (e.g. InfiniBand host names), also set
   `PLYSERVE_EXPOSE_HOST`.

   In `local.mk`, the partitions for `make cpu`, `make gpu`, `make gpu-serve` and `make serve-job`. Short-queue
   partitions are best, except `SERVE_LONG_PARTITION`, which takes multi-day jobs. An empty value leaves the choice
   to Slurm:

   ```make
   CPU_PARTITION = <a CPU partition>
   GPU_PARTITION = <any GPU partition>
   SERVE_PARTITION = <the partition with the GPU you serve on>
   SERVE_LONG_PARTITION = <the same GPU, for jobs of up to 3 days>
   ```

   In `slurm/site.sh`, the commands that put the CUDA toolkit and the compiler on `PATH`. `slurm/serve_model.sbatch`
   sources it before `plyserve serve`; `serve` stops with a message if `nvcc` or `c++` (or `$CXX`) is missing. With
   environment modules:

   ```bash
   module purge
   module load <CUDA 12.9 module> <GCC module>
   ```

4. **Runtime and weights:**

   ```bash
   make runtime                              # the pinned vLLM environment, ~10 GB
   uv run plyserve models                    # the models in models.toml
   make download MODEL=qwen-3.8-27b-fp8      # ~29 GB
   ```

   For Ollama models, also (it downloads ~1.4 GB, and uv fetches Python 3.14 once to unpack it):

   ```bash
   make runtime-ollama
   make download MODEL=qwen-3.8-27b-q8       # ~29 GB, one GGUF file
   ```

   `models.toml` lists the models we use. Add your own the same way: a Hugging Face repository, a pinned
   revision and a unique `tunnel_port`.

## Check on a GPU node

```bash
make gpu-serve
# on the node:
source slurm/site.sh && nvcc --version | tail -1 && c++ --version | head -1
nvidia-smi
runtime/vllm/.venv/bin/python -c "import torch, vllm; print(vllm.__version__, torch.version.cuda, torch.cuda.get_device_name())"
uv run plyserve serve qwen-3.8-27b-fp8 -- bash -c 'curl -s -H "Authorization: Bearer $VLLM_API_KEY" "$VLLM_BASE_URL/models"'
```

Expected:

- the vLLM version, `12.9` and your GPU's name;
- then the vLLM log, `plyserve: qwen-3.8-27b-fp8 is ready at ...`, and the model list.

The first start takes a few minutes (torch.compile, kernel compilation, warmup, CUDA graphs). Later starts reuse
the caches in `~/.cache` and are faster. In the vLLM log, `Selected FlashInferFp8DeepGEMM...` and `Using FlashInfer
GDN prefill kernel` (Qwen) show the fast kernels; `server.json` records the toolchain.

For Ollama:

```bash
uv run plyserve serve qwen-3.8-27b-q8 -- bash -c 'curl -s -H "Authorization: Bearer $VLLM_API_KEY" "$VLLM_BASE_URL/models"'
```

Expected:

- Ollama's log, with the CUDA library it picked and your GPU;
- then `plyserve: qwen-3.8-27b-q8 is ready at ...` and the model list.

In `server.json`, `memory` should show `size_vram` equal to `size`. The first start also copies the GGUF into
`$PLYSERVE_HOME/ollama`.

Do not run `serve` on a node without a GPU (e.g. the login node): vLLM fails to detect a device.

## Running

```bash
# mode 1: server and client in one job
sbatch -p <serve partition> slurm/serve_model.sbatch qwen-3.8-27b-fp8 -- uv run --project ~/classifier python classify.py

# mode 2: server in one job, clients in others
sbatch -p <serve partition> slurm/serve_model.sbatch --expose qwen-3.8-27b-fp8
# in each client job (any partition):
eval "$(<shared>/plyserve/.venv/bin/plyserve endpoint --wait qwen-3.8-27b-fp8)"

# mode 3: server in a job, client on your own machine (see "Tunnel" below)
sbatch -p <serve partition> slurm/serve_model.sbatch --tunnel qwen-3.8-27b-fp8
```

`slurm/serve_model.sbatch` asks for 1 GPU, 16 CPUs, 64 GB of memory and 4 hours. Override them on the command line
(e.g. `-t 1-00:00:00`), or submit through `make serve-job LENGTH=short|long ARGS='...'`: `short` is
`SERVE_PARTITION` for 4 hours, `long` is `SERVE_LONG_PARTITION` for 3 days, `JOB_TIME=...` overrides the time.
A client command runs in the plyserve clone, so give it its own directory (`uv run --directory <client> ...`).
Job logs go to `plyserve-<job id>.out` in the submit directory. Run records go to
`$PLYSERVE_HOME/runs/<job id>-<model>/server.json`.

Stop a server with `scancel`, not by killing its processes. `scancel` lets plyserve stop vLLM and remove the
endpoint file.

## Tunnel (mode 3)

The general setup of the client machine is in the README. The key and the saved host key must be readable from
the compute nodes. If `~/.ssh` is shared between the login and compute nodes (it usually is), make them once on
the login node:

```bash
ssh-keygen -t ed25519 -N "" -C plyserve-tunnel -f ~/.ssh/plyserve_tunnel
cat ~/.ssh/plyserve_tunnel.pub   # add it to the client machine's authorized_keys, limited as in the README
```

In `.env`:

```bash
PLYSERVE_API_KEY=<python3 -c "import secrets; print(secrets.token_urlsafe(32))">
PLYSERVE_TUNNEL_HOST=<user>@<client machine>
PLYSERVE_TUNNEL_KEY=~/.ssh/plyserve_tunnel
```

Before the first GPU job, check that a compute node reaches the client machine. This also saves the host key.
With `command="/bin/false"` it ends with exit code 1, which is expected:

```bash
make cpu
# on the node:
ssh -i ~/.ssh/plyserve_tunnel <user>@<client machine> true; echo "exit $?"
```

Then a test. It ends when the client command ends:

```bash
make gpu-serve
# on the node:
uv run plyserve serve --tunnel qwen-3.8-27b-fp8 -- sleep 600
# on the client machine, while it sleeps:
curl -s -H "Authorization: Bearer <PLYSERVE_API_KEY>" http://127.0.0.1:18001/v1/models
```

## Known issues

- **CUDA 13 packages.** vLLM's default PyPI wheels need CUDA 13, which needs a newer driver than 575.
  `runtime/vllm` pins the CUDA 12.9 builds: torchaudio comes from the cu129 index and torchcodec as the CPU build.
  After a runtime upgrade, list libraries that still need CUDA 13 (the output should be empty):

  ```bash
  cd runtime/vllm/.venv/lib/python3.12/site-packages
  find . -name '*.so*' -type f | while read -r f; do ldd "$f" 2>/dev/null | grep -q '\.so\.13 => not found' && echo "$f"; done
  ```

- **Kernel compilation.** vLLM compiles DeepGEMM and FlashInfer kernels on first use. Without a toolkit it fails
  with `NVCC compiler is not executable`, and GCC 8.5 (the EL8 system compiler) is too old for them. FlashInfer
  calls `ninja`, which plyserve finds in the runtime (`runtime/vllm/.venv/bin`, put first on `PATH`).
- **Caches in home.** vLLM's torch.compile cache goes to `~/.cache/vllm` (~31 MB per model), and the compiled
  kernels to `~/.cache` as well. Ollama creates `~/.ollama/id_ed25519` and prints its public key; this is harmless.
- **No `python` on the nodes.** On some clusters a client command `python ...` fails with "the client command
  'python' was not found". Use `uv run --project <client> python ...` or a full path.

## Adding a cluster

Copy the values you used into `docs/<cluster>.md`: the shared folder, the partitions, the GPUs, what you checked
and when, and any problems. [rci.md](rci.md) is an example.
