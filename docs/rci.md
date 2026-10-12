# plyserve on CTU RCI

How to set up and run plyserve on the CTU RCI cluster (`login2.rci.cvut.cz`).
The values below were checked on 2026-10-11; recheck them if something fails.

## What RCI has

| Item | Value |
|---|---|
| Fast shared storage | `/mnt/personal/$USER` (10 TB, WekaFS, mounted on all nodes) |
| Home | `/home/$USER`, small quota: keep weights and caches out of it |
| OS | EL8, glibc 2.28; system Python 3.6, so uv provides Python |
| GPU for serving | H200 NVL, 141 GB, driver 575.51 (CUDA 12.9) |
| Partitions | `cpufast`, `gpufast` (V100), `amdgpufast` (A100), `h200fast`, `h200`, `h200long`, `h200extralong` |
| Containers | no Apptainer or Singularity, hence the uv environment in `runtime/vllm` |
| Network | compute nodes reach huggingface.co; jobs reach each other by host name (e.g. `n01`) |

## One-time setup

1. **uv and its cache.** Install uv in `~/.local/bin` and keep its cache on `/mnt/personal`: the runtime is
   ~10 GB, and a cache on the same file system lets uv hardlink instead of copying.

   ```bash
   echo 'export UV_CACHE_DIR=/mnt/personal/$USER/uv-cache' >> ~/.bashrc
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
   cd /mnt/personal/$USER && git clone <plyserve remote> && cd plyserve
   cp .env.example .env && chmod 600 .env
   ```

   In `.env`:

   ```bash
   PLYSERVE_HOME=/mnt/personal/$USER/plyserve-home
   HF_XET_HIGH_PERFORMANCE=1
   ```

   and in `local.mk` (for `make cpu`, `make gpu`, `make gpu-serve`; the `*fast` partitions have the shortest
   queues):

   ```make
   CPU_PARTITION = cpufast
   GPU_PARTITION = gpufast
   SERVE_PARTITION = h200fast
   ```

4. **Runtime and weights**, on the login node:

   ```bash
   make runtime
   make download MODEL=qwen-3.8-27b-fp8      # ~29 GB
   make download MODEL=gemma-4-31b-fp8       # ~62 GB (BF16 weights, quantized to FP8 at load)
   ```

## Check on a GPU node

```bash
make gpu-serve
# on the node:
runtime/vllm/.venv/bin/python -c "import torch, vllm; print(vllm.__version__, torch.version.cuda, torch.cuda.get_device_name())"
uv run plyserve serve qwen-3.8-27b-fp8 -- bash -c 'curl -s -H "Authorization: Bearer $VLLM_API_KEY" "$VLLM_BASE_URL/models"'
```

Expected: `0.31.0 12.9 NVIDIA H200 NVL`, then the vLLM log, `plyserve: qwen-3.8-27b-fp8 is ready at ...` and
the model list. On 2026-10-11 Qwen3.8-27B-FP8 took ~2 min 10 s from start to ready (weights 3 s, torch.compile
36 s, warmup 44 s, CUDA graphs ~12 s), used 27.6 GiB for weights and left 95 GiB for the KV cache
(~1.45 M tokens).

Do not run `serve` or `vllm serve --help` on the login node: it has no GPU, and vLLM fails to detect a device.

## Running

Batch jobs need the partition on the command line (or `export SBATCH_PARTITION=h200fast`):

```bash
# mode 1: server and client in one job
sbatch -p h200fast slurm/serve.sbatch qwen-3.8-27b-fp8 -- uv run --project ~/classifier python classify.py

# mode 2: server in one job, clients in others
sbatch -p h200fast slurm/serve.sbatch --expose qwen-3.8-27b-fp8
# in each client job (any partition):
eval "$(/mnt/personal/$USER/plyserve/.venv/bin/plyserve endpoint --wait qwen-3.8-27b-fp8)"
```

`h200fast` is the short-queue partition; longer runs go to `h200`, `h200long` or `h200extralong` with a matching
`-t`. Job logs go to `plyserve-<job id>.out` in the submit directory; run records to
`$PLYSERVE_HOME/runs/<job id>-<model>/server.json`.

## Known issues

- **CUDA 13 packages.** vLLM's default PyPI wheels need CUDA 13, newer than the 575 driver. `runtime/vllm`
  pins the CUDA 12.9 builds; torchaudio comes from the cu129 index and torchcodec as the CPU build. After a
  runtime upgrade, list libraries that still need CUDA 13 (the output should be empty):

  ```bash
  cd runtime/vllm/.venv/lib/python3.12/site-packages
  find . -name '*.so*' -type f | while read -r f; do ldd "$f" 2>/dev/null | grep -q '\.so\.13 => not found' && echo "$f"; done
  ```

- **No CUDA compiler on the nodes.** Kernels that vLLM compiles at first use (DeepGEMM, FlashInfer) fail with
  `NVCC compiler is not executable: /usr/local/cuda/bin/nvcc`. plyserve avoids them (see `models.toml`).
- **Caches in home.** vLLM's torch.compile cache goes to `~/.cache/vllm` (~31 MB per model so far).
- **Mode 2 checked** on 2026-10-12: server job on `h02` (H200), client on `n06`; `scancel` stopped vLLM and
  removed the endpoint file. Stop an `--expose` server with `scancel`, not by killing processes.
- **Not yet checked:** RCI's policy on long-lived listeners and tunnels.
