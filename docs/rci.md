# plyserve on CTU RCI

The values and results for the CTU RCI cluster (`login2.rci.cvut.cz`). Follow [setup.md](setup.md) with the
values below. They were checked on 2026-10-11; recheck them if something fails.

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
| `~/.ssh` | the same on the login and compute nodes |
| `python` | none on the nodes: use `uv run --project <client> python ...` in client commands |

## Values for the setup

`<shared>` is `/mnt/personal/$USER`:

```bash
echo 'export UV_CACHE_DIR=/mnt/personal/$USER/uv-cache' >> ~/.bashrc
cd /mnt/personal/$USER && git clone <plyserve remote> && cd plyserve
```

In `.env`:

```bash
PLYSERVE_HOME=/mnt/personal/$USER/plyserve-home
HF_XET_HIGH_PERFORMANCE=1
```

In `local.mk` (the `*fast` partitions have the shortest queues):

```make
CPU_PARTITION = cpufast
GPU_PARTITION = gpufast
SERVE_PARTITION = h200fast
```

Batch jobs need the partition on the command line (or `export SBATCH_PARTITION=h200fast`):

```bash
sbatch -p h200fast slurm/serve.sbatch qwen-3.8-27b-fp8 -- uv run --project ~/classifier python classify.py
```

`h200fast` is the short-queue partition. Longer runs go to `h200`, `h200long` or `h200extralong` with a matching
`-t`. In client jobs of mode 2, the endpoint command is
`/mnt/personal/$USER/plyserve/.venv/bin/plyserve endpoint --wait <model>`.

## Results

- **vLLM checked** on 2026-10-11: the GPU-node check printed `0.31.0 12.9 NVIDIA H200 NVL`. Qwen3.8-27B-FP8 took
  ~2 min 10 s from start to ready (weights 3 s, torch.compile 36 s, warmup 44 s, CUDA graphs ~12 s). It used
  27.6 GiB for weights and left 95 GiB for the KV cache (~1.45 M tokens). On 2026-10-12, with the torch.compile
  cache in `~/.cache/vllm`, the same model was ready in ~45 s.
- **Mode 2 checked** on 2026-10-12: server job on `h02` (H200), client on `n06`; `scancel` stopped vLLM and
  removed the endpoint file.
- **Mode 3 checked** on 2026-10-12: `serve --tunnel` in an H200 job; `/v1/models` answered on the VPS at
  `127.0.0.1:18001` with the fixed key. `--expose --tunnel` in one job then served a client job and the VPS at
  the same time (job 11785372); the tunnel did not drop, and `scancel` stopped it. The tunnel test took ~3 min
  on `h200fast`.
- **Ollama checked** on 2026-10-13 (job 11789085, H200): Ollama 0.40.3 picked its bundled `cuda_v12` libraries
  with driver 575 (CUDA 12.9) and put all 65 layers of `qwen-3.8-27b-q8` on the GPU (~30 GiB with a 64k context).
  Ready ~61 s after start: hashing the GGUF ~15 s, copying it into `$PLYSERVE_HOME/ollama` ~25 s (first start only),
  loading 11 s. Ollama runs llama.cpp's `llama-server` and renders prompts with the GGUF's own chat template.
  Harmless message: "GPU discovery watchdog timed out" only means it kept the free-memory figures from its start.
- **Not yet checked:** reconnecting after a real drop, and RCI's policy on long-lived listeners and tunnels.
