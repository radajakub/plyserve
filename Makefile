.PHONY: test lint format fix runtime download cpu gpu gpu-h200

test:
	uv run --group dev pytest -q

lint:
	uv run --group dev ruff check .
	uv run --group dev ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

# Alias for format
fix: format

# The pinned vLLM environment (runtime/vllm/.venv, ~10 GB). Linux x86_64 only: run it on RCI
runtime:
	uv sync --project runtime/vllm

# Weights at the pinned revision into $PLYSERVE_HOME/hf: make download MODEL=qwen-3.8-27b-fp8
download:
	@test -n "$(MODEL)" || (echo "usage: make download MODEL=<served name>  (uv run plyserve models lists them)" >&2; exit 1)
	uv run plyserve download $(MODEL)

# Interactive shells on RCI compute nodes, for trying things out by hand. Run them on the login node.
# Every value can be overridden: make gpu-h200 TIME=0:30:00 MEM=128G
TIME ?= 1:00:00

# CPU node on the 4-hour partition
cpu: PARTITION ?= cpufast
cpu: CPUS ?= 4
cpu: MEM ?= 16G
cpu:
	srun -p $(PARTITION) -t $(TIME) -c $(CPUS) --mem=$(MEM) --pty bash -i

# one V100 on the 4-hour partition, the shortest queue for checking that something sees a GPU
gpu: PARTITION ?= gpufast
gpu: CPUS ?= 8
gpu: MEM ?= 32G
gpu: GPUS ?= 1
gpu:
	srun -p $(PARTITION) -t $(TIME) -c $(CPUS) --mem=$(MEM) --gres=gpu:$(GPUS) --pty bash -i

# one H200 NVL on the 4-hour partition, the GPU the models are served on
gpu-h200: PARTITION ?= h200fast
gpu-h200: CPUS ?= 16
gpu-h200: MEM ?= 64G
gpu-h200: GPUS ?= 1
gpu-h200:
	srun -p $(PARTITION) -t $(TIME) -c $(CPUS) --mem=$(MEM) --gres=gpu:$(GPUS) --pty bash -i
