.PHONY: test lint format fix runtime download cpu gpu gpu-serve

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

# Interactive shells on Slurm compute nodes, for trying things out by hand. Run them on the login node.
# Every value can be overridden: make gpu-serve TIME=0:30:00 MEM=128G
# The partitions are site settings: set CPU_PARTITION, GPU_PARTITION and SERVE_PARTITION in local.mk
# (gitignored; docs/ has the values for known clusters). An empty one leaves the choice to Slurm.
-include local.mk
TIME ?= 1:00:00

# a CPU node
cpu: PARTITION ?= $(CPU_PARTITION)
cpu: CPUS ?= 4
cpu: MEM ?= 16G
cpu:
	srun $(if $(PARTITION),-p $(PARTITION)) -t $(TIME) -c $(CPUS) --mem=$(MEM) --pty bash -i

# any GPU, the shortest queue for checking that something sees one
gpu: PARTITION ?= $(GPU_PARTITION)
gpu: CPUS ?= 8
gpu: MEM ?= 32G
gpu: GPUS ?= 1
gpu:
	srun $(if $(PARTITION),-p $(PARTITION)) -t $(TIME) -c $(CPUS) --mem=$(MEM) --gres=gpu:$(GPUS) --pty bash -i

# the GPU the models are served on, with the resources of slurm/serve.sbatch
gpu-serve: PARTITION ?= $(SERVE_PARTITION)
gpu-serve: CPUS ?= 16
gpu-serve: MEM ?= 64G
gpu-serve: GPUS ?= 1
gpu-serve:
	srun $(if $(PARTITION),-p $(PARTITION)) -t $(TIME) -c $(CPUS) --mem=$(MEM) --gres=gpu:$(GPUS) --pty bash -i
