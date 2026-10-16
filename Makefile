.PHONY: test lint typecheck check format fix release release-dry-run require-version runtime runtime-ollama download cpu gpu gpu-serve

test:
	uv run --group dev pytest -q

lint:
	uv run --group dev ruff check .
	uv run --group dev ruff format --check .
	$(MAKE) typecheck

typecheck:
	uv run --group dev pyright

# Lint, then the tests
check: lint test

format:
	uv run ruff check --fix .
	uv run ruff format .

# Alias for format
fix: format

# Fail before the slow checks run, so a missing VERSION costs nothing
require-version:
	@test -n "$(VERSION)" || (echo "usage: make $(MAKECMDGOALS) VERSION=0.2.0" >&2; exit 1)

# Cut a release: make release VERSION=0.2.0
# Bumps the version, tags it, and publishes a GitHub Release. Not uploaded to PyPI: plyserve runs from a clone
release: require-version lint test
	uv run python scripts/release.py $(VERSION)

# Print every step of a release without changing anything
release-dry-run: require-version
	uv run python scripts/release.py $(VERSION) --dry-run

# The pinned vLLM environment (runtime/vllm/.venv, ~10 GB). Linux x86_64 only: run it on the cluster
runtime:
	uv sync --project runtime/vllm

# The pinned Ollama release (runtime/ollama, with its own CUDA libraries), checked against the release's SHA-256. Linux x86_64 only: run it on the cluster.
# The archive is zstd-compressed; Python 3.14's tarfile unpacks it, so the node needs no zstd. To upgrade: change both values, delete runtime/ollama, rerun
OLLAMA_VERSION = 0.40.3
OLLAMA_SHA256 = ce6f8b671a699bf1ebafb28d60dbbabf4f37b30b499145fa308a8ba72de021d6
runtime-ollama:
	mkdir -p runtime/ollama
	curl -fL -o runtime/ollama/ollama.tar.zst https://github.com/ollama/ollama/releases/download/v$(OLLAMA_VERSION)/ollama-linux-amd64.tar.zst
	echo "$(OLLAMA_SHA256)  runtime/ollama/ollama.tar.zst" | sha256sum -c -
	uv run --no-project --python 3.14 python -m tarfile -e runtime/ollama/ollama.tar.zst runtime/ollama
	rm runtime/ollama/ollama.tar.zst
	runtime/ollama/bin/ollama --version

# Weights at the pinned revision into $PLYSERVE_HOME/hf: make download MODEL=qwen-3.8-27b-fp8 (also for Ollama models: it needs `make runtime` for the hf CLI)
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
