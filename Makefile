.PHONY: cpu gpu gpu-h200

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
