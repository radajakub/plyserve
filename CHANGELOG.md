# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-07

### Added

- `plyserve job --length short|day|long|extralong [--time ...] -- <serve arguments>` submits
  `slurm/serve_model.sbatch` from the clone with the partition and time of its length: `PLYSERVE_PARTITION_SHORT`,
  `_DAY`, `_LONG`, `_EXTRALONG` (`.env`), and the time limit `PLYSERVE_TIME_*` (default 4 hours, 1 day, 3 days, 21
  days, RCI's limits); `--time` sets the time limit of one job.
  `make serve-job LENGTH=... [JOB_TIME=...] ARGS='...'` and `plyserve remote --length/--time` call it.
  `plyserve partition <length>` prints a partition (`make gpu-serve` takes the short one).
- `plyserve.remote` as a library: `submit`, `wait_until_ready`, `job_state` and `cancel`, the steps of `remote`, for
  tools that keep a model served (plyservice).
- `PLYSERVE_MANIFEST` (environment or `.env`) names the `models.toml` to read when `--manifest` is not given, so a
  batch job (`make serve-job`) can serve a variant of a model kept outside the clone.
- `serve --monitor` samples the GPU (nvidia-smi; DCGM's SM, tensor and DRAM activity where installed) and the CPU
  (top) into the run folder while the server runs; `server.json` lists the files.
- `plyserve remote <model> -- <command>` runs mode 3 from the client machine: it submits `serve --tunnel` through
  the cluster's login node, waits until the server answers through the tunnel, runs the command and cancels the
  job when the command ends. `--notify` pushes its progress to ntfy; `--fetch` copies the job's run folder and
  Slurm log into `runs/<job id>-<model>/` of the clone (gitignored).
- `qwen-3.8-27b-fp8-mtp` in `models.toml`: the Qwen3.8 FP8 checkpoint with speculative decoding by its MTP layer
  (`--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`), tunnel port 18004.

### Changed

- `slurm/serve.sbatch` is renamed to `slurm/serve_model.sbatch`.
- The serve partitions are in `.env` (`PLYSERVE_PARTITION_*`), not in `local.mk` (`SERVE_*PARTITION`), which keeps
  `CPU_PARTITION` and `GPU_PARTITION` of the interactive shells. A `local.mk` that still names a serve partition stops
  `plyserve job` with a message, so a job never lands on Slurm's default partition unnoticed.

## [0.2.0] - 2026-10-16

### Added

- An Ollama model's client also gets `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` (same URL and key as `VLLM_*`), from
  `serve ... -- <client>` and from `plyserve endpoint`, for plyllm's `ollama` provider.
