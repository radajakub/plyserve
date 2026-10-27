# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `make serve-job LENGTH=short|long ARGS='...'` submits `slurm/serve_model.sbatch` with the partition and time for
  its length: `SERVE_PARTITION` for 4 hours, or the new `SERVE_LONG_PARTITION` (`local.mk`) for 3 days.
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

## [0.2.0] - 2026-10-16

### Added

- An Ollama model's client also gets `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` (same URL and key as `VLLM_*`), from
  `serve ... -- <client>` and from `plyserve endpoint`, for plyllm's `ollama` provider.
