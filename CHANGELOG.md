# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `make serve-job LENGTH=short|long ARGS='...'` submits `slurm/serve_model.sbatch` with the partition and time for
  its length: `SERVE_PARTITION` for 4 hours, or the new `SERVE_LONG_PARTITION` (`local.mk`) for 3 days.

### Changed

- `slurm/serve.sbatch` is renamed to `slurm/serve_model.sbatch`.

## [0.2.0] - 2026-10-16

### Added

- An Ollama model's client also gets `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` (same URL and key as `VLLM_*`), from
  `serve ... -- <client>` and from `plyserve endpoint`, for plyllm's `ollama` provider.
