from __future__ import annotations

import subprocess

from plyserve.manifest import ModelSpec
from plyserve.settings import environment, hf_home, runtime_executable


def download_command(spec: ModelSpec) -> list[str]:
    # the hf CLI of the vLLM environment, so the cache layout is the one that vLLM reads
    return [str(runtime_executable("hf")), "download", spec.repo, "--revision", spec.revision]


def download(spec: ModelSpec) -> None:
    # into the shared cache at the pinned revision; a file already there is not fetched again
    command, cache = download_command(spec), hf_home()
    cache.mkdir(parents=True, exist_ok=True)
    # HF_TOKEN and HF_XET_* come through from .env or the shell
    subprocess.run(command, env={**environment(), "HF_HOME": str(cache)}, check=True)
