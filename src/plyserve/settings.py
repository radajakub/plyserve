from __future__ import annotations

import os
from pathlib import Path

# plyserve runs from a clone of this repository (`uv sync` installs it editable)
REPOSITORY = Path(__file__).resolve().parents[2]
# the pinned vLLM environment, built by `make runtime` (uv sync --project runtime/vllm)
RUNTIME = REPOSITORY / "runtime" / "vllm" / ".venv"
# this clone's settings (gitignored; see .env.example), so a Slurm job finds them without a shell profile
ENV_FILE = REPOSITORY / ".env"


def read_env_file(path: Path) -> dict[str, str]:
    # KEY=VALUE lines; blank lines, # comments and a leading `export` are skipped, and one pair of matching
    # quotes is removed. No inline comments and no interpolation. An empty value counts as unset
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE, got {raw!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            values[key] = value
    return values


def environment() -> dict[str, str]:
    # what plyserve and the processes it starts see: .env under the process environment, which wins (as in plyllm)
    return {**read_env_file(ENV_FILE), **os.environ}


def home() -> Path:
    # weights, endpoint files and the API key. Required rather than defaulted, so ~90 GB of weights never land
    # in a home folder by accident; on RCI it is a folder on /mnt/personal. $USER and ~ are expanded
    value = environment().get("PLYSERVE_HOME")
    if not value:
        raise RuntimeError(f"PLYSERVE_HOME is not set; set it in {ENV_FILE} (see .env.example) to a folder on fast storage, e.g. /mnt/personal/$USER/plyserve-home")
    return Path(os.path.expandvars(value)).expanduser()


def hf_home() -> Path:
    # the Hugging Face cache (HF_HOME) that downloads fill and servers read
    return home() / "hf"


def runtime_executable(name: str) -> Path:
    path = RUNTIME / "bin" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; build the vLLM environment with `make runtime`")
    return path
