from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# plyserve runs from a clone of this repository (`uv sync` installs it editable), so the manifest is the one
# at the repository root
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "models.toml"

# flags plyserve sets from the served name and the run; a manifest that set them too would contradict it
RESERVED_FLAGS = frozenset({"--model", "--revision", "--tokenizer", "--tokenizer-revision", "--served-model-name", "--host", "--port", "--api-key"})
_REQUIRED_KEYS = {"repo", "revision", "tunnel_port"}
_KEYS = _REQUIRED_KEYS | {"args"}


@dataclass(frozen=True)
class ModelSpec:
    name: str  # served name, the one clients send as the model
    repo: str
    revision: str
    tunnel_port: int
    args: tuple[str, ...]

    def vllm_args(self, host: str, port: int) -> list[str]:
        # the arguments after the `vllm` executable. The tokenizer is pinned too: vLLM loads it separately,
        # and an unpinned one would follow the repository's main branch
        return [
            "serve",
            self.repo,
            "--revision",
            self.revision,
            "--tokenizer-revision",
            self.revision,
            "--served-model-name",
            self.name,
            "--host",
            host,
            "--port",
            str(port),
            *self.args,
        ]


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, ModelSpec]:
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    unknown_sections = set(data) - {"models"}
    if unknown_sections:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown_sections)}; models go under [models.\"<served name>\"]")
    specs = {name: _spec(path, name, entry) for name, entry in data.get("models", {}).items()}

    ports: dict[int, str] = {}
    for spec in specs.values():
        if spec.tunnel_port in ports:
            raise ValueError(f"{path}: {spec.name} and {ports[spec.tunnel_port]} share tunnel_port {spec.tunnel_port}")
        ports[spec.tunnel_port] = spec.name
    return specs


def _spec(path: Path, name: str, entry: object) -> ModelSpec:
    where = f"{path}: model {name}"
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a table")
    missing = _REQUIRED_KEYS - set(entry)
    if missing:
        raise ValueError(f"{where} is missing {sorted(missing)}")
    unknown = set(entry) - _KEYS
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)}")

    repo, revision, tunnel_port, args = entry["repo"], entry["revision"], entry["tunnel_port"], entry.get("args", [])
    if not isinstance(repo, str) or not repo:
        raise ValueError(f"{where}: repo must be a non-empty string")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"{where}: revision must be a commit hash")
    # bool is an int subclass, and `true` is no port
    if isinstance(tunnel_port, bool) or not isinstance(tunnel_port, int) or not 1024 <= tunnel_port <= 65535:
        raise ValueError(f"{where}: tunnel_port must be an unprivileged port (1024-65535)")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError(f"{where}: args must be a list of strings")
    reserved = sorted({arg.split("=", 1)[0] for arg in args} & RESERVED_FLAGS)
    if reserved:
        raise ValueError(f"{where}: args set {reserved}, which plyserve sets itself")
    return ModelSpec(name, repo, revision, tunnel_port, tuple(args))
