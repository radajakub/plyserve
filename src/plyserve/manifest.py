from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from plyserve.settings import REPOSITORY, environment

DEFAULT_MANIFEST = REPOSITORY / "models.toml"

# flags plyserve sets from the served name and the run; a manifest that set them too would contradict it
RESERVED_FLAGS = frozenset({"--model", "--revision", "--tokenizer", "--tokenizer-revision", "--served-model-name", "--host", "--port", "--api-key"})
_REQUIRED_KEYS = {"repo", "revision", "tunnel_port"}
# keys of each backend beyond the common ones: vLLM takes command-line flags; Ollama takes one GGUF file of the repository and Modelfile parameters
_BACKEND_KEYS = {"vllm": {"args"}, "ollama": {"file", "parameters"}}


@dataclass(frozen=True)
class ModelSpec:
    name: str  # served name, the one clients send as the model
    repo: str
    revision: str
    tunnel_port: int
    args: tuple[str, ...]
    backend: str = "vllm"
    file: str | None = None  # Ollama: the GGUF file in the repository
    parameters: tuple[tuple[str, str | int | float], ...] = ()  # Ollama: Modelfile PARAMETER lines; a name repeats for a list (e.g. stop)

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

    def modelfile(self, snapshot: Path) -> str:
        # the Ollama Modelfile that registers the downloaded GGUF under the served name; strings are quoted, so a stop sequence may hold spaces
        lines = [f"FROM {snapshot / self.file}"] if self.file else []
        lines += [f"PARAMETER {name} {json.dumps(value) if isinstance(value, str) else value}" for name, value in self.parameters]
        return "\n".join(lines) + "\n"


def default_manifest() -> Path:
    # PLYSERVE_MANIFEST, e.g. a variant of models.toml for one experiment kept outside the clone, else the clone's own. $USER and ~ are expanded
    value = environment().get("PLYSERVE_MANIFEST")
    return Path(os.path.expandvars(value)).expanduser() if value else DEFAULT_MANIFEST


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, ModelSpec]:
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    unknown_sections = set(data) - {"models"}
    if unknown_sections:
        raise ValueError(f'{path}: unknown top-level keys {sorted(unknown_sections)}; models go under [models."<served name>"]')
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
    backend = entry.get("backend", "vllm")
    if backend not in _BACKEND_KEYS:
        raise ValueError(f"{where}: backend must be one of {sorted(_BACKEND_KEYS)}")
    required = _REQUIRED_KEYS | ({"file"} if backend == "ollama" else set())
    missing = required - set(entry)
    if missing:
        raise ValueError(f"{where} is missing {sorted(missing)}")
    unknown = set(entry) - _REQUIRED_KEYS - {"backend"} - _BACKEND_KEYS[backend]
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)} for backend {backend}")

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
    if backend == "vllm":
        return ModelSpec(name, repo, revision, tunnel_port, tuple(args))

    file, parameters = entry["file"], entry.get("parameters", {})
    if not isinstance(file, str) or not file.endswith(".gguf"):
        raise ValueError(f"{where}: file must name a .gguf file in the repository")
    if not isinstance(parameters, dict):
        raise ValueError(f"{where}: parameters must be a table")
    pairs: list[tuple[str, str | int | float]] = []
    for key, value in parameters.items():
        for item in value if isinstance(value, list) else [value]:
            # bool is an int subclass, and Ollama's parameters take numbers and strings
            if isinstance(item, bool) or not isinstance(item, str | int | float):
                raise ValueError(f"{where}: parameter {key} must be a number, a string or a list of them")
            pairs.append((key, item))
    return ModelSpec(name, repo, revision, tunnel_port, (), backend, file, tuple(pairs))
