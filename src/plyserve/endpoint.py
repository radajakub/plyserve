from __future__ import annotations

import json
import os
import shlex
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from plyserve.settings import home

# requests to plyserve's servers go direct, whatever http_proxy says
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class Endpoint:
    # what a server started with --expose publishes for clients in other jobs, in $PLYSERVE_HOME/endpoints
    model: str
    base_url: str
    api_key: str
    server_id: str  # tells this server's file from the file of a later server for the same model
    hostname: str
    slurm_job_id: str | None
    ready_at: str
    run_dir: str

    @property
    def health_url(self) -> str:
        return self.base_url.removesuffix("/v1") + "/health"


def healthy(url: str, key: str) -> bool:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with _OPENER.open(request, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def endpoint_path(model: str) -> Path:
    return home() / "endpoints" / f"{model}.json"


def write_endpoint(endpoint: Endpoint) -> Path:
    target = endpoint_path(endpoint.model)
    target.parent.mkdir(parents=True, exist_ok=True)
    # the file holds the API key, so only the owner can read it; it is replaced in one step, so a reader never sees half of it
    temporary = target.with_name(f".{target.name}.{endpoint.server_id}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(asdict(endpoint), handle, indent=2)
    os.replace(temporary, target)
    return target


def read_endpoint(model: str) -> Endpoint | None:
    path = endpoint_path(model)
    try:
        return Endpoint(**json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not a plyserve endpoint file ({error})") from error


def remove_endpoint(endpoint: Endpoint) -> None:
    # only this server's file: a later server for the same model may have replaced it
    try:
        current = read_endpoint(endpoint.model)
    except ValueError:
        return
    if current is not None and current.server_id == endpoint.server_id:
        endpoint_path(endpoint.model).unlink(missing_ok=True)


def find_endpoint(model: str, *, wait: bool = False, timeout: float | None = None, interval: float = 5.0) -> Endpoint:
    # the endpoint of a running server that answers /health; with `wait`, until one does (or `timeout` passes)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        endpoint = read_endpoint(model)
        if endpoint is not None and healthy(endpoint.health_url, endpoint.api_key):
            return endpoint
        if not wait:
            if endpoint is None:
                raise RuntimeError(f"no server for {model}: {endpoint_path(model)} does not exist (start one with `plyserve serve --expose {model}`)")
            raise RuntimeError(f"the server for {model} at {endpoint.base_url} (job {endpoint.slurm_job_id or '-'}) does not answer; it is still loading or has ended")
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError(f"no server for {model} answered within {timeout:g} s")
        time.sleep(interval)


def client_variables(base_url: str, key: str, model: str, backend: str) -> dict[str, str]:
    # what a client reads: VLLM_* for every backend; for Ollama also OLLAMA_*, which plyllm's ollama provider reads
    # (its vllm provider has no entry for an Ollama model)
    values = {"VLLM_BASE_URL": base_url, "VLLM_API_KEY": key, "VLLM_MODEL": model}
    if backend == "ollama":
        values |= {"OLLAMA_BASE_URL": base_url, "OLLAMA_API_KEY": key}
    return values


def shell_exports(endpoint: Endpoint, backend: str = "vllm") -> str:
    # for `eval "$(plyserve endpoint <model>)"`: the variables a client reads, and the server's host kept off any HTTP proxy
    values = client_variables(endpoint.base_url, endpoint.api_key, endpoint.model, backend)
    lines = [f"export {name}={shlex.quote(value)}" for name, value in values.items()]
    host = shlex.quote(urlsplit(endpoint.base_url).hostname or "")
    lines += [f'export {name}="${{{name}:+${name},}}"{host}' for name in ("no_proxy", "NO_PROXY")]
    return "\n".join(lines)
