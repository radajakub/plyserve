from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

from plyserve.endpoint import Endpoint, endpoint_path, healthy, read_endpoint, remove_endpoint, write_endpoint
from plyserve.manifest import ModelSpec
from plyserve.settings import REPOSITORY, environment, hf_home, home, runtime_executable

# without --expose the server is reachable only from processes in the same job
HOST = "127.0.0.1"
# with --expose, unless PLYSERVE_EXPOSE_BIND says otherwise: every interface, so other nodes reach it
EXPOSE_BIND = "0.0.0.0"
# seconds between health checks while vLLM loads, and while the client runs
POLL_INTERVAL = 2.0
# seconds a process gets after SIGTERM before SIGKILL; Slurm itself waits 30 s (KillWait) at the time limit
STOP_GRACE = 15.0

# the runtime has no CUDA compiler: DeepGEMM and FlashInfer's sampler compile kernels on first use and fail
# without nvcc. The fallbacks are prebuilt (CUTLASS) or Triton. A variable set in .env or the shell wins
RUNTIME_ENV = {"VLLM_USE_DEEP_GEMM": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}

# versions of the runtime's packages, read without importing vLLM (which takes seconds)
_VERSIONS_SCRIPT = """
import json
from importlib.metadata import PackageNotFoundError, version

def find(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

print(json.dumps({name: find(name) for name in ("vllm", "torch", "transformers")}))
"""


class _Stopped(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


def _on_signal(signum: int, frame: FrameType | None) -> None:
    # the first SIGTERM or SIGINT starts the shutdown; later ones must not interrupt it
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, signal.SIG_IGN)
    raise _Stopped(signum)


def addresses(env: dict[str, str], expose: bool) -> tuple[str, str]:
    # (address vLLM listens on, host name clients connect to). The defaults suit a cluster whose nodes reach
    # each other by host name; PLYSERVE_EXPOSE_BIND and PLYSERVE_EXPOSE_HOST cover the rest (e.g. one network)
    if not expose:
        return HOST, HOST
    return env.get("PLYSERVE_EXPOSE_BIND") or EXPOSE_BIND, env.get("PLYSERVE_EXPOSE_HOST") or socket.gethostname()


def free_port(host: str = HOST) -> int:
    # a port the OS reports free; another process could take it before vLLM binds, and vLLM then fails to start
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def snapshot(spec: ModelSpec) -> Path:
    # where `plyserve download` put the weights (the Hugging Face cache layout)
    return hf_home() / "hub" / f"models--{spec.repo.replace('/', '--')}" / "snapshots" / spec.revision


def client_environment(env: dict[str, str], base_url: str, key: str, model: str) -> dict[str, str]:
    client = dict(env)
    # `uv run plyserve` puts plyserve's own .venv first on PATH; the client must not run in it
    venv = client.get("VIRTUAL_ENV")
    if venv and Path(venv).resolve() == (REPOSITORY / ".venv").resolve():
        del client["VIRTUAL_ENV"]
        bin_dir = str(Path(venv) / "bin")
        client["PATH"] = os.pathsep.join(part for part in client.get("PATH", "").split(os.pathsep) if part != bin_dir)
    # an HTTP client would send 127.0.0.1 through a cluster proxy too
    for name in ("no_proxy", "NO_PROXY"):
        client[name] = ",".join(part for part in (client.get(name), "127.0.0.1", "localhost") if part)
    client.update(VLLM_BASE_URL=base_url, VLLM_API_KEY=key, VLLM_MODEL=model)
    return client


def _wait_until_ready(process: subprocess.Popen[bytes], url: str, key: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not healthy(url, key):
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"vLLM exited with code {code} before it was ready; its log is above")
        if time.monotonic() > deadline:
            raise RuntimeError(f"vLLM was not ready after {timeout:g} s; if it is only loading slowly, raise --timeout")
        time.sleep(POLL_INTERVAL)


def _stop(process: subprocess.Popen[bytes], group: bool) -> None:
    # vLLM runs in its own process group, so its engine processes are stopped with it
    def send(signum: int) -> None:
        try:
            if group:
                os.killpg(process.pid, signum)
            else:
                process.send_signal(signum)
        except ProcessLookupError:
            pass

    if process.poll() is None:
        send(signal.SIGTERM)
        try:
            process.wait(STOP_GRACE)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL)
            process.wait()
    if group:
        # whatever is left of the group once its leader is gone
        send(signal.SIGKILL)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(REPOSITORY), *args], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _gpus() -> list[str]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    except FileNotFoundError:
        return []
    return result.stdout.strip().splitlines() if result.returncode == 0 else []


def _versions() -> dict[str, str | None]:
    result = subprocess.run([str(runtime_executable("python")), "-c", _VERSIONS_SCRIPT], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def serve(spec: ModelSpec, *, port: int | None = None, timeout: float = 1800.0, command: Sequence[str] = (), expose: bool = False) -> int:
    # starts vLLM, waits until it answers, then runs `command` against it and stops vLLM when the command ends;
    # without a command it serves until vLLM exits. Returns the exit code of the command (or of vLLM).
    # `expose` makes the server reachable from other nodes and publishes it in an endpoint file
    executable, weights = runtime_executable("vllm"), snapshot(spec)
    if not weights.is_dir():
        raise FileNotFoundError(f"{weights} does not exist; fetch the weights with `make download MODEL={spec.name}`")
    env = environment()
    bind, host = addresses(env, expose)
    if expose and (current := read_endpoint(spec.name)) is not None and healthy(current.health_url, current.api_key):
        raise RuntimeError(f"{spec.name} is already served at {current.base_url} (job {current.slurm_job_id or '-'} on {current.hostname}); stop that server first")
    port = port or free_port(bind)
    # a fixed key from .env lets clients outside the cluster keep theirs; otherwise a new one for every run
    key = env.get("PLYSERVE_API_KEY") or secrets.token_urlsafe(32)
    argv = [str(executable), *spec.vllm_args(bind, port)]
    # this job reaches the server on the loopback address whatever it listens on (IPv4 only)
    local = f"http://{HOST if bind == EXPOSE_BIND else bind}:{port}"
    base_url = f"http://{host}:{port}/v1"
    # the key goes through the environment so that `ps` does not show it; offline, so a job never downloads
    server_env = {**RUNTIME_ENV, **env, "HF_HOME": str(hf_home()), "HF_HUB_OFFLINE": "1", "VLLM_API_KEY": key}
    record = {
        "model": spec.name,
        "repo": spec.repo,
        "revision": spec.revision,
        "vllm_args": argv[1:],
        "vllm_env": {name: server_env[name] for name in RUNTIME_ENV},
        "versions": _versions(),
        "plyserve": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))},
        "base_url": base_url,
        "exposed": expose,
        "api_key": "fixed (PLYSERVE_API_KEY)" if env.get("PLYSERVE_API_KEY") else "per run",
        "command": list(command),
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpus": _gpus(),
        "started_at": _now(),
    }

    previous = {name: signal.signal(name, _on_signal) for name in (signal.SIGTERM, signal.SIGINT)}
    vllm, published = None, None
    try:
        vllm = subprocess.Popen(argv, env=server_env, start_new_session=True)
        _wait_until_ready(vllm, f"{local}/health", key, timeout)
        record["ready_at"] = _now()
        run_dir = home() / "runs" / f"{env.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d-%H%M%S')}-{spec.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "server.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"plyserve: {spec.name} is ready at {base_url}; run record in {run_dir}", flush=True)
        if expose:
            published = Endpoint(spec.name, base_url, key, secrets.token_hex(8), record["hostname"], record["slurm_job_id"], record["ready_at"], str(run_dir))
            write_endpoint(published)
            print(f"plyserve: endpoint published in {endpoint_path(spec.name)}", flush=True)
        if not command:
            return vllm.wait()

        client = subprocess.Popen(list(command), env=client_environment(env, f"{local}/v1", key, spec.name))
        try:
            while (code := client.poll()) is None:
                if (server_code := vllm.poll()) is not None:
                    raise RuntimeError(f"vLLM exited with code {server_code} while the client was running; its log is above")
                time.sleep(POLL_INTERVAL)
            return code
        finally:
            _stop(client, group=False)
    except _Stopped as stopped:
        print(f"plyserve: {stopped}, stopping", flush=True)
        return 128 + stopped.signum
    finally:
        # clients stop finding the server before it goes away
        if published is not None:
            remove_endpoint(published)
        if vllm is not None:
            _stop(vllm, group=True)
        for name, handler in previous.items():
            signal.signal(name, handler)
