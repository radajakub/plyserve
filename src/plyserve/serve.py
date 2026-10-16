from __future__ import annotations

import http.client
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

from plyserve.endpoint import Endpoint, client_variables, endpoint_path, healthy, read_endpoint, remove_endpoint, write_endpoint
from plyserve.manifest import ModelSpec
from plyserve.proxy import KeyProxy
from plyserve.settings import REPOSITORY, environment, hf_home, home, ollama_executable, runtime_executable

# without --expose the server is reachable only from processes in the same job
HOST = "127.0.0.1"
# with --expose, unless PLYSERVE_EXPOSE_BIND says otherwise: every interface, so other nodes reach it
EXPOSE_BIND = "0.0.0.0"
# seconds between health checks while vLLM loads, and while the client runs
POLL_INTERVAL = 2.0
# seconds a process gets after SIGTERM before SIGKILL; Slurm itself waits 30 s (KillWait) at the time limit
STOP_GRACE = 15.0
# with --tunnel, unless PLYSERVE_TUNNEL_BIND says otherwise: the remote machine's loopback, so only processes on that machine reach the server
TUNNEL_BIND = "127.0.0.1"
# seconds the first ssh must stay up to count as connected; a failure within them is a setup error and stops serve
TUNNEL_SETTLE = 10.0
# seconds before reconnecting a dropped tunnel: the first delay, doubled after each failed attempt up to the second
TUNNEL_RETRY = (1.0, 60.0)

# Ollama unloads a model 5 min after its last request; plyserve loads it once and keeps it until the server stops. Cloud features (remote models, web search) stay off.
# Without pruning, a starting server never deletes a blob that a server in another job is still registering in the shared $PLYSERVE_HOME/ollama. A variable set in .env or the shell wins
OLLAMA_ENV = {"OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_NO_CLOUD": "1", "OLLAMA_NOPRUNE": "1"}

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


def client_environment(env: dict[str, str], base_url: str, key: str, model: str, backend: str = "vllm") -> dict[str, str]:
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
    client.update(client_variables(base_url, key, model, backend))
    return client


def tunnel_argv(env: dict[str, str], remote_port: int, local: str) -> list[str]:
    # ssh -R: <bind>:<remote_port> on the remote machine forwards to `local` (host:port of vLLM) here. BatchMode never prompts, since a job has no terminal;
    # ExitOnForwardFailure exits when the remote port is taken, so a retry follows; ServerAlive notices a dead connection within ~90 s
    target = env.get("PLYSERVE_TUNNEL_HOST")
    if not target:
        raise RuntimeError("--tunnel needs PLYSERVE_TUNNEL_HOST, the SSH destination (see .env.example)")
    bind = env.get("PLYSERVE_TUNNEL_BIND") or TUNNEL_BIND
    options = ("BatchMode=yes", "ExitOnForwardFailure=yes", "ServerAliveInterval=30", "ServerAliveCountMax=3")
    argv = ["ssh", "-N", "-R", f"{bind}:{remote_port}:{local}", *(part for option in options for part in ("-o", option))]
    if key := env.get("PLYSERVE_TUNNEL_KEY"):
        argv += ["-i", str(Path(os.path.expandvars(key)).expanduser()), "-o", "IdentitiesOnly=yes"]
    return [*argv, target]


class _Tunnel:
    # the ssh behind --tunnel; when it drops, it is started again with growing delays while vLLM keeps serving
    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        self.argv, self.env = argv, env
        self.delay, self.retry_at = TUNNEL_RETRY[0], None
        self.process = self._start()

    def _start(self) -> subprocess.Popen[bytes]:
        self.started = time.monotonic()
        # its own session, so a Ctrl-C reaches only plyserve, which stops the tunnel itself
        return subprocess.Popen(self.argv, env=self.env, stdin=subprocess.DEVNULL, start_new_session=True)

    def wait_until_up(self) -> None:
        # a first attempt that fails is a setup error (host, key, known_hosts, remote port taken), which retrying does not fix
        while time.monotonic() - self.started < TUNNEL_SETTLE:
            if (code := self.process.poll()) is not None:
                raise RuntimeError(f"ssh exited with code {code} while opening the tunnel to {self.argv[-1]}; its message is above")
            time.sleep(POLL_INTERVAL)

    def check(self) -> None:
        if self.retry_at is None:
            if (code := self.process.poll()) is None:
                return
            if time.monotonic() - self.started > TUNNEL_RETRY[1]:
                self.delay = TUNNEL_RETRY[0]  # it was up for a while, so this is a new drop rather than another failed attempt
            print(f"plyserve: the tunnel to {self.argv[-1]} dropped (ssh exit code {code}); reconnecting in {self.delay:g} s", flush=True)
            self.retry_at = time.monotonic() + self.delay
            self.delay = min(2 * self.delay, TUNNEL_RETRY[1])
        elif time.monotonic() >= self.retry_at:
            self.retry_at = None
            self.process = self._start()

    def stop(self) -> None:
        _stop(self.process, group=True)


def _wait_until_ready(process: subprocess.Popen[bytes], engine: str, url: str, key: str, timeout: float, deadline: float) -> None:
    while not healthy(url, key):
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"{engine} exited with code {code} before it was ready; its log is above")
        if time.monotonic() > deadline:
            raise RuntimeError(f"{engine} was not ready after {timeout:g} s; if it is only loading slowly, raise --timeout")
        time.sleep(POLL_INTERVAL)


def _ollama_api(upstream: str, path: str, payload: dict[str, object] | None = None, timeout: float = 30.0) -> dict:
    # a call to Ollama's own API on the loopback address; http.client, unlike urllib, never goes through http_proxy
    connection = http.client.HTTPConnection(upstream, timeout=max(timeout, 1.0))
    try:
        connection.request("GET" if payload is None else "POST", path, body=None if payload is None else json.dumps(payload), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"Ollama answered {path} with {response.status}: {body.decode(errors='replace')[:500]}")
    return json.loads(body)


def _load_ollama(spec: ModelSpec, executable: Path, env: dict[str, str], upstream: str, weights: Path, timeout: float, deadline: float) -> dict[str, object]:
    # registers the GGUF under the served name, loads it and checks that all of it is on the GPU; returns what the run record keeps
    with tempfile.TemporaryDirectory() as folder:
        modelfile = Path(folder) / "Modelfile"
        modelfile.write_text(spec.modelfile(weights), encoding="utf-8")
        # the CLI hashes the GGUF and copies it into OLLAMA_MODELS, unless a blob with that hash is there already
        print(f"plyserve: registering {spec.file} with Ollama; the first start copies it into {env['OLLAMA_MODELS']}", flush=True)
        try:
            # its progress spinner would fill the job log with terminal control codes, so its output is shown only when it fails
            created = subprocess.run(
                [str(executable), "create", spec.name, "-f", str(modelfile)],
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=max(deadline - time.monotonic(), 1.0),
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Ollama was not ready after {timeout:g} s (still registering {spec.file}); raise --timeout") from error
    if created.returncode != 0:
        lines = [line.strip() for line in re.split(r"[\r\n]+", re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", created.stdout + created.stderr)) if line.strip()]
        print("\n".join(lines[-10:]), flush=True)
        raise RuntimeError(f"`ollama create` exited with code {created.returncode}; its last lines are above")
    # a request without a prompt only loads the model
    _ollama_api(upstream, "/api/generate", {"model": spec.name, "keep_alive": -1, "stream": False}, timeout=deadline - time.monotonic())
    loaded = next((model for model in _ollama_api(upstream, "/api/ps")["models"] if model["name"] in (spec.name, f"{spec.name}:latest")), None)
    if loaded is None:
        raise RuntimeError(f"Ollama does not list {spec.name} as loaded after loading it; its log is above")
    # Ollama puts what does not fit on the GPU on the CPU, which is many times slower; a run that needs the GPU stops instead
    size, on_gpu = loaded["size"], loaded["size_vram"]
    if on_gpu == 0:
        raise RuntimeError(f"Ollama runs {spec.name} on the CPU: it found no usable GPU; its log above says why")
    if on_gpu < size:
        raise RuntimeError(f"Ollama put only {on_gpu / 2**30:.1f} of {size / 2**30:.1f} GiB of {spec.name} on the GPU; lower num_ctx in models.toml or take a smaller GGUF")
    shown = _ollama_api(upstream, "/api/show", {"model": spec.name})
    # how Ollama turns messages into the prompt and parses the answer: the resolved Modelfile (with its chat template), or a built-in renderer and parser
    model = {field: shown.get(field) for field in ("modelfile", "renderer", "parser", "details")}
    return {"versions": {"ollama": _ollama_api(upstream, "/api/version")["version"]}, "ollama_model": model, "memory": {"size": size, "size_vram": on_gpu}}


def _stop(process: subprocess.Popen[bytes], group: bool) -> None:
    # the server runs in its own process group, so its engine processes (vLLM) or model runners (Ollama) are stopped with it
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


def _version_line(argv: list[str], last: bool = False) -> str:
    lines = subprocess.run(argv, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return lines[-1] if last else lines[0]


def toolchain(env: dict[str, str]) -> dict[str, str]:
    # vLLM's fastest kernels (DeepGEMM, FlashInfer) are compiled on first use, so the server needs a CUDA toolkit and a
    # C++ compiler (slurm/site.sh loads them, see docs/). Checked before the GPU loads anything; the versions go into
    # the run record, since the kernels decide the numerics of the answers
    path = env.get("PATH")
    nvcc = shutil.which("nvcc", path=path)
    if nvcc is None:
        raise RuntimeError("vLLM needs a CUDA toolkit, but nvcc is not on PATH; load one in slurm/site.sh (see docs/setup.md)")
    cxx = env.get("CXX") or shutil.which("c++", path=path)
    if cxx is None:
        raise RuntimeError("vLLM needs a C++ compiler, but CXX is unset and c++ is not on PATH; load one in slurm/site.sh (see docs/setup.md)")
    return {"nvcc": nvcc, "nvcc_version": _version_line([nvcc, "--version"], last=True), "cxx": cxx, "cxx_version": _version_line([cxx, "--version"])}


def serve(spec: ModelSpec, *, port: int | None = None, timeout: float = 1800.0, command: Sequence[str] = (), expose: bool = False, tunnel: bool = False) -> int:
    # starts vLLM or Ollama (spec.backend), waits until it answers, then runs `command` against it and stops the server when the command ends;
    # without a command it serves until the server exits. Returns the exit code of the command (or of the server).
    # `expose` makes the server reachable from other nodes and publishes it in an endpoint file;
    # `tunnel` forwards the model's tunnel_port on another machine to it over SSH
    ollama = spec.backend == "ollama"
    engine, weights = "Ollama" if ollama else "vLLM", snapshot(spec)
    required = weights / spec.file if spec.file else weights
    if not required.exists():
        raise FileNotFoundError(f"{required} does not exist; fetch the weights with `make download MODEL={spec.name}`")
    executable = ollama_executable() if ollama else runtime_executable("vllm")
    env = environment()
    bind, host = addresses(env, expose)
    if expose and (current := read_endpoint(spec.name)) is not None and healthy(current.health_url, current.api_key):
        raise RuntimeError(f"{spec.name} is already served at {current.base_url} (job {current.slurm_job_id or '-'} on {current.hostname}); stop that server first")
    # a client behind the tunnel keeps its key from run to run, so a per-run key cannot reach it
    if tunnel and not env.get("PLYSERVE_API_KEY"):
        raise RuntimeError("--tunnel needs a fixed PLYSERVE_API_KEY, the same one the remote client uses (see .env.example)")
    if tunnel and shutil.which("ssh", path=env.get("PATH")) is None:
        raise RuntimeError("--tunnel needs ssh, which is not on PATH")
    port = port or free_port(bind)
    # a fixed key from .env lets clients outside the cluster keep theirs; otherwise a new one for every run
    key = env.get("PLYSERVE_API_KEY") or secrets.token_urlsafe(32)
    # this job (and the tunnel) reaches the server on the loopback address whatever it listens on (IPv4 only)
    local_host = HOST if bind == EXPOSE_BIND else bind
    local = f"http://{local_host}:{port}"
    ssh_argv = tunnel_argv(env, spec.tunnel_port, f"{local_host}:{port}") if tunnel else None
    base_url = f"http://{host}:{port}/v1"
    upstream = ""  # Ollama's own address, behind the proxy; vLLM has none
    if ollama:
        # Ollama has no API key: it listens on the loopback address only, and plyserve's proxy on bind:port checks the key in front of it
        upstream = f"{HOST}:{free_port(HOST)}"
        argv = [str(executable), "serve"]
        server_env = {**OLLAMA_ENV, **env, "OLLAMA_HOST": upstream, "OLLAMA_MODELS": str(home() / "ollama")}
        details: dict[str, object] = {"file": spec.file, "ollama_env": {variable: server_env[variable] for variable in sorted(server_env) if variable.startswith("OLLAMA_")}}
    else:
        argv = [str(executable), *spec.vllm_args(bind, port)]
        # the key goes through the environment so that `ps` does not show it; offline, so a job never downloads.
        # The runtime's bin goes first on PATH: FlashInfer builds its kernels with the ninja installed there
        path = os.pathsep.join(part for part in (str(executable.parent), env.get("PATH")) if part)
        server_env = {**env, "PATH": path, "HF_HOME": str(hf_home()), "HF_HUB_OFFLINE": "1", "VLLM_API_KEY": key}
        details = {"vllm_args": argv[1:], "toolchain": toolchain(server_env), "versions": _versions()}
    record = {
        "model": spec.name,
        "repo": spec.repo,
        "revision": spec.revision,
        "backend": spec.backend,
        **details,
        "plyserve": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))},
        "base_url": base_url,
        "exposed": expose,
        "tunnel": ssh_argv,
        "api_key": "fixed (PLYSERVE_API_KEY)" if env.get("PLYSERVE_API_KEY") else "per run",
        "command": list(command),
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpus": _gpus(),
        "started_at": _now(),
    }

    previous = {name: signal.signal(name, _on_signal) for name in (signal.SIGTERM, signal.SIGINT)}
    server, proxy, published, forward = None, None, None, None
    try:
        deadline = time.monotonic() + timeout
        server = subprocess.Popen(argv, env=server_env, start_new_session=True)
        if ollama:
            _wait_until_ready(server, engine, f"http://{upstream}/api/version", key, timeout, deadline)
            record.update(_load_ollama(spec, executable, server_env, upstream, weights, timeout, deadline))
            proxy = KeyProxy((bind, port), upstream, key, "/api/version")
        _wait_until_ready(server, engine, f"{local}/health", key, timeout, deadline)
        record["ready_at"] = _now()
        run_dir = home() / "runs" / f"{env.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d-%H%M%S')}-{spec.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "server.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"plyserve: {spec.name} is ready at {base_url}; run record in {run_dir}", flush=True)
        if expose:
            published = Endpoint(spec.name, base_url, key, secrets.token_hex(8), record["hostname"], record["slurm_job_id"], record["ready_at"], str(run_dir))
            write_endpoint(published)
            print(f"plyserve: endpoint published in {endpoint_path(spec.name)}", flush=True)
        if ssh_argv is not None:
            forward = _Tunnel(ssh_argv, env)
            forward.wait_until_up()
            print(f"plyserve: tunnel open, {ssh_argv[-1]} forwards {ssh_argv[3].rsplit(':', 2)[0]} to this server", flush=True)

        try:
            client = subprocess.Popen(list(command), env=client_environment(env, f"{local}/v1", key, spec.name, spec.backend)) if command else None
        except FileNotFoundError as error:
            raise RuntimeError(f"the client command {command[0]!r} was not found; give a full path, or e.g. `uv run python ...` where the node has no `python`") from error
        try:
            while client is None or (code := client.poll()) is None:
                if (server_code := server.poll()) is not None:
                    if client is None:
                        return server_code
                    raise RuntimeError(f"{engine} exited with code {server_code} while the client was running; its log is above")
                if forward is not None:
                    forward.check()
                time.sleep(POLL_INTERVAL)
            return code
        finally:
            if client is not None:
                _stop(client, group=False)
    except _Stopped as stopped:
        print(f"plyserve: {stopped}, stopping", flush=True)
        return 128 + stopped.signum
    finally:
        # clients stop finding the server before it goes away
        if forward is not None:
            forward.stop()
        if published is not None:
            remove_endpoint(published)
        if proxy is not None:
            proxy.stop()
        if server is not None:
            _stop(server, group=True)
        for name, handler in previous.items():
            signal.signal(name, handler)
