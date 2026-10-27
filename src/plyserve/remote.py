from __future__ import annotations

import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

from plyserve.endpoint import healthy
from plyserve.manifest import ModelSpec
from plyserve.serve import TUNNEL_BIND, _on_signal, _stop, _Stopped, client_environment
from plyserve.settings import environment

# seconds between squeue calls while the job waits and loads; each is a new SSH login on the cluster's login node
QUEUE_INTERVAL = 30.0
# seconds between health checks through the tunnel
POLL_INTERVAL = 5.0
# seconds one SSH call to the login node may take
SSH_TIMEOUT = 120.0
# Slurm states of a job that has not ended; any other state, or none (the job left the queue), means it ended
ACTIVE_STATES = ("PENDING", "CONFIGURING", "RUNNING")


def _ssh(env: dict[str, str], target: str, command: str) -> subprocess.CompletedProcess[str]:
    # `command` in a shell on the login node. BatchMode never prompts; a call that hangs counts as a failed connection (ssh's own code 255)
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", target, command]
    try:
        return subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=SSH_TIMEOUT)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 255, "", f"ssh to {target} did not finish within {SSH_TIMEOUT:g} s")


def submit_command(repo: str, model: str, length: str, monitor: bool, manifest: str | None) -> str:
    # `make serve-job` in the cluster's clone, so the job gets the partition and time from its local.mk; PLYSERVE_MANIFEST reaches the job through sbatch.
    # In a login shell: sbatch hands the job the submitting environment, and slurm/site.sh needs what a login sets up there (e.g. Lmod's `module` function)
    args = shlex.join(["--tunnel", *(["--monitor"] if monitor else []), model])
    prefix = f"PLYSERVE_MANIFEST={shlex.quote(manifest)} " if manifest else ""
    return shlex.join(["bash", "-lc", f"cd {shlex.quote(repo)} && {prefix}make serve-job LENGTH={length} ARGS={shlex.quote(args)}"])


def job_state(env: dict[str, str], target: str, job: str) -> str | None:
    # the job's Slurm state, "" once it has left the queue (squeue then prints nothing), None when the login node did not answer
    result = _ssh(env, target, f"squeue -h -j {job} -o %T")
    return None if result.returncode == 255 else result.stdout.strip()


def push(env: dict[str, str], message: str) -> None:
    # an ntfy notification, with the settings of the other ply repositories (NTFY_URL, NTFY_TOPIC, optional NTFY_TOKEN); a failed push is only reported
    headers = {"Content-Type": "text/plain; charset=utf-8"}
    if token := env.get("NTFY_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{env['NTFY_URL'].rstrip('/')}/{env['NTFY_TOPIC']}", data=message.encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10):
            pass
    except (urllib.error.URLError, OSError) as error:
        print(f"plyserve: the ntfy push failed ({error})", flush=True)


def _wait_until_ready(env: dict[str, str], target: str, repo: str, job: str, health: str, key: str, timeout: float, say: Callable[[str, bool], None]) -> None:
    state, next_check, deadline = "", 0.0, None
    while not healthy(health, key):
        now = time.monotonic()
        if now >= next_check:
            next_check = now + QUEUE_INTERVAL
            current = job_state(env, target, job)
            if current is None:
                print(f"plyserve: {target} did not answer the squeue call; trying again in {QUEUE_INTERVAL:g} s", flush=True)
            elif current not in ACTIVE_STATES:
                raise RuntimeError(f"job {job} ended ({current or 'it left the queue'}) before the server answered; its log is plyserve-{job}.out in {repo} on {target}")
            elif current != state:
                state = current
                say(f"job {job} is {current.lower()}", current == "RUNNING")
                if current == "RUNNING":
                    deadline = now + timeout
        if deadline is not None and now > deadline:
            raise RuntimeError(
                f"job {job} runs, but the server did not answer at {health} within {timeout:g} s; check plyserve-{job}.out in {repo} on {target}, "
                "and that tunnel_port is the same in both manifests"
            )
        time.sleep(POLL_INTERVAL)


def remote(
    spec: ModelSpec, command: Sequence[str], *, length: str = "short", monitor: bool = False, manifest: str | None = None, timeout: float = 3600.0, notify: bool = False
) -> int:
    # On the client machine (mode 3): submits `serve --tunnel` as a job through the cluster's login node, waits until the server answers through the tunnel,
    # runs `command` against it and cancels the job when the command ends, fails or plyserve is stopped. Returns the exit code of the command.
    # `manifest` is a models.toml on the cluster for the job (PLYSERVE_MANIFEST there); `notify` pushes when the job runs, the server answers and the command ends
    env = environment()
    target, repo, key = env.get("PLYSERVE_REMOTE"), env.get("PLYSERVE_REMOTE_REPO"), env.get("PLYSERVE_API_KEY")
    if not target or not repo:
        raise RuntimeError("remote needs PLYSERVE_REMOTE (the cluster's login node) and PLYSERVE_REMOTE_REPO (the plyserve clone there); see .env.example")
    if not key:
        raise RuntimeError("remote needs PLYSERVE_API_KEY, the same fixed key as on the cluster (see .env.example)")
    if notify and not (env.get("NTFY_URL") and env.get("NTFY_TOPIC")):
        raise RuntimeError("--notify needs NTFY_URL and NTFY_TOPIC (see .env.example)")
    if shutil.which("ssh", path=env.get("PATH")) is None:
        raise RuntimeError("remote needs ssh, which is not on PATH")
    local = f"http://{env.get('PLYSERVE_TUNNEL_BIND') or TUNNEL_BIND}:{spec.tunnel_port}"
    # the command must not run against an older server that still holds the port
    if healthy(f"{local}/health", key):
        raise RuntimeError(f"a server already answers at {local}; stop it (or its job) first")

    def say(message: str, pushed: bool = True) -> None:
        print(f"plyserve: {message}", flush=True)
        if notify and pushed:
            push(env, f"{spec.name}: {message}")

    previous = {name: signal.signal(name, _on_signal) for name in (signal.SIGTERM, signal.SIGINT)}
    job = None
    try:
        submitted = _ssh(env, target, submit_command(repo, spec.name, length, monitor, manifest))
        print(submitted.stdout + submitted.stderr, end="", flush=True)
        if (found := re.search(r"Submitted batch job (\d+)", submitted.stdout)) is None:
            raise RuntimeError(f"`make serve-job` on {target} submitted no job (exit code {submitted.returncode}); its output is above")
        job = found.group(1)
        say(f"job {job} submitted on {target}; its log is plyserve-{job}.out in {repo}", False)
        _wait_until_ready(env, target, repo, job, f"{local}/health", key, timeout, say)
        say(f"the server of job {job} answers at {local}; running {shlex.join(command)}")
        try:
            client = subprocess.Popen(list(command), env=client_environment(env, f"{local}/v1", key, spec.name, spec.backend))
        except FileNotFoundError as error:
            raise RuntimeError(f"the command {command[0]!r} was not found") from error
        try:
            code = client.wait()
        finally:
            _stop(client, group=False)
        say(f"the command exited with code {code}")
        return code
    except _Stopped as stopped:
        say(f"{stopped}, stopping")
        return 128 + stopped.signum
    except RuntimeError as error:
        if notify:
            push(env, f"{spec.name}: {error}")
        raise
    finally:
        if job is not None:
            cancelled = _ssh(env, target, f"scancel {job}")
            if cancelled.returncode == 0:
                print(f"plyserve: cancelled job {job}", flush=True)
            else:
                say(f"scancel {job} on {target} exited with code {cancelled.returncode} ({cancelled.stderr.strip()}); if the job still runs, cancel it by hand")
        for name, handler in previous.items():
            signal.signal(name, handler)
