import http.server
import json
import os
import shlex
import signal
import sys
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from plyserve import remote as remote_module
from plyserve.cli import main
from plyserve.manifest import ModelSpec
from plyserve.remote import remote, submit_command

# stands in for ssh to the login node: records each remote command in FAKE_REMOTE_RECORD and answers it. `make serve-job` submits job 42 (FAKE_SUBMIT_FAIL: no job);
# squeue prints the next of the comma-separated FAKE_STATES (the last one repeats; SSHFAIL exits 255 as a failed connection) and, at RUNNING, starts the
# server behind the tunnel by creating FAKE_READY (unless FAKE_NEVER_READY); scancel exits with FAKE_SCANCEL_CODE
_FAKE_SSH = """
import json, os, sys
command = sys.argv[-1]
record = os.environ["FAKE_REMOTE_RECORD"]
calls = [json.loads(line) for line in open(record)] if os.path.exists(record) else []
with open(record, "a") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")
if "make serve-job" in command:
    print("sbatch -p h200fast -t 4:00:00 slurm/serve_model.sbatch ...")
    if "FAKE_SUBMIT_FAIL" in os.environ:
        print("sbatch: error: invalid partition", file=sys.stderr)
        sys.exit(2)
    print("Submitted batch job 42")
elif command.startswith("squeue"):
    states = os.environ.get("FAKE_STATES", "RUNNING").split(",")
    state = states[min(sum(call["argv"][-1].startswith("squeue") for call in calls), len(states) - 1)]
    if state == "SSHFAIL":
        sys.exit(255)
    if state == "RUNNING" and "FAKE_NEVER_READY" not in os.environ:
        open(os.environ["FAKE_READY"], "w").close()
    print(state)
elif command.startswith("scancel"):
    code = int(os.environ.get("FAKE_SCANCEL_CODE", "0"))
    if code:
        print("scancel: error: Invalid job id specified", file=sys.stderr)
    sys.exit(code)
"""

# a client that records the variables it was given and exits with code 3
_CLIENT = """
import json, os, sys
json.dump({name: os.environ.get(name) for name in ("VLLM_BASE_URL", "VLLM_API_KEY", "VLLM_MODEL")}, open(sys.argv[1], "w"))
sys.exit(3)
"""


@pytest.fixture
def tunnel(tmp_path: Path) -> Iterator[int]:
    # the server as the client machine sees it through the tunnel: /health answers once FAKE_READY exists
    ready = tmp_path / "ready"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200 if ready.exists() else 503)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


@pytest.fixture
def fake(tmp_path: Path, tunnel: int, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(f"#!{sys.executable}\n{textwrap.dedent(_FAKE_SSH)}")
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), os.environ["PATH"]]))
    monkeypatch.setenv("FAKE_REMOTE_RECORD", str(tmp_path / "ssh.jsonl"))
    monkeypatch.setenv("FAKE_READY", str(tmp_path / "ready"))
    monkeypatch.setenv("PLYSERVE_REMOTE", "me@login")
    monkeypatch.setenv("PLYSERVE_REMOTE_REPO", "/shared/me/plyserve")
    monkeypatch.setenv("PLYSERVE_API_KEY", "fixed-key")
    for name in ("PLYSERVE_TUNNEL_BIND", "NTFY_URL", "NTFY_TOPIC", "NTFY_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(remote_module, "QUEUE_INTERVAL", 0.05)
    monkeypatch.setattr(remote_module, "POLL_INTERVAL", 0.05)
    return tmp_path


def _spec(port: int) -> ModelSpec:
    return ModelSpec("m-fp8", "org/model", "abc123", port, ())


def _commands(fake: Path) -> list[str]:
    path = fake / "ssh.jsonl"
    return [json.loads(line)["argv"][-1] for line in path.read_text().splitlines()] if path.exists() else []


def test_submits_waits_runs_the_command_and_cancels(fake: Path, tunnel: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STATES", "PENDING,PENDING,RUNNING")
    out = fake / "client.json"

    assert remote(_spec(tunnel), [sys.executable, "-c", _CLIENT, str(out)]) == 3

    commands = _commands(fake)
    assert shlex.split(commands[0]) == ["bash", "-lc", "cd /shared/me/plyserve && make serve-job LENGTH=short ARGS='--tunnel m-fp8'"]
    assert commands[1:-1] == ["squeue -h -j 42 -o %T"] * 3
    assert commands[-1] == "scancel 42"
    argv = json.loads((fake / "ssh.jsonl").read_text().splitlines()[0])["argv"]
    assert argv[:-1] == ["-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "me@login"]
    assert json.loads(out.read_text()) == {"VLLM_BASE_URL": f"http://127.0.0.1:{tunnel}/v1", "VLLM_API_KEY": "fixed-key", "VLLM_MODEL": "m-fp8"}


def test_options_reach_make_serve_job() -> None:
    command = submit_command("/shared/me/plyserve", "m-fp8", "long", True, "/shared/me/exp/models mtp.toml")
    # in a login shell on the login node, so the job gets what slurm/site.sh needs from it
    assert shlex.split(command) == [
        "bash",
        "-lc",
        "cd /shared/me/plyserve && PLYSERVE_MANIFEST='/shared/me/exp/models mtp.toml' make serve-job LENGTH=long ARGS='--tunnel --monitor m-fp8'",
    ]


def test_job_that_ends_before_the_server_answers(fake: Path, tunnel: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STATES", "PENDING,RUNNING,FAILED")
    monkeypatch.setenv("FAKE_NEVER_READY", "1")
    with pytest.raises(RuntimeError, match=r"job 42 ended \(FAILED\) before the server answered; its log is plyserve-42.out in /shared/me/plyserve on me@login"):
        remote(_spec(tunnel), [sys.executable, "-c", "pass"])
    assert _commands(fake)[-1] == "scancel 42"


def test_job_that_left_the_queue(fake: Path, tunnel: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STATES", "")
    with pytest.raises(RuntimeError, match=r"job 42 ended \(it left the queue\)"):
        remote(_spec(tunnel), [sys.executable, "-c", "pass"])


def test_server_that_never_answers_times_out(fake: Path, tunnel: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_NEVER_READY", "1")
    with pytest.raises(RuntimeError, match=r"job 42 runs, but the server did not answer .* within 0.3 s"):
        remote(_spec(tunnel), [sys.executable, "-c", "pass"], timeout=0.3)
    assert _commands(fake)[-1] == "scancel 42"


def test_unreachable_login_node_is_retried(fake: Path, tunnel: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("FAKE_STATES", "SSHFAIL,RUNNING")
    assert remote(_spec(tunnel), [sys.executable, "-c", "pass"]) == 0
    assert "me@login did not answer the squeue call; trying again" in capsys.readouterr().out


def test_failed_submit_cancels_nothing(fake: Path, tunnel: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("FAKE_SUBMIT_FAIL", "1")
    with pytest.raises(RuntimeError, match=r"submitted no job \(exit code 2\)"):
        remote(_spec(tunnel), [sys.executable, "-c", "pass"])
    assert len(_commands(fake)) == 1
    assert "invalid partition" in capsys.readouterr().out


def test_a_server_already_on_the_port_stops_it_before_submitting(fake: Path, tunnel: int) -> None:
    (fake / "ready").touch()
    with pytest.raises(RuntimeError, match=f"a server already answers at http://127.0.0.1:{tunnel}"):
        remote(_spec(tunnel), [sys.executable, "-c", "pass"])
    assert _commands(fake) == []


@pytest.mark.parametrize(("unset", "message"), [("PLYSERVE_REMOTE", "PLYSERVE_REMOTE"), ("PLYSERVE_REMOTE_REPO", "PLYSERVE_REMOTE_REPO"), ("PLYSERVE_API_KEY", "PLYSERVE_API_KEY")])
def test_needs_its_settings(fake: Path, tunnel: int, monkeypatch: pytest.MonkeyPatch, unset: str, message: str) -> None:
    monkeypatch.delenv(unset)
    with pytest.raises(RuntimeError, match=message):
        remote(_spec(tunnel), [sys.executable, "-c", "pass"])
    assert _commands(fake) == []


def test_sigterm_cancels_the_job(fake: Path, tunnel: int) -> None:
    command = [sys.executable, "-c", f"import os, signal, time; os.kill({os.getpid()}, signal.SIGTERM); time.sleep(60)"]
    assert remote(_spec(tunnel), command) == 128 + signal.SIGTERM
    assert _commands(fake)[-1] == "scancel 42"


def test_failed_scancel_says_to_cancel_by_hand(fake: Path, tunnel: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("FAKE_SCANCEL_CODE", "1")
    assert remote(_spec(tunnel), [sys.executable, "-c", "pass"]) == 0
    assert "scancel 42 on me@login exited with code 1 (scancel: error: Invalid job id specified); if the job still runs, cancel it by hand" in capsys.readouterr().out


@pytest.fixture
def ntfy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, str, str | None]]]:
    # records each push as (path, body, Authorization header)
    pushes: list[tuple[str, str, str | None]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            pushes.append((self.path, body, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("NTFY_URL", f"http://127.0.0.1:{server.server_address[1]}/")
    monkeypatch.setenv("NTFY_TOPIC", "runs")
    monkeypatch.setenv("NTFY_TOKEN", "tk")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    yield pushes
    server.shutdown()


def test_notify_pushes_start_ready_and_end(fake: Path, tunnel: int, ntfy: list[tuple[str, str, str | None]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STATES", "PENDING,RUNNING")
    assert remote(_spec(tunnel), [sys.executable, "-c", "import sys; sys.exit(3)"], notify=True) == 3
    assert [body for _, body, _ in ntfy] == [
        "m-fp8: job 42 is running",
        f"m-fp8: the server of job 42 answers at http://127.0.0.1:{tunnel}; running {sys.executable} -c 'import sys; sys.exit(3)'",
        "m-fp8: the command exited with code 3",
    ]
    assert all(path == "/runs" and auth == "Bearer tk" for path, _, auth in ntfy)


def test_notify_pushes_a_failure(fake: Path, tunnel: int, ntfy: list[tuple[str, str, str | None]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STATES", "FAILED")
    with pytest.raises(RuntimeError):
        remote(_spec(tunnel), [sys.executable, "-c", "pass"], notify=True)
    assert ntfy[-1][1].startswith("m-fp8: job 42 ended (FAILED)")


def test_notify_needs_ntfy_settings(fake: Path, tunnel: int) -> None:
    with pytest.raises(RuntimeError, match="--notify needs NTFY_URL and NTFY_TOPIC"):
        remote(_spec(tunnel), [sys.executable, "-c", "pass"], notify=True)
    assert _commands(fake) == []


def test_cli_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr("plyserve.cli.remote", lambda spec, command, **kwargs: calls.append((command, kwargs)) or 0)
    monkeypatch.setattr("plyserve.cli.load_manifest", lambda path: {"m-fp8": _spec(18001)})
    with pytest.raises(SystemExit):
        main(["remote", "--length", "long", "--monitor", "--remote-manifest", "/x/models.toml", "--notify", "m-fp8", "--", "uv", "run", "bench.py", "--levels", "64"])
    command, kwargs = calls[0]
    assert command == ["uv", "run", "bench.py", "--levels", "64"]
    assert kwargs == {"length": "long", "monitor": True, "manifest": "/x/models.toml", "timeout": 3600.0, "notify": True}


def test_cli_remote_needs_a_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("plyserve.cli.load_manifest", lambda path: {"m-fp8": _spec(18001)})
    with pytest.raises(SystemExit, match="usage: plyserve remote"):
        main(["remote", "m-fp8"])
