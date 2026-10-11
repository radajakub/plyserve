import json
import os
import signal
import sys
import textwrap
from pathlib import Path

import pytest

from plyserve import serve as serve_module
from plyserve import settings
from plyserve.cli import main
from plyserve.manifest import ModelSpec
from plyserve.serve import client_environment, serve

_SPEC = ModelSpec("m-fp8", "org/model", "abc123", 18001, ("--max-model-len", "1024"))

# stands in for the runtime's vllm: answers /health and an authenticated /v1/models, and records what it got.
# FAKE_HEALTH_DELAY delays readiness, FAKE_EXIT exits before listening, FAKE_LIFETIME exits after serving that long
_FAKE_VLLM = """
import http.server, json, os, sys, threading, time

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
key = os.environ["VLLM_API_KEY"]
with open(os.environ["FAKE_RECORD"], "w") as record:
    json.dump({"argv": args, "pid": os.getpid(), "key": key, "offline": os.environ.get("HF_HUB_OFFLINE"), "hf_home": os.environ.get("HF_HOME")}, record)
if "FAKE_EXIT" in os.environ:
    sys.exit(int(os.environ["FAKE_EXIT"]))
ready_at = time.monotonic() + float(os.environ.get("FAKE_HEALTH_DELAY", "0"))

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200 if time.monotonic() >= ready_at else 503)
        elif self.path == "/v1/models":
            self.send_response(200 if self.headers.get("Authorization") == f"Bearer {key}" else 401)
        else:
            self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass

server = http.server.HTTPServer(("127.0.0.1", port), Handler)
if "FAKE_LIFETIME" in os.environ:
    threading.Timer(float(os.environ["FAKE_LIFETIME"]), lambda: os._exit(0)).start()
server.serve_forever()
"""

# a client that checks the server with the variables it was given, records them and exits with code 3
_CLIENT = """
import json, os, sys, urllib.request
request = urllib.request.Request(os.environ["VLLM_BASE_URL"] + "/models", headers={"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
status = opener.open(request).status
names = ("VLLM_BASE_URL", "VLLM_API_KEY", "VLLM_MODEL")
json.dump({"status": status, **{name: os.environ.get(name) for name in names}}, open(sys.argv[1], "w"))
sys.exit(3)
"""


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    vllm = venv / "bin" / "vllm"
    vllm.write_text(f"#!{sys.executable}\n{textwrap.dedent(_FAKE_VLLM)}")
    vllm.chmod(0o755)
    (venv / "bin" / "python").symlink_to(sys.executable)
    monkeypatch.setattr(settings, "RUNTIME", venv)

    home = tmp_path / "home"
    (home / "hf" / "hub" / "models--org--model" / "snapshots" / "abc123").mkdir(parents=True)
    monkeypatch.setenv("PLYSERVE_HOME", str(home))
    monkeypatch.setenv("FAKE_RECORD", str(tmp_path / "vllm.json"))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(serve_module, "POLL_INTERVAL", 0.05)
    monkeypatch.setattr(serve_module, "STOP_GRACE", 5.0)
    return tmp_path


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_client_runs_against_the_server_and_its_code_is_returned(fake: Path) -> None:
    out = fake / "client.json"

    code = serve(_SPEC, command=[sys.executable, "-c", _CLIENT, str(out)])

    assert code == 3
    vllm, client = json.loads((fake / "vllm.json").read_text()), json.loads(out.read_text())
    assert client["status"] == 200  # the client reached /v1/models with the key it was given
    assert client["VLLM_API_KEY"] == vllm["key"]
    assert client["VLLM_MODEL"] == "m-fp8"
    port = vllm["argv"][vllm["argv"].index("--port") + 1]
    assert client["VLLM_BASE_URL"] == f"http://127.0.0.1:{port}/v1"
    assert vllm["argv"][:2] == ["serve", "org/model"] and vllm["argv"][-2:] == ["--max-model-len", "1024"]
    assert vllm["key"] not in vllm["argv"]
    assert vllm["offline"] == "1"
    assert vllm["hf_home"] == str(fake / "home" / "hf")
    assert _gone(vllm["pid"])


def test_run_record(fake: Path) -> None:
    serve(_SPEC, command=[sys.executable, "-c", "pass"])

    [record_file] = (fake / "home" / "runs").glob("*-m-fp8/server.json")
    record = json.loads(record_file.read_text())
    vllm = json.loads((fake / "vllm.json").read_text())
    assert record["model"] == "m-fp8" and record["repo"] == "org/model" and record["revision"] == "abc123"
    assert record["vllm_args"] == vllm["argv"]
    assert record["command"] == [sys.executable, "-c", "pass"]
    assert set(record["versions"]) == {"vllm", "torch", "transformers"}
    assert {"started_at", "ready_at", "plyserve", "gpus", "hostname"} <= set(record)
    assert vllm["key"] not in record_file.read_text()


def test_run_directory_is_named_after_the_slurm_job(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    serve(_SPEC, command=[sys.executable, "-c", "pass"])
    record = json.loads((fake / "home" / "runs" / "4242-m-fp8" / "server.json").read_text())
    assert record["slurm_job_id"] == "4242"


def test_waits_for_health(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HEALTH_DELAY", "0.5")
    out = fake / "client.json"
    assert serve(_SPEC, command=[sys.executable, "-c", _CLIENT, str(out)]) == 3
    assert json.loads(out.read_text())["status"] == 200


def test_vllm_that_exits_early_is_reported(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_EXIT", "7")
    with pytest.raises(RuntimeError, match="exited with code 7 before it was ready"):
        serve(_SPEC, command=[sys.executable, "-c", "pass"])
    assert not (fake / "home" / "runs").exists()


def test_timeout_stops_vllm(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HEALTH_DELAY", "60")
    with pytest.raises(RuntimeError, match="not ready after 0.3 s"):
        serve(_SPEC, timeout=0.3, command=[sys.executable, "-c", "pass"])
    assert _gone(json.loads((fake / "vllm.json").read_text())["pid"])


def test_vllm_dying_under_the_client_stops_the_client(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_LIFETIME", "0.5")
    marker = fake / "client-pid"
    client = f"import os, time; open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    with pytest.raises(RuntimeError, match="exited with code 0 while the client was running"):
        serve(_SPEC, command=[sys.executable, "-c", client])
    assert _gone(int(marker.read_text()))


def test_without_a_client_it_serves_until_vllm_exits(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_LIFETIME", "0.5")
    assert serve(_SPEC) == 0


def test_sigterm_stops_both(fake: Path) -> None:
    # the client sends plyserve the SIGTERM that Slurm sends at the time limit
    client = f"import os, signal, time; os.kill({os.getpid()}, signal.SIGTERM); time.sleep(60)"
    assert serve(_SPEC, command=[sys.executable, "-c", client]) == 128 + signal.SIGTERM
    assert _gone(json.loads((fake / "vllm.json").read_text())["pid"])
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_missing_weights_say_how_to_fetch_them(fake: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make download MODEL=other"):
        serve(ModelSpec("other", "org/other", "def456", 18002, ()))


def test_client_leaves_plyserve_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    venv = settings.REPOSITORY / ".venv"
    env = {"VIRTUAL_ENV": str(venv), "PATH": os.pathsep.join([str(venv / "bin"), "/usr/bin"]), "no_proxy": "example.org"}
    client = client_environment(env, "http://127.0.0.1:1/v1", "k", "m")
    assert "VIRTUAL_ENV" not in client
    assert client["PATH"] == "/usr/bin"
    assert client["no_proxy"] == "example.org,127.0.0.1,localhost" and client["NO_PROXY"] == "127.0.0.1,localhost"


def test_client_keeps_its_own_venv() -> None:
    env = {"VIRTUAL_ENV": "/elsewhere/.venv", "PATH": "/elsewhere/.venv/bin:/usr/bin"}
    client = client_environment(env, "http://127.0.0.1:1/v1", "k", "m")
    assert client["VIRTUAL_ENV"] == "/elsewhere/.venv" and client["PATH"] == env["PATH"]


@pytest.mark.parametrize("argv", [["serve", "m-fp8", "--", "cmd", "-x"], ["serve", "--port", "9", "m-fp8", "--", "cmd", "-x"]])
def test_cli_passes_the_client_command(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[ModelSpec, dict[str, object]]] = []
    monkeypatch.setattr("plyserve.cli.serve", lambda spec, **kwargs: calls.append((spec, kwargs)) or 5)
    monkeypatch.setattr("plyserve.cli.load_manifest", lambda path: {"m-fp8": _SPEC})
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 5
    [(spec, kwargs)] = calls
    assert spec == _SPEC and kwargs["command"] == ["cmd", "-x"]


def test_cli_without_a_client_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("plyserve.cli.serve", lambda spec, **kwargs: calls.append(kwargs) or 0)
    monkeypatch.setattr("plyserve.cli.load_manifest", lambda path: {"m-fp8": _SPEC})
    with pytest.raises(SystemExit):
        main(["serve", "m-fp8", "--"])
    assert calls[0]["command"] == [] and calls[0]["port"] is None
