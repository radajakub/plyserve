import http.server
import json
import os
import signal
import sys
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from plyserve import serve as serve_module
from plyserve import settings
from plyserve.cli import main
from plyserve.endpoint import Endpoint, endpoint_path, write_endpoint
from plyserve.manifest import ModelSpec
from plyserve.serve import addresses, client_environment, serve

_SPEC = ModelSpec("m-fp8", "org/model", "abc123", 18001, ("--max-model-len", "1024"))

# stands in for the runtime's vllm: answers /health and an authenticated /v1/models, and records what it got.
# FAKE_HEALTH_DELAY delays readiness, FAKE_EXIT exits before listening, FAKE_LIFETIME exits after serving that long
_FAKE_VLLM = """
import http.server, json, os, sys, threading, time

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
key = os.environ["VLLM_API_KEY"]
with open(os.environ["FAKE_RECORD"], "w") as record:
    json.dump({"argv": args, "pid": os.getpid(), "key": key, "offline": os.environ.get("HF_HUB_OFFLINE"), "hf_home": os.environ.get("HF_HOME"), "deep_gemm": os.environ.get("VLLM_USE_DEEP_GEMM")}, record)
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

server = http.server.HTTPServer((args[args.index("--host") + 1], port), Handler)
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
    monkeypatch.delenv("PLYSERVE_API_KEY", raising=False)
    # --expose on the loopback address, so the tests open no port to the network
    monkeypatch.setenv("PLYSERVE_EXPOSE_BIND", "127.0.0.1")
    monkeypatch.setenv("PLYSERVE_EXPOSE_HOST", "127.0.0.1")
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
    assert vllm["deep_gemm"] == "0"
    assert _gone(vllm["pid"])


def test_run_record(fake: Path) -> None:
    serve(_SPEC, command=[sys.executable, "-c", "pass"])

    [record_file] = (fake / "home" / "runs").glob("*-m-fp8/server.json")
    record = json.loads(record_file.read_text())
    vllm = json.loads((fake / "vllm.json").read_text())
    assert record["model"] == "m-fp8" and record["repo"] == "org/model" and record["revision"] == "abc123"
    assert record["vllm_args"] == vllm["argv"]
    assert record["command"] == [sys.executable, "-c", "pass"]
    assert record["vllm_env"] == {"VLLM_USE_DEEP_GEMM": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    assert set(record["versions"]) == {"vllm", "torch", "transformers"}
    assert {"started_at", "ready_at", "plyserve", "gpus", "hostname"} <= set(record)
    assert vllm["key"] not in record_file.read_text()


def test_run_directory_is_named_after_the_slurm_job(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    serve(_SPEC, command=[sys.executable, "-c", "pass"])
    record = json.loads((fake / "home" / "runs" / "4242-m-fp8" / "server.json").read_text())
    assert record["slurm_job_id"] == "4242"


def test_shell_can_turn_a_runtime_default_back_on(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # e.g. once a CUDA toolkit is available
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    serve(_SPEC, command=[sys.executable, "-c", "pass"])
    assert json.loads((fake / "vllm.json").read_text())["deep_gemm"] == "1"


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


def test_fixed_api_key_from_the_environment(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLYSERVE_API_KEY", "fixed-key")
    out = fake / "client.json"
    serve(_SPEC, command=[sys.executable, "-c", _CLIENT, str(out)])
    assert json.loads(out.read_text())["VLLM_API_KEY"] == "fixed-key"
    [record_file] = (fake / "home" / "runs").glob("*/server.json")
    assert json.loads(record_file.read_text())["api_key"] == "fixed (PLYSERVE_API_KEY)"
    assert "fixed-key" not in record_file.read_text()


def test_expose_publishes_an_endpoint_while_serving(fake: Path) -> None:
    # the client stands in for a client job: it finds the server through `plyserve endpoint` and calls it
    out, manifest = fake / "client.json", fake / "models.toml"
    manifest.write_text('[models."m-fp8"]\nrepo = "org/model"\nrevision = "abc123"\ntunnel_port = 18001\n')
    client_py = fake / "client.py"
    client_py.write_text(_CLIENT)
    src = Path(__file__).resolve().parents[1] / "src"
    script = f'eval "$(PYTHONPATH={src} {sys.executable} -m plyserve --manifest {manifest} endpoint m-fp8)" && {sys.executable} {client_py} {out}'

    assert serve(_SPEC, expose=True, command=["bash", "-c", script]) == 3

    vllm, client = json.loads((fake / "vllm.json").read_text()), json.loads(out.read_text())
    port = vllm["argv"][vllm["argv"].index("--port") + 1]
    assert vllm["argv"][vllm["argv"].index("--host") + 1] == "127.0.0.1"
    assert client["status"] == 200 and client["VLLM_API_KEY"] == vllm["key"]
    assert client["VLLM_BASE_URL"] == f"http://127.0.0.1:{port}/v1"
    assert not endpoint_path("m-fp8").exists()  # removed once the server stopped
    [record_file] = (fake / "home" / "runs").glob("*/server.json")
    assert json.loads(record_file.read_text())["exposed"] is True


@pytest.fixture
def healthy_server() -> Iterator[str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def _endpoint(base_url: str, server_id: str = "other") -> Endpoint:
    return Endpoint("m-fp8", base_url, "k", server_id, "elsewhere", "99", "2026-10-11T00:00:00+00:00", "/runs/99")


def test_expose_refuses_a_second_server_for_the_model(fake: Path, healthy_server: str) -> None:
    write_endpoint(_endpoint(healthy_server))
    with pytest.raises(RuntimeError, match=r"already served at .*\(job 99 on elsewhere\)"):
        serve(_SPEC, expose=True)
    assert not (fake / "vllm.json").exists()  # vLLM was never started


def test_expose_replaces_a_stale_endpoint(fake: Path) -> None:
    write_endpoint(_endpoint("http://127.0.0.1:9/v1"))  # nothing answers there
    assert serve(_SPEC, expose=True, command=[sys.executable, "-c", "pass"]) == 0
    assert not endpoint_path("m-fp8").exists()


def test_expose_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("socket.gethostname", lambda: "node7")
    assert addresses({}, expose=False) == ("127.0.0.1", "127.0.0.1")
    assert addresses({}, expose=True) == ("0.0.0.0", "node7")
    assert addresses({"PLYSERVE_EXPOSE_BIND": "10.0.0.5", "PLYSERVE_EXPOSE_HOST": "gpu7.ib"}, expose=True) == ("10.0.0.5", "gpu7.ib")


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
    assert calls[0]["command"] == [] and calls[0]["port"] is None and calls[0]["expose"] is False


def test_cli_expose(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("plyserve.cli.serve", lambda spec, **kwargs: calls.append(kwargs) or 0)
    monkeypatch.setattr("plyserve.cli.load_manifest", lambda path: {"m-fp8": _SPEC})
    with pytest.raises(SystemExit):
        main(["serve", "--expose", "m-fp8"])
    assert calls[0]["expose"] is True and calls[0]["command"] == []


# stands in for ssh: records each start (argv, pid) as a line in FAKE_SSH_RECORD, stays up for the next of the comma-separated FAKE_SSH_LIFETIMES
# (the last one repeats) and exits with 255, as ssh does when its connection fails
_FAKE_SSH = """
import json, os, sys, time
record = os.environ["FAKE_SSH_RECORD"]
starts = open(record).read().count("\\n") if os.path.exists(record) else 0
with open(record, "a") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:], "pid": os.getpid()}) + "\\n")
lifetimes = [float(value) for value in os.environ.get("FAKE_SSH_LIFETIMES", "60").split(",")]
time.sleep(lifetimes[min(starts, len(lifetimes) - 1)])
sys.exit(255)
"""


@pytest.fixture
def fake_ssh(fake: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = fake / "ssh-bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(f"#!{sys.executable}\n{textwrap.dedent(_FAKE_SSH)}")
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), os.environ["PATH"]]))
    monkeypatch.setenv("FAKE_SSH_RECORD", str(fake / "ssh.jsonl"))
    monkeypatch.setenv("PLYSERVE_API_KEY", "fixed-key")
    monkeypatch.setenv("PLYSERVE_TUNNEL_HOST", "tunnel@vps")
    monkeypatch.delenv("PLYSERVE_TUNNEL_KEY", raising=False)
    monkeypatch.delenv("PLYSERVE_TUNNEL_BIND", raising=False)
    monkeypatch.setattr(serve_module, "TUNNEL_SETTLE", 0.3)
    monkeypatch.setattr(serve_module, "TUNNEL_RETRY", (0.05, 0.2))
    return fake / "ssh.jsonl"


def _starts(record: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in record.read_text().splitlines()]


def test_tunnel_forwards_the_model_port_while_serving(fake: Path, fake_ssh: Path) -> None:
    # the client checks that the tunnel is up while it runs
    client = f"import os, sys; sys.exit(0 if os.path.exists({str(fake_ssh)!r}) else 9)"
    assert serve(_SPEC, tunnel=True, command=[sys.executable, "-c", client]) == 0

    vllm = json.loads((fake / "vllm.json").read_text())
    port = vllm["argv"][vllm["argv"].index("--port") + 1]
    [start] = _starts(fake_ssh)
    assert start["argv"][:3] == ["-N", "-R", f"127.0.0.1:18001:127.0.0.1:{port}"]
    assert "BatchMode=yes" in start["argv"] and "ExitOnForwardFailure=yes" in start["argv"]
    assert start["argv"][-1] == "tunnel@vps"
    assert _gone(start["pid"]) and _gone(vllm["pid"])
    [record_file] = (fake / "home" / "runs").glob("*/server.json")
    assert json.loads(record_file.read_text())["tunnel"] == ["ssh", *start["argv"]]


def test_tunnel_key_and_bind_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/u")
    env = {"PLYSERVE_TUNNEL_HOST": "vps", "PLYSERVE_TUNNEL_KEY": "~/.ssh/plyserve", "PLYSERVE_TUNNEL_BIND": "172.17.0.1"}
    argv = serve_module.tunnel_argv(env, 18002, "127.0.0.1:5000")
    assert argv[3] == "172.17.0.1:18002:127.0.0.1:5000"
    assert argv[-5:] == ["-i", "/home/u/.ssh/plyserve", "-o", "IdentitiesOnly=yes", "vps"]


@pytest.mark.parametrize(("unset", "message"), [("PLYSERVE_API_KEY", "fixed PLYSERVE_API_KEY"), ("PLYSERVE_TUNNEL_HOST", "PLYSERVE_TUNNEL_HOST")])
def test_tunnel_needs_its_settings(fake: Path, fake_ssh: Path, monkeypatch: pytest.MonkeyPatch, unset: str, message: str) -> None:
    monkeypatch.delenv(unset)
    with pytest.raises(RuntimeError, match=message):
        serve(_SPEC, tunnel=True)
    assert not (fake / "vllm.json").exists()  # vLLM was never started


def test_tunnel_that_fails_at_first_stops_serve(fake: Path, fake_ssh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SSH_LIFETIMES", "0")
    with pytest.raises(RuntimeError, match="ssh exited with code 255 while opening the tunnel to tunnel@vps"):
        serve(_SPEC, tunnel=True, command=[sys.executable, "-c", "pass"])
    assert len(_starts(fake_ssh)) == 1  # no retry
    assert _gone(json.loads((fake / "vllm.json").read_text())["pid"])


def test_tunnel_reconnects_after_a_drop(fake: Path, fake_ssh: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # up long enough to count as connected, then two quick failures, then up for good
    monkeypatch.setenv("FAKE_SSH_LIFETIMES", "0.5,0,0,60")
    client = f"import time, sys\nwhile open({str(fake_ssh)!r}).read().count('\\n') < 4: time.sleep(0.05)\ntime.sleep(0.3)"
    assert serve(_SPEC, tunnel=True, command=[sys.executable, "-c", client]) == 0

    starts = _starts(fake_ssh)
    assert len(starts) == 4 and all(start["argv"] == starts[0]["argv"] for start in starts)
    assert all(_gone(start["pid"]) for start in starts)
    out = capsys.readouterr().out
    assert "dropped (ssh exit code 255); reconnecting in 0.05 s" in out and "reconnecting in 0.1 s" in out and "reconnecting in 0.2 s" in out


def test_sigterm_stops_the_tunnel(fake: Path, fake_ssh: Path) -> None:
    assert serve(_SPEC, tunnel=True, command=[sys.executable, "-c", f"import os, signal, time; os.kill({os.getpid()}, signal.SIGTERM); time.sleep(60)"]) == 128 + signal.SIGTERM
    [start] = _starts(fake_ssh)
    assert _gone(start["pid"])


def test_cli_tunnel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("plyserve.cli.serve", lambda spec, **kwargs: calls.append(kwargs) or 0)
    monkeypatch.setattr("plyserve.cli.load_manifest", lambda path: {"m-fp8": _SPEC})
    with pytest.raises(SystemExit):
        main(["serve", "--tunnel", "m-fp8"])
    assert calls[0]["tunnel"] is True and calls[0]["expose"] is False


_OLLAMA_SPEC = ModelSpec("m-q8", "org/model-gguf", "def456", 18003, (), "ollama", "model-Q8_0.gguf", (("num_ctx", 4096),))

# stands in for the Ollama release: `create` records its Modelfile; `serve` answers Ollama's API without a key on OLLAMA_HOST and records its environment.
# FAKE_VRAM is how many of the model's 100 bytes end up on the GPU
_FAKE_OLLAMA = """
import http.server, json, os, sys

record = os.environ["FAKE_RECORD"]
if sys.argv[1] == "create":
    with open(record + ".create", "w") as handle:
        json.dump({"argv": sys.argv[2:], "modelfile": open(sys.argv[sys.argv.index("-f") + 1]).read(), "host": os.environ["OLLAMA_HOST"]}, handle)
    # a spinner with terminal control codes, as the real CLI writes it, then FAKE_CREATE_EXIT
    sys.stderr.write("\\x1b[?25l\\x1b[1Ggathering model components \\u280b \\x1b[K\\r\\x1b[1Ggathering model components \\u2819 \\x1b[K\\n")
    if "FAKE_CREATE_EXIT" in os.environ:
        sys.stderr.write("Error: invalid file magic\\n")
    sys.exit(int(os.environ.get("FAKE_CREATE_EXIT", "0")))
assert sys.argv[1:] == ["serve"]
with open(record, "w") as handle:
    json.dump({"pid": os.getpid(), "env": {name: value for name, value in os.environ.items() if name.startswith("OLLAMA_")}}, handle)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.40.3"})
        elif self.path == "/api/ps":
            self._send({"models": [{"name": "m-q8:latest", "model": "m-q8:latest", "size": 100, "size_vram": int(os.environ.get("FAKE_VRAM", "100"))}]})
        elif self.path == "/v1/models":
            self._send({"object": "list", "data": [{"id": "m-q8:latest"}], "authorization": self.headers.get("Authorization")})
        else:
            self._send({}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/generate":
            with open(record + ".load", "w") as handle:
                json.dump(body, handle)
            self._send({"model": body["model"], "done": True, "done_reason": "load"})
        elif self.path == "/api/show":
            self._send({"modelfile": "FROM /blobs/sha256-x", "renderer": "qwen3", "details": {"format": "gguf"}})
        else:
            self._send({}, 404)

    def _send(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass

host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
http.server.HTTPServer((host, int(port)), Handler).serve_forever()
"""

# a client that calls /v1/models with the key it was given and without one, and records both answers
_KEY_CLIENT = """
import json, os, sys, urllib.error, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def call(headers):
    try:
        response = opener.open(urllib.request.Request(os.environ["VLLM_BASE_URL"] + "/models", headers=headers))
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, None
with_key, without_key = call({"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}), call({})
json.dump({"with_key": with_key, "without_key": without_key[0], "VLLM_MODEL": os.environ["VLLM_MODEL"]}, open(sys.argv[1], "w"))
"""


@pytest.fixture
def fake_ollama(fake: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runtime = fake / "ollama-runtime"
    (runtime / "bin").mkdir(parents=True)
    ollama = runtime / "bin" / "ollama"
    ollama.write_text(f"#!{sys.executable}\n{textwrap.dedent(_FAKE_OLLAMA)}")
    ollama.chmod(0o755)
    monkeypatch.setattr(settings, "OLLAMA_RUNTIME", runtime)
    snapshot = fake / "home" / "hf" / "hub" / "models--org--model-gguf" / "snapshots" / "def456"
    snapshot.mkdir(parents=True)
    (snapshot / "model-Q8_0.gguf").write_bytes(b"GGUF")
    for name in ("OLLAMA_KEEP_ALIVE", "OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_NUM_PARALLEL", "FAKE_CREATE_EXIT"):
        monkeypatch.delenv(name, raising=False)
    return snapshot


def test_ollama_serves_behind_the_key_proxy(fake: Path, fake_ollama: Path) -> None:
    out = fake / "client.json"
    assert serve(_OLLAMA_SPEC, command=[sys.executable, "-c", _KEY_CLIENT, str(out)]) == 0

    client, ollama = json.loads(out.read_text()), json.loads((fake / "vllm.json").read_text())
    status, models = client["with_key"]
    assert status == 200 and models["authorization"] is None  # the key stops at the proxy
    assert client["without_key"] == 401 and client["VLLM_MODEL"] == "m-q8"
    host = ollama["env"]["OLLAMA_HOST"]
    assert host.startswith("127.0.0.1:")  # Ollama itself is reachable only on the loopback address
    assert ollama["env"]["OLLAMA_MODELS"] == str(fake / "home" / "ollama")
    assert ollama["env"]["OLLAMA_KEEP_ALIVE"] == "-1" and ollama["env"]["OLLAMA_NOPRUNE"] == "1"
    created = json.loads((fake / "vllm.json.create").read_text())
    assert created["argv"][0] == "m-q8" and created["host"] == host
    assert created["modelfile"] == f"FROM {fake_ollama / 'model-Q8_0.gguf'}\nPARAMETER num_ctx 4096\n"
    assert json.loads((fake / "vllm.json.load").read_text()) == {"model": "m-q8", "keep_alive": -1, "stream": False}
    assert _gone(ollama["pid"])

    [record_file] = (fake / "home" / "runs").glob("*-m-q8/server.json")
    record = json.loads(record_file.read_text())
    assert record["backend"] == "ollama" and record["file"] == "model-Q8_0.gguf" and record["versions"] == {"ollama": "0.40.3"}
    assert record["ollama_model"]["renderer"] == "qwen3" and record["memory"] == {"size": 100, "size_vram": 100}
    assert record["ollama_env"]["OLLAMA_HOST"] == host and "vllm_args" not in record


def test_ollama_reachable_from_another_job_through_the_endpoint(fake: Path, fake_ollama: Path) -> None:
    out, manifest = fake / "client.json", fake / "models.toml"
    manifest.write_text('[models."m-q8"]\nbackend = "ollama"\nrepo = "org/model-gguf"\nrevision = "def456"\nfile = "model-Q8_0.gguf"\ntunnel_port = 18003\n')
    client_py = fake / "client.py"
    client_py.write_text(_KEY_CLIENT)
    src = Path(__file__).resolve().parents[1] / "src"
    script = f'eval "$(PYTHONPATH={src} {sys.executable} -m plyserve --manifest {manifest} endpoint m-q8)" && {sys.executable} {client_py} {out}'

    assert serve(_OLLAMA_SPEC, expose=True, command=["bash", "-c", script]) == 0

    client = json.loads(out.read_text())
    assert client["with_key"][0] == 200 and client["without_key"] == 401
    assert not endpoint_path("m-q8").exists()


@pytest.mark.parametrize(("vram", "message"), [("60", "only 0.0 of 0.0 GiB of m-q8 on the GPU"), ("0", "found no usable GPU")])
def test_ollama_not_fully_on_the_gpu_stops(fake: Path, fake_ollama: Path, monkeypatch: pytest.MonkeyPatch, vram: str, message: str) -> None:
    monkeypatch.setenv("FAKE_VRAM", vram)
    with pytest.raises(RuntimeError, match=message):
        serve(_OLLAMA_SPEC, command=[sys.executable, "-c", "pass"])
    assert _gone(json.loads((fake / "vllm.json").read_text())["pid"])
    assert not (fake / "home" / "runs").exists()


def test_ollama_missing_gguf_says_how_to_fetch_it(fake: Path, fake_ollama: Path) -> None:
    (fake_ollama / "model-Q8_0.gguf").unlink()
    with pytest.raises(FileNotFoundError, match="model-Q8_0.gguf does not exist; fetch the weights with `make download MODEL=m-q8`"):
        serve(_OLLAMA_SPEC)


def test_ollama_missing_runtime_says_how_to_unpack_it(fake: Path, fake_ollama: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OLLAMA_RUNTIME", fake / "missing")
    with pytest.raises(FileNotFoundError, match="make runtime-ollama"):
        serve(_OLLAMA_SPEC)


def test_ollama_create_output_only_when_it_fails(fake: Path, fake_ollama: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]) -> None:
    serve(_OLLAMA_SPEC, command=[sys.executable, "-c", "pass"])
    assert "gathering model components" not in capfd.readouterr().out
    monkeypatch.setenv("FAKE_CREATE_EXIT", "1")
    with pytest.raises(RuntimeError, match="`ollama create` exited with code 1"):
        serve(_OLLAMA_SPEC, command=[sys.executable, "-c", "pass"])
    out = capfd.readouterr().out
    assert "Error: invalid file magic" in out and "\x1b" not in out
    assert _gone(json.loads((fake / "vllm.json").read_text())["pid"])


def test_missing_client_command_is_named(fake: Path) -> None:
    with pytest.raises(RuntimeError, match="the client command 'no-such-client' was not found"):
        serve(_SPEC, command=["no-such-client", "x"])
    assert _gone(json.loads((fake / "vllm.json").read_text())["pid"])
