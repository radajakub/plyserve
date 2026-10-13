import http.server
import os
import stat
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from plyserve.cli import main
from plyserve.endpoint import Endpoint, endpoint_path, find_endpoint, read_endpoint, remove_endpoint, shell_exports, write_endpoint
from plyserve.manifest import ModelSpec


@pytest.fixture(autouse=True)
def plyserve_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PLYSERVE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def server() -> Iterator[str]:
    # answers /health with 200 when the key matches
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200 if self.path == "/health" and self.headers.get("Authorization") == "Bearer k" else 401)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def _endpoint(base_url: str, server_id: str = "a1", key: str = "k") -> Endpoint:
    return Endpoint("m-fp8", base_url, key, server_id, "node7", "42", "2026-10-11T00:00:00+00:00", "/runs/42-m-fp8")


def test_round_trip_and_owner_only(plyserve_home: Path) -> None:
    endpoint = _endpoint("http://node7:4242/v1")
    path = write_endpoint(endpoint)
    assert path == plyserve_home / "endpoints" / "m-fp8.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600  # it holds the API key
    assert read_endpoint("m-fp8") == endpoint
    assert endpoint.health_url == "http://node7:4242/health"
    assert [p.name for p in path.parent.iterdir()] == ["m-fp8.json"]  # no temporary file left behind


def test_garbage_file_is_reported() -> None:
    path = endpoint_path("m-fp8")
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "m-fp8"}')
    with pytest.raises(ValueError, match="not a plyserve endpoint file"):
        read_endpoint("m-fp8")


def test_remove_leaves_a_later_servers_file() -> None:
    write_endpoint(_endpoint("http://a:1/v1", server_id="later"))
    remove_endpoint(_endpoint("http://a:1/v1", server_id="earlier"))
    assert read_endpoint("m-fp8") is not None
    remove_endpoint(_endpoint("http://a:1/v1", server_id="later"))
    assert read_endpoint("m-fp8") is None


def test_find_a_running_server(server: str) -> None:
    write_endpoint(_endpoint(server))
    assert find_endpoint("m-fp8").base_url == server


def test_find_without_a_server() -> None:
    with pytest.raises(RuntimeError, match="no server for m-fp8: .* does not exist"):
        find_endpoint("m-fp8")


def test_find_a_server_that_does_not_answer(server: str) -> None:
    write_endpoint(_endpoint(server, key="wrong"))  # answers 401, as a server of someone else would
    with pytest.raises(RuntimeError, match="does not answer"):
        find_endpoint("m-fp8")


def test_wait_until_the_server_appears(server: str) -> None:
    timer = threading.Timer(0.3, lambda: write_endpoint(_endpoint(server)))
    timer.start()
    started = time.monotonic()
    assert find_endpoint("m-fp8", wait=True, timeout=10, interval=0.05).base_url == server
    assert time.monotonic() - started >= 0.3


def test_wait_gives_up_after_the_timeout() -> None:
    with pytest.raises(RuntimeError, match="within 0.2 s"):
        find_endpoint("m-fp8", wait=True, timeout=0.2, interval=0.05)


def test_shell_exports_survive_eval() -> None:
    endpoint = _endpoint("http://node7:4242/v1", key="it's $secret")
    script = f'no_proxy=example.org; unset NO_PROXY; {shell_exports(endpoint)}\nprintf "%s|" "$VLLM_BASE_URL" "$VLLM_API_KEY" "$VLLM_MODEL" "$no_proxy" "$NO_PROXY"'
    output = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert output == "http://node7:4242/v1|it's $secret|m-fp8|example.org,node7|node7|"


def test_cli_prints_the_exports(server: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("plyserve.cli.load_manifest", lambda path: {"m-fp8": ModelSpec("m-fp8", "org/model", "abc123", 18001, ())})
    write_endpoint(_endpoint(server))
    main(["endpoint", "m-fp8"])
    assert f"export VLLM_BASE_URL={server}" in capsys.readouterr().out


def test_cli_reports_a_missing_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("plyserve.cli.load_manifest", lambda path: {"m-fp8": ModelSpec("m-fp8", "org/model", "abc123", 18001, ())})
    with pytest.raises(SystemExit, match="plyserve: no server for m-fp8"):
        main(["endpoint", "m-fp8"])
