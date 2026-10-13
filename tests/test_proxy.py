import http.client
import http.server
import json
import threading
import time
from collections.abc import Iterator

import pytest

from plyserve.proxy import KeyProxy
from plyserve.serve import free_port


# stands in for Ollama: no key of its own; echoes POST requests, and streams /stream in three chunks 0.2 s apart
class _Upstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    up = True

    def do_GET(self) -> None:
        if self.path == "/api/version":
            self._send(200 if _Upstream.up else 500, b'{"version": "0.0.0"}')
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for index in range(3):
                data = f"data: {index}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                time.sleep(0.2)
            self.wfile.write(b"0\r\n\r\n")
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self._send(200, json.dumps({"path": self.path, "body": body.decode(), "authorization": self.headers.get("Authorization")}).encode())

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def proxy() -> Iterator[int]:
    _Upstream.up = True
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    port = free_port()
    key_proxy = KeyProxy(("127.0.0.1", port), f"127.0.0.1:{upstream.server_address[1]}", "secret", "/api/version")
    yield port
    key_proxy.stop()
    upstream.shutdown()


def _request(port: int, method: str, path: str, key: str | None = "secret", body: bytes | None = None) -> http.client.HTTPResponse:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, path, body=body, headers={"Authorization": f"Bearer {key}"} if key else {})
    return connection.getresponse()


@pytest.mark.parametrize("key", [None, "wrong", "secretX"])
def test_requests_without_the_key_are_refused(proxy: int, key: str | None) -> None:
    response = _request(proxy, "POST", "/v1/chat/completions", key=key, body=b"{}")
    assert response.status == 401 and b"API key" in response.read()


def test_requests_with_the_key_reach_the_upstream_without_it(proxy: int) -> None:
    response = _request(proxy, "POST", "/v1/chat/completions?x=1", body=b'{"model": "m"}')
    assert response.status == 200
    assert json.loads(response.read()) == {"path": "/v1/chat/completions?x=1", "body": '{"model": "m"}', "authorization": None}


def test_health_follows_the_upstream(proxy: int) -> None:
    assert _request(proxy, "GET", "/health").status == 200
    _Upstream.up = False
    assert _request(proxy, "GET", "/health").status == 503
    assert _request(proxy, "GET", "/health", key=None).status == 401


def test_streamed_answers_arrive_chunk_by_chunk(proxy: int) -> None:
    response = _request(proxy, "GET", "/stream")
    started = time.monotonic()
    first = response.read1(65536)
    assert first == b"data: 0\n\n" and time.monotonic() - started < 0.15  # before the upstream sent the rest
    assert first + response.read() == b"data: 0\n\ndata: 1\n\ndata: 2\n\n"


def test_upstream_that_is_gone_gives_502() -> None:
    port = free_port()
    key_proxy = KeyProxy(("127.0.0.1", port), f"127.0.0.1:{free_port()}", "secret", "/api/version")
    try:
        assert _request(port, "GET", "/v1/models").status == 502
    finally:
        key_proxy.stop()
