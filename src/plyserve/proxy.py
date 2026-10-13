from __future__ import annotations

import hmac
import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from plyserve.endpoint import healthy

# headers of one connection rather than of the request or the response; the proxy speaks HTTP/1.0, so the end of the connection ends a streamed body
_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"})


class KeyProxy:
    # puts an API key in front of a server that has none (Ollama): requests with `Authorization: Bearer <key>` go on to `upstream` (host:port), others get 401.
    # /health answers 200 while the upstream answers `health_path`, as vLLM's /health does, so endpoint files and readiness checks work unchanged
    def __init__(self, address: tuple[str, int], upstream: str, key: str, health_path: str) -> None:
        handler = type("_Handler", (_Handler,), {"upstream": upstream, "expected": f"Bearer {key}".encode(), "health_path": health_path})
        self.server = ThreadingHTTPServer(address, handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class _Handler(BaseHTTPRequestHandler):
    upstream: str
    expected: bytes
    health_path: str

    def _forward(self) -> None:
        # the body is read before any answer, so a client that is refused still reads the answer rather than a reset connection
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            return self._reply(411, b'{"error": "send the request body with Content-Length"}')
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), self.expected):
            return self._reply(401, b'{"error": "missing or wrong API key"}')
        if self.path == "/health":
            return self._reply(200 if healthy(f"http://{self.upstream}{self.health_path}", "") else 503, b"")

        headers = {name: value for name, value in self.headers.items() if name.lower() not in _HOP_BY_HOP | {"host", "authorization", "content-length"}}
        # no timeout: a long prompt can take minutes before the first byte of the answer
        connection = http.client.HTTPConnection(self.upstream)
        try:
            try:
                connection.request(self.command, self.path, body=body or None, headers=headers)
                response = connection.getresponse()
            except OSError:
                return self._reply(502, b'{"error": "the server behind plyserve does not answer"}')
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                if name.lower() not in _HOP_BY_HOP | {"date", "server"}:
                    self.send_header(name, value)
            self.end_headers()
            # a streamed answer (server-sent events) goes on chunk by chunk as it arrives
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
        except OSError:
            pass  # the client went away; closing the upstream connection lets the server stop generating
        finally:
            connection.close()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = _forward

    def _reply(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass  # Ollama logs every request itself
