"""HTTP API (stdlib only).

GET  /health            liveness and index size
GET  /manifest          config, corpus file hashes, last reload stats
POST /search            {"query": str, "k": int?} -> {"hits": [...]}
POST /ask               {"query": str} -> {"answer", "mode", "citations"}
POST /admin/reload      incremental re-index; needs header X-Admin-Token when RAGTUNER_ADMIN_TOKEN is set
GET  /widget.js         the embeddable chat widget
GET  /                  a demo page with the widget

Identity for permission filtering comes from the X-Tenant and X-Principal headers, which
a trusted gateway in front of this service must set; this server does not authenticate.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from ragtuner.serve.runtime import Runtime

MAX_BODY = 64 * 1024


def make_handler(rt: Runtime, *, cors_origin: str = "*") -> type[BaseHTTPRequestHandler]:
    admin_token = os.environ.get("RAGTUNER_ADMIN_TOKEN")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ragtuner"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            if os.environ.get("RAGTUNER_HTTP_LOG") == "1":
                super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", cors_origin)
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Tenant, X-Principal, X-Admin-Token")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode())

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("request body too large")
            raw = self.rfile.read(n) if n else b"{}"
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
            return data

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(204, b"")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(200, {"status": "ok", "chunks": len(rt.index.chunks) if rt.index else 0, "config": rt.config.key})
            elif self.path == "/manifest":
                self._json(200, rt.manifest())
            elif self.path == "/widget.js":
                self._send(200, resources.files("ragtuner.serve").joinpath("widget.js").read_bytes(), "application/javascript")
            elif self.path in ("/", "/index.html"):
                self._send(200, resources.files("ragtuner.serve").joinpath("demo.html").read_bytes(), "text/html; charset=utf-8")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                tenant, principal = self.headers.get("X-Tenant"), self.headers.get("X-Principal")
                if self.path == "/search":
                    k = body.get("k")
                    if k is not None and (not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 100):
                        raise ValueError("k must be an integer between 1 and 100")
                    self._json(200, {"hits": rt.search(str(body.get("query", "")), k=k, tenant=tenant, principal=principal)})
                elif self.path == "/ask":
                    q = str(body.get("query", ""))
                    if not q.strip():
                        raise ValueError("query must not be empty")
                    self._json(200, rt.ask(q, tenant=tenant, principal=principal))
                elif self.path == "/admin/reload":
                    if admin_token and self.headers.get("X-Admin-Token") != admin_token:
                        self._json(403, {"error": "forbidden"})
                        return
                    self._json(200, rt.reload())
                else:
                    self._json(404, {"error": "not found"})
            except ValueError as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def make_server(rt: Runtime, host: str = "127.0.0.1", port: int = 8080, *, cors_origin: str = "*") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(rt, cors_origin=cors_origin))
