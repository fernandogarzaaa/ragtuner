"""MCP server over stdio (JSON-RPC 2.0, one message per line).

Tools: `search` (ranked passages with document, page and offsets) and `ask` (answer with
citations). stdout carries only protocol messages; logs go to stderr. The index builds in
a background thread so `initialize` answers immediately; tool calls wait for it.

Register with an MCP client, for example in a project's .mcp.json:
  {"mcpServers": {"docs": {"command": "ragtuner", "args": ["mcp", "/path/corpus", "--config", "/path/best_config.yaml"]}}}
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, Callable, TextIO

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "search",
        "description": "Search the document corpus. Returns the most relevant passages with document id, page and score.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "What to look for"},
                                                         "k": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Number of passages"}},
                        "required": ["query"], "additionalProperties": False},
    },
    {
        "name": "ask",
        "description": "Answer a question from the document corpus with numbered citations to the passages used.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "The question"}},
                        "required": ["query"], "additionalProperties": False},
    },
]


class MCPServer:
    def __init__(self, runtime_factory: Callable[[], Any], *, name: str = "ragtuner", version: str = "0.0.1",
                 tenant: str | None = None, principal: str | None = None) -> None:
        self._factory = runtime_factory
        self._rt: Any = None
        self._err: BaseException | None = None
        self._ready = threading.Event()
        self.name, self.version = name, version
        self.tenant, self.principal = tenant, principal
        threading.Thread(target=self._build, daemon=True).start()

    def _build(self) -> None:
        try:
            self._rt = self._factory()
        except BaseException as e:  # noqa: BLE001
            self._err = e
        finally:
            self._ready.set()

    def runtime(self) -> Any:
        self._ready.wait()
        if self._err is not None:
            raise RuntimeError(f"index failed to build: {self._err}")
        return self._rt

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # notification
            return None
        try:
            if method == "initialize":
                requested = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION
                result: Any = {"protocolVersion": requested, "capabilities": {"tools": {"listChanged": False}},
                               "serverInfo": {"name": self.name, "version": self.version}}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call(msg.get("params") or {})
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
            return {"jsonrpc": "2.0", "id": mid, "result": result}
        except Exception as e:  # noqa: BLE001
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        name, args = params.get("name"), params.get("arguments") or {}
        query = str(args.get("query", "")).strip()
        if not query:
            return {"content": [{"type": "text", "text": "query must not be empty"}], "isError": True}
        rt = self.runtime()
        if name == "search":
            k = args.get("k")
            hits = rt.search(query, k=int(k) if k else None, tenant=self.tenant, principal=self.principal)
            text = "\n\n".join(f"[{i + 1}] {h['doc_id']} p{h['page']} score={h['score']}\n{h['text']}" for i, h in enumerate(hits)) or "No results."
            return {"content": [{"type": "text", "text": text}], "structuredContent": {"hits": hits}, "isError": False}
        if name == "ask":
            out = rt.ask(query, tenant=self.tenant, principal=self.principal)
            cites = "\n".join(f"[{c['n']}] {c['doc_id']} p{c['page']}" for c in out["citations"])
            return {"content": [{"type": "text", "text": f"{out['answer']}\n\nSources:\n{cites}"}], "structuredContent": out, "isError": False}
        return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                reply: dict[str, Any] | None = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            else:
                reply = self.handle(msg) if isinstance(msg, dict) else {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
            if reply is not None:
                stdout.write(json.dumps(reply) + "\n")
                stdout.flush()
