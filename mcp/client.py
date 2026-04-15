"""
Async HTTP client for the Ticket Management System MCP server.

The MCP server exposes a JSON-RPC 2.0 endpoint (POST /api/mcp).
This client wraps list_tools() and call_tool() with proper auth.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MCPToolDefinition:
    """Minimal representation of an MCP tool returned by tools/list."""

    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def __repr__(self) -> str:  # pragma: no cover
        return f"MCPToolDefinition(name={self.name!r})"

    def to_prompt_description(self) -> str:
        """Human-readable one-liner for use in system prompts."""
        params = list(self.input_schema.get("properties", {}).keys())
        required = self.input_schema.get("required", [])
        param_str = ", ".join(
            f"{p}{'*' if p in required else '?'}" for p in params
        )
        return f"- {self.name}({param_str}): {self.description}"


class MCPHTTPClient:
    """
    Thin async wrapper around the Ticket Management System MCP HTTP endpoint.

    Usage
    -----
    async with MCPHTTPClient(url, api_key) as client:
        tools = await client.list_tools()
        result = await client.call_tool("project_list", {})
    """

    def __init__(self, server_url: str, api_key: str) -> None:
        self._server_url = server_url.rstrip("/")
        self._api_key = api_key
        self._request_id = 0
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MCPHTTPClient":
        self._http = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Internal JSON-RPC 2.0 request
    # ------------------------------------------------------------------

    async def _rpc(self, method: str, params: Any = None) -> Any:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        assert self._http is not None, "Client not started — use `async with`"

        try:
            resp = await self._http.post(
                self._server_url, content=json.dumps(payload), headers=headers
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"MCP server returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"MCP server unreachable: {exc}") from exc

        body = resp.json()
        if "error" in body:
            err = body["error"]
            raise RuntimeError(
                f"MCP error {err.get('code')}: {err.get('message', 'unknown')}"
            )
        return body.get("result")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[MCPToolDefinition]:
        """Return all tools exposed by the MCP server."""
        result = await self._rpc("tools/list")
        tools_raw: list[dict[str, Any]] = result.get("tools", [])
        return [
            MCPToolDefinition(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in tools_raw
        ]

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """
        Call a named MCP tool and return its result.

        The MCP server returns ``{ content: [{ type: 'text', text: '...' }] }``.
        We parse the text field as JSON when possible so the agents can work
        with native Python dicts.
        """
        result = await self._rpc("tools/call", {"name": name, "arguments": args})

        # Unwrap MCP content envelope
        content_list: list[dict[str, Any]] = result.get("content", [])
        text_parts = [
            c["text"] for c in content_list if c.get("type") == "text" and "text" in c
        ]
        raw_text = "\n".join(text_parts)

        try:
            return json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            return raw_text
