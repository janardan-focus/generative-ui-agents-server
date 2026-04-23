"""
Generative-UI Agent Server
==========================
FastAPI application that:
  1. Accepts chat queries from the React frontend.
  2. Runs the LangGraph multi-agent pipeline.
  3. Streams progress + the final UI schema back over Server-Sent Events.

SSE event types emitted
-----------------------
agent_thinking  {"agent": str, "message": str}
tool_selected   {"tool": str, "args": dict, "reasoning": str}
tool_executing  {"tool": str}
tool_result     {"tool": str, "result": any}
ui_generating   {"message": str}
ui_schema       {"schema": UISchema}
token           {"text": str, "agent": str}
done            {"message": "complete"}
error           {"message": str}
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from agents.graph import agent_graph
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Generative-UI Agent Server",
    description="LangGraph multi-agent pipeline with SSE streaming",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: Any) -> dict[str, str]:
    """Format an SSE message dict consumed by sse-starlette."""
    return {"event": event, "data": json.dumps(data)}


# ---------------------------------------------------------------------------
# Core streaming generator
# ---------------------------------------------------------------------------

async def _run_agent(
    query: str,
    mcp_api_key: str,
    mcp_server_url: str,
) -> AsyncGenerator[dict[str, str], None]:
    """
    Runs the LangGraph agent pipeline and yields SSE events.

    We use astream_events(version="v2") to get fine-grained lifecycle events:
      - on_chain_start / on_chain_end per node
      - on_chat_model_stream for token-level streaming
    """

    initial_state = {
        "query": query,
        "tool_name": None,
        "tool_args": {},
        "reasoning": "",
        "tool_result": None,
        "ui_schema": None,
        "messages": [],
        "mcp_api_key": mcp_api_key,
        "mcp_server_url": mcp_server_url,
        "error": None,
    }

    # Map node names to friendly labels
    agent_labels: dict[str, str] = {
        "orchestrator": "Orchestrator Agent",
        "data_fetcher": "DataFetcher Agent",
        "ui_renderer": "UIRenderer Agent",
    }

    # Track which node is currently active for token attribution
    current_node: str = "orchestrator"

    try:
        async for event in agent_graph.astream_events(initial_state, version="v2"):
            kind: str = event["event"]
            name: str = event.get("name", "")
            metadata: dict = event.get("metadata", {})
            node_name: str = metadata.get("langgraph_node", current_node)

            # ── Node lifecycle ────────────────────────────────────────────
            if kind == "on_chain_start" and name in agent_labels:
                current_node = name
                yield _sse(
                    "agent_thinking",
                    {
                        "agent": name,
                        "message": f"{agent_labels[name]} is working…",
                    },
                )

            elif kind == "on_chain_end" and name == "orchestrator":
                output: dict = event["data"].get("output", {})
                if output.get("tool_name"):
                    yield _sse(
                        "tool_selected",
                        {
                            "tool": output["tool_name"],
                            "args": output.get("tool_args", {}),
                            "reasoning": output.get("reasoning", ""),
                        },
                    )

            elif kind == "on_chain_start" and name == "data_fetcher":
                # Grab tool_name from the input snapshot if available
                inp: dict = event["data"].get("input", {})
                tool = inp.get("tool_name", "")
                if tool:
                    yield _sse("tool_executing", {"tool": tool})

            elif kind == "on_chain_end" and name == "data_fetcher":
                output = event["data"].get("output", {})
                tool_result = output.get("tool_result")
                if tool_result is not None:
                    # Grab tool_name from graph state snapshot via input
                    inp = event["data"].get("input", {})
                    yield _sse(
                        "tool_result",
                        {
                            "tool": inp.get("tool_name", ""),
                            "result": tool_result,
                        },
                    )

            elif kind == "on_chain_start" and name == "ui_renderer":
                yield _sse("ui_generating", {"message": "Generating UI schema…"})

            elif kind == "on_chain_end" and name == "ui_renderer":
                output = event["data"].get("output", {})
                ui_schema = output.get("ui_schema")
                if ui_schema:
                    yield _sse("ui_schema", {"schema": ui_schema})
                # Pipeline is complete — emit done immediately and exit.
                # Do NOT wait for LangGraph's internal graph-wrapper events
                # (e.g. on_chain_end for "LangGraph") which keep the async-for
                # loop alive and delay — or permanently block — the done event.
                yield _sse("done", {"message": "complete"})
                return

            # ── Token streaming ───────────────────────────────────────────
            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk is None:
                    continue
                text = (
                    chunk.content
                    if isinstance(chunk.content, str)
                    else "".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in chunk.content
                    )
                )
                if text:
                    yield _sse(
                        "token",
                        {"text": text, "agent": node_name or current_node},
                    )

        yield _sse("done", {"message": "complete"})

    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent pipeline failed: %s", exc)
        yield _sse("error", {"message": str(exc)})
        yield _sse("done", {"message": "error"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "generative-ui-agent-server"})


@app.get("/chat/stream")
async def chat_stream(
    query: str = Query(..., description="User's natural-language query"),
    api_key: str = Query(..., description="MCP server API key (Bearer token)"),
    mcp_url: str = Query(
        default=settings.mcp_server_url,
        description="MCP server HTTP URL override",
    ),
) -> EventSourceResponse:
    """
    SSE endpoint consumed by the React AgentClient.

    GET /chat/stream?query=<...>&api_key=<...>&mcp_url=<...>
    """
    return EventSourceResponse(
        _run_agent(query=query, mcp_api_key=api_key, mcp_server_url=mcp_url)
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.agent_server_port,
        reload=True,
    )
