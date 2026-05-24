"""
Generative-UI Agent Server
==========================
FastAPI application that:
  1. Accepts chat queries from the React frontend.
  2. Runs the LangGraph multi-agent pipeline.
  3. Streams progress + the final UI schema back over Server-Sent Events.

SSE event types emitted
-----------------------
session         {"session_id": str}           — FIRST event; client stores and re-sends id
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

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from sse_starlette.sse import EventSourceResponse

import agents.graph as _agents_graph
import sessions.store as _sessions_store
from agents.graph import build_graph
from config import settings
from sessions.store import SessionRegistry, hash_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App lifespan — wires saver, registry, graph, and background sweep
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build checkpointer, registry, and compiled graph.
    Shutdown: cancel background sweep task.

    ⚠️  Single-worker constraint: sessions are process-sticky (in-process store).
    Do NOT run uvicorn with --workers >1 or behind a round-robin load balancer
    until the store is migrated to Redis/Mongo.  See CLAUDE.md §Session Management.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    # 1. Create the shared checkpointer (persists graph state per thread_id)
    saver = InMemorySaver()

    # 2. Create the session registry (lifecycle metadata + reference to saver)
    registry = SessionRegistry(
        saver=saver,
        idle_timeout=settings.idle_timeout_seconds,
        max_sessions=settings.max_sessions,
    )
    _sessions_store.registry = registry

    # 3. Compile the graph with the checkpointer (done once, reused per request)
    _agents_graph.agent_graph = build_graph(saver)
    logger.info("[Lifespan] LangGraph compiled with InMemorySaver checkpointer")

    # 4. Start background sweep task for memory hygiene
    async def _sweep_loop() -> None:
        while True:
            await asyncio.sleep(settings.session_sweep_interval_seconds)
            try:
                await registry.sweep()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Lifespan] Sweep error: %s", exc)

    sweep_task = asyncio.create_task(_sweep_loop())
    logger.info(
        "[Lifespan] Session sweep started (interval=%ds)", settings.session_sweep_interval_seconds
    )

    yield  # ── server is running ──

    # Shutdown
    sweep_task.cancel()
    try:
        await sweep_task
    except asyncio.CancelledError:
        pass
    logger.info("[Lifespan] Session sweep stopped")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Generative-UI Agent Server",
    description="LangGraph multi-agent pipeline with SSE streaming",
    version="1.0.0",
    lifespan=lifespan,
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
    session_id: str,
) -> AsyncGenerator[dict[str, str], None]:
    """
    Runs the LangGraph agent pipeline and yields SSE events.

    The session_id is passed as thread_id in the LangGraph config so the
    InMemorySaver checkpointer restores prior turn state (messages, etc.).

    We use astream_events(version="v2") to get fine-grained lifecycle events:
      - on_chain_start / on_chain_end per node
      - on_chat_model_stream for token-level streaming
    """
    # Seed the new turn with only the human message.  The add_messages reducer
    # in AgentState will APPEND to the restored history — NOT replace it.
    # We deliberately do NOT pass "messages": [] to avoid wiping the history.
    # mcp_api_key / mcp_server_url are re-injected fresh from the request every
    # turn so the raw key doesn't linger stale inside the checkpoint.
    initial_state = {
        "query": query,
        "tool_name": None,
        "tool_args": {},
        "reasoning": "",
        "tool_result": None,
        "ui_schema": None,
        "messages": [HumanMessage(content=query)],
        "mcp_api_key": mcp_api_key,
        "mcp_server_url": mcp_server_url,
        "error": None,
    }

    # thread_id binds this invocation to the session's checkpoint history
    config = {"configurable": {"thread_id": session_id}}

    # Map node names to friendly labels
    agent_labels: dict[str, str] = {
        "orchestrator": "Orchestrator Agent",
        "data_fetcher": "DataFetcher Agent",
        "ui_renderer": "UIRenderer Agent",
    }

    current_node: str = "orchestrator"

    try:
        async for event in _agents_graph.agent_graph.astream_events(
            initial_state, config=config, version="v2"
        ):
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
                inp: dict = event["data"].get("input", {})
                tool = inp.get("tool_name", "")
                if tool:
                    yield _sse("tool_executing", {"tool": tool})

            elif kind == "on_chain_end" and name == "data_fetcher":
                output = event["data"].get("output", {})
                tool_result = output.get("tool_result")
                if tool_result is not None:
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
                # Pipeline complete — touch the session then signal done.
                registry = _sessions_store.registry
                if registry:
                    await registry.touch(session_id)
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
    session_id: str | None = Query(
        default=None,
        description=(
            "Existing session ID to resume.  Omit (or pass an expired/unknown id) "
            "to start a new session.  The server echoes the active session_id in the "
            "first SSE 'session' event and the X-Session-Id response header."
        ),
    ),
) -> EventSourceResponse:
    """
    SSE endpoint consumed by the React AgentClient.

    GET /chat/stream?query=<...>&api_key=<...>[&session_id=<...>][&mcp_url=<...>]

    Session handling
    ----------------
    1. If session_id is absent or the session is unknown/expired/closed, a new
       session is minted and its id is returned.
    2. The active session_id is emitted as the FIRST SSE event ('session') so
       the EventSource client can read it (EventSource cannot read response headers).
    3. The id is also set in the X-Session-Id response header as a convenience
       for non-EventSource callers.
    """
    registry = _sessions_store.registry
    owner_hash = hash_api_key(api_key)

    active_session_id, was_created = await registry.resolve_or_create(
        session_id=session_id,
        owner_hash=owner_hash,
    )

    if was_created:
        logger.info(
            "[chat_stream] New session created session=%s (requested=%s)",
            active_session_id,
            session_id,
        )
    else:
        logger.info("[chat_stream] Resuming session=%s", active_session_id)

    async def _stream() -> AsyncGenerator[dict[str, str], None]:
        # Always emit the session id as the very first event so the client
        # can capture it regardless of whether it sent one in the request.
        yield _sse("session", {"session_id": active_session_id})
        async for evt in _run_agent(
            query=query,
            mcp_api_key=api_key,
            mcp_server_url=mcp_url,
            session_id=active_session_id,
        ):
            yield evt

    return EventSourceResponse(
        _stream(),
        headers={"X-Session-Id": active_session_id},
    )


@app.post("/chat/session/{session_id}/close")
async def close_session(session_id: str) -> JSONResponse:
    """
    Explicitly close a chat session.

    Marks the session closed, removes it from the registry, and deletes its
    LangGraph checkpoints (frees in-process memory immediately).

    The React client should call this on 'New chat', chat-close, and via
    navigator.sendBeacon inside the 'beforeunload' event handler.
    """
    registry = _sessions_store.registry
    await registry.close(session_id)
    logger.info("[close_session] Closed session=%s", session_id)
    return JSONResponse({"status": "closed", "session_id": session_id})


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
