"""
The three agent nodes that make up the LangGraph workflow.

OrchestratorNode   — reads user query + prior conversation, picks the right MCP tool
DataFetcherNode    — executes the chosen MCP tool
UIRendererNode     — converts the raw JSON result into a UI schema

Message-accumulation contract
------------------------------
The `messages` field in AgentState uses the `add_messages` reducer, which
APPENDS whatever each node returns.  Because the checkpointer now persists
messages across turns, we must be disciplined about what we put there:

  ✅ Persist per turn:
       - The user's HumanMessage (already added by main.py before the graph runs)
       - The orchestrator's AIMessage response (tool decision + reasoning summary)

  ❌ Do NOT persist per turn:
       - SystemMessage prompts (large, re-built every turn, would accumulate linearly)
       - DataFetcher / UIRenderer LLM calls (intermediate; not conversational)

This keeps the persisted history as a clean conversational log:
    [HumanMessage(t1), AIMessage(t1), HumanMessage(t2), AIMessage(t2), …]

and bounds growth to two small messages per turn regardless of how large the
system prompt or tool-result payloads are.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from agents.prompts import (
    ORCHESTRATOR_HUMAN,
    ORCHESTRATOR_SYSTEM,
    UI_RENDERER_HUMAN,
    UI_RENDERER_SYSTEM,
)
from agents.state import AgentState
from config import settings
from mcp.client import MCPHTTPClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared LLM instance
# ---------------------------------------------------------------------------

_llm = ChatGoogleGenerativeAI(
    model=settings.google_model,
    google_api_key=settings.google_api_key,
    temperature=0,
    max_output_tokens=2048,
)

# Maximum number of prior conversational messages to include in the
# orchestrator's context window.  Caps the token cost for long sessions.
_MAX_HISTORY_MESSAGES = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    """
    Robustly extract a JSON object from LLM output.
    Handles cases where the model wraps the JSON in markdown fences.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _build_history_context(messages: list[AnyMessage]) -> str:
    """
    Serialize the most recent conversational messages into a plain-text block
    suitable for injection into the orchestrator's system prompt.

    Only HumanMessage and AIMessage entries are included; SystemMessages
    (which are large per-turn artefacts) are intentionally excluded.
    Returns an empty string when there is no prior history.
    """
    conversational = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
    ]
    # Drop the last message — it's the HumanMessage for the *current* turn,
    # already present in the human_msg we're about to build.
    prior = conversational[:-1] if conversational else []
    if not prior:
        return ""

    # Apply sliding-window cap
    prior = prior[-_MAX_HISTORY_MESSAGES:]

    lines = ["Previous conversation (for context only):"]
    for msg in prior:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Truncate very long assistant messages (e.g. raw JSON reasoning) to
        # keep the system prompt size bounded.
        if len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node 1 — OrchestratorAgent
# ---------------------------------------------------------------------------

async def orchestrator_node(state: AgentState) -> dict[str, Any]:
    """
    Analyses the user query and decides which MCP tool to invoke.

    Prior conversation context (restored from the InMemorySaver checkpointer)
    is injected into the system prompt so references like "change *its* status"
    resolve correctly across turns.

    Writes to state:
        tool_name  — MCP tool to call
        tool_args  — arguments for that tool
        reasoning  — one-sentence rationale
        messages   — AIMessage with the decision summary (persisted; NOT the SystemMessage)
    """
    logger.info("[Orchestrator] Analysing query: %s", state["query"])

    # Fetch live tool definitions from the MCP server so the prompt is
    # always up-to-date with whatever tools the server exposes.
    async with MCPHTTPClient(state["mcp_server_url"], state["mcp_api_key"]) as client:
        tools = await client.list_tools()

    tools_description = "\n".join(t.to_prompt_description() for t in tools)

    # Build history context from restored messages (excludes SystemMessages)
    history_context = _build_history_context(state.get("messages", []))

    # Append prior-turn context to the system prompt when available.
    # This is injected at call time and NOT persisted into the checkpoint.
    system_content = ORCHESTRATOR_SYSTEM.format(tools_description=tools_description)
    if history_context:
        system_content = f"{system_content}\n\n{history_context}"

    system_msg = SystemMessage(content=system_content)
    human_msg = HumanMessage(content=ORCHESTRATOR_HUMAN.format(query=state["query"]))

    response = await _llm.ainvoke([system_msg, human_msg])
    raw: str = response.content if isinstance(response.content, str) else str(response.content)

    try:
        parsed = _extract_json(raw)
        tool_name: str = parsed["tool_name"]
        tool_args: dict[str, Any] = parsed.get("tool_args", {})
        reasoning: str = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("[Orchestrator] Failed to parse LLM response: %s", raw)
        # Persist a brief AI summary (NOT the full system prompt)
        error_summary = AIMessage(content=f"[error] Could not parse tool decision: {exc}")
        return {
            "tool_name": None,
            "tool_args": {},
            "reasoning": "",
            "error": f"Orchestrator could not parse LLM output: {exc}",
            "messages": [error_summary],
        }

    logger.info("[Orchestrator] Selected tool=%s args=%s", tool_name, tool_args)

    # Persist only a compact AI summary per turn — NOT the system prompt.
    # This keeps the stored history small and prevents token-cost blow-up.
    decision_summary = AIMessage(
        content=f"[tool={tool_name}] {reasoning}"
    )
    return {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "reasoning": reasoning,
        "error": None,
        "messages": [decision_summary],
    }


# ---------------------------------------------------------------------------
# Node 2 — DataFetcherAgent
# ---------------------------------------------------------------------------

async def data_fetcher_node(state: AgentState) -> dict[str, Any]:
    """
    Executes the MCP tool chosen by the OrchestratorAgent.

    Writes to state:
        tool_result — raw JSON result from the MCP server

    Does NOT write to `messages` — this is an I/O node, not a conversational
    turn, and its output would only inflate the persisted history.
    """
    if state.get("error"):
        return {"tool_result": None}

    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args", {})

    if not tool_name:
        return {
            "tool_result": None,
            "error": "No tool was selected by the Orchestrator.",
        }

    logger.info("[DataFetcher] Calling tool=%s args=%s", tool_name, tool_args)

    async with MCPHTTPClient(state["mcp_server_url"], state["mcp_api_key"]) as client:
        try:
            result = await client.call_tool(tool_name, tool_args)
        except RuntimeError as exc:
            logger.error("[DataFetcher] MCP call failed: %s", exc)
            return {
                "tool_result": None,
                "error": str(exc),
            }

    logger.info("[DataFetcher] Got result type=%s", type(result).__name__)
    return {"tool_result": result, "error": None}


# ---------------------------------------------------------------------------
# Node 3 — UIRendererAgent
# ---------------------------------------------------------------------------

async def ui_renderer_node(state: AgentState) -> dict[str, Any]:
    """
    Converts the raw MCP tool result into a declarative UI schema.

    Writes to state:
        ui_schema — dict matching schemas.ui_schema.UISchema

    Does NOT write to `messages` — the UI rendering step is not conversational
    and its LLM call context (raw tool JSON) would bloat the persisted history.
    """
    if state.get("error"):
        return {
            "ui_schema": {
                "layout": "error",
                "title": "Something went wrong",
                "subtitle": state["error"],
                "components": [
                    {"type": "error-banner", "props": {"message": state["error"]}}
                ],
                "data": {},
                "actions": [],
                "metadata": {},
            }
        }

    tool_name = state.get("tool_name", "unknown_tool")
    tool_result = state.get("tool_result")

    logger.info("[UIRenderer] Generating schema for tool=%s", tool_name)

    system_msg = SystemMessage(content=UI_RENDERER_SYSTEM)
    human_msg = HumanMessage(
        content=UI_RENDERER_HUMAN.format(
            tool_name=tool_name,
            tool_result=json.dumps(tool_result, indent=2),
        )
    )

    response = await _llm.ainvoke([system_msg, human_msg])
    raw: str = response.content if isinstance(response.content, str) else str(response.content)

    try:
        ui_schema = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("[UIRenderer] Failed to parse schema: %s\nRaw: %s", exc, raw[:500])
        ui_schema = {
            "layout": "error",
            "title": "Render error",
            "subtitle": "The agent returned malformed UI schema.",
            "components": [
                {
                    "type": "json-viewer",
                    "props": {"data": tool_result, "label": tool_name},
                }
            ],
            "data": {},
            "actions": [],
            "metadata": {},
        }

    # Return ui_schema only — do NOT append to messages (see module docstring).
    return {"ui_schema": ui_schema}
