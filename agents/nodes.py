"""
The three agent nodes that make up the LangGraph workflow.

OrchestratorNode   — reads user query, picks the right MCP tool
DataFetcherNode    — executes the chosen MCP tool
UIRendererNode     — converts the raw JSON result into a UI schema
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    """
    Robustly extract a JSON object from LLM output.
    Handles cases where the model wraps the JSON in markdown fences.
    """
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last line (``` or ```json)
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


# ---------------------------------------------------------------------------
# Node 1 — OrchestratorAgent
# ---------------------------------------------------------------------------

async def orchestrator_node(state: AgentState) -> dict[str, Any]:
    """
    Analyses the user query and decides which MCP tool to invoke.

    Writes to state:
        tool_name  — MCP tool to call
        tool_args  — arguments for that tool
        reasoning  — one-sentence rationale
    """
    logger.info("[Orchestrator] Analysing query: %s", state["query"])

    # Fetch live tool definitions from the MCP server so the prompt is
    # always up-to-date with whatever tools the server exposes.
    async with MCPHTTPClient(state["mcp_server_url"], state["mcp_api_key"]) as client:
        tools = await client.list_tools()

    tools_description = "\n".join(t.to_prompt_description() for t in tools)

    system_msg = SystemMessage(
        content=ORCHESTRATOR_SYSTEM.format(tools_description=tools_description)
    )
    human_msg = HumanMessage(
        content=ORCHESTRATOR_HUMAN.format(query=state["query"])
    )

    response = await _llm.ainvoke([system_msg, human_msg])
    raw: str = response.content if isinstance(response.content, str) else str(response.content)

    try:
        parsed = _extract_json(raw)
        tool_name: str = parsed["tool_name"]
        tool_args: dict[str, Any] = parsed.get("tool_args", {})
        reasoning: str = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("[Orchestrator] Failed to parse LLM response: %s", raw)
        return {
            "tool_name": None,
            "tool_args": {},
            "reasoning": "",
            "error": f"Orchestrator could not parse LLM output: {exc}",
            "messages": [system_msg, human_msg, response],
        }

    logger.info("[Orchestrator] Selected tool=%s args=%s", tool_name, tool_args)
    return {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "reasoning": reasoning,
        "error": None,
        "messages": [system_msg, human_msg, response],
    }


# ---------------------------------------------------------------------------
# Node 2 — DataFetcherAgent
# ---------------------------------------------------------------------------

async def data_fetcher_node(state: AgentState) -> dict[str, Any]:
    """
    Executes the MCP tool chosen by the OrchestratorAgent.

    Writes to state:
        tool_result — raw JSON result from the MCP server
    """
    if state.get("error"):
        # Propagate upstream error without executing anything
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
    """
    if state.get("error"):
        # Return an error UI schema so the frontend can display the message
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

    return {
        "ui_schema": ui_schema,
        "messages": [system_msg, human_msg, response],
    }
