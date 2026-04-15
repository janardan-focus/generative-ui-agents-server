"""
LangGraph state definition shared across all agent nodes.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # User's original query
    query: str

    # Set by OrchestratorAgent
    tool_name: str | None
    tool_args: dict[str, Any]
    reasoning: str

    # Set by DataFetcherAgent
    tool_result: Any | None

    # Set by UIRendererAgent
    ui_schema: dict[str, Any] | None

    # Running message history (for LLM context continuity)
    messages: Annotated[list[AnyMessage], add_messages]

    # MCP auth credentials forwarded from the request
    mcp_api_key: str
    mcp_server_url: str

    # Errors
    error: str | None
