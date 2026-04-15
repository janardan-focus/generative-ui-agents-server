"""
LangGraph workflow: Orchestrator → DataFetcher → UIRenderer.

The compiled graph is the single import used by main.py.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.nodes import data_fetcher_node, orchestrator_node, ui_renderer_node
from agents.state import AgentState

# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

_builder = StateGraph(AgentState)

_builder.add_node("orchestrator", orchestrator_node)
_builder.add_node("data_fetcher", data_fetcher_node)
_builder.add_node("ui_renderer", ui_renderer_node)

_builder.set_entry_point("orchestrator")
_builder.add_edge("orchestrator", "data_fetcher")
_builder.add_edge("data_fetcher", "ui_renderer")
_builder.add_edge("ui_renderer", END)

# Compile once at import time; reused for every request
agent_graph = _builder.compile()
