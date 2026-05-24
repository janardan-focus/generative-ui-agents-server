"""
LangGraph workflow: Orchestrator → DataFetcher → UIRenderer.

Graph factory
-------------
build_graph(checkpointer) compiles the graph with the provided InMemorySaver so
that multi-turn sessions persist across requests under the same thread_id.

The module-level `agent_graph` starts as None and is set during app lifespan
startup in main.py — do not use it before startup completes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph

from agents.nodes import data_fetcher_node, orchestrator_node, ui_renderer_node
from agents.state import AgentState

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

# ---------------------------------------------------------------------------
# Graph definition (nodes + edges, no checkpointer yet)
# ---------------------------------------------------------------------------

_builder = StateGraph(AgentState)

_builder.add_node("orchestrator", orchestrator_node)
_builder.add_node("data_fetcher", data_fetcher_node)
_builder.add_node("ui_renderer", ui_renderer_node)

_builder.set_entry_point("orchestrator")
_builder.add_edge("orchestrator", "data_fetcher")
_builder.add_edge("data_fetcher", "ui_renderer")
_builder.add_edge("ui_renderer", END)


# ---------------------------------------------------------------------------
# Factory — called once at app startup
# ---------------------------------------------------------------------------

def build_graph(checkpointer: "InMemorySaver") -> Any:
    """Compile the graph with the shared InMemorySaver checkpointer.

    The checkpointer persists full AgentState under thread_id == session_id,
    enabling multi-turn conversation context to survive across SSE requests.

    Call this exactly once during the app lifespan and assign the result to
    `agents.graph.agent_graph` so all requests share one compiled instance.
    """
    return _builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Module-level singleton — set during lifespan startup in main.py
# ---------------------------------------------------------------------------

agent_graph: Any = None  # type: ignore[assignment]
