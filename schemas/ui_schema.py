"""
UI Schema types — the contract between the Python agent server
and the React frontend's DynamicRenderer.

The UIRenderer agent produces a UISchema.  The React frontend
deserialises it and renders the appropriate components.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Leaf-level component models
# ---------------------------------------------------------------------------

class UIComponent(BaseModel):
    """A single renderable component inside a layout."""

    type: Literal[
        "card",           # Project / entity card
        "ticket-card",    # Ticket-specific card
        "table",          # Tabular data
        "badge",          # Status / priority badge
        "stat",           # Key-value statistic
        "list-item",      # Simple list row
        "kanban-column",  # Kanban board column
        "success-banner", # Action-confirmed banner
        "error-banner",   # Error display
        "json-viewer",    # Raw JSON fallback
    ]
    props: dict[str, Any] = Field(default_factory=dict)


class UIAction(BaseModel):
    """An action button rendered below the component group."""

    label: str
    tool: str                              # MCP tool name to invoke
    args: dict[str, Any] = Field(default_factory=dict)
    style: Literal["primary", "secondary", "danger"] = "primary"


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------

class UISchema(BaseModel):
    """
    The complete UI schema returned by the UIRenderer agent.

    layout         — which top-level layout to use
    title          — heading shown above the components
    subtitle       — optional sub-heading / context message
    components     — list of rendered child components
    data           — raw data blob (available to the renderer for drilling down)
    actions        — optional action buttons
    metadata       — arbitrary key/value bag (e.g. total counts, pagination info)
    """

    layout: Literal[
        "card-grid",   # Grid of cards (e.g. project list)
        "table",       # Full-width data table
        "detail",      # Single-entity detail view
        "kanban",      # Kanban column view
        "success",     # Confirmation / success state
        "error",       # Error state
        "list",        # Simple vertical list
        "empty",       # Empty state placeholder
    ]
    title: str
    subtitle: str | None = None
    components: list[UIComponent] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    actions: list[UIAction] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SSE event payload types  (serialised to JSON and sent as SSE data)
# ---------------------------------------------------------------------------

class SSEAgentThinking(BaseModel):
    agent: str
    message: str


class SSEToolSelected(BaseModel):
    tool: str
    args: dict[str, Any]
    reasoning: str


class SSEToolResult(BaseModel):
    tool: str
    result: Any


class SSEToken(BaseModel):
    text: str
    agent: str


class SSEUiSchema(BaseModel):
    schema_: UISchema = Field(alias="schema")

    model_config = {"populate_by_name": True}


class SSEDone(BaseModel):
    message: str = "complete"


class SSEError(BaseModel):
    message: str
