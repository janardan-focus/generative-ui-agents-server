# CLAUDE.md — generative-ui-agents-server

## Project Overview

**Python FastAPI** backend that powers the Generative UI chat experience. It:
1. Accepts natural-language queries from the React chat client via SSE (`GET /chat/stream`)
2. Runs a **LangGraph multi-agent pipeline** (Orchestrator → DataFetcher → UIRenderer)
3. Streams progress events + a final **UISchema** back over Server-Sent Events
4. Connects to the **Ticket Management System MCP HTTP endpoint** to fetch/mutate data

This server is the "brain" — it takes queries in natural language, decides which MCP tool to call, fetches data, then generates a structured UI schema for the React frontend to render.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Web framework | FastAPI 0.115+, uvicorn, sse-starlette |
| Agent framework | LangGraph 0.3+, LangChain Core 0.3+ |
| LLM | Google Gemini 2.0 Flash (`langchain-google-genai`) |
| Config | pydantic-settings 2.6+ |
| HTTP client | httpx 0.27+ (async) |
| Python | 3.13, virtual env at `.venv/` |

---

## Directory Structure

```
generative-ui-agents-server/
├── main.py                  # FastAPI app, SSE endpoint, startup
├── config.py                # Pydantic Settings (env vars)
├── requirements.txt         # All Python dependencies
├── .env                     # Local env vars (see below)
├── .env.example
│
├── agents/                  # LangGraph agent pipeline
│   ├── __init__.py
│   ├── state.py             # AgentState TypedDict (shared graph state)
│   ├── graph.py             # Compiled LangGraph: orchestrator→data_fetcher→ui_renderer
│   └── nodes.py             # Node implementations (3 agent functions)
│       # orchestrator_node  — picks the right MCP tool + args
│       # data_fetcher_node  — calls MCPHTTPClient.call_tool()
│       # ui_renderer_node   — generates UISchema from tool_result
│
├── mcp/                     # MCP client
│   ├── __init__.py
│   └── client.py            # MCPHTTPClient: list_tools(), call_tool() over JSON-RPC 2.0
│
└── schemas/                 # Pydantic models (shared contract with React frontend)
    ├── __init__.py
    └── ui_schema.py         # UISchema, UIComponent, UIAction, SSE event types
```

---

## Environment Variables (`.env`)

```
GOOGLE_API_KEY=              # Google AI Studio API key (Gemini)
GOOGLE_MODEL=gemini-2.0-flash  # LLM model override (optional)
MCP_SERVER_URL=http://localhost:3000/api/mcp  # Ticket Management System MCP endpoint
AGENT_SERVER_PORT=8000       # FastAPI listen port
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```

---

## Agent Pipeline (LangGraph)

```
User Query
    │
    ▼
[orchestrator_node]           — Decides: which MCP tool? what args? logs reasoning.
    │  sets: tool_name, tool_args, reasoning
    ▼
[data_fetcher_node]           — Calls MCPHTTPClient.call_tool(tool_name, tool_args)
    │  sets: tool_result
    ▼
[ui_renderer_node]            — Builds UISchema from tool_result using Gemini
    │  sets: ui_schema
    ▼
SSE: ui_schema event → React frontend renders dynamically
```

Graph is compiled once at import time in `agents/graph.py` and reused per request.

---

## AgentState (agents/state.py)

```python
class AgentState(TypedDict):
    query: str             # Original user query
    tool_name: str | None  # Set by orchestrator
    tool_args: dict        # Set by orchestrator
    reasoning: str         # Set by orchestrator (for transparency)
    tool_result: Any       # Set by data_fetcher
    ui_schema: dict | None # Set by ui_renderer
    messages: list[AnyMessage]  # LangChain message history
    mcp_api_key: str       # Forwarded from HTTP request
    mcp_server_url: str    # Forwarded from HTTP request (overridable)
    error: str | None
```

---

## UISchema Contract (schemas/ui_schema.py)

The `UIRendererAgent` produces a `UISchema` that the React `DynamicRenderer` deserialises:

```python
class UISchema(BaseModel):
    layout: Literal["card-grid","table","detail","kanban","success","error","list","empty"]
    title: str
    subtitle: str | None
    components: list[UIComponent]   # leaf-level renderable items
    data: dict                      # raw data blob for drilling down
    actions: list[UIAction]         # optional action buttons
    metadata: dict                  # pagination, totals, etc.

class UIComponent(BaseModel):
    type: Literal["card","ticket-card","table","badge","stat",
                  "list-item","kanban-column","success-banner","error-banner","json-viewer"]
    props: dict

class UIAction(BaseModel):
    label: str
    tool: str            # MCP tool name to invoke on click
    args: dict
    style: Literal["primary","secondary","danger"]
```

---

## SSE Events Emitted (GET /chat/stream)

| Event | Payload | Description |
|---|---|---|
| `agent_thinking` | `{agent, message}` | Node started |
| `tool_selected` | `{tool, args, reasoning}` | Orchestrator decision |
| `tool_executing` | `{tool}` | DataFetcher starting |
| `tool_result` | `{tool, result}` | Raw MCP result |
| `ui_generating` | `{message}` | UIRenderer starting |
| `ui_schema` | `{schema: UISchema}` | **Final UI — render this** |
| `token` | `{text, agent}` | LLM token stream |
| `done` | `{message: "complete"}` | Stream end |
| `error` | `{message}` | Pipeline failure |

---

## MCP HTTP Client (mcp/client.py)

```python
async with MCPHTTPClient(url=mcp_server_url, api_key=mcp_api_key) as client:
    tools = await client.list_tools()   # → list[MCPToolDefinition]
    result = await client.call_tool("ticket_list", {"project_id": "..."})
```

Sends `Authorization: Bearer <api_key>` header. Wraps JSON-RPC 2.0. Auto-parses JSON text responses from MCP content envelope.

---

## API Endpoints

```
GET  /health               → {"status": "ok"}
GET  /chat/stream          → SSE stream
     ?query=<string>       (required) Natural language query
     ?api_key=<string>     (required) MCP API key
     ?mcp_url=<string>     (optional) Override MCP server URL
```

---

## Running Locally

```bash
# Create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and fill env
cp .env.example .env

# Start server (reload mode)
python main.py
# or
uvicorn main:app --reload --port 8000
```

---

## Inter-Service Communication

```
mcp-chat-client (Vite :5173)
    └─ GET /chat/stream ──► This server (:8000)
                                └─ POST /api/mcp ──► Ticket-Management-System (:3000)
```

The `mcp_api_key` and `mcp_server_url` are forwarded from the frontend request through the entire pipeline — agents never store credentials, they're passed through `AgentState`.

---

## Key Files to Know First

1. `agents/state.py` — understand the data contract between agents
2. `agents/graph.py` — see the pipeline structure
3. `agents/nodes.py` — the actual agent logic
4. `schemas/ui_schema.py` — the React rendering contract
5. `mcp/client.py` — how MCP tools are called
6. `config.py` — all configurable settings
7. `main.py` — SSE endpoint and event routing
