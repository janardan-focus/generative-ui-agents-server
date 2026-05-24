# CLAUDE.md — generative-ui-agents-server

## Project Overview

**Python FastAPI** backend that powers the Generative UI chat experience. It:
1. Accepts natural-language queries from the React chat client via SSE (`GET /chat/stream`)
2. Runs a **LangGraph multi-agent pipeline** (Orchestrator → DataFetcher → UIRenderer)
3. Streams progress events + a final **UISchema** back over Server-Sent Events
4. Connects to the **Ticket Management System MCP HTTP endpoint** to fetch/mutate data
5. Maintains **multi-turn conversation sessions** via an in-process LangGraph checkpointer

This server is the "brain" — it takes queries in natural language, decides which MCP tool to call, fetches data, then generates a structured UI schema for the React frontend to render.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Web framework | FastAPI 0.115+, uvicorn, sse-starlette |
| Agent framework | LangGraph 1.1+, LangChain Core 0.3+ |
| LLM | Google Gemini 2.0 Flash (`langchain-google-genai`) |
| Session persistence | `langgraph-checkpoint` 4.0+ `InMemorySaver` |
| Config | pydantic-settings 2.6+ |
| HTTP client | httpx 0.27+ (async) |
| Python | 3.13, virtual env at `.venv/` |

---

## Directory Structure

```
generative-ui-agents-server/
├── main.py                  # FastAPI app, lifespan, SSE endpoint, session endpoints
├── config.py                # Pydantic Settings (env vars, incl. session lifecycle)
├── requirements.txt         # All Python dependencies
├── .env                     # Local env vars (see below)
├── .env.example
│
├── agents/                  # LangGraph agent pipeline
│   ├── __init__.py
│   ├── state.py             # AgentState TypedDict (shared graph state)
│   ├── graph.py             # build_graph(checkpointer) factory + agent_graph singleton
│   ├── nodes.py             # Node implementations (3 agent functions)
│   │   # orchestrator_node  — picks the right MCP tool + args; injects prior context
│   │   # data_fetcher_node  — calls MCPHTTPClient.call_tool()
│   │   # ui_renderer_node   — generates UISchema from tool_result
│   └── prompts.py           # System + human prompt templates
│
├── sessions/                # In-process chat session lifecycle
│   ├── __init__.py
│   └── store.py             # SessionRecord, SessionRegistry, hash_api_key, singleton
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

# Session lifecycle (in-process store)
IDLE_TIMEOUT_SECONDS=1800    # Seconds idle before session expires (default: 30 min)
SESSION_SWEEP_INTERVAL_SECONDS=300  # Background eviction cadence (default: 5 min)
MAX_SESSIONS=1000            # Hard cap; oldest evicted first when exceeded
```

---

## Agent Pipeline (LangGraph)

```
User Query + session_id
    │
    ▼
[InMemorySaver]               — Restores prior AgentState for this thread_id
    │
    ▼
[orchestrator_node]           — Decides: which MCP tool? what args? (with prior context)
    │  sets: tool_name, tool_args, reasoning
    │  persists: AIMessage(decision summary) only — NOT the system prompt
    ▼
[data_fetcher_node]           — Calls MCPHTTPClient.call_tool(tool_name, tool_args)
    │  sets: tool_result
    ▼
[ui_renderer_node]            — Builds UISchema from tool_result using Gemini
    │  sets: ui_schema
    ▼
[InMemorySaver]               — Saves updated AgentState under thread_id
    ▼
SSE: ui_schema event → React frontend renders dynamically
```

Graph is compiled once during app lifespan startup in `agents/graph.py` and reused per request.

---

## What is a Checkpointer?

A **checkpointer** is LangGraph's built-in persistence layer. Every time a graph runs, it saves a snapshot of the full `AgentState` — called a **checkpoint** — after each **super-step** (one "tick" of the graph where all scheduled nodes execute). These checkpoints are keyed by a `thread_id`, which maps directly to our `session_id`.

### How it works in this server

```
Turn 1:  client sends query (no session_id)
         → server mints session_id = "abc-123"
         → graph runs with config={"configurable": {"thread_id": "abc-123"}}
         → InMemorySaver saves AgentState snapshot under thread "abc-123"

Turn 2:  client sends query + session_id="abc-123"
         → graph runs again with the SAME thread_id
         → InMemorySaver restores the prior AgentState (including messages history)
         → orchestrator sees past messages and resolves references like "change ITS status"
         → InMemorySaver saves updated snapshot
```

### Checkpoints vs Memory Store

| | Checkpointer (`InMemorySaver`) | Memory Store (`InMemoryStore`) |
|---|---|---|
| Scope | **Per thread** (per session) | **Cross-thread** (per user, global) |
| Holds | Full `AgentState` snapshot | Arbitrary key-value memories |
| Used for | Conversation continuity within one session | Long-term facts across all sessions |
| This project | ✅ Used — `InMemorySaver` | ❌ Not used (yet) |

We use only the checkpointer. A Memory Store would be the next step for things like "remember the user always works on project TIC".

### Super-steps and what gets saved

For our sequential pipeline `orchestrator → data_fetcher → ui_renderer`, LangGraph creates one checkpoint per node plus an input checkpoint, e.g.:

```
[input checkpoint]      → state: {query, messages: [HumanMsg]}
[after orchestrator]    → state: + tool_name, tool_args, AIMessage(decision)
[after data_fetcher]    → state: + tool_result
[after ui_renderer]     → state: + ui_schema
```

On the next turn, the graph restores from the **latest** checkpoint for that `thread_id` and the `add_messages` reducer **appends** the new turn's messages to the existing history rather than replacing it.

### What we persist per turn (bounded growth design)

To prevent the token cost from growing unboundedly each turn, only two small messages are appended to the persistent `messages` list per turn:

```
Turn 1 → HumanMessage("list tickets")     + AIMessage("[tool=ticket_list] ...")
Turn 2 → HumanMessage("create one called X") + AIMessage("[tool=ticket_create] ...")
Turn N → ...
```

`SystemMessage` prompts (which embed the full MCP tool catalogue) and UIRenderer LLM call context (raw JSON) are constructed fresh each turn and **never** stored in the checkpoint.

### Available checkpointer backends

| Backend | Package | Use case |
|---|---|---|
| `InMemorySaver` | `langgraph-checkpoint` (built-in) | Dev / single-instance (current) |
| `SqliteSaver` / `AsyncSqliteSaver` | `langgraph-checkpoint-sqlite` | Local persistence, single worker |
| `PostgresSaver` / `AsyncPostgresSaver` | `langgraph-checkpoint-postgres` | Production, multi-worker |
| `CosmosDBSaver` | `langgraph-checkpoint-cosmosdb` | Production on Azure |

Swapping backends only touches `sessions/store.py` and the lifespan wiring in `main.py` — the graph nodes and `AgentState` do not change.

### The `thread_id` contract

The single most important detail: **every graph invocation must pass `thread_id` in the config**, otherwise the checkpointer cannot associate the state with a session:

```python
config = {"configurable": {"thread_id": session_id}}
await agent_graph.astream_events(initial_state, config=config, version="v2")
```

Without this, each request gets a brand-new empty state — which is exactly the bug this feature fixes.

---

## Session Management

### Mechanism
Sessions use LangGraph's `InMemorySaver` checkpointer keyed by `thread_id == session_id` (UUID4). The `SessionRegistry` in `sessions/store.py` tracks lifecycle metadata (status, idle time, owner hash).

### Lifecycle
- **Create:** first request with no `session_id` → server mints UUID4, returns it in the `session` SSE event.
- **Resume:** subsequent requests send `&session_id=<id>` → checkpointer restores prior messages.
- **Idle expiry:** session status → `expired` after `IDLE_TIMEOUT_SECONDS` of inactivity. Checked on each request; server mints a fresh session automatically.
- **Explicit close:** client calls `POST /chat/session/{id}/close` → registry entry removed, checkpoints deleted.
- **Background sweep:** runs every `SESSION_SWEEP_INTERVAL_SECONDS` to evict idle/closed sessions and enforce `MAX_SESSIONS`.

### Message accumulation strategy
To avoid history balloon growth across turns, only these messages are persisted per turn:
- `HumanMessage` (added by `main.py` before the graph runs)
- `AIMessage` with a short decision summary (added by `orchestrator_node`)

`SystemMessage` prompts and UIRenderer LLM calls are **not** persisted — they are rebuilt fresh each turn.

### ⚠️ Single-worker constraint
Sessions live in process memory and are sticky to the worker that created them. **Run uvicorn with a single worker** (`python main.py` or `uvicorn main:app --port 8000` — no `--workers >1`) until the store is migrated to Redis/Mongo. Multi-worker deployments will silently break multi-turn context.

### Future migration
All lifecycle logic is in `sessions/store.py` behind a storage-agnostic interface. To migrate to Redis or Mongo, replace `InMemorySaver` + the registry dict — `main.py`, `agents/graph.py`, and nodes stay unchanged.

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
    messages: Annotated[list[AnyMessage], add_messages]  # Persisted conversation history
    mcp_api_key: str       # Forwarded from HTTP request (re-injected each turn)
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
| `session` | `{session_id: str}` | **FIRST event** — client must store and re-send this id |
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
GET  /health                           → {"status": "ok"}

GET  /chat/stream                      → SSE stream
     ?query=<string>                   (required) Natural language query
     ?api_key=<string>                 (required) MCP API key
     ?session_id=<string>              (optional) Resume existing session
     ?mcp_url=<string>                 (optional) Override MCP server URL
     Response header: X-Session-Id     Active session id (also in first SSE 'session' event)

POST /chat/session/{session_id}/close  → {"status": "closed", "session_id": ...}
     Call on 'New chat' / tab close to free in-process memory immediately.
     Use navigator.sendBeacon in beforeunload so the request survives unload.
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

# Start server — single worker only (see Session Management warning above)
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

The `mcp_api_key` and `mcp_server_url` are forwarded from the frontend request through the entire pipeline — agents never store credentials, they're passed through `AgentState` and re-injected fresh each turn.

---

## Key Files to Know First

1. `agents/state.py` — understand the data contract between agents
2. `agents/graph.py` — see the pipeline structure + `build_graph()` factory
3. `agents/nodes.py` — the actual agent logic + message accumulation rules
4. `sessions/store.py` — session lifecycle, registry, sweep
5. `schemas/ui_schema.py` — the React rendering contract
6. `mcp/client.py` — how MCP tools are called
7. `config.py` — all configurable settings
8. `main.py` — lifespan wiring, SSE endpoint, close endpoint
