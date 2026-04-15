# Generative-UI Agent Server

Python FastAPI + LangGraph multi-agent server that bridges the React chat
frontend and the Ticket Management System MCP server, then synthesises a
**dynamic UI schema** so the frontend can render rich components instead of
raw JSON.

---

## Architecture

```
Browser (React Chat)
        │  GET /chat/stream?query=…  (SSE)
        ▼
┌─────────────────────────────────┐
│   FastAPI Agent Server :8000    │
│                                 │
│  ┌──────────────────────────┐   │
│  │   LangGraph Workflow     │   │
│  │                          │   │
│  │  1. OrchestratorAgent    │   │
│  │     – reads user query   │   │
│  │     – fetches live tool  │   │
│  │       list from MCP      │   │
│  │     – picks tool + args  │   │
│  │                          │   │
│  │  2. DataFetcherAgent     │   │
│  │     – calls MCP server   │   │
│  │     – returns raw JSON   │   │
│  │                          │   │
│  │  3. UIRendererAgent      │   │
│  │     – generates layout   │   │
│  │       + component list   │   │
│  └──────────────────────────┘   │
└────────────┬────────────────────┘
             │  POST /api/mcp  (JSON-RPC 2.0)
             ▼
┌────────────────────────────────────┐
│  Ticket Management System (Next.js)│
│  MCP HTTP endpoint :3000/api/mcp   │
│  Tools: project_*, ticket_*,       │
│         kanban_*                   │
└────────────────────────────────────┘
```

### SSE Event Stream

| Event           | Payload                      | Description                        |
|-----------------|------------------------------|------------------------------------|
| agent_thinking  | {agent, message}             | Node started processing            |
| tool_selected   | {tool, args, reasoning}      | Orchestrator chose a tool          |
| tool_executing  | {tool}                       | DataFetcher is calling MCP         |
| tool_result     | {tool, result}               | Raw MCP response received          |
| ui_generating   | {message}                    | UIRenderer started                 |
| ui_schema       | {schema: UISchema}           | Final renderable UI schema         |
| token           | {text, agent}                | LLM token (streamed)               |
| done            | {message}                    | Stream complete                    |
| error           | {message}                    | Pipeline error                     |

### UI Schema Contract

```jsonc
{
  "layout": "card-grid | table | detail | kanban | success | error | list | empty",
  "title": "string",
  "subtitle": "string | null",
  "components": [
    {
      "type": "card | ticket-card | table | badge | stat | list-item | kanban-column | success-banner | error-banner | json-viewer",
      "props": { "...": "..." }
    }
  ],
  "data": {},
  "actions": [
    { "label": "string", "tool": "mcp_tool_name", "args": {}, "style": "primary" }
  ],
  "metadata": {}
}
```

---

## Setup

### 1 – Python server

```bash
cd generative-ui-agents-server

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Set ANTHROPIC_API_KEY and MCP_SERVER_URL in .env

python main.py
# Listening on http://localhost:8000
```

### 2 – React chat client

```bash
cd Ticket-Management-System/mcp-chat-client

cp .env.example .env
# Defaults: VITE_AGENT_SERVER_URL=http://localhost:8000
#           VITE_MCP_SERVER_URL=http://localhost:3000/api/mcp

npm install
npm run dev
# http://localhost:5173
```

### 3 – Ticket Management System (MCP + API server)

```bash
cd Ticket-Management-System
npm install
# Set up .env (MongoDB URI, NextAuth secret, Google OAuth, etc.)
npm run dev
# http://localhost:3000
```

---

## File Structure

```
generative-ui-agents-server/
├── main.py              FastAPI app + /chat/stream SSE endpoint
├── config.py            pydantic-settings config
├── requirements.txt
├── .env.example
├── agents/
│   ├── graph.py         LangGraph StateGraph (compiled at import)
│   ├── nodes.py         orchestrator_node, data_fetcher_node, ui_renderer_node
│   ├── prompts.py       System prompts for each LLM call
│   └── state.py         AgentState TypedDict
├── mcp/
│   └── client.py        Async JSON-RPC 2.0 HTTP client for the MCP server
└── schemas/
    └── ui_schema.py     Pydantic models for UISchema + SSE payloads

Ticket-Management-System/mcp-chat-client/src/
├── App.tsx              Root — creates AgentClient, handles auth
├── types/chat.ts        ChatMessage, UISchema, AgentLog types
├── lib/
│   └── agent-client.ts  SSE client for the Python agent server
└── components/
    ├── ChatInterface.tsx SSE-driven chat loop
    ├── MessageBubble.tsx Renders text OR DynamicRenderer + AgentTrace
    └── DynamicRenderer.tsx Renders UISchema (card-grid, table, kanban, …)
```

---

## Extending

**Add a new MCP tool** — no server changes needed; the Orchestrator dynamically
fetches the tool list from the MCP server on every request.

**Add a new UI component type** — add to `UIComponentType` in
`schemas/ui_schema.py` (Python) and `types/chat.ts` (TypeScript), then add a
renderer in `DynamicRenderer.tsx`.

**Add a new layout** — add to the `UILayout` union, handle in the UIRenderer
system prompt, and add a dispatcher case in `DynamicRenderer.tsx`.

---

## Key Dependencies

| Package              | Purpose                            |
|----------------------|------------------------------------|
| fastapi + uvicorn    | HTTP server                        |
| sse-starlette        | Server-Sent Events                 |
| langgraph            | Agent workflow graph               |
| langchain-anthropic  | Claude LLM calls                   |
| httpx                | Async HTTP client for MCP          |
| pydantic-settings    | Typed env config                   |
