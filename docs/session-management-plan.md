# Implementation Plan — Chat Session Management

> **Audience:** Claude Sonnet (implementing engineer).
> **Repo:** `generative-ui-agents-server` (Python 3.13, FastAPI + LangGraph).
> **Goal:** Maintain ONE conversation session for the whole chat (multi-turn context
> continuity) instead of starting a fresh, contextless stream per query. A session ends
> on **idle timeout OR explicit close** ("till the user leaves the chat").
>
> **Store decision (made by the project owner):** **in-process memory store** — no new
> infrastructure, no database connection added to the agents server. This is a deliberate
> choice for a single-instance / research-preview deployment. See §2 for why, and §9 for
> the limitations you must respect.
>
> **Identity decision:** sessions are **anonymous** (server-minted UUID). We additionally
> stamp a **SHA-256 hash of the MCP api_key** on each session as an `owner_hash` tag so a
> future "list my chats" feature is possible without a rewrite. We do NOT store the raw key.

---

## 0. Verified facts (confirmed against the repo on 2026-05-23 — do not re-assume)

Read these before you touch anything. They overturn assumptions an earlier draft of this
plan made:

1. **The agents server has NO database connection.** There is no Mongo/Redis client, no
   `MONGODB_URI`, nothing. Only the **Next.js app** (`Ticket-Management-System/lib/db.ts`,
   mongoose) talks to MongoDB Atlas. The **MCP server was deliberately refactored** into a
   thin HTTP proxy and its `config.py` states *"MongoDB settings have been removed."* So
   "just reuse the Mongo cluster" is **not free** for this server — it would mean a brand-new
   Atlas dependency. We are intentionally NOT doing that. Use the in-process store.

2. **Installed `langgraph` is `1.1.6`** (`.venv/.../langgraph-1.1.6.dist-info`), with
   `langgraph-checkpoint==4.0.2`. This is **LangGraph 1.x**, not the `0.3.x` in
   `requirements.txt`'s lower bound.

3. **`AsyncMongoDBSaver` is deprecated/removed in LangGraph 1.x** (GitHub issue
   `langchain-ai/langgraph#6506`, `langchain-mongodb#285`). The `from
   langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver` import does not resolve. Even if
   we *were* using Mongo, that old API guidance would be wrong. We are not using it — noted
   here only so you don't resurrect it from older tutorials.

4. **The frontend** (`Ticket-Management-System/mcp-chat-client/src/lib/agent-client.ts`) uses
   a native browser `EventSource` (GET-only) against `/chat/stream`. It has **no session
   concept** yet. `EventSource` cannot send custom headers, so the session id round-trips via
   the URL query string and (optionally) the first SSE event. Plan for GET + query params.

5. `agents/state.py` **already declares** `messages: Annotated[list[AnyMessage], add_messages]`.
   The reducer is ready; it just never accumulates because state is never persisted between
   requests and `main.py` resets `"messages": []` every turn.

---

## 1. Problem Statement (why this is needed)

Today every query is fully independent:

- `main.py → _run_agent()` builds a brand-new `initial_state` on **every** request, with
  `"messages": []`. Nothing is persisted between requests.
- `agents/graph.py` compiles the graph with **no checkpointer** (`_builder.compile()` with no args).
- `GET /chat/stream` takes only `query`, `api_key`, `mcp_url` — there is no session/thread identifier.

Net effect: the orchestrator can never see what the user asked before. "Show me ticket TIC-1"
followed by "change its status to Done" cannot work, because the second turn has lost "TIC-1".

**Fix:** Introduce a stable `session_id` per chat, persist the LangGraph state per session in
an in-process checkpointer keyed by `thread_id == session_id`, restore it on each turn, and
expire/close sessions on idle timeout or explicit close.

---

## 2. Chosen Approach (and why)

### Mechanism: LangGraph `InMemorySaver` checkpointer + a small in-process session registry

LangGraph's persistence layer is the idiomatic way to get multi-turn memory: a *checkpointer*
saves the full graph state under a `thread_id` after every super-step and restores it on the
next invocation. We map **`thread_id == session_id`**. We do NOT hand-roll message
serialization — the checkpointer + the existing `add_messages` reducer handle it.

For the store we use the checkpointer that ships **inside `langgraph-checkpoint` (already
installed, v4.0.2)** — no new dependency:

```python
from langgraph.checkpoint.memory import InMemorySaver   # 1.x name
```

> **Verify the import name at install time.** Across versions this class has been exposed as
> `InMemorySaver` and (older) `MemorySaver`, both in `langgraph.checkpoint.memory`. Open
> `.venv/lib/python3.13/site-packages/langgraph/checkpoint/memory/__init__.py` and use whatever
> the installed 4.0.2 exports. If both exist, prefer `InMemorySaver`.

The checkpointer holds **graph state** (including `messages`). It does **not** track session
*lifecycle* (created/active/closed, idle expiry, owner). For that we add a tiny in-process
**`SessionRegistry`** (a dict guarded by an `asyncio.Lock`) — see §5.3. Lifecycle logic stays
explicit and unit-testable instead of being smeared across `main.py`.

### Why in-process memory (the owner's decision), and what we gave up

- **No new infra, no new DB connection.** Preserves the clean boundary where only the Next.js
  app owns MongoDB and this server stays stateless-by-network. Fastest path to working
  multi-turn chat.
- **Trade-offs we accept (see §9):** state is lost on process restart, and it does **not**
  work across multiple uvicorn workers/replicas (a session is sticky to the process that
  created it). This is acceptable for single-instance / research-preview. The code is
  structured (§5.3) so swapping in Redis or a Mongo checkpointer later touches **one module**.

### Why not the alternatives (recorded for context)

- **Mongo checkpointer:** durable + multi-worker, but adds a direct Atlas dependency to a
  server that deliberately has none, and pulls in `langgraph-checkpoint-mongodb` +
  `pymongo`. Deferred by decision.
- **Redis:** clean TTL semantics and horizontal scale, but brand-new infra not run today.
  Deferred.

---

## 3. Session Lifecycle (idle timeout + explicit close)

```
create  ──► active ──(each turn refreshes last_active_at)──► active
                │                                              │
                │ idle > IDLE_TIMEOUT_SECONDS                  │ client sends close
                ▼                                              ▼
             expired  ◄─────────── both terminal ───────────► closed
```

- **Create:** on a turn where the client sends no `session_id` (or an unknown/expired/closed
  one), the server mints a UUID4, registers it, and tells the client the id (see §5.5c).
- **Resume:** subsequent turns send the same `session_id`; the checkpointer restores prior
  `messages` via the `thread_id`.
- **Idle timeout:** `IDLE_TIMEOUT_SECONDS` (default **1800s / 30 min**, configurable). Enforced
  at request time: on each turn the registry checks `now - last_active_at > IDLE_TIMEOUT`; if
  exceeded, the session is marked `expired` and a fresh one is minted (new id returned). A
  background sweep (§5.3) also evicts idle/closed sessions so memory doesn't grow unbounded.
- **Explicit close:** new endpoint `POST /chat/session/{session_id}/close` marks the session
  `closed`, evicts it from the registry, and deletes its checkpoints (so memory is reclaimed).
  The React client calls this on "New chat" / chat-close / tab unload.

> Expiry is **deterministic at request time** (the per-turn staleness check), so the
> background sweep is only for memory hygiene, not correctness.

---

## 4. Data Model (in-process)

There is no database. State lives in two in-memory places, both inside the agents-server
process:

### 4.1 Checkpoints — owned by `InMemorySaver` (do not touch directly)

`InMemorySaver` keeps full graph state keyed by `thread_id` (= our `session_id`). You never
read/write it directly — only via `agent_graph` invocation with `configurable.thread_id`, and
deletion via the saver's delete-thread method at close time (verify the exact method name in
4.0.2, see §5.6).

### 4.2 `SessionRegistry` — our lifecycle metadata (one dict)

Each entry (a small dataclass, `SessionRecord`):

| Field            | Type            | Notes                                                       |
|------------------|-----------------|-------------------------------------------------------------|
| `session_id`     | `str`           | UUID4 (also the checkpointer `thread_id`)                   |
| `status`         | `str`           | `"active" \| "closed" \| "expired"`                         |
| `created_at`     | `float`         | `time.monotonic()` or epoch seconds (UTC) — be consistent   |
| `last_active_at` | `float`         | refreshed every turn; basis for idle expiry                 |
| `turn_count`     | `int`           | incremented per query                                       |
| `owner_hash`     | `str`           | SHA-256 hex of the MCP api_key (never the raw key)          |

Guard the dict with a single `asyncio.Lock` — SSE handlers are async and concurrent.

---

## 5. Concrete Code Changes

All paths relative to `generative-ui-agents-server/`.

### 5.1 `requirements.txt` — align the lower bound, add NOTHING new

No new packages. The in-process saver lives in already-installed `langgraph-checkpoint`.

Do bump the stale lower bounds so a fresh `pip install` can't pull a pre-1.x LangGraph whose
checkpointer API differs:

```
langgraph>=1.1.0,<2.0
langgraph-checkpoint>=4.0.0
```

> Confirm these match what's in `.venv` (`langgraph 1.1.6`, `langgraph-checkpoint 4.0.2`).
> Run `pip show langgraph langgraph-checkpoint` to verify before editing the file.

### 5.2 `config.py` — new settings (lifecycle only)

Add to `Settings`:

```python
# Session lifecycle (in-process store)
idle_timeout_seconds: int = Field(default=1800, description="Idle expiry for a chat session")
session_sweep_interval_seconds: int = Field(default=300, description="Background eviction cadence")
max_sessions: int = Field(default=1000, description="Safety cap; evict oldest when exceeded")
```

Add matching keys to `.env` and `.env.example`:

```
IDLE_TIMEOUT_SECONDS=1800
SESSION_SWEEP_INTERVAL_SECONDS=300
MAX_SESSIONS=1000
```

No DB/connection settings — by design.

### 5.3 New package: `sessions/` (`__init__.py` + `store.py`)

Encapsulate ALL lifecycle here so `main.py` stays thin and this is unit-testable. **This is the
one module you'd swap to move to Redis/Mongo later — keep its public interface storage-agnostic.**

`sessions/store.py` contents:

- `SessionRecord` dataclass (fields per §4.2).
- `class SessionRegistry:`
  - holds `self._sessions: dict[str, SessionRecord]`, `self._lock = asyncio.Lock()`,
    and a reference to the `InMemorySaver` (so close can delete checkpoints).
  - `async def create(self, owner_hash: str) -> str` — mint UUID4, insert active record, return id.
  - `async def touch(self, session_id: str) -> None` — set `last_active_at = now`, `turn_count += 1`.
  - `async def get(self, session_id: str) -> SessionRecord | None`.
  - `async def is_active(self, session_id: str) -> bool` — exists AND status active AND
    `now - last_active_at <= idle_timeout`. If it exists but is idle, mark it `expired` here.
  - `async def close(self, session_id: str) -> None` — mark `closed`, pop from dict, and
    `await delete_thread(session_id)` on the saver (verify method name, §5.6).
  - `async def sweep(self) -> None` — evict `closed`/`expired` and idle-past-timeout entries;
    also enforce `max_sessions` (evict oldest `last_active_at`). Delete their checkpoints too.
  - `def resolve_or_create(...)` convenience that returns `(session_id, was_created: bool)`.
- `def hash_api_key(raw: str) -> str` — `hashlib.sha256(raw.encode()).hexdigest()`. Mirror the
  MCP server's SHA-256 convention (`ticket-management-mcp/auth/api_key.py`).
- A module-level singleton `registry: SessionRegistry | None` set during app startup, plus a
  background sweep task started/stopped by the app lifespan.

> Keep `time` handling consistent: use timezone-aware UTC epoch seconds (`time.time()`), not
> `monotonic()`, if you ever log timestamps to humans. Pick one and document it in the file.

### 5.4 `agents/graph.py` — compile WITH the checkpointer

Currently compiles at import time with no persistence. Refactor so the **same node/edge
definition** is compiled once at startup, bound to the shared saver. Preferred shape:

```python
_builder = StateGraph(AgentState)
# ... add_node / add_edge / set_entry_point unchanged ...

def build_graph(checkpointer):
    return _builder.compile(checkpointer=checkpointer)

agent_graph = None  # set during app lifespan startup (see main.py)
```

`main.py`'s lifespan sets `agents.graph.agent_graph = build_graph(saver)`.

> Build ONE saver and ONE compiled graph at startup; reuse for all requests. Do not create a
> saver per request.

### 5.5 `main.py` — lifespan, endpoint changes, state restore

**(a) App lifespan.** Replace the bare `app = FastAPI(...)` with an `@asynccontextmanager`
lifespan that:
1. Creates the `InMemorySaver`.
2. Creates the `SessionRegistry(saver=...)` and assigns the module singleton in `sessions.store`.
3. Builds the graph: `agents.graph.agent_graph = build_graph(saver)`.
4. Starts the background `sweep` task (`asyncio.create_task`).
5. On shutdown: cancel the sweep task.

Pass `lifespan=lifespan` to `FastAPI(...)`.

**(b) `_run_agent()` — accept `session_id`, stop hardcoding empty messages.**
- New signature: `_run_agent(query, mcp_api_key, mcp_server_url, session_id)`.
- **Remove `"messages": []`** from `initial_state`. Seed the new turn as
  `"messages": [HumanMessage(content=query)]` and let the `add_messages` reducer append to the
  restored history. NEVER reset to `[]` — that wipes the conversation. (Keep the other
  fields — `tool_name`, `tool_args`, etc. — as a fresh per-turn scratch; they're recomputed
  each turn and don't need to persist.)
- Invoke the graph **with a thread config** so it loads/saves the right session:

  ```python
  config = {"configurable": {"thread_id": session_id}}
  async for event in agent_graph.astream_events(initial_state, config=config, version="v2"):
      ...
  ```

  > This `config=` argument is the single most important change — without `thread_id` the
  > checkpointer cannot associate state with a session, and you'll silently keep getting
  > fresh, contextless turns.

- Call `await registry.touch(session_id)` once per turn.
- **Credential hygiene:** `mcp_api_key` / `mcp_server_url` still flow through `AgentState`.
  With the in-process saver they'll sit in memory inside the checkpoint. To avoid persisting
  the raw key in restored state, re-inject it into `initial_state` fresh from the request
  every turn (you already pass it in), and treat the restored value as overwritten. If you
  want to be strict, drop `mcp_api_key` from the persisted channels and pass it out-of-band;
  but for in-process this is low-risk — note it and move on.

**(c) `GET /chat/stream` — session handling.**
- New optional query param: `session_id: str | None = Query(default=None)`.
- At the top of the handler, before streaming:
  - `owner_hash = hash_api_key(api_key)`
  - If `session_id is None` or `not await registry.is_active(session_id)`:
    `session_id = await registry.create(owner_hash)` and remember `was_created = True`.
- **Tell the client the id — do BOTH:**
  1. Emit it as the **first** SSE event: `yield _sse("session", {"session_id": session_id})`.
  2. Set an `X-Session-Id` response header (`EventSourceResponse(..., headers={"X-Session-Id": session_id})`).
     (Header is a convenience; the SSE event is the source of truth the current `EventSource`
     client can actually read — see §6.)
- Then run `_run_agent(..., session_id=session_id)`.

**(d) New endpoint — explicit close:**

```python
@app.post("/chat/session/{session_id}/close")
async def close_session(session_id: str):
    await registry.close(session_id)
    return JSONResponse({"status": "closed", "session_id": session_id})
```

**(e) New SSE event type** — document `session` in the module docstring and CLAUDE.md SSE
table: `session  {"session_id": str}` — emitted first, tells the client which session to reuse.

### 5.6 `agents/nodes.py` — make the orchestrator USE the restored history

This is essential — restoring `messages` is pointless if no node reads them. **Read the file
first**, then:
- In `orchestrator_node`, the LLM is currently called with only a fresh
  `SystemMessage` + `HumanMessage(query)`. Include the **prior conversation** so references
  like "change *its* status" resolve. Two acceptable approaches:
  - Pass `state["messages"]` (the restored history) ahead of the new system/human messages in
    the `_llm.ainvoke([...])` call; or
  - Build a short text summary of prior turns and interpolate it into `ORCHESTRATOR_HUMAN`.
- Be careful not to double-append (the context-balloon trap): the nodes currently return
  `"messages": [system_msg, human_msg, response]`, and `add_messages` **appends** rather than
  replaces. With no persistence today that list is discarded after each request, so it's
  harmless. Once the checkpointer persists across turns, that same line re-stores the **full
  system prompt every turn** — and `ORCHESTRATOR_SYSTEM` is large because it embeds the MCP
  tool descriptions fetched per request. So history grows by a big fixed chunk each turn, all
  of it re-sent to the LLM, driving up token cost/latency and eventually risking the context
  window:

  ```
  Bad  (system re-stored each turn):   t1 [system, human1, resp1]  →  t2 [system, human1, resp1, system, human2, resp2]  → …
  Good (only conversational content):  t1 [human1, resp1]          →  t2 [human1, resp1, human2, resp2]                  → …
  ```

  Fix: persist only the **human query** and the **assistant's final answer/summary** per turn;
  keep the system prompt as a per-turn, non-persisted message prepended at call time (the same
  fix as the bullet above). Adjust what each node returns into `messages` accordingly, and
  verify history grows by a small bounded amount per turn (see §8).

> Do not blindly rewrite prompts. Read `agents/prompts.py`, understand the current contract,
> and make the minimal change that gives the orchestrator prior-turn context. Note that
> restoring history is inert unless a node actually feeds `state["messages"]` into its LLM
> call — the checkpointer saving history is necessary but not sufficient.

### 5.7 Verify the saver's delete-thread API (do this during step 1)

Open `.venv/lib/python3.13/site-packages/langgraph/checkpoint/memory/__init__.py` (and the
`base` module). Confirm:
- the class name (`InMemorySaver` vs `MemorySaver`),
- the async delete-by-thread method name (e.g. `adelete_thread(thread_id)` — name varies by
  version). Use whatever 4.0.2 exposes; if there's no public delete, dropping the registry
  entry is enough for correctness and the saver's own memory for a dead thread is small —
  note the limitation rather than reaching into private state.

---

## 6. Frontend (`mcp-chat-client`) — coordination notes

The client lives at `Ticket-Management-System/mcp-chat-client/`. Today
`src/lib/agent-client.ts` uses a native `EventSource` and has no session concept. Contract to
implement (and a constraint to respect):

1. **`EventSource` is GET-only and cannot set request headers.** So the session id must travel
   as a **query param** outbound and be read from the **`session` SSE event** inbound (the
   `X-Session-Id` header is not readable from `EventSource`). Add an `onSession` callback to
   `AgentStreamCallbacks` and a listener for the `session` event that stores the id.
2. Keep `session_id` in React state/context for the life of the chat — **not** `localStorage`
   if you want it to die with the tab (matches "till the user leaves"). Send
   `&session_id=<id>` on every subsequent `streamChat` call once known.
3. On "New chat" / chat-close / `beforeunload`, `POST /chat/session/{id}/close`. Use
   `navigator.sendBeacon` inside `beforeunload` so the request survives unload.
4. On receiving a `session` event whose id differs from the one held (i.e. the server expired
   the old one and minted a new one), update the stored id silently.

> The agents-server work (§5) is the deliverable for this task. Treat §6 as the contract the
> frontend must follow; implement it only if the same task explicitly includes the client.

---

## 7. Step-by-Step Build Order (for Sonnet)

1. **Confirm the environment.** `pip show langgraph langgraph-checkpoint`. Open
   `langgraph/checkpoint/memory/__init__.py` to confirm the saver class name and the
   delete-thread method (§5.7). No package installs expected.
2. **Config + env.** Add lifecycle settings (§5.2) to `config.py`, `.env`, `.env.example`.
3. **Sessions package.** Create `sessions/store.py` with `SessionRecord`, `SessionRegistry`,
   `hash_api_key`, the module singleton, and the sweep coroutine (§5.3).
4. **Graph.** Refactor `agents/graph.py` to `build_graph(checkpointer)` (§5.4).
5. **App lifespan + endpoints.** Wire the lifespan, `session_id` param, create/restore logic,
   `config={"configurable":{"thread_id": session_id}}`, `touch`, the first-event `session`
   emit + `X-Session-Id` header, and the close endpoint (§5.5).
6. **Nodes.** Give the orchestrator restored-history context and fix message accumulation (§5.6).
7. **Docs.** Update `CLAUDE.md` (SSE events table → add `session`; endpoints → add close route;
   env vars → add the three lifecycle keys; directory structure → add `sessions/`; add a
   short "Session management" section describing the in-process store and its limits).
8. **Verify** (§8).

---

## 8. Verification & Testing (required before declaring done)

- **Unit — registry:** create → `is_active` true → `touch` increments `turn_count` →
  back-date `last_active_at` beyond `IDLE_TIMEOUT` → `is_active` false and status becomes
  `expired`; `close` sets `closed`, removes the entry, and (if supported) deletes the thread.
- **Unit — sweep:** populate with mixed active/idle/closed entries, run `sweep`, assert only
  live ones remain and `max_sessions` cap is enforced.
- **Multi-turn integration (the key proof):** start a stream with NO `session_id`; capture the
  id from the `session` event. Send a follow-up that references the prior turn (turn 1: "list
  tickets for project TIC"; turn 2: "create one titled 'Login bug' in that project"). Assert
  turn 2's orchestrator had prior context — e.g. `tool_args` carries the project from turn 1,
  and the restored `messages` length on turn 2 > turn 1. This is the regression that proves
  the whole feature.
- **Bounded history growth:** assert `messages` grows by a small fixed amount per turn (no
  system-prompt accumulation blow-up, per §5.6).
- **Idle expiry:** set `IDLE_TIMEOUT_SECONDS=2`, wait >2s, send another turn, assert a NEW
  `session_id` is issued.
- **Explicit close:** call the close endpoint; assert a subsequent turn with the old id starts
  fresh (new id, empty history).
- **Regression:** existing single-query behavior still streams
  `agent_thinking → … → ui_schema → done`, now preceded by a `session` event.
- **Restart caveat (document, don't fix):** note in the test summary that restarting the
  server drops all sessions — expected for the in-process store.

---

## 9. Risks / Gotchas / Limitations (the honest list)

- **No durability:** server restart wipes all sessions. Acceptable for now; called out so it
  isn't a surprise in a demo.
- **Single-instance only (sessions are process-sticky):** each worker process has its own
  private memory, so a session lives only in the RAM of the worker that created it. With
  multiple workers/replicas behind a load balancer, turn 1 may land on Worker A (which holds
  the history) while turn 2 lands on Worker B (which has never seen that `session_id` → empty
  history → broken multi-turn). Round-robin routing makes this fail intermittently and
  unpredictably. **Run uvicorn with a single worker** (don't pass `--workers >1`, run a single
  replica) until/unless the store moves to shared storage (Redis/Mongo). Add a one-line
  warning in the README/CLAUDE.md.
- **Memory growth:** without the sweep + `max_sessions` cap, abandoned sessions accumulate.
  The sweep (§5.3) and per-turn expiry handle this; make sure the sweep task is actually
  started in the lifespan and cancelled on shutdown.
- **Message accumulation:** the biggest correctness trap — if nodes keep appending system
  prompts to `messages`, history balloons and the LLM context cost grows every turn. Persist
  only what's needed (§5.6) and assert bounded growth (§8).
- **Version drift:** the saver class/method names vary across LangGraph versions. Verify
  against the installed 4.0.2 source (§5.7); do not trust tutorial snippets that import from
  `langgraph.checkpoint.mongodb.aio` (deprecated/removed) or assume `MemorySaver` vs
  `InMemorySaver`.
- **`EventSource` header limitation:** the session id MUST come back via the `session` SSE
  event for the current client to read it; the `X-Session-Id` header is a nice-to-have only.

---

## 10. Future migration path (out of scope now — keep the door open)

Because all lifecycle lives in `sessions/store.py` behind a storage-agnostic interface, moving
to a durable/multi-worker store later is a localized change:
- **Redis:** swap `InMemorySaver` for a Redis-backed checkpointer (or persist `messages`
  yourself) and back the registry with Redis hashes + native key TTL.
- **Mongo:** use `langgraph-checkpoint-mongodb`'s **`MongoDBSaver`** (NOT the deprecated
  `AsyncMongoDBSaver`), pointed at the Atlas cluster the Next.js app already uses; verify the
  current constructor + TTL support against the installed version at that time.
Either way, `main.py`, `agents/graph.py`, and the nodes should not need to change — only the
store module and the lifespan wiring.

## 11. Out of Scope (do not implement unless asked)

- Durable / cross-restart session storage (see §10).
- Multi-worker / horizontally-scaled deployment of sessions.
- Cross-device / cross-login session resume and a "my chats" listing UI (the `owner_hash` tag
  is the only forward-looking hook we add now).
- Long-term semantic memory / summarization of old turns (LangGraph Store, separate concern).
