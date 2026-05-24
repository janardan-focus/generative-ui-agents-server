"""
In-process chat session lifecycle management.

Responsibilities
----------------
- Mint / retrieve / expire / close chat sessions (identified by UUID4 session_id).
- Persist nothing to disk — all state lives in process memory.
- Provide a storage-agnostic public interface so swapping to Redis/Mongo later
  only touches THIS module and the lifespan wiring in main.py.

Limitations (by design — see docs/session-management-plan.md §9)
-----------------------------------------------------------------
- State is lost on process restart.
- Sessions are sticky to the worker process that created them.  Run uvicorn with
  a single worker (no --workers >1) until the store is migrated to shared storage.

Thread safety
-------------
All public methods are async and serialise access through a single asyncio.Lock.
Do NOT call sync variants from concurrent coroutines without the lock.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SessionRecord:
    """Lifecycle metadata for one chat session."""

    session_id: str
    """UUID4 — also used as the LangGraph checkpointer thread_id."""

    status: str
    """One of: 'active' | 'closed' | 'expired'."""

    created_at: float
    """UTC epoch seconds (time.time()) when the session was minted."""

    last_active_at: float
    """UTC epoch seconds; refreshed by touch() on every turn."""

    turn_count: int = 0
    """Number of completed turns in this session."""

    owner_hash: str = ""
    """SHA-256 hex of the MCP api_key — never the raw key.
    Stored for future 'list my chats' queries; not used today."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hash_api_key(raw: str) -> str:
    """Return the SHA-256 hex digest of a raw API key string.

    Mirrors the convention used in ticket-management-mcp/auth/api_key.py.
    We never store the raw key anywhere in session metadata.
    """
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SessionRegistry:
    """
    In-process store for session lifecycle metadata + a reference to the
    LangGraph InMemorySaver so we can delete thread checkpoints on close/evict.

    Public interface is intentionally storage-agnostic:  every caller goes
    through async methods; the internal dict + lock is an implementation detail.
    """

    def __init__(self, saver: "InMemorySaver", idle_timeout: int, max_sessions: int) -> None:
        """
        Parameters
        ----------
        saver:
            The shared InMemorySaver instance compiled into the LangGraph graph.
            SessionRegistry holds a reference so it can delete thread checkpoints
            when a session is closed or evicted.
        idle_timeout:
            Seconds of inactivity before a session is considered expired.
        max_sessions:
            Hard cap on concurrent sessions.  When exceeded, the oldest
            (by last_active_at) session is evicted to make room.
        """
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()
        self._saver = saver
        self._idle_timeout = idle_timeout
        self._max_sessions = max_sessions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create(self, owner_hash: str = "") -> str:
        """Mint a new active session and return its session_id.

        If the registry is at capacity, the oldest idle session is evicted first.
        """
        async with self._lock:
            await self._enforce_cap_locked()
            session_id = str(uuid.uuid4())
            now = time.time()
            self._sessions[session_id] = SessionRecord(
                session_id=session_id,
                status="active",
                created_at=now,
                last_active_at=now,
                turn_count=0,
                owner_hash=owner_hash,
            )
            logger.info("[SessionRegistry] Created session=%s", session_id)
            return session_id

    async def touch(self, session_id: str) -> None:
        """Refresh last_active_at and increment turn_count for an existing session."""
        async with self._lock:
            record = self._sessions.get(session_id)
            if record and record.status == "active":
                record.last_active_at = time.time()
                record.turn_count += 1
                logger.debug(
                    "[SessionRegistry] Touched session=%s turn=%d",
                    session_id,
                    record.turn_count,
                )

    async def get(self, session_id: str) -> SessionRecord | None:
        """Return the SessionRecord for session_id, or None if not found."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def is_active(self, session_id: str) -> bool:
        """Return True iff the session exists, is 'active', and is within idle timeout.

        Side-effect: if the session exists but has exceeded idle_timeout, its
        status is set to 'expired' here (deterministic expiry at request time).
        """
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return False
            if record.status != "active":
                return False
            if time.time() - record.last_active_at > self._idle_timeout:
                record.status = "expired"
                logger.info(
                    "[SessionRegistry] Session expired (idle) session=%s", session_id
                )
                return False
            return True

    async def close(self, session_id: str) -> None:
        """Explicitly close a session: mark it closed, remove from registry, delete checkpoints."""
        async with self._lock:
            record = self._sessions.pop(session_id, None)
            if record is None:
                return
            record.status = "closed"
            await self._delete_thread_locked(session_id)
            logger.info("[SessionRegistry] Closed session=%s", session_id)

    async def sweep(self) -> None:
        """Evict all closed/expired/idle sessions and enforce the max_sessions cap.

        Called by the background task in main.py on a configurable cadence.
        This is a memory hygiene pass only — correctness does not depend on it
        because expiry is also checked deterministically in is_active().
        """
        async with self._lock:
            now = time.time()
            to_evict = [
                sid
                for sid, rec in list(self._sessions.items())
                if rec.status != "active"
                or (now - rec.last_active_at) > self._idle_timeout
            ]
            for sid in to_evict:
                self._sessions.pop(sid, None)
                await self._delete_thread_locked(sid)

            if to_evict:
                logger.info(
                    "[SessionRegistry] Sweep evicted %d session(s); %d remaining",
                    len(to_evict),
                    len(self._sessions),
                )

            # Enforce hard cap: evict oldest by last_active_at
            await self._enforce_cap_locked()

    async def resolve_or_create(
        self,
        session_id: str | None,
        owner_hash: str = "",
    ) -> tuple[str, bool]:
        """Return (session_id, was_created).

        If session_id is None, unknown, or expired/closed, a new session is
        minted and was_created=True is returned.  Otherwise the existing active
        session is returned with was_created=False.

        NOTE: This method does NOT call touch(); the caller (_run_agent) must
        call touch() after the turn completes.
        """
        if session_id is not None:
            # is_active acquires the lock internally; we can't hold it here
            # without nesting, so call it as a public method.
            if await self.is_active(session_id):
                return session_id, False

        new_id = await self.create(owner_hash=owner_hash)
        return new_id, True

    # ------------------------------------------------------------------
    # Private helpers (must be called with self._lock held)
    # ------------------------------------------------------------------

    async def _delete_thread_locked(self, session_id: str) -> None:
        """Delete the LangGraph checkpoints for a thread (best-effort)."""
        try:
            await self._saver.adelete_thread(session_id)
        except Exception as exc:  # noqa: BLE001
            # Not fatal — worst case a small amount of checkpoint memory lingers
            logger.warning(
                "[SessionRegistry] Could not delete thread=%s: %s", session_id, exc
            )

    async def _enforce_cap_locked(self) -> None:
        """If over max_sessions, evict the oldest session(s) by last_active_at."""
        while len(self._sessions) >= self._max_sessions:
            oldest_sid = min(
                self._sessions, key=lambda sid: self._sessions[sid].last_active_at
            )
            self._sessions.pop(oldest_sid, None)
            await self._delete_thread_locked(oldest_sid)
            logger.warning(
                "[SessionRegistry] Max sessions cap hit — evicted oldest session=%s",
                oldest_sid,
            )


# ---------------------------------------------------------------------------
# Module-level singleton (set during app lifespan startup in main.py)
# ---------------------------------------------------------------------------

registry: SessionRegistry | None = None
