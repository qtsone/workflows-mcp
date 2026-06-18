"""Shared context types for MCP server.

This module contains context types used across server and tools modules,
separated to avoid circular imports.
"""

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from .engine import ExecutionContext, WorkflowRegistry
from .engine.execution_memory import ExecutionMemory
from .engine.executor_base import ExecutorRegistry
from .engine.io_queue import IOQueue
from .engine.job_queue import JobQueue
from .engine.llm_config import LLMConfigLoader
from .engine.secrets import SecretProvider

if TYPE_CHECKING:
    from .engine.memory_scope_resolver import SyncContextCandidate
    from .engine.workflow_source_loader import WorkflowSourceReloadSummary
    from .watcher.manager import WatcherManager


@dataclass
class MemoryBackendUnavailableError:
    """Typed fail-closed error descriptor for unavailable memory backend."""

    code: str
    message: str
    retryable: bool
    actionable_fix: str


@dataclass(frozen=True)
class SessionProjectContext:
    """Session/token-scoped project binding used for memory scope defaults."""

    project_id: str
    slug: str
    palace: str
    default_wing: str | None
    default_room: str | None
    source: str  # token_bound | token_unbound | session_selected
    fs_root: str | None = None
    fs_allowlist: tuple[str, ...] = ()


@dataclass
class AppContext:
    """Application context containing shared resources for MCP tools.

    This context is created during server startup and made available to all tools
    via dependency injection through the Context parameter.

    Post ADR-008: Stores shared resources for ExecutionContext creation.

    Session-scoped active context:
        _active_contexts maps id(session) → SyncContextCandidate.
        Set by onboard/select; read by memory/sync when no explicit scope is provided.
        Keyed by session object identity (id()) which is stable for the session lifetime
        and does not require any MCP SDK changes.
    """

    registry: WorkflowRegistry
    executor_registry: ExecutorRegistry
    llm_config_loader: LLMConfigLoader
    io_queue: IOQueue | None  # Optional IO queue for serialized file operations
    job_queue: JobQueue | None = None  # Optional job queue for async execution
    memory_backend: Any | None = None  # Optional shared memory backend (lifespan-scoped)
    memory_backend_lock: asyncio.Lock | None = None  # Serialize shared memory backend access
    memory_backend_unavailable_error: MemoryBackendUnavailableError | None = None
    metadata_base_dir: Path | None = None
    metadata_db_path: Path | None = None
    secret_provider: SecretProvider | None = None
    watcher_manager: "WatcherManager | None" = None
    max_recursion_depth: int = 50  # Default recursion depth limit
    # Optional callback for user context resolution.
    # OSS standalone: leave as None → tools fall back to OS/env user detection.
    # Platform: pass a function that reads platform-internal ContextVars.
    # Signature: () -> (user_uuid | None, user_string | None, auth_method)
    get_user_context: Callable[[], tuple[uuid.UUID | None, str | None, str]] | None = field(
        default=None, repr=False
    )
    # Optional callback used by MCP reload_workflows tool.
    reload_workflows: Callable[[], "WorkflowSourceReloadSummary"] | None = field(
        default=None,
        repr=False,
    )
    # Session-scoped active context: keyed by id(session) → SyncContextCandidate.
    _active_contexts: dict[int, "SyncContextCandidate"] = field(default_factory=dict, repr=False)
    # Session/token-scoped project bindings.
    _allowed_projects: dict[int, list[SessionProjectContext]] = field(
        default_factory=dict, repr=False
    )
    _active_projects: dict[int, SessionProjectContext] = field(default_factory=dict, repr=False)
    _session_onboard_contexts: dict[int, dict[str, "SyncContextCandidate"]] = field(
        default_factory=dict,
        repr=False,
    )
    _transport_sessions: dict[str, tuple[Any, str | None]] = field(default_factory=dict, repr=False)

    def get_active_context(self, session: Any) -> "SyncContextCandidate | None":
        """Return the active context for the given session, or None."""
        return self._active_contexts.get(id(session))

    def set_active_context(self, session: Any, candidate: "SyncContextCandidate") -> None:
        """Set the active context for the given session."""
        self._active_contexts[id(session)] = candidate

    def register_allowed_projects(
        self,
        session: Any,
        projects: list[SessionProjectContext],
    ) -> None:
        """Register allowed projects for a single session/token context."""
        self._allowed_projects[id(session)] = list(projects)

    def list_allowed_projects(self, session: Any) -> list[SessionProjectContext]:
        """Return allowed projects for the given session/token context."""
        return list(self._allowed_projects.get(id(session), []))

    def set_active_project(self, session: Any, project: SessionProjectContext) -> None:
        """Set one active project for the given session/token context."""
        self._active_projects[id(session)] = project

    def clear_active_project(self, session: Any) -> None:
        """Clear active project for a session, if any."""
        self._active_projects.pop(id(session), None)

    def get_active_project(self, session: Any) -> SessionProjectContext | None:
        """Return active project for a session, or None when unset."""
        return self._active_projects.get(id(session))

    def register_onboard_context_candidate(
        self,
        session: Any,
        candidate: "SyncContextCandidate",
    ) -> None:
        """Upsert one session-scoped onboard/sync context candidate."""
        session_id = id(session)
        scoped = self._session_onboard_contexts.setdefault(session_id, {})
        scoped[candidate.scope_key_value] = candidate

    def list_onboard_context_candidates(self, session: Any) -> list["SyncContextCandidate"]:
        """Return onboard/sync candidates visible to this session only."""
        scoped = self._session_onboard_contexts.get(id(session), {})
        return list(scoped.values())

    def bind_transport_session(
        self,
        transport_session_id: str,
        session: Any,
        *,
        token_id: str | None,
    ) -> None:
        """Map MCP transport session-id to (session, owning SQLite token id)."""
        if not transport_session_id:
            return
        self._transport_sessions[transport_session_id] = (session, token_id)

    def clear_session_state_by_transport_id(
        self,
        transport_session_id: str,
        *,
        requester_token_id: str | None,
    ) -> bool:
        """Clear session state when requester is owner or mapping is non-SQLite-owned.

        Security policy:
        - SQLite-owned mapping (owner token id set): requester token id must match.
        - Non-SQLite-owned mapping (owner token id None): allow clear.
        """
        if not transport_session_id:
            return False
        bound = self._transport_sessions.get(transport_session_id)
        if bound is None:
            return False
        session, owner_token_id = bound
        if owner_token_id is not None and requester_token_id != owner_token_id:
            return False
        self._transport_sessions.pop(transport_session_id, None)
        self.clear_session_state(session)
        return True

    def clear_session_state(self, session: Any) -> None:
        """Clear all session-scoped state to avoid stale/leaked context reuse."""
        session_id = id(session)
        self._active_contexts.pop(session_id, None)
        self._allowed_projects.pop(session_id, None)
        self._active_projects.pop(session_id, None)
        self._session_onboard_contexts.pop(session_id, None)
        stale_transport_ids = [
            transport_id
            for transport_id, (mapped_session, _owner_token_id) in self._transport_sessions.items()
            if id(mapped_session) == session_id
        ]
        for transport_id in stale_transport_ids:
            self._transport_sessions.pop(transport_id, None)

    def create_execution_context(
        self,
        workflow_stack: list[str] | None = None,
        execution_memory: ExecutionMemory | None = None,
        user_id: uuid.UUID | None = None,
        auth_method: str | None = None,
    ) -> ExecutionContext:
        """Create ExecutionContext for workflow execution.

        Args:
            workflow_stack: Optional workflow stack for resume (default: empty for new executions).
                           When resuming paused workflows, pass the saved workflow_stack to ensure
                           correct depth tracking for recursive workflows.
            execution_memory: Optional ephemeral SQLite memory for the execution.
            user_id: UUID of the user initiating the execution (for audit trails).
            auth_method: Authentication method used (PAT, SSO, SYSTEM) (for audit trails).

        Returns:
            ExecutionContext with access to all shared resources and configured recursion limit
        """
        return ExecutionContext(
            workflow_registry=self.registry,
            executor_registry=self.executor_registry,
            llm_config_loader=self.llm_config_loader,
            io_queue=self.io_queue,
            parent=None,
            workflow_stack=workflow_stack or [],
            max_recursion_depth=self.max_recursion_depth,
            execution_memory=execution_memory,
            secret_provider=self.secret_provider,
            user_id=user_id,
            auth_method=auth_method,
            memory_backend=self.memory_backend,
            memory_backend_lock=self.memory_backend_lock,
        )


# Type alias for MCP tool context parameter
AppContextType = Context[ServerSession, AppContext]


__all__ = [
    "AppContext",
    "AppContextType",
    "MemoryBackendUnavailableError",
    "SessionProjectContext",
]
