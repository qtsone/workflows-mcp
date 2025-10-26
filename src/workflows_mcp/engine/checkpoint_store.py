"""Checkpoint storage implementations.

Provides storage backends for persisting checkpoint state.
Initial implementation uses in-memory storage.
"""

import asyncio
import time
from abc import ABC, abstractmethod

from workflows_mcp.engine.checkpoint import CheckpointState


class CheckpointStore(ABC):
    """Abstract base class for checkpoint storage."""

    @abstractmethod
    async def save_checkpoint(self, state: CheckpointState) -> str:
        """Save checkpoint state and return checkpoint ID."""
        ...

    @abstractmethod
    async def load_checkpoint(self, checkpoint_id: str) -> CheckpointState | None:
        """Load checkpoint state by ID, return None if not found."""
        ...

    @abstractmethod
    async def list_checkpoints(self, workflow_name: str | None = None) -> list[CheckpointState]:
        """List all checkpoints, optionally filtered by workflow name."""
        ...

    @abstractmethod
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint by ID, return True if deleted."""
        ...

    @abstractmethod
    async def cleanup_expired(self, max_age_seconds: int) -> int:
        """Remove checkpoints older than max_age_seconds, return count deleted."""
        ...


class InMemoryCheckpointStore(CheckpointStore):
    """In-memory checkpoint storage for development and testing.

    Thread-safe implementation using asyncio.Lock.
    """

    def __init__(self) -> None:
        """Initialize empty checkpoint store."""
        self._checkpoints: dict[str, CheckpointState] = {}
        self._lock = asyncio.Lock()

    async def save_checkpoint(self, state: CheckpointState) -> str:
        """Save checkpoint state in memory."""
        async with self._lock:
            self._checkpoints[state.checkpoint_id] = state
            return state.checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> CheckpointState | None:
        """Load checkpoint from memory."""
        async with self._lock:
            return self._checkpoints.get(checkpoint_id)

    async def list_checkpoints(self, workflow_name: str | None = None) -> list[CheckpointState]:
        """List checkpoints from memory, optionally filtered."""
        async with self._lock:
            checkpoints = list(self._checkpoints.values())

            if workflow_name is not None:
                checkpoints = [c for c in checkpoints if c.workflow_name == workflow_name]

            return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory."""
        async with self._lock:
            if checkpoint_id in self._checkpoints:
                del self._checkpoints[checkpoint_id]
                return True
            return False

    async def cleanup_expired(self, max_age_seconds: int) -> int:
        """Remove expired checkpoints from memory."""
        async with self._lock:
            current_time = time.time()
            cutoff_time = current_time - max_age_seconds

            expired_ids = [
                checkpoint_id
                for checkpoint_id, state in self._checkpoints.items()
                if state.created_at < cutoff_time
            ]

            for checkpoint_id in expired_ids:
                del self._checkpoints[checkpoint_id]

            return len(expired_ids)
