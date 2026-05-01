from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WatcherEventType = Literal[
    "created",
    "modified",
    "deleted",
    "renamed",
    "overflow",
    "permission_error",
    "dropped",
    "unknown",
]

_RECONCILIATION_EVENT_TYPES: frozenset[WatcherEventType] = frozenset(
    {"overflow", "permission_error", "dropped", "unknown"}
)


@dataclass(frozen=True)
class WatcherEvent:
    project_id: str
    path: str
    event_type: WatcherEventType
    previous_path: str | None = None
    message: str | None = None

    @property
    def requires_reconciliation(self) -> bool:
        return self.event_type in _RECONCILIATION_EVENT_TYPES
