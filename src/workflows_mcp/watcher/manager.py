from __future__ import annotations

from workflows_mcp.metadata.repos.watcher_repo import DirtyQueueEntry, SQLiteWatcherRepository

from .events import WatcherEvent


class WatcherManager:
    def __init__(self, repository: SQLiteWatcherRepository) -> None:
        self._repository = repository
        self._active_projects: set[str] = set()
        self._is_started = False
        self._backend_mode = "polling_fallback"
        self._backend_native_available = False

    @property
    def active_project_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_projects))

    @property
    def is_started(self) -> bool:
        return self._is_started

    @property
    def backend_mode(self) -> str:
        return self._backend_mode

    @property
    def backend_native_available(self) -> bool:
        return self._backend_native_available

    def start(self) -> list[str]:
        if self._is_started:
            return []
        restored = self.restore_enabled_watchers()
        self._is_started = True
        return restored

    def stop(self) -> None:
        if not self._is_started and not self._active_projects:
            return
        self._active_projects.clear()
        self._is_started = False

    def start_project_watcher(self, project_id: str) -> str:
        self._active_projects.add(project_id)
        return project_id

    def stop_project_watcher(self, project_id: str) -> str:
        self._active_projects.discard(project_id)
        return project_id

    def enable_project_by_default(self, project_id: str) -> str:
        status = self._repository.get_status(project_id)
        if status is None:
            status = self._repository.set_status(project_id=project_id, state="enabled")

        if status.state == "enabled" and self._is_started:
            self.start_project_watcher(project_id)

        return project_id

    def restore_enabled_watchers(self) -> list[str]:
        statuses = self._repository.list_statuses(state="enabled")
        restored: list[str] = []
        for status in statuses:
            restored.append(self.start_project_watcher(status.project_id))
        return restored

    def record_event(self, event: WatcherEvent) -> DirtyQueueEntry:
        reason = self._reason_for_event(event)
        return self._repository.enqueue_dirty(
            project_id=event.project_id,
            path=event.path,
            event_type=event.event_type,
            reason=reason,
        )

    @staticmethod
    def _reason_for_event(event: WatcherEvent) -> str:
        if event.requires_reconciliation:
            return f"reconciliation_required:{event.event_type}"
        return "file_event"
