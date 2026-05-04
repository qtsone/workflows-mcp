from __future__ import annotations

from pathlib import Path

import pytest

from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.run_history_repo import (
    SUMMARY_TRUNCATION_MARKER,
    InvalidRunHistoryPaginationError,
    RunHistoryTransactionOwnershipError,
    RunNotFoundError,
    SQLiteRunHistoryRepository,
)


def _create_repo(tmp_path: Path, *, summary_limit: int = 24) -> SQLiteRunHistoryRepository:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    migrate_metadata_db(conn)
    return SQLiteRunHistoryRepository(conn, summary_limit=summary_limit)


@pytest.mark.parametrize(
    ("raw", "limit", "expected"),
    [
        ("ok", 12, "ok"),
        ("x" * 12, 12, "x" * 12),
        ("abcdefghijklmnopqrstuvwxyz", 12, "abcdefg[...]"),
    ],
)
def test_create_run_compacts_result_summary_deterministically(
    tmp_path: Path,
    raw: str,
    limit: int,
    expected: str,
) -> None:
    repo = _create_repo(tmp_path, summary_limit=limit)
    record = repo.create_run(
        run_id="run-1",
        workflow_name="wf",
        status="running",
        timeout_seconds=300,
        created_at="2026-04-29T08:00:00Z",
        started_at="2026-04-29T08:00:00Z",
        updated_at="2026-04-29T08:00:00Z",
        result_summary=raw,
    )
    assert record.result_summary == expected
    assert len(record.result_summary) <= limit


@pytest.mark.parametrize(
    ("raw", "limit", "expected"),
    [
        ("err", 11, "err"),
        ("y" * 11, 11, "y" * 11),
        ("0123456789abc", 11, "012345[...]"),
    ],
)
def test_create_run_compacts_error_summary_deterministically(
    tmp_path: Path,
    raw: str,
    limit: int,
    expected: str,
) -> None:
    repo = _create_repo(tmp_path, summary_limit=limit)
    record = repo.create_run(
        run_id="run-2",
        workflow_name="wf",
        status="failed",
        timeout_seconds=300,
        created_at="2026-04-29T08:01:00Z",
        started_at="2026-04-29T08:01:00Z",
        updated_at="2026-04-29T08:01:00Z",
        error_summary=raw,
    )
    assert record.error_summary == expected
    assert len(record.error_summary) <= limit


def test_compaction_marker_contract_is_stable() -> None:
    assert SUMMARY_TRUNCATION_MARKER == "[...]"


def test_run_history_repo_create_update_get_and_list(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    created = repo.create_run(
        run_id="run-a",
        workflow_name="build-and-test",
        status="running",
        execution_mode="sync",
        timeout_seconds=120,
        created_at="2026-04-29T08:59:00Z",
        started_at="2026-04-29T09:00:00Z",
        updated_at="2026-04-29T09:00:00Z",
        inputs_json='{"branch":"main"}',
        project_id=None,
        token_id=None,
        cancellable=True,
        result_summary="starting",
        execution_json='{"status":"running","blocks":{}}',
    )
    assert created.run_id == "run-a"
    assert created.execution_mode == "sync"
    assert created.project_id is None
    assert created.token_id is None
    assert created.cancellable is True
    assert created.error_summary is None
    assert created.execution_json == '{"status":"running","blocks":{}}'
    assert created.timeout_seconds == 120
    assert created.created_at == "2026-04-29T08:59:00Z"
    assert created.started_at == "2026-04-29T09:00:00Z"
    assert created.updated_at == "2026-04-29T09:00:00Z"
    assert created.inputs_json == '{"branch":"main"}'

    updated = repo.update_run(
        run_id="run-a",
        status="completed",
        finished_at="2026-04-28T10:00:00Z",
        updated_at="2026-04-29T09:01:00Z",
        inputs_json='{"branch":"release"}',
        result_summary="done",
        execution_json='{"status":"success","outputs":{"ok":true}}',
    )
    assert updated.status == "completed"
    assert updated.finished_at == "2026-04-28T10:00:00Z"
    assert updated.result_summary == "done"
    assert updated.execution_json == '{"status":"success","outputs":{"ok":true}}'
    assert updated.updated_at == "2026-04-29T09:01:00Z"
    assert updated.inputs_json == '{"branch":"release"}'

    fetched = repo.get_run("run-a")
    assert fetched is not None
    assert fetched.run_id == "run-a"
    assert fetched.timeout_seconds == 120
    assert fetched.execution_mode == "sync"
    assert fetched.execution_json == '{"status":"success","outputs":{"ok":true}}'
    assert fetched.updated_at == "2026-04-29T09:01:00Z"
    assert fetched.inputs_json == '{"branch":"release"}'

    repo.create_run(
        run_id="run-b",
        workflow_name="deploy",
        status="queued",
        timeout_seconds=3600,
        created_at="2026-04-29T09:02:00Z",
        started_at=None,
        updated_at="2026-04-29T09:02:00Z",
    )
    listed = repo.list_runs(limit=10, offset=0)
    assert [item.run_id for item in listed] == ["run-b", "run-a"]
    assert listed[0].timeout_seconds == 3600
    assert listed[1].timeout_seconds == 120

    sync_runs = repo.list_runs(limit=10, offset=0, execution_mode="sync")
    assert [item.run_id for item in sync_runs] == ["run-a"]

    workflow_runs = repo.list_runs(limit=10, offset=0, workflow_name="deploy")
    assert [item.run_id for item in workflow_runs] == ["run-b"]

    assert repo.count_runs(status="completed") == 1
    assert repo.count_runs(execution_mode="sync") == 1


def test_update_run_keeps_inputs_json_when_not_provided(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    repo.create_run(
        run_id="run-inputs",
        workflow_name="wf",
        status="running",
        timeout_seconds=300,
        created_at="2026-04-29T10:00:00Z",
        started_at="2026-04-29T10:00:00Z",
        updated_at="2026-04-29T10:00:00Z",
        inputs_json='{"k":"v1"}',
    )

    updated = repo.update_run(
        run_id="run-inputs",
        status="running",
        updated_at="2026-04-29T10:01:00Z",
    )

    assert updated.updated_at == "2026-04-29T10:01:00Z"
    assert updated.inputs_json == '{"k":"v1"}'


def test_list_running_runs_for_stale_check_returns_only_running_rows(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    repo.create_run(
        run_id="run-running",
        workflow_name="wf",
        status="running",
        timeout_seconds=45,
        created_at="2026-04-29T11:00:00Z",
        started_at="2026-04-29T11:00:00Z",
        updated_at="2026-04-29T11:00:00Z",
    )
    repo.create_run(
        run_id="run-completed",
        workflow_name="wf",
        status="completed",
        timeout_seconds=90,
        created_at="2026-04-29T11:01:00Z",
        started_at="2026-04-29T11:01:00Z",
        updated_at="2026-04-29T11:01:00Z",
    )

    stale_rows = repo.list_running_runs_for_stale_check()
    assert [row.run_id for row in stale_rows] == ["run-running"]
    assert stale_rows[0].timeout_seconds == 45
    assert stale_rows[0].updated_at == "2026-04-29T11:00:00Z"


def test_update_missing_run_raises_domain_error(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    with pytest.raises(RunNotFoundError, match="run not found"):
        repo.update_run(
            run_id="missing",
            status="failed",
            updated_at="2026-04-29T12:00:00Z",
        )


def test_create_run_rejects_nested_transaction_ownership(tmp_path: Path) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    migrate_metadata_db(conn)
    repo = SQLiteRunHistoryRepository(conn)

    conn.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RunHistoryTransactionOwnershipError, match="caller-owned transaction"):
            repo.create_run(
                run_id="run-3",
                workflow_name="wf",
                status="running",
                timeout_seconds=300,
                created_at="2026-04-29T12:01:00Z",
                started_at="2026-04-29T12:01:00Z",
                updated_at="2026-04-29T12:01:00Z",
            )
    finally:
        conn.rollback()


def test_queued_run_started_at_can_transition_from_none_to_timestamp(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)

    created = repo.create_run(
        run_id="run-queued",
        workflow_name="wf",
        status="queued",
        timeout_seconds=180,
        created_at="2026-04-29T12:10:00Z",
        started_at=None,
        updated_at="2026-04-29T12:10:00Z",
    )
    assert created.created_at == "2026-04-29T12:10:00Z"
    assert created.started_at is None

    running = repo.update_run(
        run_id="run-queued",
        status="running",
        started_at="2026-04-29T12:10:05Z",
        updated_at="2026-04-29T12:10:05Z",
        cancellable=True,
    )
    assert running.status == "running"
    assert running.created_at == "2026-04-29T12:10:00Z"
    assert running.started_at == "2026-04-29T12:10:05Z"


def test_list_runs_rejects_non_positive_limit(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    with pytest.raises(InvalidRunHistoryPaginationError, match="limit must be > 0"):
        repo.list_runs(limit=0, offset=0)


def test_list_runs_rejects_negative_offset(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    with pytest.raises(InvalidRunHistoryPaginationError, match="offset must be >= 0"):
        repo.list_runs(limit=1, offset=-1)
