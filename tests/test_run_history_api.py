from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.run_history_repo import SQLiteRunHistoryRepository
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase2-admin-password"


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _MCP_BOOTSTRAP_TOKEN)
    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )
    return TestClient(build_app(base_dir=base_dir), raise_server_exceptions=False)


def _login_and_csrf(client: TestClient) -> str:
    response = client.post("/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD})
    assert response.status_code == 200
    csrf_token = response.headers.get("X-CSRF-Token") or response.json().get("csrf_token")
    assert csrf_token
    return str(csrf_token)


def _seed_run(
    client: TestClient,
    *,
    run_id: str,
    status: str,
    workflow_name: str,
    created_at: str,
    started_at: str | None,
    updated_at: str,
    finished_at: str | None,
    cancellable: bool,
    result_summary: str | None = None,
    error_summary: str | None = None,
    execution_mode: str = "async",
    execution_json: str | None = None,
    project_id: str | None = None,
    token_id: str | None = None,
) -> None:
    resources = client.app.state.resources
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    migrate_metadata_db(conn)
    repo = SQLiteRunHistoryRepository(conn)
    repo.create_run(
        run_id=run_id,
        workflow_name=workflow_name,
        status=status,
        execution_mode=execution_mode,
        timeout_seconds=300,
        created_at=created_at,
        started_at=started_at,
        updated_at=updated_at,
        project_id=project_id,
        token_id=token_id,
        cancellable=cancellable,
        result_summary=result_summary,
        error_summary=error_summary,
        execution_json=execution_json,
    )
    if finished_at is not None:
        repo.update_run(
            run_id=run_id,
            status=status,
            updated_at=updated_at,
            finished_at=finished_at,
            cancellable=cancellable,
            result_summary=result_summary,
            error_summary=error_summary,
            execution_json=execution_json,
        )
    conn.close()


def _error_code(payload: dict[str, object]) -> str | None:
    detail = payload.get("detail")
    if isinstance(detail, dict):
        code = detail.get("code")
        if isinstance(code, str):
            return code
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str):
            return code
    code = payload.get("code")
    if isinstance(code, str):
        return code
    return None


def test_runs_routes_require_admin_session_and_csrf_for_mutations(app_client: TestClient) -> None:
    unauth_list = app_client.get("/api/admin/v1/runs")
    assert unauth_list.status_code == 401

    unauth_detail = app_client.get("/api/admin/v1/runs/run-unknown")
    assert unauth_detail.status_code == 401

    unauth_cancel = app_client.post("/api/admin/v1/runs/run-unknown/cancel")
    assert unauth_cancel.status_code == 401

    unauth_resume = app_client.post("/api/admin/v1/runs/run-unknown/resume")
    assert unauth_resume.status_code == 401

    _login_and_csrf(app_client)

    missing_csrf_cancel = app_client.post("/api/admin/v1/runs/run-unknown/cancel")
    assert missing_csrf_cancel.status_code == 403

    missing_csrf_resume = app_client.post("/api/admin/v1/runs/run-unknown/resume")
    assert missing_csrf_resume.status_code == 403


def test_list_runs_supports_status_filter_and_deterministic_pagination(
    app_client: TestClient,
) -> None:
    _login_and_csrf(app_client)
    _seed_run(
        app_client,
        run_id="run-001",
        status="queued",
        workflow_name="wf-a",
        created_at="2026-04-29T09:00:00Z",
        started_at=None,
        updated_at="2026-04-29T09:00:00Z",
        finished_at=None,
        cancellable=True,
        execution_mode="sync",
        project_id=None,
        token_id=None,
    )
    _seed_run(
        app_client,
        run_id="run-002",
        status="completed",
        workflow_name="wf-b",
        created_at="2026-04-29T09:01:00Z",
        started_at="2026-04-29T09:01:05Z",
        updated_at="2026-04-29T09:01:08Z",
        finished_at="2026-04-29T09:01:08Z",
        cancellable=False,
        project_id=None,
        token_id=None,
    )
    _seed_run(
        app_client,
        run_id="run-003",
        status="queued",
        workflow_name="wf-c",
        created_at="2026-04-29T09:02:00Z",
        started_at=None,
        updated_at="2026-04-29T09:02:00Z",
        finished_at=None,
        cancellable=True,
        project_id=None,
        token_id=None,
    )

    filtered = app_client.get(
        "/api/admin/v1/runs",
        params={"status": "queued", "mode": "sync", "limit": 10, "offset": 0},
    )
    assert filtered.status_code == 200
    payload = filtered.json()
    runs = payload["runs"]
    assert payload["total"] == 1
    assert payload["limit"] == 10
    assert payload["offset"] == 0
    assert [item["run_id"] for item in runs] == ["run-001"]
    row = runs[0]
    assert row["job_id"] == row["run_id"]
    assert row["workflow_name"] == "wf-a"
    assert row["execution_mode"] == "sync"
    assert row["status"] == "queued"
    assert row["created_at"] == "2026-04-29T09:00:00Z"
    assert row["started_at"] is None
    assert row["finished_at"] is None
    assert row["updated_at"] == "2026-04-29T09:00:00Z"
    assert row["cancellable"] is True
    assert row["project_id"] is None
    assert row["token_id"] is None

    paged = app_client.get("/api/admin/v1/runs", params={"limit": 1, "offset": 1})
    assert paged.status_code == 200
    paged_payload = paged.json()
    paged_runs = paged_payload["runs"]
    assert paged_payload["total"] == 3
    assert [item["run_id"] for item in paged_runs] == ["run-002"]


def test_get_run_detail_returns_compact_summaries_and_missing_is_404(
    app_client: TestClient,
) -> None:
    _login_and_csrf(app_client)
    _seed_run(
        app_client,
        run_id="run-det-1",
        status="failed",
        workflow_name="wf-detail",
        created_at="2026-04-29T10:00:00Z",
        started_at="2026-04-29T10:00:01Z",
        updated_at="2026-04-29T10:00:03Z",
        finished_at="2026-04-29T10:00:03Z",
        cancellable=False,
        result_summary="result-summary",
        error_summary="error-summary",
        execution_mode="sync",
        execution_json=(
            '{"status":"failure","outputs":null,"error":"boom",'
            '"metadata":{"workflow_name":"wf-detail","execution_time_seconds":1.25},'
            '"blocks":{"setup":{"inputs":{"x":1},"outputs":{"stdout":"hi"},'
            '"metadata":{"id":"setup","type":"Shell","status":"completed",'
            '"duration_ms":12,"outcome":"success"}}}}'
        ),
    )

    detail = app_client.get("/api/admin/v1/runs/run-det-1")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["run_id"] == "run-det-1"
    assert payload["job_id"] == "run-det-1"
    assert payload["execution_mode"] == "sync"
    assert payload["result_summary"] == "result-summary"
    assert payload["error_summary"] == "error-summary"
    assert payload["inputs"] == {}
    assert payload["outputs"] is None
    assert payload["error"] == "boom"
    assert payload["metadata"]["workflow_name"] == "wf-detail"
    assert payload["blocks"][0]["block_id"] == "setup"
    assert payload["blocks"][0]["status"] == "completed"
    assert payload["blocks"][0]["duration_ms"] == 12
    assert payload["technical_json"]["status"] == "failure"

    missing = app_client.get("/api/admin/v1/runs/does-not-exist")
    assert missing.status_code == 404
    missing_payload = missing.json()
    assert _error_code(missing_payload) == "run_not_found"


def test_cancel_endpoint_returns_stable_queue_contract(app_client: TestClient) -> None:
    csrf_token = _login_and_csrf(app_client)
    _seed_run(
        app_client,
        run_id="run-cancel-1",
        status="running",
        workflow_name="wf-cancel",
        created_at="2026-04-29T11:00:00Z",
        started_at="2026-04-29T11:00:02Z",
        updated_at="2026-04-29T11:00:02Z",
        finished_at=None,
        cancellable=True,
    )

    async def _fake_cancel(job_id: str) -> dict[str, object]:
        return {
            "job_id": job_id,
            "cancelled": False,
            "outcome": "already_terminal",
            "error_code": "JOB_ALREADY_TERMINAL",
            "message": "Job is already in terminal state and cannot be cancelled",
        }

    app_client.app.state.resources.job_queue.cancel_job = _fake_cancel  # type: ignore[method-assign]

    response = app_client.post(
        "/api/admin/v1/runs/run-cancel-1/cancel",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "job_id": "run-cancel-1",
        "cancelled": False,
        "outcome": "already_terminal",
        "error_code": "JOB_ALREADY_TERMINAL",
        "message": "Job is already in terminal state and cannot be cancelled",
    }


def test_resume_endpoint_returns_conflict_for_non_paused_or_non_resumable_runs(
    app_client: TestClient,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    _seed_run(
        app_client,
        run_id="run-resume-1",
        status="running",
        workflow_name="wf-resume",
        created_at="2026-04-29T12:00:00Z",
        started_at="2026-04-29T12:00:01Z",
        updated_at="2026-04-29T12:00:01Z",
        finished_at=None,
        cancellable=True,
    )

    response = app_client.post(
        "/api/admin/v1/runs/run-resume-1/resume",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 409
    payload = response.json()
    assert _error_code(payload) == "run_not_resumable"


def test_resume_endpoint_uses_public_job_queue_resume_api(app_client: TestClient) -> None:
    csrf_token = _login_and_csrf(app_client)
    _seed_run(
        app_client,
        run_id="run-resume-public-1",
        status="paused",
        workflow_name="wf-resume-public",
        created_at="2026-04-29T13:00:00Z",
        started_at="2026-04-29T13:00:01Z",
        updated_at="2026-04-29T13:00:01Z",
        finished_at=None,
        cancellable=True,
    )

    called: list[tuple[str, str]] = []

    async def _fake_resume(job_id: str, response: str) -> dict[str, object]:
        called.append((job_id, response))
        return {
            "job_id": job_id,
            "resumed": False,
            "outcome": "not_resumable",
            "error_code": "JOB_NOT_RESUMABLE",
            "message": "Job is not paused or has no resumable execution state",
        }

    app_client.app.state.resources.job_queue.resume_job = _fake_resume  # type: ignore[method-assign]

    response = app_client.post(
        "/api/admin/v1/runs/run-resume-public-1/resume",
        headers={"X-CSRF-Token": csrf_token},
        json={"response": "approved"},
    )
    assert response.status_code == 409
    assert called == [("run-resume-public-1", "approved")]


def test_runs_route_source_has_no_private_queue_or_runner_calls() -> None:
    route_file = (
        Path(__file__).resolve().parents[1]
        / "src/workflows_mcp/http/routes/admin_v1/runs.py"
    )
    source = route_file.read_text(encoding="utf-8")
    assert "._store" not in source
    assert "._extract_execution_state" not in source
    assert "._build_debug_data" not in source
