"""Unified memory MCP tool — conditionally registered at startup."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field, ValidationError

from .context import AppContextType
from .engine.memory_service import (
    MemoryContractError,
    MemoryRequest,
    MemoryResponseInput,
    MemoryResult,
    MemoryService,
)
from .engine.sql.postgres_backend import PostgresBackend

logger = logging.getLogger(__name__)

_PROJECT_FLOW_VERSION = "oss-r2"
_PROJECT_FLOW_OPERATIONS: tuple[str, ...] = ("ingest", "supersede", "archive", "maintain")


def _json_response(data: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, separators=(",", ":")))],
        structuredContent=data,
    )


def _tool_error_payload(tool: str, err: Exception) -> dict[str, Any]:
    correlation_id = str(uuid.uuid4())

    if isinstance(err, MemoryContractError):
        code = err.code
        message = err.message
        retryable = err.retryable
        logger.warning(
            "memory tool contract error tool=%s code=%s retryable=%s correlation_id=%s",
            tool,
            code,
            retryable,
            correlation_id,
        )
    elif isinstance(err, ValidationError):
        contract_error: MemoryContractError | None = None
        for item in err.errors(include_url=False):
            ctx = item.get("ctx")
            if not isinstance(ctx, dict):
                continue
            nested = ctx.get("error")
            if isinstance(nested, MemoryContractError):
                contract_error = nested
                break
        if contract_error is not None:
            code = contract_error.code
            message = contract_error.message
            retryable = contract_error.retryable
            logger.warning(
                (
                    "memory tool contract validation error "
                    "tool=%s code=%s retryable=%s correlation_id=%s"
                ),
                tool,
                code,
                retryable,
                correlation_id,
            )
        else:
            code = "MEM_SCHEMA_VALIDATION_FAILED"
            message = "Request schema validation failed"
            retryable = False
            logger.warning(
                "memory tool schema validation failed tool=%s code=%s correlation_id=%s",
                tool,
                code,
                correlation_id,
            )
    else:
        code = "MEM_INTERNAL_ERROR"
        message = f"{tool} failed"
        retryable = False
        logger.exception(
            "memory tool internal error tool=%s code=%s correlation_id=%s",
            tool,
            code,
            correlation_id,
        )

    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "correlation_id": correlation_id,
        }
    }


def _error_envelope(
    *,
    code: str,
    message: str,
    retryable: bool = False,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        }
    }
    if correlation_id is not None:
        envelope["error"]["correlation_id"] = correlation_id
    return envelope


def _shape_scope_fields(result: MemoryResult) -> dict[str, Any]:
    if result.resolved_scope is None:
        return {}
    return {
        "resolved_scope": result.resolved_scope.model_dump(exclude_none=False),
        "scope_source": result.scope_source,
    }


def _lean_memory_item(m: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {"content": m.get("content", "")}
    if m.get("path"):
        item["path"] = m["path"]
    if m.get("source"):
        item["source"] = m["source"]
    return item


def _shape_memory_response(result: MemoryResult, response: MemoryResponseInput) -> dict[str, Any]:
    if response.debug:
        return result.model_dump(by_alias=True)

    scope_fields = _shape_scope_fields(result)

    if result.query is not None:
        q = result.query
        if response.mode == "graph":
            out: dict[str, Any] = {"paths": q.paths, "diagnostics": q.diagnostics}
            if q.evidence:
                first = q.evidence[0]
                if "nodes" in first:
                    out["nodes"] = first["nodes"]
                if "edges" in first:
                    out["edges"] = first["edges"]
            return {**out, **scope_fields}

        payload: dict[str, Any] = {}
        if q.facts:
            payload["facts"] = [_lean_memory_item(item) for item in q.facts]
        if q.memories:
            payload["memories"] = [_lean_memory_item(item) for item in q.memories]
        if q.communities:
            payload["communities"] = [{"content": c.get("content", "")} for c in q.communities]
        if response.mode == "evidence":
            payload["diagnostics"] = q.diagnostics
            payload["evidence"] = q.evidence
        if not payload:
            return {"found": False, **scope_fields}
        return {**payload, **scope_fields}

    if result.manage is not None:
        m = result.manage
        if not m.success:
            return {
                **_error_envelope(
                    code="MEM_OPERATION_FAILED",
                    message=m.error or "operation failed",
                    retryable=False,
                ),
                **scope_fields,
            }

        op = result.operation
        if op == "ingest":
            ids = m.memory_ids
            base: dict[str, Any] = {"stored": m.stored_count}
            if len(ids) == 1:
                base["id"] = ids[0]
            else:
                base["ids"] = ids
            if m.entity_ids:
                base["entity_ids"] = m.entity_ids
            if m.relation_ids:
                base["relation_ids"] = m.relation_ids
            if m.entities_stored_count:
                base["entities_stored"] = m.entities_stored_count
            if m.relations_stored_count:
                base["relations_stored"] = m.relations_stored_count
            return {**base, **scope_fields}
        if op == "validate":
            return {"validated": m.validated_count, **scope_fields}
        if op == "supersede":
            return {"superseded": len(m.superseded_ids), **scope_fields}
        if op == "archive":
            archive_out = {"archived": m.archived_count}
            if m.skipped_count:
                archive_out["skipped"] = m.skipped_count
            return {**archive_out, **scope_fields}
        if op == "maintain":
            maintain_out: dict[str, Any] = {}
            if m.communities_updated:
                maintain_out["communities_updated"] = m.communities_updated
            if m.assessed_count:
                maintain_out["assessed_count"] = m.assessed_count
            if m.expired_count:
                maintain_out["expired"] = m.expired_count
            if m.resolved_count:
                maintain_out["resolved"] = m.resolved_count
            if m.needs_review:
                maintain_out["needs_review_count"] = len(m.needs_review)
            if m.auto_archive_ids:
                maintain_out["auto_archived"] = len(m.auto_archive_ids)
            if response.include_candidates and m.prune_candidates:
                maintain_out["candidates"] = m.prune_candidates
            return {**(maintain_out or {"status": "ok"}), **scope_fields}
        if op == "graph_upsert":
            if m.entity_id:
                return {"entity_id": m.entity_id, **scope_fields}
            if m.relation_id:
                return {"relation_id": m.relation_id, **scope_fields}
            return {"status": "ok", **scope_fields}
        if op == "graph_delete":
            if m.deleted_entity_count:
                return {"deleted_places": m.deleted_entity_count, **scope_fields}
            return {"deleted_links": m.deleted_relation_count, **scope_fields}
        return {**m.model_dump(), **scope_fields}

    return {
        **_error_envelope(
            code="MEM_EMPTY_RESULT",
            message="empty result",
            retryable=False,
        ),
        **scope_fields,
    }


def _get_standalone_user_context() -> tuple[uuid.UUID | None, str | None, str]:
    import getpass

    from .engine.memory_service import SYSTEM_USER_UUID

    for env_var in [
        "WORKFLOWS_USER_ID",
        "WORKFLOWS_USER",
        "MCP_USER_ID",
        "USER",
        "USERNAME",
        "LOGNAME",
    ]:
        if os_user := os.environ.get(env_var):
            os_user = os_user.strip()
            try:
                return (uuid.UUID(os_user), os_user, "ENV_UUID")
            except ValueError:
                det_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"workflows-user:{os_user}")
                return (det_uuid, os_user, "OS_USER")

    try:
        os_user = getpass.getuser()
        det_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"workflows-user:{os_user}")
        return (det_uuid, os_user, "OS_USER")
    except Exception:
        return (SYSTEM_USER_UUID, "system", "SYSTEM")


def _create_memory_execution(ctx: AppContextType) -> Any:
    from .engine.execution import Execution

    app_ctx = ctx.request_context.lifespan_context
    if app_ctx.get_user_context:
        user_uuid, user_string, auth_method = app_ctx.get_user_context()
    else:
        user_uuid, user_string, auth_method = _get_standalone_user_context()

    exec_context = app_ctx.create_execution_context(user_id=user_uuid, auth_method=auth_method)
    if user_string:
        exec_context.user_string_id = user_string

    execution = Execution()
    execution.set_execution_context(exec_context)
    return execution


async def _execute_memory_request(
    *,
    app_ctx: Any,
    execution: Any,
    operation: str,
    scope: dict[str, Any] | None,
    scope_token: str | None,
    context_id: str | None,
    query: dict[str, Any] | None,
    record: dict[str, Any] | None,
    graph: dict[str, Any] | None,
    maintenance: dict[str, Any] | None,
    response: dict[str, Any] | None,
) -> dict[str, Any]:
    from .engine.sql import ConnectionConfig, DatabaseEngine

    shared_backend = getattr(app_ctx, "memory_backend", None)
    shared_backend_lock = getattr(app_ctx, "memory_backend_lock", None)
    uses_ephemeral_backend = shared_backend is None
    backend = shared_backend if shared_backend is not None else PostgresBackend()

    try:
        if uses_ephemeral_backend:
            await backend.connect(
                ConnectionConfig(
                    dialect=DatabaseEngine.POSTGRESQL,
                    host=os.environ.get("MEMORY_DB_HOST", "localhost"),
                    port=int(os.environ.get("MEMORY_DB_PORT", "5432")),
                    database=os.environ.get("MEMORY_DB_NAME", "memory_db"),
                    username=os.environ.get("MEMORY_DB_USER"),
                    password=os.environ.get("MEMORY_DB_PASSWORD"),
                )
            )

        service = MemoryService(backend, execution)
        request = MemoryRequest.model_validate(
            {
                "operation": operation,
                "scope": scope or {},
                "scope_token": scope_token,
                "context_id": context_id,
                "query": query,
                "record": record,
                "graph": graph,
                "maintenance": maintenance,
                "response": response or {},
            }
        )

        if shared_backend is not None and shared_backend_lock is not None:
            async with shared_backend_lock:
                result = await service.execute(request)
        else:
            result = await service.execute(request)

        response_cfg = MemoryResponseInput.model_validate(response or {})
        return _shape_memory_response(result, response_cfg)
    finally:
        if uses_ephemeral_backend:
            await backend.disconnect()


def _project_flow_plan(
    *,
    ingest: dict[str, Any] | None,
    supersede: dict[str, Any] | None,
    archive: dict[str, Any] | None,
    maintain: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    if ingest is not None:
        plan.append({"operation": "ingest", "payload": ingest})
    if supersede is not None:
        plan.append({"operation": "supersede", "payload": supersede})
    if archive is not None:
        plan.append({"operation": "archive", "payload": archive})
    if maintain is not None:
        plan.append({"operation": "maintain", "payload": maintain})
    return plan


def _checkpoint_error(message: str) -> MemoryContractError:
    return MemoryContractError(
        code="MEM_CHECKPOINT_INVALID",
        message=f"MEM_CHECKPOINT_INVALID: {message}",
        retryable=False,
    )


def _restore_checkpoint(
    checkpoint: dict[str, Any],
    *,
    require_ingest: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, Any]]]:
    if checkpoint.get("version") != _PROJECT_FLOW_VERSION:
        raise _checkpoint_error("unsupported checkpoint version")

    raw_scope = checkpoint.get("scope", {})
    if not isinstance(raw_scope, dict):
        raise _checkpoint_error("checkpoint.scope must be an object")

    raw_plan = checkpoint.get("plan")
    if not isinstance(raw_plan, list) or not raw_plan:
        raise _checkpoint_error("checkpoint.plan must be a non-empty list")

    normalized_plan: list[dict[str, Any]] = []
    for idx, step in enumerate(raw_plan):
        if not isinstance(step, dict):
            raise _checkpoint_error(f"checkpoint.plan[{idx}] must be an object")
        operation = step.get("operation")
        if operation not in _PROJECT_FLOW_OPERATIONS:
            allowed = ", ".join(_PROJECT_FLOW_OPERATIONS)
            raise _checkpoint_error(f"checkpoint.plan[{idx}].operation must be one of: {allowed}")
        payload = step.get("payload")
        if payload is not None and not isinstance(payload, dict):
            raise _checkpoint_error(f"checkpoint.plan[{idx}].payload must be an object")
        normalized_plan.append({"operation": operation, "payload": payload or {}})

    if require_ingest and normalized_plan[0]["operation"] != "ingest":
        raise _checkpoint_error("project_onboard checkpoint plan must start with ingest")

    raw_next_index = checkpoint.get("next_index", 0)
    if (
        not isinstance(raw_next_index, int)
        or raw_next_index < 0
        or raw_next_index > len(normalized_plan)
    ):
        raise _checkpoint_error("checkpoint.next_index is out of range")

    raw_completed = checkpoint.get("completed", [])
    if not isinstance(raw_completed, list):
        raise _checkpoint_error("checkpoint.completed must be a list")

    normalized_completed: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_completed):
        if not isinstance(item, dict):
            raise _checkpoint_error(f"checkpoint.completed[{idx}] must be an object")
        if idx >= len(normalized_plan):
            raise _checkpoint_error(
                f"checkpoint.completed[{idx}] has no corresponding checkpoint.plan step"
            )
        completed_operation = item.get("operation")
        if completed_operation is None:
            raise _checkpoint_error(f"checkpoint.completed[{idx}].operation must be present")
        if not isinstance(completed_operation, str):
            raise _checkpoint_error(f"checkpoint.completed[{idx}].operation must be a string")
        expected_operation = str(normalized_plan[idx]["operation"])
        if completed_operation != expected_operation:
            raise _checkpoint_error(
                f"checkpoint.completed[{idx}].operation must match checkpoint.plan[{idx}].operation"
            )
        normalized_completed.append(item)

    if len(normalized_completed) != raw_next_index:
        raise _checkpoint_error("checkpoint.completed length must equal checkpoint.next_index")

    return raw_scope, normalized_plan, raw_next_index, normalized_completed


def _build_new_checkpoint(
    *,
    scope: dict[str, Any] | None,
    ingest: dict[str, Any] | None,
    supersede: dict[str, Any] | None,
    archive: dict[str, Any] | None,
    maintain: dict[str, Any] | None,
    require_ingest: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, Any]]]:
    plan = _project_flow_plan(
        ingest=ingest,
        supersede=supersede,
        archive=archive,
        maintain=maintain,
    )
    if not plan:
        raise MemoryContractError(
            code="MEM_PROJECT_FLOW_EMPTY",
            message=(
                "MEM_PROJECT_FLOW_EMPTY: provide at least one of ingest, "
                "supersede, archive, or maintain"
            ),
            retryable=False,
        )
    if require_ingest and plan[0]["operation"] != "ingest":
        raise MemoryContractError(
            code="MEM_PROJECT_ONBOARD_REQUIRES_INGEST",
            message=("MEM_PROJECT_ONBOARD_REQUIRES_INGEST: project_onboard must start with ingest"),
            retryable=False,
        )
    return scope or {}, plan, 0, []


def _normalize_project_checkpoint(
    *,
    checkpoint: dict[str, Any] | None,
    scope: dict[str, Any] | None,
    ingest: dict[str, Any] | None,
    supersede: dict[str, Any] | None,
    archive: dict[str, Any] | None,
    maintain: dict[str, Any] | None,
    require_ingest: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, Any]]]:
    if checkpoint is not None:
        has_new_plan_input = (
            any(payload is not None for payload in (ingest, supersede, archive, maintain))
            or scope is not None
        )
        if has_new_plan_input:
            raise MemoryContractError(
                code="MEM_CHECKPOINT_CONFLICT",
                message=(
                    "MEM_CHECKPOINT_CONFLICT: provide either checkpoint OR "
                    "scope/plan payloads, not both"
                ),
                retryable=False,
            )
        return _restore_checkpoint(checkpoint, require_ingest=require_ingest)

    return _build_new_checkpoint(
        scope=scope,
        ingest=ingest,
        supersede=supersede,
        archive=archive,
        maintain=maintain,
        require_ingest=require_ingest,
    )


def _step_sections(
    operation: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if operation in {"ingest", "supersede", "archive"}:
        return None, payload, None
    if operation == "maintain":
        return None, None, payload
    raise MemoryContractError(
        code="MEM_INVALID_OPERATION",
        message=f"MEM_INVALID_OPERATION: unsupported project flow operation {operation!r}",
        retryable=False,
    )


def _extract_error_envelope(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return deterministic error envelope object when present, else None."""
    maybe_error = payload.get("error")
    if not isinstance(maybe_error, dict):
        return None

    code = maybe_error.get("code")
    message = maybe_error.get("message")
    retryable = maybe_error.get("retryable")
    if not isinstance(code, str) or not isinstance(message, str) or not isinstance(retryable, bool):
        return None

    envelope_error: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    correlation_id = maybe_error.get("correlation_id")
    if isinstance(correlation_id, str):
        envelope_error["correlation_id"] = correlation_id
    return envelope_error


def _build_project_checkpoint_payload(
    *,
    scope: dict[str, Any],
    plan: list[dict[str, Any]],
    next_index: int,
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": _PROJECT_FLOW_VERSION,
        "scope": scope,
        "plan": plan,
        "next_index": next_index,
        "completed": completed,
    }


def _build_project_failed_checkpoint_payload(
    *,
    error: dict[str, Any],
    failed_operation: str,
    scope: dict[str, Any],
    plan: list[dict[str, Any]],
    next_index: int,
    completed: list[dict[str, Any]],
    completed_ops: list[str],
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build deterministic checkpoint payload for failed project flow steps."""
    checkpoint_payload = _build_project_checkpoint_payload(
        scope=scope,
        plan=plan,
        next_index=next_index,
        completed=completed,
    )
    return {
        "status": "checkpoint",
        "failed_operation": failed_operation,
        "error": error,
        "remaining_operations": [
            str(remaining_step["operation"]) for remaining_step in plan[next_index:]
        ],
        "completed_operations": completed_ops,
        "last_operation": completed_ops[-1] if completed_ops else None,
        "result": result,
        "checkpoint": checkpoint_payload,
    }


def register_memory_tools(
    mcp_server: FastMCP,
    *,
    enable_project_tools: bool = True,
) -> None:
    """Register unified memory MCP tool (and optional project flow tools)."""

    @mcp_server.tool(
        description=(
            "Query and update memory in one tool (search, ingest, maintain, validate, "
            "and graph updates). Use this when you need direct memory operations."
        ),
        annotations=ToolAnnotations(
            title="Memory",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def memory(
        operation: Annotated[
            str,
            Field(
                description=(
                    "Operation: query, ingest, validate, supersede, archive, maintain, "
                    "graph_upsert, graph_delete"
                )
            ),
        ],
        scope: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        scope_token: Annotated[str | None, Field(default=None)] = None,
        context_id: Annotated[str | None, Field(default=None)] = None,
        query: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        record: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        graph: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        maintenance: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        response: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """Run one memory operation and return compact JSON results."""
        app_ctx = ctx.request_context.lifespan_context
        execution = _create_memory_execution(ctx)
        try:
            payload = await _execute_memory_request(
                app_ctx=app_ctx,
                execution=execution,
                operation=operation,
                scope=scope,
                scope_token=scope_token,
                context_id=context_id,
                query=query,
                record=record,
                graph=graph,
                maintenance=maintenance,
                response=response,
            )
            return _json_response(payload)
        except Exception as e:
            return _json_response(_tool_error_payload("memory", e))

    if not enable_project_tools:
        return

    @mcp_server.tool(
        description=(
            "Start or continue project memory onboarding using resumable checkpoints. "
            "Call this first to run ingest/supersede/archive/maintain in order."
        ),
        annotations=ToolAnnotations(
            title="Project Onboard",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def project_onboard(
        scope: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Project scope for onboarding (for example palace/wing/room/compartment)."
                ),
            ),
        ] = None,
        ingest: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Memory ingest payload (record section for memory(operation='ingest')). "
                    "Required when starting a new onboarding flow."
                ),
            ),
        ] = None,
        supersede: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional supersede payload (record section for memory(operation='supersede'))."
                ),
            ),
        ] = None,
        archive: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional archive payload (record section for memory(operation='archive'))."
                ),
            ),
        ] = None,
        maintain: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Optional maintain payload "
                    "(maintenance section for memory(operation='maintain'))."
                ),
            ),
        ] = None,
        checkpoint: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Checkpoint returned by project_onboard or project_sync to resume progress."
                ),
            ),
        ] = None,
        response: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional memory response shaping options."),
        ] = None,
        max_operations: Annotated[
            int,
            Field(
                default=1,
                ge=1,
                le=20,
                description=(
                    "Maximum onboarding steps to execute before returning. "
                    "Use 1 for strict checkpoint progression."
                ),
            ),
        ] = 1,
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """Run project onboarding steps and return a checkpoint for the next call."""
        app_ctx = ctx.request_context.lifespan_context
        execution = _create_memory_execution(ctx)
        try:
            resolved_scope, plan, next_index, completed = _normalize_project_checkpoint(
                checkpoint=checkpoint,
                scope=scope,
                ingest=ingest,
                supersede=supersede,
                archive=archive,
                maintain=maintain,
                require_ingest=True,
            )

            completed_ops: list[str] = []
            last_result: dict[str, Any] | None = None

            for _ in range(max_operations):
                if next_index >= len(plan):
                    break
                step = plan[next_index]
                operation_name = str(step["operation"])
                step_payload = step.get("payload")
                if not isinstance(step_payload, dict):
                    raise _checkpoint_error("checkpoint step payload must be an object")
                query_payload, record_payload, maintenance_payload = _step_sections(
                    operation_name, step_payload
                )
                try:
                    step_result = await _execute_memory_request(
                        app_ctx=app_ctx,
                        execution=execution,
                        operation=operation_name,
                        scope=resolved_scope,
                        scope_token=None,
                        context_id=None,
                        query=query_payload,
                        record=record_payload,
                        graph=None,
                        maintenance=maintenance_payload,
                        response=response,
                    )
                except Exception as step_exc:
                    step_error_payload = _tool_error_payload("project_onboard", step_exc)
                    step_error = step_error_payload.get("error")
                    if not isinstance(step_error, dict):
                        step_error = {
                            "code": "MEM_INTERNAL_ERROR",
                            "message": "project_onboard failed",
                            "retryable": False,
                        }
                    return _json_response(
                        _build_project_failed_checkpoint_payload(
                            error=step_error,
                            failed_operation=operation_name,
                            scope=resolved_scope,
                            plan=plan,
                            next_index=next_index,
                            completed=completed,
                            completed_ops=completed_ops,
                            result=last_result,
                        )
                    )
                step_error = _extract_error_envelope(step_result)
                if step_error is not None:
                    return _json_response(
                        _build_project_failed_checkpoint_payload(
                            error=step_error,
                            failed_operation=operation_name,
                            scope=resolved_scope,
                            plan=plan,
                            next_index=next_index,
                            completed=completed,
                            completed_ops=completed_ops,
                            result=step_result,
                        )
                    )
                completed.append({"operation": operation_name, "result": step_result})
                completed_ops.append(operation_name)
                last_result = step_result
                next_index += 1

            completed_results: list[dict[str, Any]] = completed
            checkpoint_payload = _build_project_checkpoint_payload(
                scope=resolved_scope,
                plan=plan,
                next_index=next_index,
                completed=completed_results,
            )

            if next_index < len(plan):
                return _json_response(
                    {
                        "status": "checkpoint",
                        "remaining_operations": [
                            str(step["operation"]) for step in plan[next_index:]
                        ],
                        "completed_operations": completed_ops,
                        "last_operation": completed_ops[-1] if completed_ops else None,
                        "result": last_result,
                        "checkpoint": checkpoint_payload,
                    }
                )

            return _json_response(
                {
                    "status": "completed",
                    "completed_operations": [
                        str(item.get("operation")) for item in completed_results
                    ],
                    "results": completed_results,
                    "checkpoint": checkpoint_payload,
                }
            )
        except Exception as e:
            return _json_response(_tool_error_payload("project_onboard", e))

    @mcp_server.tool(
        description=(
            "Continue project memory synchronization from a checkpoint, or start a new sync plan. "
            "Call this after project_onboard returns status='checkpoint'."
        ),
        annotations=ToolAnnotations(
            title="Project Sync",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def project_sync(
        checkpoint: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description=(
                    "Checkpoint returned by project_onboard or project_sync."
                ),
            ),
        ] = None,
        scope: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description="Scope for starting a new sync flow when checkpoint is omitted.",
            ),
        ] = None,
        ingest: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional ingest payload for new sync flows."),
        ] = None,
        supersede: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional supersede payload for new sync flows."),
        ] = None,
        archive: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional archive payload for new sync flows."),
        ] = None,
        maintain: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional maintain payload for new sync flows."),
        ] = None,
        response: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Optional memory response shaping options."),
        ] = None,
        max_operations: Annotated[
            int,
            Field(
                default=1,
                ge=1,
                le=20,
                description="Maximum sync steps to execute before returning.",
            ),
        ] = 1,
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """Advance project sync steps and return the next checkpoint or final result."""
        app_ctx = ctx.request_context.lifespan_context
        execution = _create_memory_execution(ctx)
        try:
            resolved_scope, plan, next_index, completed = _normalize_project_checkpoint(
                checkpoint=checkpoint,
                scope=scope,
                ingest=ingest,
                supersede=supersede,
                archive=archive,
                maintain=maintain,
                require_ingest=False,
            )

            completed_ops: list[str] = []
            last_result: dict[str, Any] | None = None

            for _ in range(max_operations):
                if next_index >= len(plan):
                    break
                step = plan[next_index]
                operation_name = str(step["operation"])
                step_payload = step.get("payload")
                if not isinstance(step_payload, dict):
                    raise _checkpoint_error("checkpoint step payload must be an object")
                query_payload, record_payload, maintenance_payload = _step_sections(
                    operation_name, step_payload
                )
                try:
                    step_result = await _execute_memory_request(
                        app_ctx=app_ctx,
                        execution=execution,
                        operation=operation_name,
                        scope=resolved_scope,
                        scope_token=None,
                        context_id=None,
                        query=query_payload,
                        record=record_payload,
                        graph=None,
                        maintenance=maintenance_payload,
                        response=response,
                    )
                except Exception as step_exc:
                    step_error_payload = _tool_error_payload("project_sync", step_exc)
                    step_error = step_error_payload.get("error")
                    if not isinstance(step_error, dict):
                        step_error = {
                            "code": "MEM_INTERNAL_ERROR",
                            "message": "project_sync failed",
                            "retryable": False,
                        }
                    return _json_response(
                        _build_project_failed_checkpoint_payload(
                            error=step_error,
                            failed_operation=operation_name,
                            scope=resolved_scope,
                            plan=plan,
                            next_index=next_index,
                            completed=completed,
                            completed_ops=completed_ops,
                            result=last_result,
                        )
                    )
                step_error = _extract_error_envelope(step_result)
                if step_error is not None:
                    return _json_response(
                        _build_project_failed_checkpoint_payload(
                            error=step_error,
                            failed_operation=operation_name,
                            scope=resolved_scope,
                            plan=plan,
                            next_index=next_index,
                            completed=completed,
                            completed_ops=completed_ops,
                            result=step_result,
                        )
                    )
                completed.append({"operation": operation_name, "result": step_result})
                completed_ops.append(operation_name)
                last_result = step_result
                next_index += 1

            completed_results: list[dict[str, Any]] = completed
            checkpoint_payload = _build_project_checkpoint_payload(
                scope=resolved_scope,
                plan=plan,
                next_index=next_index,
                completed=completed_results,
            )

            if next_index < len(plan):
                return _json_response(
                    {
                        "status": "checkpoint",
                        "remaining_operations": [
                            str(step["operation"]) for step in plan[next_index:]
                        ],
                        "completed_operations": completed_ops,
                        "last_operation": completed_ops[-1] if completed_ops else None,
                        "result": last_result,
                        "checkpoint": checkpoint_payload,
                    }
                )

            return _json_response(
                {
                    "status": "completed",
                    "completed_operations": [
                        str(item.get("operation")) for item in completed_results
                    ],
                    "results": completed_results,
                    "checkpoint": checkpoint_payload,
                }
            )
        except Exception as e:
            return _json_response(_tool_error_payload("project_sync", e))
