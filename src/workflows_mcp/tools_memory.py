"""Unified memory MCP tool — conditionally registered at startup."""

from __future__ import annotations

import json
import os
import uuid
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from .context import AppContextType
from .engine.memory_service import MemoryRequest, MemoryResponseInput, MemoryResult, MemoryService
from .engine.sql.postgres_backend import PostgresBackend


def _json_response(data: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, separators=(",", ":")))],
        structuredContent=data,
    )


def _tool_error_payload(tool: str, err: Exception, *, debug: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": f"{tool} failed"}
    if debug:
        details = str(err).strip()
        if not details:
            details = f"{err.__class__.__name__} (empty error message)"
        payload["details"] = details
    return payload


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
            return out

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
            return {"found": False}
        return payload

    if result.manage is not None:
        m = result.manage
        if not m.success:
            return {"error": m.error or "operation failed"}

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
            return base
        if op == "validate":
            return {"validated": m.validated_count}
        if op == "supersede":
            return {"superseded": len(m.superseded_ids)}
        if op == "archive":
            archive_out = {"archived": m.archived_count}
            if m.skipped_count:
                archive_out["skipped"] = m.skipped_count
            return archive_out
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
            return maintain_out or {"status": "ok"}
        if op == "graph_upsert":
            if m.entity_id:
                return {"entity_id": m.entity_id}
            if m.relation_id:
                return {"relation_id": m.relation_id}
            return {"status": "ok"}
        if op == "graph_delete":
            if m.deleted_entity_count:
                return {"deleted_places": m.deleted_entity_count}
            return {"deleted_links": m.deleted_relation_count}
        return m.model_dump()

    return {"error": "empty result"}


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


def register_memory_tools(mcp_server: FastMCP) -> None:
    """Register unified memory MCP tool on server instance."""

    @mcp_server.tool(
        description=(
            "Unified memory operations for query, ingest, maintenance, validation, "
            "and knowledge graph upsert/delete actions."
        ),
        annotations=ToolAnnotations(
            title="Memory",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        )
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
        query: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        record: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        graph: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        maintenance: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        response: Annotated[dict[str, Any] | None, Field(default=None)] = None,
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """Execute a unified memory operation and return a compact JSON payload."""
        from .engine.sql import ConnectionConfig, DatabaseEngine

        app_ctx = ctx.request_context.lifespan_context
        execution = _create_memory_execution(ctx)
        shared_backend = getattr(app_ctx, "memory_backend", None)
        shared_backend_lock = getattr(app_ctx, "memory_backend_lock", None)
        uses_ephemeral_backend = shared_backend is None
        backend = shared_backend if shared_backend is not None else PostgresBackend()
        debug = bool((response or {}).get("debug", False))

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
            return _json_response(_shape_memory_response(result, response_cfg))
        except Exception as e:
            return _json_response(_tool_error_payload("memory", e, debug=debug))
        finally:
            if uses_ephemeral_backend:
                await backend.disconnect()
