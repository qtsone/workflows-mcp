"""Resumable project-flow checkpoint state machine.

``ProjectFlowService`` owns the multi-step onboard/sync flow: a checkpointed
plan of memory operations (ingest → supersede → archive → maintain) that can
pause between steps and resume from the returned checkpoint. The checkpoint
dict IS the state — it is passed in and out, so the service holds no
server-side flow state and is safe to share across concurrent agent sessions
(see ADR-014).

The single impure dependency — running one memory operation — is injected as an
``OperationExecutor`` port. The service imports nothing from the heavy memory
backend stack: transport, session, scan IO, and graph persistence stay in the
caller (the MCP tool builds an executor adapter wrapping the memory service;
tests inject a pure-dict fake).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from .memory_errors import MemoryContractError
from .memory_scope_resolver import scope_key

PROJECT_FLOW_VERSION = "oss-r3"
PROJECT_FLOW_OPERATIONS: tuple[str, ...] = ("ingest", "supersede", "archive", "maintain")

#: Routed sections for one memory operation: ``query``, ``record``, ``maintenance``.
StepSections = tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]

#: Injected port: run one memory operation against a scope and return its result.
#: Transport, session, response shaping, and backend selection live in the adapter.
OperationExecutor = Callable[[str, dict[str, Any], StepSections], Awaitable[dict[str, Any]]]

#: Injected port: shape a raised step exception into an error envelope ``{"error": {...}}``.
#: Keeps memory-specific error classification (codes, correlation ids) in the caller.
StepErrorShaper = Callable[[Exception, str], dict[str, Any]]


# ---------------------------------------------------------------------------
# Sync delta classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncDelta:
    """Deterministic delta between two file entry sets.

    Output of :func:`compute_sync_delta`. All path lists are sorted for
    deterministic ordering across calls.

    Attributes:
        added: Paths present in new entries but absent in prior entries.
        modified: Paths present in both sets with differing content_hash.
        deleted: Paths present in prior entries but absent in new entries.
        unchanged: Paths present in both sets with identical content_hash.
    """

    added: list[str]
    modified: list[str]
    deleted: list[str]
    unchanged: list[str]

    @property
    def has_semantic_delta(self) -> bool:
        """True when at least one file was added, modified, or deleted."""
        return bool(self.added or self.modified or self.deleted)

    @property
    def total_files(self) -> int:
        """Total file count in new snapshot (added + modified + unchanged)."""
        return len(self.added) + len(self.modified) + len(self.unchanged)

    def to_debug_dict(self) -> dict[str, Any]:
        """Serialise the delta for debug diagnostics in sync responses."""
        return {
            "added": sorted(self.added),
            "modified": sorted(self.modified),
            "deleted": sorted(self.deleted),
            "unchanged": sorted(self.unchanged),
            "has_semantic_delta": self.has_semantic_delta,
            "total_files": self.total_files,
            "counts": {
                "added": len(self.added),
                "modified": len(self.modified),
                "deleted": len(self.deleted),
                "unchanged": len(self.unchanged),
            },
        }


def compute_sync_delta(
    prior_entries: list[dict[str, Any]],
    new_entries: list[dict[str, Any]],
) -> SyncDelta:
    """Compute a deterministic four-class delta between two file entry sets.

    Each entry must have at minimum a ``path`` field and a ``content_hash``
    field.  Entries with missing ``path`` are silently ignored.

    Classification rules:
    - ``added``:     path in new_entries but not in prior_entries.
    - ``modified``:  path in both sets with differing content_hash values.
    - ``deleted``:   path in prior_entries but not in new_entries.
    - ``unchanged``: path in both sets with identical content_hash values.

    Args:
        prior_entries: File entries from the previous scan (snapshot).
        new_entries: File entries from the current scan.

    Returns:
        SyncDelta with all four classified path lists, sorted for determinism.
    """
    old_by_path: dict[str, str] = {}
    for e in prior_entries:
        p = e.get("path") or ""
        if p:
            old_by_path[p] = str(e.get("content_hash") or "")

    new_by_path: dict[str, str] = {}
    for e in new_entries:
        p = e.get("path") or ""
        if p:
            new_by_path[p] = str(e.get("content_hash") or "")

    added: list[str] = sorted(p for p in new_by_path if p not in old_by_path)
    deleted: list[str] = sorted(p for p in old_by_path if p not in new_by_path)
    modified: list[str] = []
    unchanged: list[str] = []
    for path, new_hash in new_by_path.items():
        if path in old_by_path:
            if old_by_path[path] != new_hash:
                modified.append(path)
            else:
                unchanged.append(path)

    return SyncDelta(
        added=added,
        modified=sorted(modified),
        deleted=deleted,
        unchanged=sorted(unchanged),
    )


def classify_deletion_policy(
    deletion_policy: str,
) -> Literal["archive", "supersede", "ignore"]:
    """Normalise deletion policy aliases to canonical storage actions.

    Two semantic aliases collapse onto ``archive``:
    - ``mark_missing``    → ``archive`` (marks files as inactive).
    - ``archive_missing`` → ``archive`` (explicit alias for clarity).

    The three original policies are passed through unchanged.

    Args:
        deletion_policy: Raw deletion policy string from ScanConfig.

    Returns:
        Canonical action: ``"archive"``, ``"supersede"``, or ``"ignore"``.

    Raises:
        ValueError: When the policy string is not recognised.
    """
    _policy_map: dict[str, Literal["archive", "supersede", "ignore"]] = {
        "archive": "archive",
        "supersede": "supersede",
        "ignore": "ignore",
        "mark_missing": "archive",
        "archive_missing": "archive",
    }
    canonical = _policy_map.get(deletion_policy)
    if canonical is None:
        allowed = ", ".join(sorted(_policy_map.keys()))
        raise ValueError(f"Unknown deletion_policy {deletion_policy!r}. Allowed values: {allowed}.")
    return canonical


# ---------------------------------------------------------------------------
# Pure checkpoint helpers
# ---------------------------------------------------------------------------


def project_flow_plan(
    *,
    ingest: dict[str, Any] | None,
    supersede: dict[str, Any] | None,
    archive: dict[str, Any] | None,
    maintain: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build an ordered plan of operation steps from the provided payloads."""
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


def checkpoint_error(message: str) -> MemoryContractError:
    return MemoryContractError(
        code="MEM_CHECKPOINT_INVALID",
        message=f"MEM_CHECKPOINT_INVALID: {message}",
        retryable=False,
    )


def restore_checkpoint(
    checkpoint: dict[str, Any],
    *,
    require_ingest: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Validate and normalise a resume checkpoint into ``(scope, plan, next_index, completed)``."""
    if checkpoint.get("version") != PROJECT_FLOW_VERSION:
        raise checkpoint_error("unsupported checkpoint version")

    raw_scope = checkpoint.get("scope", {})
    if not isinstance(raw_scope, dict):
        raise checkpoint_error("checkpoint.scope must be an object")

    raw_plan = checkpoint.get("plan")
    if not isinstance(raw_plan, list) or not raw_plan:
        raise checkpoint_error("checkpoint.plan must be a non-empty list")

    normalized_plan: list[dict[str, Any]] = []
    for idx, step in enumerate(raw_plan):
        if not isinstance(step, dict):
            raise checkpoint_error(f"checkpoint.plan[{idx}] must be an object")
        operation = step.get("operation")
        if operation not in PROJECT_FLOW_OPERATIONS:
            allowed = ", ".join(PROJECT_FLOW_OPERATIONS)
            raise checkpoint_error(f"checkpoint.plan[{idx}].operation must be one of: {allowed}")
        payload = step.get("payload")
        if payload is not None and not isinstance(payload, dict):
            raise checkpoint_error(f"checkpoint.plan[{idx}].payload must be an object")
        normalized_plan.append({"operation": operation, "payload": payload or {}})

    if require_ingest and normalized_plan[0]["operation"] != "ingest":
        raise checkpoint_error("onboard checkpoint plan must start with ingest")

    raw_next_index = checkpoint.get("next_index", 0)
    # Accept integral floats (e.g. 1.0 from JSON deserialisation); reject non-integral floats.
    if isinstance(raw_next_index, float):
        if raw_next_index != int(raw_next_index):
            raise checkpoint_error("checkpoint.next_index is out of range")
        raw_next_index = int(raw_next_index)
    if (
        not isinstance(raw_next_index, int)
        or raw_next_index < 0
        or raw_next_index > len(normalized_plan)
    ):
        raise checkpoint_error("checkpoint.next_index is out of range")

    raw_completed = checkpoint.get("completed", [])
    if not isinstance(raw_completed, list):
        raise checkpoint_error("checkpoint.completed must be a list")

    normalized_completed: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_completed):
        if not isinstance(item, dict):
            raise checkpoint_error(f"checkpoint.completed[{idx}] must be an object")
        if idx >= len(normalized_plan):
            raise checkpoint_error(
                f"checkpoint.completed[{idx}] has no corresponding checkpoint.plan step"
            )
        completed_operation = item.get("operation")
        if completed_operation is None:
            raise checkpoint_error(f"checkpoint.completed[{idx}].operation must be present")
        if not isinstance(completed_operation, str):
            raise checkpoint_error(f"checkpoint.completed[{idx}].operation must be a string")
        expected_operation = str(normalized_plan[idx]["operation"])
        if completed_operation != expected_operation:
            raise checkpoint_error(
                f"checkpoint.completed[{idx}].operation must match checkpoint.plan[{idx}].operation"
            )
        normalized_completed.append(item)

    if len(normalized_completed) != raw_next_index:
        raise checkpoint_error("checkpoint.completed length must equal checkpoint.next_index")

    return raw_scope, normalized_plan, raw_next_index, normalized_completed


def build_new_checkpoint(
    *,
    scope: dict[str, Any] | None,
    ingest: dict[str, Any] | None,
    supersede: dict[str, Any] | None,
    archive: dict[str, Any] | None,
    maintain: dict[str, Any] | None,
    require_ingest: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Build a fresh ``(scope, plan, next_index, completed)`` state from new payloads."""
    plan = project_flow_plan(
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
            message=("MEM_PROJECT_ONBOARD_REQUIRES_INGEST: onboard must start with ingest"),
            retryable=False,
        )
    return scope or {}, plan, 0, []


def normalize_project_checkpoint(
    *,
    checkpoint: dict[str, Any] | None,
    scope: dict[str, Any] | None,
    ingest: dict[str, Any] | None,
    supersede: dict[str, Any] | None,
    archive: dict[str, Any] | None,
    maintain: dict[str, Any] | None,
    require_ingest: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Resolve a resume checkpoint or new-plan inputs into normalised flow state.

    Exactly one source must be provided: a ``checkpoint`` to resume, or fresh
    plan payloads to start. Supplying both raises ``MEM_CHECKPOINT_CONFLICT``.
    """
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
        return restore_checkpoint(checkpoint, require_ingest=require_ingest)

    return build_new_checkpoint(
        scope=scope,
        ingest=ingest,
        supersede=supersede,
        archive=archive,
        maintain=maintain,
        require_ingest=require_ingest,
    )


def step_sections(
    operation: str,
    payload: dict[str, Any],
) -> StepSections:
    """Route a step payload to the (query, record, maintenance) section it drives."""
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


def _compact_plan_step_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a compacted copy of a *completed* plan step payload.

    Only ``memories[].content`` is stripped — it is the dominant source of
    payload bloat and is no longer needed once the step has been executed.
    All other keys (format, metadata, ids, …) are preserved so the checkpoint
    remains human-readable and auditable.

    This function is intentionally conservative: it only modifies the
    ``memories`` list and leaves every other key intact.
    """
    if "memories" not in payload:
        return payload
    compacted = dict(payload)
    compacted["memories"] = [
        {k: v for k, v in mem.items() if k != "content"} if isinstance(mem, dict) else mem
        for mem in payload["memories"]
    ]
    return compacted


def compact_checkpoint(
    checkpoint: dict[str, Any],
    *,
    debug: bool,
) -> dict[str, Any]:
    """Return a copy of *checkpoint* shaped for the given verbosity level.

    Compact (debug=False):
    - ``completed[].result`` blobs are stripped (not needed for resume; only
      ``operation`` is required by :func:`restore_checkpoint`).
    - ``plan[i].payload.memories[].content`` is stripped for *completed* steps
      (index < next_index) — these steps will not be re-executed, so their
      heavy memory bodies are dead weight.  Pending steps (index >= next_index)
      keep their full payload so resume can pass it through unchanged.
    - ``scan_snapshot`` is kept fully intact because ``sync`` reads
      entries to compute delta between scans.  Stripping entries would break
      resume functionality for scan-based flows.
    - Everything else (plan, scope, next_index, scan config, version) is kept
      intact so the checkpoint remains fully usable for resume.

    Debug (debug=True):
    - Checkpoint is returned as-is with all blobs present.
    """
    if debug:
        return checkpoint

    result = dict(checkpoint)

    # Strip result blobs from completed items; keep operation for resume.
    if isinstance(result.get("completed"), list):
        result["completed"] = [
            {"operation": item["operation"]}
            if isinstance(item, dict) and "operation" in item
            else item
            for item in result["completed"]
        ]

    # Strip memories content from completed plan steps (index < next_index).
    # Pending steps (index >= next_index) keep their payload intact for resume.
    next_index = result.get("next_index", 0)
    if not isinstance(next_index, int):
        # Accept integral floats produced by JSON round-trips
        try:
            next_index = int(next_index)
        except (TypeError, ValueError):
            next_index = 0

    if isinstance(result.get("plan"), list) and next_index > 0:
        compacted_plan: list[dict[str, Any]] = []
        for i, step in enumerate(result["plan"]):
            if not isinstance(step, dict):
                compacted_plan.append(step)
                continue
            if i < next_index:
                # Completed step — strip heavy payload content
                raw_payload = step.get("payload")
                compacted_step = dict(step)
                if isinstance(raw_payload, dict):
                    compacted_step["payload"] = _compact_plan_step_payload(raw_payload)
                compacted_plan.append(compacted_step)
            else:
                # Pending step — keep as-is (needed for resume)
                compacted_plan.append(step)
        result["plan"] = compacted_plan

    return result


def build_project_checkpoint_payload(
    *,
    scope: dict[str, Any],
    plan: list[dict[str, Any]],
    next_index: int,
    completed: list[dict[str, Any]],
    scan: dict[str, Any] | None = None,
    scan_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical checkpoint dict carried between flow calls.

    ``scan`` / ``scan_snapshot`` are accepted as already-serialised dicts so the
    service never depends on the tool-layer scan models.
    """
    payload: dict[str, Any] = {
        "version": PROJECT_FLOW_VERSION,
        "scope": scope,
        # Include stable scope_key for deterministic context lookup.
        "scope_key": scope_key(scope),
        "plan": plan,
        "next_index": next_index,
        "completed": completed,
    }
    if scan is not None:
        payload["scan"] = scan
    if scan_snapshot is not None:
        payload["scan_snapshot"] = scan_snapshot
    return payload


def build_project_failed_checkpoint_payload(
    *,
    error: dict[str, Any],
    failed_operation: str,
    scope: dict[str, Any],
    plan: list[dict[str, Any]],
    next_index: int,
    completed: list[dict[str, Any]],
    completed_ops: list[str],
    result: dict[str, Any] | None,
    scan: dict[str, Any] | None = None,
    scan_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic checkpoint payload for failed project flow steps."""
    checkpoint_payload = build_project_checkpoint_payload(
        scope=scope,
        plan=plan,
        next_index=next_index,
        completed=completed,
        scan=scan,
        scan_snapshot=scan_snapshot,
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


# ---------------------------------------------------------------------------
# Flow state + service
# ---------------------------------------------------------------------------


class FlowState:
    """Resolved, in-flight state of one project flow.

    Built by the tool's routing/scan layer (which owns scope resolution and the
    filesystem scan) and handed to :meth:`ProjectFlowService.advance`. ``scan``
    and ``scan_snapshot`` are pre-serialised dicts persisted verbatim into the
    returned checkpoint; ``on_ingest`` lets the caller fold step results back
    into the snapshot (memory-id back-population) without the service knowing
    the snapshot model.
    """

    __slots__ = ("scope", "plan", "next_index", "completed", "scan", "scan_snapshot", "on_ingest")

    def __init__(
        self,
        *,
        scope: dict[str, Any],
        plan: list[dict[str, Any]],
        next_index: int,
        completed: list[dict[str, Any]],
        scan: dict[str, Any] | None = None,
        scan_snapshot: dict[str, Any] | None = None,
        on_ingest: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    ) -> None:
        self.scope = scope
        self.plan = plan
        self.next_index = next_index
        self.completed = completed
        self.scan = scan
        self.scan_snapshot = scan_snapshot
        # Called after a successful ``ingest`` step with the step result; returns
        # an updated scan_snapshot dict (or None to leave it unchanged).
        self.on_ingest = on_ingest


class AdvanceResult:
    """Outcome of one :meth:`ProjectFlowService.advance` call.

    Exactly one resolution holds:
    - ``response`` is the terminal payload to return (completed / checkpoint /
      failed-checkpoint); ``checkpoint`` carries the full uncompacted checkpoint
      so the caller can register session context on completion.
    - ``completed`` is True only when the whole plan ran without error.
    """

    __slots__ = ("response", "checkpoint", "completed")

    def __init__(
        self,
        *,
        response: dict[str, Any],
        checkpoint: dict[str, Any],
        completed: bool,
    ) -> None:
        self.response = response
        self.checkpoint = checkpoint
        self.completed = completed


class ProjectFlowService:
    """Stateless transition over the project-flow checkpoint state machine.

    Holds no per-call state — every invocation receives the full flow state and
    returns the next one, so a single instance is safe to share across all agent
    sessions of the central server.
    """

    async def advance(
        self,
        state: FlowState,
        *,
        executor: OperationExecutor,
        error_shaper: StepErrorShaper,
        flow_name: str,
        max_operations: int,
        debug: bool,
    ) -> AdvanceResult:
        """Execute up to ``max_operations`` plan steps, then return the next state.

        Steps run one at a time through ``executor``; the unit of atomicity is the
        single step (the executor owns each operation's transaction). The first
        failing step emits a failed checkpoint at its index and halts — resume
        re-runs from there.

        Args:
            state: Resolved flow state (scope, plan, cursor, scan snapshot).
            executor: Port running one memory operation.
            error_shaper: Port converting a raised step exception into an error
                envelope (memory error classification lives in the caller).
            flow_name: Flow label for error staging (e.g. ``"onboard"``).
            max_operations: Maximum steps to execute before pausing.
            debug: When True, keep full checkpoint internals and per-step results.
        """
        scope = state.scope
        plan = state.plan
        next_index = state.next_index
        completed = state.completed
        scan_snapshot = state.scan_snapshot

        completed_ops: list[str] = []
        last_result: dict[str, Any] | None = None

        for _ in range(max_operations):
            if next_index >= len(plan):
                break
            step = plan[next_index]
            operation_name = str(step["operation"])
            step_payload = step.get("payload")
            if not isinstance(step_payload, dict):
                raise checkpoint_error("checkpoint step payload must be an object")
            sections = step_sections(operation_name, step_payload)

            try:
                step_result = await executor(operation_name, scope, sections)
            except Exception as step_exc:
                step_error_payload = error_shaper(step_exc, operation_name)
                step_error = step_error_payload.get("error")
                if not isinstance(step_error, dict):
                    step_error = {
                        "code": "MEM_INTERNAL_ERROR",
                        "message": f"{flow_name} failed",
                        "retryable": False,
                        "stage": operation_name,
                        "actionable_fix": None,
                    }
                return self._failed(
                    error=step_error,
                    operation_name=operation_name,
                    scope=scope,
                    plan=plan,
                    next_index=next_index,
                    completed=completed,
                    completed_ops=completed_ops,
                    last_result=last_result,
                    state=state,
                    scan_snapshot=scan_snapshot,
                )

            step_error = _extract_error_envelope(step_result)
            if step_error is not None:
                return self._failed(
                    error=step_error,
                    operation_name=operation_name,
                    scope=scope,
                    plan=plan,
                    next_index=next_index,
                    completed=completed,
                    completed_ops=completed_ops,
                    last_result=step_result,
                    state=state,
                    scan_snapshot=scan_snapshot,
                )

            if operation_name == "ingest" and state.on_ingest is not None:
                updated = state.on_ingest(step_result)
                if updated is not None:
                    scan_snapshot = updated

            completed.append({"operation": operation_name, "result": step_result})
            completed_ops.append(operation_name)
            last_result = step_result
            next_index += 1

        checkpoint_payload = build_project_checkpoint_payload(
            scope=scope,
            plan=plan,
            next_index=next_index,
            completed=completed,
            scan=state.scan,
            scan_snapshot=scan_snapshot,
        )

        if next_index < len(plan):
            checkpoint_response: dict[str, Any] = {
                "status": "checkpoint",
                "remaining_operations": [str(s["operation"]) for s in plan[next_index:]],
                "completed_operations": completed_ops,
                "last_operation": completed_ops[-1] if completed_ops else None,
                "result": last_result,
                "checkpoint": compact_checkpoint(checkpoint_payload, debug=debug),
            }
            return AdvanceResult(
                response=checkpoint_response, checkpoint=checkpoint_payload, completed=False
            )

        response: dict[str, Any] = {
            "status": "completed",
            "completed_operations": [str(item.get("operation")) for item in completed],
            "checkpoint": compact_checkpoint(checkpoint_payload, debug=debug),
        }
        if debug:
            response["results"] = completed
        return AdvanceResult(response=response, checkpoint=checkpoint_payload, completed=True)

    def _failed(
        self,
        *,
        error: dict[str, Any],
        operation_name: str,
        scope: dict[str, Any],
        plan: list[dict[str, Any]],
        next_index: int,
        completed: list[dict[str, Any]],
        completed_ops: list[str],
        last_result: dict[str, Any] | None,
        state: FlowState,
        scan_snapshot: dict[str, Any] | None,
    ) -> AdvanceResult:
        response = build_project_failed_checkpoint_payload(
            error=error,
            failed_operation=operation_name,
            scope=scope,
            plan=plan,
            next_index=next_index,
            completed=completed,
            completed_ops=completed_ops,
            result=last_result,
            scan=state.scan,
            scan_snapshot=scan_snapshot,
        )
        return AdvanceResult(response=response, checkpoint=response["checkpoint"], completed=False)
