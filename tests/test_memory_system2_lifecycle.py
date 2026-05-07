"""Task 10: Semantic lifecycle transitions and archive gate.

Tests for `reconcile_semantic_lifecycle` state-machine transitions:
  active_evidenced -> degraded  (evidence weakens)
  degraded -> archived          (two successful absent-evidence System 1 cycles)

Negative cases:
  - failed verification cycles do not increment archival counters
  - partial-scope cycles do not count for claims in a different scope
  - scope-unrelated cycles do not count
  - direct active_evidenced -> archived is rejected (must transit through degraded first)

Scope identity:
  - Cycle applicability is keyed by normalized scope_key identity
    (exact match, not prefix/contains)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from workflows_mcp.engine.knowledge.schema import ensure_schema
from workflows_mcp.engine.sql.backend import ConnectionConfig, DatabaseEngine
from workflows_mcp.engine.sql.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio

PALACE = "palace_lifecycle_test"
WING = "default"
ROOM = "default"
COMPARTMENT = "lifecycle"


def _scope() -> dict[str, str]:
    return {"palace": PALACE, "wing": WING, "room": ROOM, "compartment": COMPARTMENT}


def _scope_key(
    palace: str = PALACE,
    wing: str = WING,
    room: str = ROOM,
    compartment: str = COMPARTMENT,
) -> str:
    from workflows_mcp.engine.memory_scope_resolver import scope_key as _sk
    return _sk({"palace": palace, "wing": wing, "room": room, "compartment": compartment})


def _make_config() -> ConnectionConfig:
    return ConnectionConfig(
        dialect=DatabaseEngine.POSTGRESQL,
        host=os.environ.get("MEMORY_DB_HOST", "localhost"),
        port=int(os.environ.get("MEMORY_DB_PORT", "5432")),
        database=os.environ.get("MEMORY_DB_NAME", "workflows"),
        username=os.environ.get("MEMORY_DB_USER", "workflows"),
        password=os.environ.get("MEMORY_DB_PASSWORD", "supersecret"),
    )


@pytest_asyncio.fixture
async def knowledge_backend() -> AsyncIterator[PostgresBackend]:
    backend = PostgresBackend()
    await backend.connect(_make_config())
    await ensure_schema(backend)
    try:
        yield backend
    finally:
        await backend.disconnect()


@pytest_asyncio.fixture
async def memory_service(knowledge_backend: PostgresBackend) -> Any:
    from unittest.mock import MagicMock

    from workflows_mcp.engine.executor_base import Execution
    from workflows_mcp.engine.memory_service import MemoryService

    context = MagicMock(spec=Execution)
    context.execution_context = MagicMock()
    context.execution_context.get = MagicMock(return_value=None)
    context.execution_context.user_id = None
    context.execution_context.user_string_id = None
    context.execution_context.auth_method = None
    return MemoryService(backend=knowledge_backend, context=context)


@pytest_asyncio.fixture
async def clean_palace(knowledge_backend: PostgresBackend) -> AsyncIterator[None]:
    async def _wipe() -> None:
        palace_pattern = f"{PALACE}%"
        await knowledge_backend.execute(
            "DELETE FROM knowledge_relations "
            "WHERE source_entity_id IN (SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_memories "
            "WHERE memory_id IN (SELECT id FROM knowledge_memories WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_embeddings "
            "WHERE entity_id IN (SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_memories WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entities WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_items WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_sources WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_structural_evidence WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_verification_cycles WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            """
            DELETE FROM knowledge_semantic_corridors
             WHERE from_claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace LIKE $1
             )
               OR to_claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace LIKE $1
             )
            """,
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_claim_evidence_links WHERE claim_id IN "
            "(SELECT id FROM knowledge_semantic_claims WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_semantic_claims WHERE palace LIKE $1", (palace_pattern,)
        )

    await _wipe()
    yield
    await _wipe()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_claim(
    backend: PostgresBackend,
    *,
    lifecycle_state: str = "active_evidenced",
    palace: str = PALACE,
    wing: str = WING,
    room: str = ROOM,
    compartment: str = COMPARTMENT,
) -> str:
    """Insert a semantic claim directly into the DB and return its UUID string."""
    from workflows_mcp.engine.memory_scope_resolver import scope_key as _sk
    scope_key = _sk({"palace": palace, "wing": wing, "room": room, "compartment": compartment})
    result = await backend.query(
        """
        INSERT INTO knowledge_semantic_claims
            (palace, wing, room, compartment, claim_type, lifecycle_state,
             claim_text, scope_key, created_at, updated_at)
        VALUES ($1, $2, $3, $4, 'room_intent', $5, 'test claim', $6, NOW(), NOW())
        RETURNING id::text
        """,
        (palace, wing, room, compartment, lifecycle_state, scope_key),
    )
    return str(result.rows[0]["id"])


async def _record_verification_cycle(
    memory_service: Any,
    *,
    success: bool = True,
    palace: str = PALACE,
    wing: str = WING,
    room: str = ROOM,
    compartment: str = COMPARTMENT,
) -> str:
    """Record a System 1 verification cycle and return its cycle_id."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    result = await memory_service.execute(MemoryRequest.model_validate({
        "operation": "record_system1_verification_cycle",
        "scope": {"palace": palace, "wing": wing, "room": room, "compartment": compartment},
        "record": {
            "format": "structured",
            "verification_cycle": {
                "success": success,
                "covered_scope": {
                    "palace": palace,
                    "wing": wing,
                    "room": room,
                    "compartment": compartment,
                },
            },
        },
    }))
    assert result.manage is not None and result.manage.cycle_id is not None, (
        f"Failed to record verification cycle: {result.manage}"
    )
    return result.manage.cycle_id


# ---------------------------------------------------------------------------
# State machine: active_evidenced -> degraded
# ---------------------------------------------------------------------------


async def test_reconcile_transitions_active_evidenced_to_degraded(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """reconcile_semantic_lifecycle must transition a claim from active_evidenced
    to degraded when the caller signals evidence has weakened.

    The claim must exist in the DB with lifecycle_state='degraded' after the call.
    reconciled_count must be >= 1 (the transitioned claim was counted).
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    claim_id = await _insert_claim(knowledge_backend, lifecycle_state="active_evidenced")

    result = await memory_service.execute(MemoryRequest.model_validate({
        "operation": "reconcile_semantic_lifecycle",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "lifecycle_reconciliation": {
                "scope_key": _scope_key(),
                "degrade_claim_ids": [claim_id],
            },
        },
    }))

    assert result.manage is not None
    assert result.manage.success, f"Expected success, got error: {result.manage.error!r}"
    assert result.manage.reconciled_count is not None
    assert result.manage.reconciled_count >= 1, (
        f"Expected reconciled_count >= 1, got {result.manage.reconciled_count}"
    )

    # Verify DB state.
    row = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert row.rows, f"Claim {claim_id!r} not found in DB after reconciliation"
    assert row.rows[0]["lifecycle_state"] == "degraded", (
        f"Expected 'degraded', got {row.rows[0]['lifecycle_state']!r}"
    )


# ---------------------------------------------------------------------------
# State machine: degraded -> archived (happy path with two successful cycles)
# ---------------------------------------------------------------------------


async def test_reconcile_archives_degraded_claim_after_two_successful_absent_cycles(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """A degraded claim must be archived when force_archive_claim_ids is supplied
    with two successful absent-evidence cycle IDs.

    After the call:
    - lifecycle_state in DB == 'archived'
    - reconciled_count >= 1
    - The claim row still exists (no deletion — archived in-place)
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    claim_id = await _insert_claim(knowledge_backend, lifecycle_state="degraded")

    # Record two successful absent-evidence cycles (success=True means absence confirmed).
    cycle_id_1 = await _record_verification_cycle(memory_service, success=True)
    cycle_id_2 = await _record_verification_cycle(memory_service, success=True)

    result = await memory_service.execute(MemoryRequest.model_validate({
        "operation": "reconcile_semantic_lifecycle",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "lifecycle_reconciliation": {
                "scope_key": _scope_key(),
                "force_archive_claim_ids": [claim_id],
                "absent_verification_cycle_ids": [cycle_id_1, cycle_id_2],
            },
        },
    }))

    assert result.manage is not None
    assert result.manage.success, f"Expected success, got error: {result.manage.error!r}"
    assert result.manage.reconciled_count is not None
    assert result.manage.reconciled_count >= 1

    row = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert row.rows, f"Claim {claim_id!r} must still exist in DB after archival (no deletion)"
    assert row.rows[0]["lifecycle_state"] == "archived", (
        f"Expected 'archived', got {row.rows[0]['lifecycle_state']!r}"
    )


# ---------------------------------------------------------------------------
# Invalid transition: direct active_evidenced -> archived must be rejected
# ---------------------------------------------------------------------------


async def test_direct_active_evidenced_to_archived_is_rejected(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """A claim in active_evidenced state must NOT be directly archived.
    The state machine requires active_evidenced -> degraded -> archived.

    Attempting to force-archive an active_evidenced claim (even with two valid
    absent cycles) must fail with MEM_ARCHIVE_GATE_NOT_MET or similar guard.
    """
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    claim_id = await _insert_claim(knowledge_backend, lifecycle_state="active_evidenced")

    cycle_id_1 = await _record_verification_cycle(memory_service, success=True)
    cycle_id_2 = await _record_verification_cycle(memory_service, success=True)

    try:
        result = await memory_service.execute(MemoryRequest.model_validate({
            "operation": "reconcile_semantic_lifecycle",
            "scope": _scope(),
            "record": {
                "format": "structured",
                "lifecycle_reconciliation": {
                    "scope_key": _scope_key(),
                    "force_archive_claim_ids": [claim_id],
                    "absent_verification_cycle_ids": [cycle_id_1, cycle_id_2],
                },
            },
        }))
        assert result.manage is not None
        assert not result.manage.success, (
            "Direct active_evidenced -> archived must not succeed; "
            "claim must transit through degraded first"
        )
        assert result.manage.error is not None and (
            "MEM_ARCHIVE_GATE_NOT_MET" in result.manage.error
            or "INVALID_TRANSITION" in result.manage.error
        ), f"Expected transition guard error, got: {result.manage.error!r}"
    except MemoryContractError as exc:
        assert exc.code in ("MEM_ARCHIVE_GATE_NOT_MET", "MEM_INVALID_TRANSITION"), (
            f"Expected transition guard, got: {exc.code!r}"
        )

    # Verify DB state unchanged.
    row = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert row.rows[0]["lifecycle_state"] == "active_evidenced", (
        "DB lifecycle_state must remain active_evidenced after rejected direct-archive"
    )


# ---------------------------------------------------------------------------
# Negative: failed verification cycles do NOT count toward archive gate
# ---------------------------------------------------------------------------


async def test_failed_cycles_do_not_satisfy_archive_gate(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Two failed verification cycles (success=False) must not satisfy the archive gate.

    Even when supplied as absent_verification_cycle_ids, the gate must reject them.
    """
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    claim_id = await _insert_claim(knowledge_backend, lifecycle_state="degraded")

    # Two failed cycles (evidence still present = absence NOT confirmed).
    cycle_id_1 = await _record_verification_cycle(memory_service, success=False)
    cycle_id_2 = await _record_verification_cycle(memory_service, success=False)

    try:
        result = await memory_service.execute(MemoryRequest.model_validate({
            "operation": "reconcile_semantic_lifecycle",
            "scope": _scope(),
            "record": {
                "format": "structured",
                "lifecycle_reconciliation": {
                    "scope_key": _scope_key(),
                    "force_archive_claim_ids": [claim_id],
                    "absent_verification_cycle_ids": [cycle_id_1, cycle_id_2],
                },
            },
        }))
        assert result.manage is not None
        assert not result.manage.success, (
            "Failed cycles must not satisfy archive gate"
        )
        assert "MEM_ARCHIVE_GATE_NOT_MET" in (result.manage.error or ""), (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {result.manage.error!r}"
        )
    except MemoryContractError as exc:
        assert exc.code == "MEM_ARCHIVE_GATE_NOT_MET", (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {exc.code!r}"
        )

    # State must remain degraded.
    row = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert row.rows[0]["lifecycle_state"] == "degraded"


# ---------------------------------------------------------------------------
# Negative: scope-unrelated cycles do NOT count (exact scope_key identity)
# ---------------------------------------------------------------------------


async def test_scope_unrelated_cycles_do_not_count_for_archive_gate(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Cycles recorded under a different scope must not count toward archival of
    claims in the target scope.

    Scope identity is keyed by normalized scope_key (exact match). A cycle for
    palace/wing/room/OTHER_COMPARTMENT must not satisfy the gate for claims under
    palace/wing/room/COMPARTMENT.
    """
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    claim_id = await _insert_claim(knowledge_backend, lifecycle_state="degraded")

    # Cycles recorded for a different compartment — scope_key will differ.
    other_compartment = "other_compartment"
    cycle_id_1 = await _record_verification_cycle(
        memory_service, success=True, compartment=other_compartment
    )
    cycle_id_2 = await _record_verification_cycle(
        memory_service, success=True, compartment=other_compartment
    )

    try:
        result = await memory_service.execute(MemoryRequest.model_validate({
            "operation": "reconcile_semantic_lifecycle",
            "scope": _scope(),
            "record": {
                "format": "structured",
                "lifecycle_reconciliation": {
                    "scope_key": _scope_key(),
                    "force_archive_claim_ids": [claim_id],
                    "absent_verification_cycle_ids": [cycle_id_1, cycle_id_2],
                },
            },
        }))
        assert result.manage is not None
        assert not result.manage.success, (
            "Scope-unrelated cycles must not satisfy archive gate"
        )
        assert "MEM_ARCHIVE_GATE_NOT_MET" in (result.manage.error or ""), (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {result.manage.error!r}"
        )
    except MemoryContractError as exc:
        assert exc.code == "MEM_ARCHIVE_GATE_NOT_MET", (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {exc.code!r}"
        )

    # State must remain degraded.
    row = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert row.rows[0]["lifecycle_state"] == "degraded"


# ---------------------------------------------------------------------------
# Negative: one cycle (not two) is insufficient for archive gate
# ---------------------------------------------------------------------------


async def test_single_successful_absent_cycle_insufficient_for_archive(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Archive gate requires at least two successful absent cycles. One is not enough."""
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    claim_id = await _insert_claim(knowledge_backend, lifecycle_state="degraded")
    cycle_id = await _record_verification_cycle(memory_service, success=True)

    try:
        result = await memory_service.execute(MemoryRequest.model_validate({
            "operation": "reconcile_semantic_lifecycle",
            "scope": _scope(),
            "record": {
                "format": "structured",
                "lifecycle_reconciliation": {
                    "scope_key": _scope_key(),
                    "force_archive_claim_ids": [claim_id],
                    "absent_verification_cycle_ids": [cycle_id],
                },
            },
        }))
        assert result.manage is not None
        assert not result.manage.success
        assert "MEM_ARCHIVE_GATE_NOT_MET" in (result.manage.error or "")
    except MemoryContractError as exc:
        assert exc.code == "MEM_ARCHIVE_GATE_NOT_MET"

    row = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert row.rows[0]["lifecycle_state"] == "degraded"


# ---------------------------------------------------------------------------
# Archived claims remain in DB (no deletion)
# ---------------------------------------------------------------------------


async def test_archived_claims_remain_in_knowledge_semantic_claims(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Archived claims must stay in knowledge_semantic_claims with
    lifecycle_state='archived'. The archival operation must NOT delete the row.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    claim_id = await _insert_claim(knowledge_backend, lifecycle_state="degraded")
    cycle_id_1 = await _record_verification_cycle(memory_service, success=True)
    cycle_id_2 = await _record_verification_cycle(memory_service, success=True)

    result = await memory_service.execute(MemoryRequest.model_validate({
        "operation": "reconcile_semantic_lifecycle",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "lifecycle_reconciliation": {
                "scope_key": _scope_key(),
                "force_archive_claim_ids": [claim_id],
                "absent_verification_cycle_ids": [cycle_id_1, cycle_id_2],
            },
        },
    }))

    assert result.manage is not None and result.manage.success

    count = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert count.rows[0]["n"] == 1, (
        "Archived claim must remain in knowledge_semantic_claims (not deleted)"
    )

    row = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert row.rows[0]["lifecycle_state"] == "archived"


# ---------------------------------------------------------------------------
# reconciled_count reflects actual transitions made
# ---------------------------------------------------------------------------


async def test_reconciled_count_reflects_degraded_claims(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """When two claims are degraded in one call, reconciled_count must be 2."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    claim_id_1 = await _insert_claim(knowledge_backend, lifecycle_state="active_evidenced")
    claim_id_2 = await _insert_claim(knowledge_backend, lifecycle_state="active_evidenced")

    result = await memory_service.execute(MemoryRequest.model_validate({
        "operation": "reconcile_semantic_lifecycle",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "lifecycle_reconciliation": {
                "scope_key": _scope_key(),
                "degrade_claim_ids": [claim_id_1, claim_id_2],
            },
        },
    }))

    assert result.manage is not None
    assert result.manage.success
    assert result.manage.reconciled_count == 2, (
        f"Expected reconciled_count=2, got {result.manage.reconciled_count}"
    )


# ---------------------------------------------------------------------------
# No-op call (no claim IDs supplied) returns success with reconciled_count=0
# ---------------------------------------------------------------------------


async def test_reconcile_no_op_returns_zero_count(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """reconcile_semantic_lifecycle with no claim IDs supplied must succeed
    and return reconciled_count=0 (nothing to do).
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    result = await memory_service.execute(MemoryRequest.model_validate({
        "operation": "reconcile_semantic_lifecycle",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "lifecycle_reconciliation": {
                "scope_key": _scope_key(),
            },
        },
    }))

    assert result.manage is not None
    assert result.manage.success
    assert result.manage.reconciled_count == 0


# ---------------------------------------------------------------------------
# Archival metadata: archived_at IS NOT NULL and absent_cycle_count == 2
# ---------------------------------------------------------------------------


async def test_archive_sets_archived_at_and_absent_cycle_count(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """After a successful archival the DB row must have:
    - archived_at IS NOT NULL (required by ck_ksc_archive_requires_two_cycles constraint)
    - absent_cycle_count == 2 (the number of proven absent-evidence cycles supplied)
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    claim_id = await _insert_claim(knowledge_backend, lifecycle_state="degraded")
    cycle_id_1 = await _record_verification_cycle(memory_service, success=True)
    cycle_id_2 = await _record_verification_cycle(memory_service, success=True)

    result = await memory_service.execute(MemoryRequest.model_validate({
        "operation": "reconcile_semantic_lifecycle",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "lifecycle_reconciliation": {
                "scope_key": _scope_key(),
                "force_archive_claim_ids": [claim_id],
                "absent_verification_cycle_ids": [cycle_id_1, cycle_id_2],
            },
        },
    }))

    assert result.manage is not None and result.manage.success

    row = await knowledge_backend.query(
        "SELECT lifecycle_state, archived_at, absent_cycle_count"
        " FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert row.rows, f"Claim {claim_id!r} not found after archival"
    claim = row.rows[0]
    assert claim["lifecycle_state"] == "archived"
    assert claim["archived_at"] is not None, (
        "archived_at must be set on archival (required by DB constraint)"
    )
    assert claim["absent_cycle_count"] == 2, (
        f"Expected absent_cycle_count=2, got {claim['absent_cycle_count']}"
    )


# ---------------------------------------------------------------------------
# Combined degrade + force_archive in one request (transactional)
# ---------------------------------------------------------------------------


async def test_combined_degrade_and_archive_in_one_request(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """A single reconcile_semantic_lifecycle request may supply both
    degrade_claim_ids and force_archive_claim_ids.

    Contract:
    - Phase 1: degrade_claim_ids transitions active_evidenced -> degraded.
    - Phase 2: force_archive_claim_ids transitions degraded -> archived (gate enforced).
    - Both phases succeed atomically; reconciled_count reflects total transitions.
    - The degrade and archive sets must be disjoint (each claim goes through one
      transition per call).
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Claim A: starts active_evidenced — will be degraded in this call.
    claim_a = await _insert_claim(knowledge_backend, lifecycle_state="active_evidenced")
    # Claim B: already degraded — will be archived in this call.
    claim_b = await _insert_claim(knowledge_backend, lifecycle_state="degraded")

    cycle_id_1 = await _record_verification_cycle(memory_service, success=True)
    cycle_id_2 = await _record_verification_cycle(memory_service, success=True)

    result = await memory_service.execute(MemoryRequest.model_validate({
        "operation": "reconcile_semantic_lifecycle",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "lifecycle_reconciliation": {
                "scope_key": _scope_key(),
                "degrade_claim_ids": [claim_a],
                "force_archive_claim_ids": [claim_b],
                "absent_verification_cycle_ids": [cycle_id_1, cycle_id_2],
            },
        },
    }))

    assert result.manage is not None
    assert result.manage.success, f"Expected success, got: {result.manage.error!r}"
    assert result.manage.reconciled_count == 2, (
        f"Expected reconciled_count=2 (1 degraded + 1 archived), "
        f"got {result.manage.reconciled_count}"
    )

    row_a = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_a,),
    )
    assert row_a.rows[0]["lifecycle_state"] == "degraded", (
        f"Claim A must be degraded, got {row_a.rows[0]['lifecycle_state']!r}"
    )

    row_b = await knowledge_backend.query(
        "SELECT lifecycle_state, archived_at FROM knowledge_semantic_claims"
        " WHERE id = $1::uuid",
        (claim_b,),
    )
    assert row_b.rows[0]["lifecycle_state"] == "archived", (
        f"Claim B must be archived, got {row_b.rows[0]['lifecycle_state']!r}"
    )
    assert row_b.rows[0]["archived_at"] is not None


async def test_combined_request_rolls_back_degrade_when_archive_gate_fails(
    memory_service: Any, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """When a combined request (degrade + force_archive) fails the archive gate,
    the Phase 1 degradation must be rolled back atomically.

    After rejection, claim_a must remain active_evidenced (not degraded), proving
    the transaction was rolled back and no partial state mutation occurred.
    """
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    claim_a = await _insert_claim(knowledge_backend, lifecycle_state="active_evidenced")
    claim_b = await _insert_claim(knowledge_backend, lifecycle_state="degraded")

    # Only one successful cycle — gate requires two.
    cycle_id = await _record_verification_cycle(memory_service, success=True)

    try:
        result = await memory_service.execute(MemoryRequest.model_validate({
            "operation": "reconcile_semantic_lifecycle",
            "scope": _scope(),
            "record": {
                "format": "structured",
                "lifecycle_reconciliation": {
                    "scope_key": _scope_key(),
                    "degrade_claim_ids": [claim_a],
                    "force_archive_claim_ids": [claim_b],
                    "absent_verification_cycle_ids": [cycle_id],
                },
            },
        }))
        assert result.manage is not None
        assert not result.manage.success, (
            "Combined request with insufficient cycles must be rejected"
        )
        assert "MEM_ARCHIVE_GATE_NOT_MET" in (result.manage.error or "")
    except MemoryContractError as exc:
        assert exc.code == "MEM_ARCHIVE_GATE_NOT_MET"

    # Critical: claim_a must still be active_evidenced — Phase 1 rolled back.
    row_a = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_a,),
    )
    assert row_a.rows[0]["lifecycle_state"] == "active_evidenced", (
        "Phase 1 degradation must be rolled back when archive gate fails; "
        f"got {row_a.rows[0]['lifecycle_state']!r}"
    )

    # claim_b must also remain degraded — not archived.
    row_b = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_b,),
    )
    assert row_b.rows[0]["lifecycle_state"] == "degraded"
