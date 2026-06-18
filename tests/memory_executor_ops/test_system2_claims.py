"""ADR-013 System 1 / System 2 operation tests: semantic claim derivation.

Covers MemoryRequest acceptance of the System 1 / System 2 operations, the
derive_system2_semantic_claims behaviors (room intent, reasoning unit, semantic
corridors, evidence resolution, atomic rollback) and override accountability
wired into the verification-cycle dispatch path.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from _ops_helpers import COMPARTMENT, PALACE, ROOM, WING, _scope, _scope_key

pytestmark = pytest.mark.asyncio


def test_system1_store_structural_evidence_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'store_system1_structural_evidence' without raising."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_evidence",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "store_system1_structural_evidence"


def test_system1_record_verification_cycle_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'record_system1_verification_cycle' without raising."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "record_system1_verification_cycle",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "record_system1_verification_cycle"


def test_system2_derive_semantic_claims_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'derive_system2_semantic_claims' without raising."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "derive_system2_semantic_claims",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "derive_system2_semantic_claims"


def test_system2_apply_semantic_override_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'apply_semantic_override' without raising."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "apply_semantic_override",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "apply_semantic_override"


def test_system2_reconcile_semantic_lifecycle_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'reconcile_semantic_lifecycle' without raising."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "reconcile_semantic_lifecycle",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "reconcile_semantic_lifecycle"


async def test_system1_store_structural_evidence_persists_evidence_rows(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """store_system1_structural_evidence must write structural evidence rows keyed by
    scope/entity and return success with stored evidence IDs."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/mod.py::MyClass",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/mod.py", "line": 1},
                        },
                        {
                            "entity_stable_id": "src/mod.py::MyClass.method",
                            "entity_type": "function",
                            "evidence_category": "structural_function",
                            "evidence_data": {"file": "src/mod.py", "line": 10},
                        },
                    ],
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.stored_count >= 2
    assert result.manage.stored_evidence_ids is not None
    assert len(result.manage.stored_evidence_ids) == 2, (
        "Expected exactly 2 stored evidence IDs, one per submitted item"
    )


@pytest.mark.asyncio
async def test_system1_store_structural_evidence_row_exists_in_db(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """store_system1_structural_evidence must write a real row to the DB.
    Querying knowledge_structural_evidence after the call must return exactly
    one row matching the submitted scope/entity/category key.
    """
    from workflows_mcp.engine.memory_schema import MemoryRequest

    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/db_test.py::DbCheck",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/db_test.py", "line": 5},
                        },
                    ],
                },
            }
        )
    )

    rows = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_structural_evidence
         WHERE palace = $1
           AND wing = $2
           AND room = $3
           AND compartment = $4
           AND entity_stable_id = $5
           AND evidence_category = $6
        """,
        (PALACE, WING, ROOM, COMPARTMENT, "src/db_test.py::DbCheck", "structural_class"),
    )
    assert len(rows.rows) == 1, (
        f"Expected exactly 1 DB row in knowledge_structural_evidence,"
        f" got {len(rows.rows)}: {rows.rows!r}"
    )


async def test_system1_record_verification_cycle_persists_cycle_metadata(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """record_system1_verification_cycle must persist a verification cycle row
    with scope_key identity and success/failure status."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "record_system1_verification_cycle",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "verification_cycle": {
                        "success": True,
                        "covered_scope": {
                            "palace": PALACE,
                            "wing": WING,
                            "room": ROOM,
                            "compartment": COMPARTMENT,
                        },
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.cycle_id is not None


async def test_system2_derive_semantic_claims_returns_claim_ids(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must derive claims backed by System 1 evidence
    and return claim IDs in the response."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    # Pre-store structural evidence so the derivation can resolve stable IDs.
    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/mod.py::MyClass",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/mod.py", "line": 1},
                        },
                    ],
                },
            }
        )
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "data_access_layer",
                        "evidence_entity_stable_ids": ["src/mod.py::MyClass"],
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.claim_ids is not None
    assert len(result.manage.claim_ids) >= 1


async def test_system2_apply_semantic_override_activates_immediately_with_provenance(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """apply_semantic_override must activate the override immediately and persist
    provenance fields (override_reason, overridden_by, activated_at)."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    # First, store structural evidence so a real claim can be derived.
    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/override_test.py::OverrideClass",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/override_test.py", "line": 1},
                        },
                    ],
                },
            }
        )
    )

    # Derive a claim to obtain a real claim_id (FK constraint on overrides table).
    derive_result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "data_access_layer",
                        "evidence_entity_stable_ids": ["src/override_test.py::OverrideClass"],
                    },
                },
            }
        )
    )
    assert derive_result.manage is not None
    assert derive_result.manage.claim_ids and len(derive_result.manage.claim_ids) >= 1
    real_claim_id = derive_result.manage.claim_ids[0]

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "apply_semantic_override",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "override": {
                        "claim_id": real_claim_id,
                        "override_reason": "manual correction by architect",
                        "overridden_by": "alice",
                        "new_lifecycle_state": "active_evidenced",
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.override_id is not None


async def test_system2_reconcile_semantic_lifecycle_returns_transition_summary(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """reconcile_semantic_lifecycle must evaluate lifecycle transitions and return
    a summary of affected claims and their new states."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "reconcile_semantic_lifecycle",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "lifecycle_reconciliation": {
                        "scope_key": _scope_key(),
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.reconciled_count is not None


@pytest.mark.asyncio
async def test_system1_failed_verification_cycle_does_not_count_for_archive_gate(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """ADR-013: only *successful* absent-evidence verification cycles count toward
    the archive eligibility gate. Recording two cycles with success=False and then
    attempting a force-archive must still be rejected with MEM_ARCHIVE_GATE_NOT_MET.

    This is a runtime service invariant — failed cycles must not satisfy the gate
    even when two cycle IDs are technically present.
    """
    from workflows_mcp.engine.memory_schema import MemoryRequest
    from workflows_mcp.engine.memory_service import MemoryContractError

    scope_key = _scope_key()

    # Record two verification cycles, both with success=False (i.e., evidence was
    # found — the absence condition was NOT met).
    cycle_ids: list[str] = []
    for _ in range(2):
        cycle_result = await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "record_system1_verification_cycle",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "verification_cycle": {
                            "success": False,  # Evidence still present — absence NOT confirmed.
                            "covered_scope": {
                                "palace": PALACE,
                                "wing": WING,
                                "room": ROOM,
                                "compartment": COMPARTMENT,
                            },
                        },
                    },
                }
            )
        )
        # The operation itself succeeds (cycle is recorded), but the cycle is
        # marked as failed (absence not confirmed). Collect the cycle IDs.
        assert cycle_result.manage is not None
        assert cycle_result.manage.cycle_id is not None
        cycle_ids.append(cycle_result.manage.cycle_id)

    # Now attempt to archive using those two failed-cycle IDs as proof.
    # The service must reject this with MEM_ARCHIVE_GATE_NOT_MET because the
    # cycles did not confirm absence of evidence.
    try:
        result = await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "reconcile_semantic_lifecycle",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "lifecycle_reconciliation": {
                            "scope_key": scope_key,
                            "force_archive_claim_ids": ["claim-from-test"],
                            "absent_verification_cycle_ids": cycle_ids,
                        },
                    },
                }
            )
        )
        assert result.manage is not None, (
            "Archive using failed verification cycles must not succeed silently; "
            "expected MEM_ARCHIVE_GATE_NOT_MET"
        )
        assert not result.manage.success, (
            "Archive gate must reject force-archive backed only by failed verification cycles"
        )
        assert "MEM_ARCHIVE_GATE_NOT_MET" in (result.manage.error or ""), (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {result.manage.error!r}"
        )
    except MemoryContractError as exc:
        assert exc.code == "MEM_ARCHIVE_GATE_NOT_MET", (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {exc.code!r}"
        )


@pytest.mark.asyncio
async def test_system1_store_structural_evidence_is_idempotent_for_same_scope_entity_category(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """ADR-013 Task 4: storing the same structural evidence (same scope +
    entity_stable_id + evidence_category) twice must be idempotent.

    Semantics locked in by this test:
    - stored_count: number of submitted items processed (insert or upsert).
      It is NOT a count of newly inserted rows.  Callers use it to confirm
      every submitted item reached the DB, whether new or refreshed.
    - stored_evidence_ids: the UUIDs of the persisted rows, stable across
      calls for the same key — the second call returns the same row UUID
      as the first, proving ON CONFLICT DO UPDATE hit the existing row.
    - DB row count: exactly 1 row must exist in knowledge_structural_evidence
      for the conflict key after two store calls, proving no duplicate was
      created.
    """
    from workflows_mcp.engine.memory_schema import MemoryRequest

    evidence_payload = {
        "operation": "store_system1_structural_evidence",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "structural_evidence": [
                {
                    "entity_stable_id": "src/idempotent.py::MyClass",
                    "entity_type": "class",
                    "evidence_category": "structural_class",
                    "evidence_data": {"file": "src/idempotent.py", "line": 1},
                },
            ],
        },
    }

    first = await memory_service.execute(MemoryRequest.model_validate(evidence_payload))
    assert first.manage is not None
    assert first.manage.success
    # stored_count == number of submitted items (1), not number of new rows.
    assert first.manage.stored_count == 1
    first_ids = first.manage.stored_evidence_ids
    assert first_ids is not None and len(first_ids) == 1

    second = await memory_service.execute(MemoryRequest.model_validate(evidence_payload))
    assert second.manage is not None
    assert second.manage.success
    # Second call: still 1 item submitted, still reports stored_count == 1.
    assert second.manage.stored_count == 1
    second_ids = second.manage.stored_evidence_ids
    assert second_ids is not None and len(second_ids) == 1

    # Row UUID must be identical — ON CONFLICT returned the existing row id.
    assert first_ids == second_ids, (
        "Idempotency violation: re-storing the same structural evidence"
        f" returned different IDs: first={first_ids!r}, second={second_ids!r}"
    )

    # Direct DB assertion: exactly 1 row for the conflict key, no duplicate.
    rows = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_structural_evidence
         WHERE palace = $1
           AND wing = $2
           AND room = $3
           AND compartment = $4
           AND entity_stable_id = $5
           AND evidence_category = $6
        """,
        (
            PALACE,
            WING,
            ROOM,
            COMPARTMENT,
            "src/idempotent.py::MyClass",
            "structural_class",
        ),
    )
    assert len(rows.rows) == 1, (
        f"Expected exactly 1 DB row after two idempotent store calls,"
        f" got {len(rows.rows)}: {rows.rows!r}"
    )
    # The DB row UUID must match the returned evidence ID.
    assert str(rows.rows[0]["id"]) == first_ids[0], (
        "DB row UUID does not match the ID returned by store operation: "
        f"db={rows.rows[0]['id']!r}, returned={first_ids[0]!r}"
    )


@pytest.mark.asyncio
async def test_archive_gate_rejects_unregistered_fabricated_cycle_ids(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """ADR-013 archive gate must be fail-closed: cycle IDs that were never registered
    via record_system1_verification_cycle must NOT count as successful absent-evidence
    cycles.  Passing two fabricated/unregistered IDs must be rejected with
    MEM_ARCHIVE_GATE_NOT_MET — unknown IDs are not countable.
    """
    from workflows_mcp.engine.memory_schema import MemoryRequest
    from workflows_mcp.engine.memory_service import MemoryContractError

    scope_key = _scope_key()
    # Fabricated UUIDs — never registered with the service (not in DB).
    fake_cycle_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    try:
        result = await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "reconcile_semantic_lifecycle",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "lifecycle_reconciliation": {
                            "scope_key": scope_key,
                            "force_archive_claim_ids": ["claim-fabricated"],
                            "absent_verification_cycle_ids": fake_cycle_ids,
                        },
                    },
                }
            )
        )
        assert result.manage is not None, (
            "Archive with fabricated cycle IDs must not succeed silently; "
            "expected MEM_ARCHIVE_GATE_NOT_MET"
        )
        assert not result.manage.success, (
            "Archive gate must reject force-archive backed by unregistered cycle IDs"
        )
        assert "MEM_ARCHIVE_GATE_NOT_MET" in (result.manage.error or ""), (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {result.manage.error!r}"
        )
    except MemoryContractError as exc:
        assert exc.code == "MEM_ARCHIVE_GATE_NOT_MET", (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {exc.code!r}"
        )


# ---------------------------------------------------------------------------
# Task 7: System 2 claim derivation — room intent, reasoning unit, corridors
# ---------------------------------------------------------------------------


async def _store_evidence(memory_service: Any, stable_id: str) -> None:
    """Helper: store one structural evidence row in the test palace/scope."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": stable_id,
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/t7.py", "line": 1},
                        },
                    ],
                },
            }
        )
    )
    assert result.manage is not None and result.manage.success, (
        f"Pre-condition failed: could not store evidence for {stable_id!r}"
    )


@pytest.mark.asyncio
async def test_system2_room_intent_label_persists_claim_and_evidence_link(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must persist a room_intent claim backed by
    at least one evidence link.  The returned claim ID must map to a real DB row
    in knowledge_semantic_claims with claim_type='room_intent' and
    lifecycle_state='active_evidenced'.  A matching knowledge_claim_evidence_links
    row must also exist."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    stable_id = "src/t7_room.py::RoomClass"
    await _store_evidence(memory_service, stable_id)

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "data_access_layer",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success, f"Expected success, got error: {result.manage.error!r}"
    assert result.manage.claim_ids is not None
    assert len(result.manage.claim_ids) >= 1
    claim_id = result.manage.claim_ids[0]

    # Must be a real UUID (not a stub hash).
    try:
        uuid.UUID(claim_id)
    except ValueError:
        pytest.fail(f"claim_id is not a UUID: {claim_id!r}")

    # DB row must exist with correct type and lifecycle.
    claim_row = await knowledge_backend.query(
        "SELECT claim_type, lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert len(claim_row.rows) == 1, f"No DB row for claim_id={claim_id!r}"
    assert claim_row.rows[0]["claim_type"] == "room_intent"
    assert claim_row.rows[0]["lifecycle_state"] == "active_evidenced"

    # Evidence link must exist.
    link_row = await knowledge_backend.query(
        "SELECT count(*) AS n FROM knowledge_claim_evidence_links WHERE claim_id = $1::uuid",
        (claim_id,),
    )
    assert int(link_row.rows[0]["n"]) >= 1, (
        f"Expected at least 1 evidence link for claim {claim_id!r}, got 0"
    )


@pytest.mark.asyncio
async def test_system2_compartment_reasoning_unit_persists_claim_and_evidence_link(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must persist a compartment_reasoning_unit claim
    backed by at least one evidence link when compartment_reasoning_unit is supplied."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    stable_id = "src/t7_comp.py::CompClass"
    await _store_evidence(memory_service, stable_id)

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "compartment_reasoning_unit": "query_builder",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success, f"Expected success, got error: {result.manage.error!r}"
    assert result.manage.claim_ids is not None
    assert len(result.manage.claim_ids) >= 1
    claim_id = result.manage.claim_ids[0]

    uuid.UUID(claim_id)  # must be a real UUID

    claim_row = await knowledge_backend.query(
        "SELECT claim_type, lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert len(claim_row.rows) == 1, f"No DB row for claim_id={claim_id!r}"
    assert claim_row.rows[0]["claim_type"] == "compartment_reasoning_unit"
    assert claim_row.rows[0]["lifecycle_state"] == "active_evidenced"

    link_row = await knowledge_backend.query(
        "SELECT count(*) AS n FROM knowledge_claim_evidence_links WHERE claim_id = $1::uuid",
        (claim_id,),
    )
    assert int(link_row.rows[0]["n"]) >= 1, (
        f"Expected at least 1 evidence link for claim {claim_id!r}, got 0"
    )


@pytest.mark.asyncio
async def test_system2_semantic_corridor_persists_claim_edge_and_canonical_type(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims with corridor fields must:
    1. Persist a semantic_corridor claim row in knowledge_semantic_claims.
    2. Persist a directed edge row in knowledge_semantic_corridors with
       canonicalized corridor_type and original raw type in corridor_type_raw.
    3. Return the corridor claim ID in claim_ids."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    stable_id = "src/t7_corr.py::CorrClass"
    await _store_evidence(memory_service, stable_id)

    # Derive two endpoint claims first.
    r_from = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "from_room",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert r_from.manage is not None and r_from.manage.success
    from_claim_id = r_from.manage.claim_ids[0]

    r_to = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "to_room",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert r_to.manage is not None and r_to.manage.success
    to_claim_id = r_to.manage.claim_ids[0]

    # Derive the corridor claim.
    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "corridor_from_claim_id": from_claim_id,
                        "corridor_to_claim_id": to_claim_id,
                        "corridor_type": "calls-into",  # raw: should canonicalize to CALLS_INTO
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success, f"Expected success, got error: {result.manage.error!r}"
    assert result.manage.claim_ids is not None
    assert len(result.manage.claim_ids) >= 1
    corridor_claim_id = result.manage.claim_ids[0]
    uuid.UUID(corridor_claim_id)

    # Claim row must be semantic_corridor.
    claim_row = await knowledge_backend.query(
        "SELECT claim_type, lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (corridor_claim_id,),
    )
    assert len(claim_row.rows) == 1, f"No claim row for {corridor_claim_id!r}"
    assert claim_row.rows[0]["claim_type"] == "semantic_corridor"
    assert claim_row.rows[0]["lifecycle_state"] == "active_evidenced"

    # Edge row must exist with canonical type and raw type preserved.
    edge_row = await knowledge_backend.query(
        """
        SELECT from_claim_id, to_claim_id, corridor_type, corridor_type_raw
          FROM knowledge_semantic_corridors
         WHERE claim_id = $1::uuid
        """,
        (corridor_claim_id,),
    )
    assert len(edge_row.rows) == 1, f"Expected 1 edge row for {corridor_claim_id!r}"
    row = edge_row.rows[0]
    assert str(row["from_claim_id"]) == from_claim_id
    assert str(row["to_claim_id"]) == to_claim_id
    assert row["corridor_type"] == "CALLS_INTO", (
        f"Expected canonical type 'CALLS_INTO', got {row['corridor_type']!r}"
    )
    assert row["corridor_type_raw"] == "calls-into", (
        f"Expected raw type 'calls-into', got {row['corridor_type_raw']!r}"
    )


@pytest.mark.asyncio
async def test_system2_partial_corridor_fields_rejected(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must reject a request that supplies only
    some corridor fields (only from_claim_id, no to_claim_id/corridor_type).
    The model validator must raise ValidationError with a message that names
    all three required fields."""
    from pydantic import ValidationError

    from workflows_mcp.engine.memory_schema import MemoryRequest

    stable_id = "src/t7_partial.py::PartialClass"
    await _store_evidence(memory_service, stable_id)

    with pytest.raises(ValidationError) as exc_info:
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "corridor_from_claim_id": str(uuid.uuid4()),
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    error_text = str(exc_info.value)
    assert "corridor_from_claim_id" in error_text, (
        f"Expected 'corridor_from_claim_id' in error; got: {error_text!r}"
    )
    assert "corridor_to_claim_id" in error_text, (
        f"Expected 'corridor_to_claim_id' in error; got: {error_text!r}"
    )
    assert "corridor_type" in error_text, f"Expected 'corridor_type' in error; got: {error_text!r}"


@pytest.mark.asyncio
async def test_system2_corridor_self_loop_rejected(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must reject corridor edges where
    from_claim_id == to_claim_id.  The model validator must raise ValidationError
    with a message that mentions self-loop."""
    from pydantic import ValidationError

    from workflows_mcp.engine.memory_schema import MemoryRequest

    stable_id = "src/t7_loop.py::LoopClass"
    await _store_evidence(memory_service, stable_id)

    same_id = str(uuid.uuid4())
    with pytest.raises(ValidationError) as exc_info:
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "corridor_from_claim_id": same_id,
                        "corridor_to_claim_id": same_id,
                        "corridor_type": "depends_on",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    error_text = str(exc_info.value)
    assert "self-loop" in error_text.lower(), (
        f"Expected 'self-loop' in error message; got: {error_text!r}"
    )


@pytest.mark.asyncio
async def test_system2_derive_without_resolvable_evidence_fails(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must fail when no matching structural evidence
    rows exist in the scope — claims must not be born without resolvable evidence."""
    from workflows_mcp.engine.memory_schema import MemoryRequest
    from workflows_mcp.engine.memory_service import MemoryContractError

    # Do NOT store evidence — stable ID will not resolve.
    try:
        result = await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "derive_system2_semantic_claims",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "derivation": {
                            "room_intent_label": "orphan_room",
                            "evidence_entity_stable_ids": ["nonexistent::StableId"],
                        },
                    },
                }
            )
        )
        assert result.manage is not None
        assert not result.manage.success, "Derivation with unresolvable evidence must not succeed"
        assert result.manage.error is not None
    except MemoryContractError:
        pass  # contract error is also acceptable


# ---------------------------------------------------------------------------
# Task 7 atomicity: duplicate directed edge rolls back — no orphaned claims
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_corridor_edge_rolls_back_and_leaves_no_orphaned_claim(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """Atomicity invariant: when the semantic_corridor edge insert fails due to a
    duplicate directed-edge unique constraint (from_claim_id, to_claim_id,
    corridor_type), the entire transaction must roll back.

    After the failed second attempt:
    - Exactly one claim row exists (the first successful one).
    - Exactly one corridor edge row exists.
    - No orphaned claim row (half-inserted state) was left in
      knowledge_semantic_claims from the second call.
    """
    from workflows_mcp.engine.memory_schema import MemoryRequest

    stable_id = "src/t7_atomicity.py::AtomicityClass"
    await _store_evidence(memory_service, stable_id)

    # Derive two endpoint claims to act as corridor endpoints.
    r_from = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "atomicity_from",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert r_from.manage is not None and r_from.manage.success
    from_claim_id = r_from.manage.claim_ids[0]

    r_to = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "atomicity_to",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert r_to.manage is not None and r_to.manage.success
    to_claim_id = r_to.manage.claim_ids[0]

    corridor_payload = {
        "operation": "derive_system2_semantic_claims",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "derivation": {
                "corridor_from_claim_id": from_claim_id,
                "corridor_to_claim_id": to_claim_id,
                "corridor_type": "calls-into",
                "evidence_entity_stable_ids": [stable_id],
            },
        },
    }

    # First derivation: must succeed.
    first = await memory_service.execute(MemoryRequest.model_validate(corridor_payload))
    first_error = first.manage.error if first.manage else "no manage result"
    assert first.manage is not None and first.manage.success, (
        f"First corridor derivation must succeed; error: {first_error!r}"
    )
    first_claim_id = first.manage.claim_ids[0]

    # Count claims and edges before second attempt.
    claims_before = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_semantic_claims WHERE palace = $1",
        (PALACE,),
    )
    count_before = claims_before.rows[0]["n"]

    # Second derivation: same directed+typed edge — unique constraint must fire,
    # transaction must roll back, operation must return failure.
    second = await memory_service.execute(MemoryRequest.model_validate(corridor_payload))
    assert second.manage is not None, "Second derivation must return a manage result"
    assert not second.manage.success, (
        "Duplicate corridor edge must not succeed; expected rollback and failure"
    )
    assert second.manage.error is not None
    assert "MEM_DB_ERROR" in second.manage.error, (
        f"Expected MEM_DB_ERROR in error; got: {second.manage.error!r}"
    )

    # Claim count must be unchanged — no orphaned row from the rolled-back attempt.
    claims_after = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_semantic_claims WHERE palace = $1",
        (PALACE,),
    )
    count_after = claims_after.rows[0]["n"]
    assert count_after == count_before, (
        f"Orphaned claim detected: claim count changed from {count_before} to {count_after}"
        f" after a rolled-back corridor insert"
    )

    # Exactly one corridor edge must exist for this directed+typed pair.
    edge_count = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
          FROM knowledge_semantic_corridors
         WHERE from_claim_id = $1::uuid
           AND to_claim_id   = $2::uuid
           AND corridor_type = 'CALLS_INTO'
        """,
        (from_claim_id, to_claim_id),
    )
    assert edge_count.rows[0]["n"] == 1, (
        f"Expected exactly 1 corridor edge, got {edge_count.rows[0]['n']}"
    )

    # The first claim must still be intact (not rolled back by the second attempt).
    first_claim_row = await knowledge_backend.query(
        "SELECT id FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (first_claim_id,),
    )
    assert len(first_claim_row.rows) == 1, (
        f"First corridor claim {first_claim_id!r} must survive the second attempt"
    )


# ---------------------------------------------------------------------------
# Override accountability wired into verification-cycle dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_cycle_updates_override_accountability(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """record_system1_verification_cycle must automatically update override
    accountability_status for active/pending overrides in the covered scope,
    based on live evidence links — not on the cycle.success boolean.

    Acceptance criteria (evidence-based, ADR-013 v1 deterministic semantics):
    - Before any cycle: override accountability_status == 'pending'.
    - After a successful cycle (success=True) when live evidence links exist in the
      covered scope: the override transitions to 'supported'.
    - After a successful cycle (success=True) when no live evidence links exist:
      the override transitions to 'unsupported'.
    - After a failed cycle (success=False): accountability_status does NOT change.
    - Trajectory via cycle-operation path: pending → supported → unsupported → supported.
    """
    from workflows_mcp.engine.memory_schema import MemoryRequest

    # Step 1: Store structural evidence so we can derive a real claim.
    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/accountability_test.py::AccountabilityClass",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/accountability_test.py", "line": 1},
                        },
                    ],
                },
            }
        )
    )

    # Step 2: Derive a semantic claim with the evidence entity — this creates
    # knowledge_claim_evidence_links rows linking claim_id → evidence_id.
    derive_result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "accountability_test_layer",
                        "evidence_entity_stable_ids": [
                            "src/accountability_test.py::AccountabilityClass"
                        ],
                    },
                },
            }
        )
    )
    assert derive_result.manage is not None and derive_result.manage.success
    assert derive_result.manage.claim_ids
    real_claim_id = derive_result.manage.claim_ids[0]

    # Step 3: Apply an override — starts as 'pending'.
    override_result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "apply_semantic_override",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "override": {
                        "claim_id": real_claim_id,
                        "override_reason": "architect manual correction for accountability test",
                        "overridden_by": "test-architect",
                        "new_lifecycle_state": "active_evidenced",
                    },
                },
            }
        )
    )
    assert override_result.manage is not None and override_result.manage.success
    override_id = override_result.manage.override_id
    assert override_id is not None

    # Verify initial accountability_status is 'pending'.
    before_row = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert before_row.rows, "Override row must exist in DB"
    assert before_row.rows[0]["accountability_status"] == "pending", (
        f"Expected 'pending' before cycle; got {before_row.rows[0]['accountability_status']!r}"
    )

    def _covered_cycle(success: bool) -> dict:  # type: ignore[type-arg]
        return {
            "operation": "record_system1_verification_cycle",
            "scope": _scope(),
            "record": {
                "format": "structured",
                "verification_cycle": {
                    "success": success,
                    "covered_scope": {
                        "palace": PALACE,
                        "wing": WING,
                        "room": ROOM,
                        "compartment": COMPARTMENT,
                    },
                },
            },
        }

    # Step 4: Successful cycle with live evidence links in scope → 'supported'.
    # Evidence links were created in Step 2 (derive_system2_semantic_claims inserts them).
    cycle_result = await memory_service.execute(
        MemoryRequest.model_validate(_covered_cycle(success=True))
    )
    assert cycle_result.manage is not None and cycle_result.manage.success

    after_row = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert after_row.rows[0]["accountability_status"] == "supported", (
        "After success=True cycle with live evidence links, override must be 'supported'; "
        f"got {after_row.rows[0]['accountability_status']!r}"
    )

    # Step 5: Remove all evidence links for the claim to simulate absence of evidence.
    await knowledge_backend.execute(
        "DELETE FROM knowledge_claim_evidence_links WHERE claim_id = $1::uuid",
        (real_claim_id,),
    )

    # Successful cycle with NO live evidence links in scope → 'unsupported'.
    cycle_result2 = await memory_service.execute(
        MemoryRequest.model_validate(_covered_cycle(success=True))
    )
    assert cycle_result2.manage is not None and cycle_result2.manage.success

    after_row2 = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert after_row2.rows[0]["accountability_status"] == "unsupported", (
        "After success=True cycle with NO evidence links, override must be 'unsupported'; "
        f"got {after_row2.rows[0]['accountability_status']!r}"
    )

    # Step 6: Failed cycle (success=False) — accountability must NOT change (stays 'unsupported').
    cycle_result3 = await memory_service.execute(
        MemoryRequest.model_validate(_covered_cycle(success=False))
    )
    assert cycle_result3.manage is not None and cycle_result3.manage.success

    after_row3 = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert after_row3.rows[0]["accountability_status"] == "unsupported", (
        "After success=False (failed) cycle, accountability_status must NOT change; "
        f"got {after_row3.rows[0]['accountability_status']!r}"
    )

    # Step 7: Re-add evidence link and run another successful cycle → back to 'supported'.
    # Re-query the evidence_id we stored in Step 1.
    evidence_row = await knowledge_backend.query(
        """
        SELECT id, evidence_category
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = $2
        """,
        (PALACE, "src/accountability_test.py::AccountabilityClass"),
    )
    assert evidence_row.rows, "Structural evidence row must still exist"
    evidence_id = evidence_row.rows[0]["id"]
    evidence_category = evidence_row.rows[0]["evidence_category"]

    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_claim_evidence_links
            (claim_id, evidence_id, evidence_category, linked_at)
        VALUES ($1::uuid, $2::uuid, $3, NOW())
        ON CONFLICT DO NOTHING
        """,
        (real_claim_id, evidence_id, evidence_category),
    )

    # Final successful cycle with re-linked evidence → 'supported' again (trajectory complete).
    cycle_result4 = await memory_service.execute(
        MemoryRequest.model_validate(_covered_cycle(success=True))
    )
    assert cycle_result4.manage is not None and cycle_result4.manage.success

    after_row4 = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert after_row4.rows[0]["accountability_status"] == "supported", (
        "After success=True cycle with re-added evidence links, override must be 'supported'; "
        f"got {after_row4.rows[0]['accountability_status']!r}"
    )
