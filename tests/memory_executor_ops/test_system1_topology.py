"""ADR-013 derive_system1_topology tests.

Covers the System 1 topology derivation paths: fail-closed behavior, explicit
override, the structural heuristic, the proof-bundle new-wing gate, atomic
persistence, and inline candidate passthrough.
"""

from __future__ import annotations

from typing import Any

import pytest
from _ops_helpers import PALACE

from workflows_mcp.engine.sql.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# ADR-013 Task 1: derive_system1_topology fail-closed behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_fails_closed_when_evidence_rows_are_indeterminate(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Evidence IDs that reference no persisted rows → MEM_INSUFFICIENT_EVIDENCE.

    ADR-013 constraint: no fallback to resolved_scope.*, no literal 'default'.
    When the provided evidence_ids do not match any persisted structural evidence
    rows in this palace, the heuristic has nothing to work from and must fail closed.
    This is the authoritative indeterminate case: zero evidence rows loaded.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest
    from workflows_mcp.memory.memory_service import MemoryContractError

    # Use a fabricated UUID that does not exist in knowledge_structural_evidence.
    nonexistent_evidence_id = "00000000-dead-beef-0000-000000000099"

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [nonexistent_evidence_id],
            },
        }
    )

    with pytest.raises(MemoryContractError) as exc:
        await memory_service.execute(derive_req)  # type: ignore[union-attr]

    error_code = exc.value.code
    assert "MEM_INSUFFICIENT_EVIDENCE" in error_code, (
        f"Expected MEM_INSUFFICIENT_EVIDENCE (or stricter descendant), got: {error_code!r}. "
        "Topology derivation must fail closed when evidence is insufficient — "
        "no fallback to resolved_scope.* or literals like 'default'."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_does_not_fallback_to_default_literal(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """derive_system1_topology must never produce 'default' as a wing/topology output.

    Even if evidence rows exist, the result must not silently assign a default
    topology value. The operation must either produce a proven topology or fail closed.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest
    from workflows_mcp.memory.memory_service import MemoryContractError

    # Attempt derivation with a fabricated evidence_id (not actually stored).
    # The operation should fail closed, not silently return 'default'.
    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": ["00000000-dead-beef-0000-000000000001"],
            },
        }
    )

    with pytest.raises(MemoryContractError) as exc:
        await memory_service.execute(derive_req)  # type: ignore[union-attr]

    error_code = exc.value.code
    assert "MEM_INSUFFICIENT_EVIDENCE" in error_code, (
        f"Expected fail-closed MEM_INSUFFICIENT_EVIDENCE, got: {error_code!r}. "
        "Operation must never silently produce 'default' topology — no hidden fallback permitted."
    )
    # Extra guard: confirm no 'default' leaks into the error message either
    assert "default" not in exc.value.message.lower(), (
        "Error message must not contain 'default' — that would suggest a fallback was attempted."
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 2b: derive_system1_topology explicit override path
# ---------------------------------------------------------------------------


async def _store_structural_evidence_and_get_id(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    *,
    entity_stable_id: str,
    wing: str = "owl-wing",
    room: str = "owl-room",
    compartment: str = "owl-compartment",
) -> str:
    """Helper: store one structural evidence row and return its UUID string."""
    from workflows_mcp.memory.memory_schema import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_evidence",
            "scope": {
                "palace": PALACE,
                "wing": wing,
                "room": room,
                "compartment": compartment,
            },
            "record": {
                "format": "structured",
                "structural_evidence": [
                    {
                        "entity_stable_id": entity_stable_id,
                        "entity_type": "module",
                        "evidence_category": "structural_module",
                        "evidence_data": {"path": f"src/{entity_stable_id}.py"},
                    }
                ],
            },
        }
    )
    result = await memory_service.execute(req)  # type: ignore[union-attr]
    assert result.manage is not None and result.manage.success

    row = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_structural_evidence
         WHERE palace = $1 AND entity_stable_id = $2
        """,
        (PALACE, entity_stable_id),
    )
    assert row.rows, f"Evidence row for {entity_stable_id!r} must exist after store"
    return str(row.rows[0]["id"])


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_success(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Explicit complete override succeeds, persists all rows, returns typed result fields.

    ADR-013 Task 2b: when topology_override is fully populated and evidence IDs
    reference persisted rows, the operation must:
    - Write/upsert a semantic claim row.
    - Write a semantic override accountability row (override_id).
    - Append a knowledge_topology_provenance row.
    - Append knowledge_topology_provenance_evidence rows for each evidence ID.
    - Return derivation_source='explicit_override', derived_wing/room/compartment,
      provenance_id, claim_id.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    evidence_id = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/owl.py::OwlClass"
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                "topology_override": {
                    "wing": "owl-wing",
                    "room": "owl-room",
                    "compartment": "owl-compartment",
                    "override_reason": "Authoritative placement confirmed by architect review.",
                    "applied_by": "architect-agent-v1",
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]

    assert result.manage is not None, "manage section must be present"
    manage = result.manage
    assert manage.success, f"Operation must succeed; error={manage.error!r}"

    # Typed result fields
    assert manage.derived_wing == "owl-wing", (
        f"derived_wing must equal override wing; got {manage.derived_wing!r}"
    )
    assert manage.derived_room == "owl-room", (
        f"derived_room must equal override room; got {manage.derived_room!r}"
    )
    assert manage.derived_compartment == "owl-compartment", (
        f"derived_compartment must equal override compartment; got {manage.derived_compartment!r}"
    )
    assert manage.derivation_source == "explicit_override", (
        f"derivation_source must be 'explicit_override'; got {manage.derivation_source!r}"
    )
    assert manage.provenance_id is not None, "provenance_id must be set to a persisted row ID"
    assert manage.claim_id is not None, "claim_id must reference persisted semantic claim"

    # Durable DB inspectability: provenance row must exist
    prov_row = await knowledge_backend.query(
        """
        SELECT id, derivation_source, wing, room, compartment, override_reason, applied_by
          FROM knowledge_topology_provenance
         WHERE id = $1::uuid
        """,
        (manage.provenance_id,),
    )
    assert prov_row.rows, "knowledge_topology_provenance row must be persisted"
    prov = prov_row.rows[0]
    assert prov["derivation_source"] == "explicit_override"
    assert prov["wing"] == "owl-wing"
    assert prov["room"] == "owl-room"
    assert prov["compartment"] == "owl-compartment"
    assert prov["override_reason"] == "Authoritative placement confirmed by architect review."
    assert prov["applied_by"] == "architect-agent-v1"

    # Evidence link row must exist
    link_row = await knowledge_backend.query(
        """
        SELECT provenance_id, evidence_id
          FROM knowledge_topology_provenance_evidence
         WHERE provenance_id = $1::uuid AND evidence_id = $2::uuid
        """,
        (manage.provenance_id, evidence_id),
    )
    assert link_row.rows, "knowledge_topology_provenance_evidence link row must be persisted"

    # Semantic claim must exist
    claim_row = await knowledge_backend.query(
        """
        SELECT id, claim_type, palace, wing, room, compartment
          FROM knowledge_semantic_claims
         WHERE id = $1::uuid
        """,
        (manage.claim_id,),
    )
    assert claim_row.rows, "knowledge_semantic_claims row must exist"
    claim = claim_row.rows[0]
    assert claim["palace"] == PALACE
    assert claim["wing"] == "owl-wing"

    # Semantic override accountability row must exist (links override to claim)
    override_row = await knowledge_backend.query(
        """
        SELECT id, claim_id, override_reason, applied_by
          FROM knowledge_semantic_overrides
         WHERE claim_id = $1::uuid
        """,
        (manage.claim_id,),
    )
    assert override_row.rows, "knowledge_semantic_overrides accountability row must exist"
    ov = override_row.rows[0]
    assert ov["override_reason"] == "Authoritative placement confirmed by architect review."
    assert ov["applied_by"] == "architect-agent-v1"


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_provenance_is_append_only(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Repeated accepted overrides each append a new provenance row (not update).

    ADR-013 append-only requirement: every accepted override call must produce
    a distinct knowledge_topology_provenance row, preserving full history.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    evidence_id = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/append.py::AppendClass"
    )

    def _derive_req(reason: str) -> dict:  # type: ignore[type-arg]
        return {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                "topology_override": {
                    "wing": "owl-wing",
                    "room": "owl-room",
                    "compartment": "owl-compartment",
                    "override_reason": reason,
                    "applied_by": "architect-agent-v1",
                },
            },
        }

    result1 = await memory_service.execute(  # type: ignore[union-attr]
        MemoryRequest.model_validate(_derive_req("First override"))
    )
    result2 = await memory_service.execute(  # type: ignore[union-attr]
        MemoryRequest.model_validate(_derive_req("Second override"))
    )

    assert result1.manage is not None and result1.manage.success
    assert result2.manage is not None and result2.manage.success

    prov_id_1 = result1.manage.provenance_id
    prov_id_2 = result2.manage.provenance_id
    assert prov_id_1 != prov_id_2, (
        "Each accepted override must produce a distinct provenance_id (append-only)."
    )

    # Both rows must be present in DB
    count_row = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance
         WHERE palace = $1 AND derivation_source = 'explicit_override'
        """,
        (PALACE,),
    )
    count = count_row.rows[0]["cnt"]
    assert count >= 2, f"Expected at least 2 provenance rows for append-only; found {count}"

    # Idempotent claim: repeated overrides for the same scope must NOT create
    # duplicate knowledge_semantic_claims rows — exactly 1 claim row expected.
    # Re-query using scope_key derived from the placement
    from workflows_mcp.memory.memory_service import _scope_key_fn  # type: ignore[import]

    scope_key = _scope_key_fn(
        {"palace": PALACE, "wing": "owl-wing", "room": "owl-room", "compartment": "owl-compartment"}
    )
    claim_count_row2 = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_claims
         WHERE palace = $1
           AND scope_key = $2
           AND claim_type = 'room_intent'
        """,
        (PALACE, scope_key),
    )
    claim_count = claim_count_row2.rows[0]["cnt"]
    assert claim_count == 1, (
        f"Repeated overrides for same scope must produce exactly 1 claim row "
        f"(idempotent upsert); found {claim_count}"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_multiple_evidence_ids(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """All provided evidence IDs are linked in knowledge_topology_provenance_evidence.

    When multiple evidence IDs are supplied, each must have a corresponding
    link row — no partial linking permitted.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    ev1 = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/multi1.py::ClassOne"
    )
    ev2 = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/multi2.py::ClassTwo"
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev1, ev2],
                "topology_override": {
                    "wing": "owl-wing",
                    "room": "owl-room",
                    "compartment": "owl-compartment",
                    "override_reason": "Multi-evidence authoritative placement.",
                    "applied_by": "architect-agent-v1",
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    assert result.manage is not None and result.manage.success

    prov_id = result.manage.provenance_id
    link_rows = await knowledge_backend.query(
        """
        SELECT evidence_id::text
          FROM knowledge_topology_provenance_evidence
         WHERE provenance_id = $1::uuid
        ORDER BY evidence_id
        """,
        (prov_id,),
    )
    linked_ids = {r["evidence_id"] for r in link_rows.rows}
    assert ev1 in linked_ids, f"Evidence ID {ev1!r} not linked in provenance_evidence"
    assert ev2 in linked_ids, f"Evidence ID {ev2!r} not linked in provenance_evidence"


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_nonexistent_evidence_id_fails(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Nonexistent evidence ID causes fail-closed error and full rollback.

    ADR-013 Task 2b: evidence IDs must reference persisted structural evidence rows.
    A nonexistent ID must produce an error and no orphaned rows (atomic rollback).
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest
    from workflows_mcp.memory.memory_service import MemoryContractError

    fake_id = "00000000-dead-beef-0000-000000000099"

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [fake_id],
                "topology_override": {
                    "wing": "owl-wing",
                    "room": "owl-room",
                    "compartment": "owl-compartment",
                    "override_reason": "Override with fake evidence.",
                    "applied_by": "bad-agent",
                },
            },
        }
    )

    with pytest.raises(MemoryContractError) as exc:
        await memory_service.execute(derive_req)  # type: ignore[union-attr]

    assert "MEM_" in exc.value.code, (
        f"Expected a MEM_ error code for nonexistent evidence ID; got {exc.value.code!r}"
    )

    # Rollback: all four write tables must have zero committed rows for this attempt.
    prov_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance
         WHERE palace = $1 AND applied_by = 'bad-agent'
        """,
        (PALACE,),
    )
    assert prov_count.rows[0]["cnt"] == 0, (
        "No knowledge_topology_provenance row must be committed when evidence ID is nonexistent."
    )

    claim_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_claims
         WHERE palace = $1
        """,
        (PALACE,),
    )
    assert claim_count.rows[0]["cnt"] == 0, (
        "No knowledge_semantic_claims row must be committed when evidence ID is nonexistent."
    )

    override_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_overrides so
          JOIN knowledge_semantic_claims sc ON sc.id = so.claim_id
         WHERE sc.palace = $1
        """,
        (PALACE,),
    )
    assert override_count.rows[0]["cnt"] == 0, (
        "No knowledge_semantic_overrides row must be committed when evidence ID is nonexistent."
    )

    prov_ev_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance_evidence kpe
          JOIN knowledge_topology_provenance ktp ON ktp.id = kpe.provenance_id
         WHERE ktp.palace = $1 AND ktp.applied_by = 'bad-agent'
        """,
        (PALACE,),
    )
    assert prov_ev_count.rows[0]["cnt"] == 0, (
        "No knowledge_topology_provenance_evidence row must be committed "
        "when evidence ID is nonexistent."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_without_override_succeeds_via_structural_heuristic(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """No topology_override supplied → structural heuristic derives topology (Task 3).

    Task 3 implements the structural derivation path. Without topology_override,
    the operation must now succeed when sufficient structural evidence is present,
    returning derivation_source='system1_derived'.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    evidence_id = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/no_override.py::NoOverrideClass"
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                # No topology_override — structural heuristic path
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, (
        f"Non-override path must succeed via structural heuristic (Task 3); "
        f"got error={manage.error!r}"
    )
    assert manage.derivation_source == "system1_derived", (
        f"derivation_source must be 'system1_derived'; got {manage.derivation_source!r}"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 3: derive_system1_topology structural heuristic (system1_derived)
# ---------------------------------------------------------------------------


async def _store_structural_evidence_typed(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    *,
    entity_stable_id: str,
    entity_type: str,
    evidence_category: str,
    wing: str,
    room: str,
    compartment: str,
) -> str:
    """Helper: store a structural evidence row with explicit type/category and return its UUID."""
    from workflows_mcp.memory.memory_schema import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_evidence",
            "scope": {
                "palace": PALACE,
                "wing": wing,
                "room": room,
                "compartment": compartment,
            },
            "record": {
                "format": "structured",
                "structural_evidence": [
                    {
                        "entity_stable_id": entity_stable_id,
                        "entity_type": entity_type,
                        "evidence_category": evidence_category,
                        "evidence_data": {"path": f"src/{entity_stable_id}"},
                    }
                ],
            },
        }
    )
    result = await memory_service.execute(req)  # type: ignore[union-attr]
    assert result.manage is not None and result.manage.success

    row = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_structural_evidence
         WHERE palace = $1 AND entity_stable_id = $2
        """,
        (PALACE, entity_stable_id),
    )
    assert row.rows, f"Evidence row for {entity_stable_id!r} must exist after store"
    return str(row.rows[0]["id"])


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_success_returns_system1_derived(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Sufficient structural evidence → successful derivation with system1_derived source.

    ADR-013 Task 3: when evidence rows carry deterministic structural signals that allow
    wing/room/compartment to be resolved, the operation must:
    - Return success=True.
    - Return derivation_source='system1_derived'.
    - Return derivation_algorithm_version='system1.v1'.
    - Return non-empty derived_wing, derived_room, derived_compartment.
    - Return non-None provenance_id and claim_id.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    # Store a class entity (highest anchor priority) in a known wing/room/compartment.
    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_success/core.py::CoreEngine",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-wing",
        room="t3-room",
        compartment="t3-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_id],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, f"Expected success=True; got error={manage.error!r}"

    assert manage.derivation_source == "system1_derived", (
        f"derivation_source must be 'system1_derived'; got {manage.derivation_source!r}"
    )
    assert manage.derived_wing, f"derived_wing must be non-empty; got {manage.derived_wing!r}"
    assert manage.derived_room, f"derived_room must be non-empty; got {manage.derived_room!r}"
    assert manage.derived_compartment, (
        f"derived_compartment must be non-empty; got {manage.derived_compartment!r}"
    )
    assert manage.provenance_id is not None, "provenance_id must be returned"
    assert manage.claim_id is not None, "claim_id must be returned"


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_algorithm_version_is_system1_v1(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Derived topology provenance row carries derivation_algorithm_version='system1.v1'.

    ADR-013 Task 3: algorithm version must be persisted for inspectability and future
    upgrade tracking.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_version/mod.py::VersionMod",
        entity_type="module",
        evidence_category="structural_module",
        wing="t3-ver-wing",
        room="t3-ver-room",
        compartment="t3-ver-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_id],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success

    prov_row = await knowledge_backend.query(
        """
        SELECT derivation_algorithm_version, derivation_source
          FROM knowledge_topology_provenance
         WHERE id = $1::uuid
        """,
        (manage.provenance_id,),
    )
    assert prov_row.rows, "knowledge_topology_provenance row must be persisted"
    prov = prov_row.rows[0]
    assert prov["derivation_algorithm_version"] == "system1.v1", (
        f"algorithm version must be 'system1.v1'; got {prov['derivation_algorithm_version']!r}"
    )
    assert prov["derivation_source"] == "system1_derived", (
        f"derivation_source must be 'system1_derived'; got {prov['derivation_source']!r}"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_provenance_is_append_only(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Two identical structural derivation calls produce two distinct provenance rows.

    ADR-013 Task 3: provenance is history, not mutable state. Each accepted derivation
    must produce a fresh knowledge_topology_provenance row.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_append/util.py::UtilClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-append-wing",
        room="t3-append-room",
        compartment="t3-append-comp",
    )

    async def _derive() -> str:
        req = MemoryRequest.model_validate(
            {
                "operation": "derive_system1_topology",
                "derivation": {
                    "palace": PALACE,
                    "evidence_ids": [ev_id],
                },
            }
        )
        res = await memory_service.execute(req)  # type: ignore[union-attr]
        assert res.manage is not None and res.manage.success
        return str(res.manage.provenance_id)

    prov_id_1 = await _derive()
    prov_id_2 = await _derive()

    assert prov_id_1 != prov_id_2, (
        "Each structural derivation must produce a distinct provenance_id (append-only); "
        f"got same id={prov_id_1!r} for both calls"
    )

    count_row = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance
         WHERE palace = $1 AND derivation_source = 'system1_derived'
           AND id IN ($2::uuid, $3::uuid)
        """,
        (PALACE, prov_id_1, prov_id_2),
    )
    assert count_row.rows[0]["cnt"] == 2, (
        "Both provenance rows must be persisted independently (append-only)"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_evidence_links_persisted(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Structural derivation links evidence IDs in knowledge_topology_provenance_evidence.

    ADR-013 Task 3: evidence IDs used for structural derivation must be linked in the
    provenance evidence table for accountability.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_evlink/svc.py::SvcClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-evlink-wing",
        room="t3-evlink-room",
        compartment="t3-evlink-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_id],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success

    link_row = await knowledge_backend.query(
        """
        SELECT provenance_id, evidence_id
          FROM knowledge_topology_provenance_evidence
         WHERE provenance_id = $1::uuid AND evidence_id = $2::uuid
        """,
        (manage.provenance_id, ev_id),
    )
    assert link_row.rows, (
        "knowledge_topology_provenance_evidence row must link the evidence ID used in derivation"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_class_anchor_priority(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Class entity is preferred as anchor over function when both are present.

    ADR-013 Task 3 spec: anchor selection priority is class/module first, then
    function/doc unit. The derived topology must reflect the class entity's placement
    rather than the function entity's placement.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    # Store a class entity in one wing/room/compartment.
    cls_ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_anchor/engine.py::Engine",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-anchor-wing",
        room="t3-anchor-room",
        compartment="t3-anchor-class-comp",
    )
    # Store a function entity in a different compartment.
    fn_ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_anchor/utils.py::helper_fn",
        entity_type="function",
        evidence_category="structural_function",
        wing="t3-anchor-wing",
        room="t3-anchor-room",
        compartment="t3-anchor-fn-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [cls_ev_id, fn_ev_id],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success is True

    # Anchor is the class entity → compartment must reflect class entity identity.
    assert manage.derived_wing == "t3-anchor-wing", (
        f"derived_wing must be 't3-anchor-wing'; got {manage.derived_wing!r}"
    )
    assert manage.derived_room == "t3-anchor-room", (
        f"derived_room must be 't3-anchor-room'; got {manage.derived_room!r}"
    )
    # Compartment is derived from the class anchor's compartment column.
    assert manage.derived_compartment == "t3-anchor-class-comp", (
        f"derived_compartment should reflect class anchor placement; "
        f"got {manage.derived_compartment!r}"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_no_semantic_labels_influence(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Structural derivation result must not depend on semantic intent labels.

    ADR-013 Task 3 spec: 'No semantic intent labels in System 1.' Two derivation calls
    with identical structural evidence but different derivation contexts (one with a
    semantic-sounding entity_stable_id prefix, one without) must produce the same
    wing/room/compartment topology from structural signals only.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    # Evidence rows in identical structural positions but with names that could
    # be mistaken for semantic categories.  The heuristic must use only structural
    # columns (wing/room/compartment, entity_type), not parse entity_stable_id
    # for intent/semantic signals.
    ev_structural = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_semantic/plain.py::PlainClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-semantic-wing",
        room="t3-semantic-room",
        compartment="t3-semantic-comp",
    )
    ev_intent_named = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_semantic/intent_domain_model.py::IntentDomainClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-semantic-wing",
        room="t3-semantic-room",
        compartment="t3-semantic-comp",
    )

    async def _derive(ev_id: str) -> tuple[str, str, str]:
        req = MemoryRequest.model_validate(
            {
                "operation": "derive_system1_topology",
                "derivation": {
                    "palace": PALACE,
                    "evidence_ids": [ev_id],
                },
            }
        )
        res = await memory_service.execute(req)  # type: ignore[union-attr]
        assert res.manage is not None and res.manage.success
        m = res.manage
        return (m.derived_wing or "", m.derived_room or "", m.derived_compartment or "")

    topo_plain = await _derive(ev_structural)
    topo_intent = await _derive(ev_intent_named)

    assert topo_plain == topo_intent, (
        "Structural derivation must produce identical topology for evidence rows in the same "
        f"structural position regardless of entity name semantics; "
        f"plain={topo_plain!r} intent={topo_intent!r}"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 3 (fix): modal wing/room tie-break is lexicographic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_modal_wing_room_tie_break_is_lexicographic(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Equal-count wing and room values must resolve via lexicographic tie-break.

    This test constructs two structural evidence rows where:
    - wing "z-wing" and wing "a-wing" each appear exactly once (equal count).
    - room "z-room" and room "a-room" each appear exactly once (equal count).

    The rows are inserted in order that would cause Counter.most_common(1) to
    non-deterministically (or insertion-order-biased) select the non-lexicographic
    winner.  The correct implementation must always select "a-room" and "a-wing".

    To force the non-lexicographic value to appear first in insertion/DB order,
    the "z-*" row is stored first so that naive Counter iteration or DB row order
    would return it before "a-*".
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    # Store "z-wing"/"z-room" with a stable_id that sorts BEFORE "a-wing" entity's stable_id.
    # The DB query returns rows ORDER BY entity_stable_id ASC, so "aaa_..." sorts first.
    # This means "z-wing" is encountered first by Counter — naive most_common(1) in CPython
    # will return "z-wing" (first-seen wins ties), causing the non-lexicographic value to win.
    ev_z = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_tiebreak/aaa_first.py::FirstEntityZWing",
        entity_type="class",
        evidence_category="structural_class",
        wing="z-wing",
        room="z-room",
        compartment="tiebreak-comp",
    )
    # Store "a-wing"/"a-room" with a stable_id that sorts AFTER — it is seen second by Counter.
    ev_a = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_tiebreak/zzz_second.py::SecondEntityAWing",
        entity_type="class",
        evidence_category="structural_class",
        wing="a-wing",
        room="a-room",
        compartment="tiebreak-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_z, ev_a],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, f"Expected success=True; got error={manage.error!r}"

    # Lexicographically first wins the tie: "a-wing" < "z-wing", "a-room" < "z-room".
    assert manage.derived_wing == "a-wing", (
        f"Equal-count wing tie must select lexicographically first value 'a-wing'; "
        f"got {manage.derived_wing!r}"
    )
    assert manage.derived_room == "a-room", (
        f"Equal-count room tie must select lexicographically first value 'a-room'; "
        f"got {manage.derived_room!r}"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 4: proof-bundle new-wing gate against persisted evidence rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_new_wing_insufficient_bundle_one_row_two_declared_categories(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Single persisted evidence row cannot satisfy two declared categories.

    ADR-013 Task 4: is_new_wing=True with proof_bundle listing two categories but
    only one persisted evidence artifact backing them must fail with
    MEM_INSUFFICIENT_EVIDENCE_BUNDLE.

    The gate must count distinct persisted rows per declared category — one
    artifact can satisfy at most one category. Two category labels with one row
    must not pass.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    # Store exactly ONE evidence row under one category.
    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t4_bundle/single.py::SingleClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t4-wing",
        room="t4-room",
        compartment="t4-comp",
    )

    # Declare two categories in proof_bundle but only supply the one evidence ID.
    # 'structural_class' is backed by the one row; 'structural_module' has no row.
    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_id],
                "is_new_wing": True,
                "proof_bundle": {
                    "evidence_categories": ["structural_class", "structural_module"],
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is False, (
        "New-wing derivation with one persisted row must fail; "
        f"got success=True with derivation_source={manage.derivation_source!r}"
    )
    error_code = manage.error or ""
    assert "MEM_INSUFFICIENT_EVIDENCE_BUNDLE" in error_code, (
        f"Expected MEM_INSUFFICIENT_EVIDENCE_BUNDLE in error; got {error_code!r}. "
        "Gate must reject when distinct persisted evidence rows per category < 2."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_new_wing_sufficient_bundle_two_rows_two_categories(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Two distinct persisted evidence rows across two categories passes the gate.

    ADR-013 Task 4: is_new_wing=True with proof_bundle listing two categories,
    each backed by one distinct persisted evidence artifact, must succeed.

    Verifies: one artifact per category, >=2 categories, >=2 distinct rows.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    ev1 = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t4_bundle/class_a.py::ClassA",
        entity_type="class",
        evidence_category="structural_class",
        wing="t4b-wing",
        room="t4b-room",
        compartment="t4b-comp",
    )
    ev2 = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t4_bundle/mod_a.py::ModA",
        entity_type="module",
        evidence_category="structural_module",
        wing="t4b-wing",
        room="t4b-room",
        compartment="t4b-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev1, ev2],
                "is_new_wing": True,
                "proof_bundle": {
                    "evidence_categories": ["structural_class", "structural_module"],
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, (
        f"New-wing derivation with two distinct evidence rows across two categories "
        f"must succeed; got error={manage.error!r}"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_bypasses_proof_bundle_gate(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """topology_override bypasses the new-wing proof-bundle gate even when is_new_wing=True.

    ADR-013 Task 4: when topology_override is supplied (explicit override path),
    the proof-bundle gate must NOT be enforced regardless of is_new_wing value.
    An override is authoritative by definition; requiring a proof bundle for an
    override would break the explicit-override contract from Task 2b.

    Verifies: result succeeds with is_new_wing=True, no proof_bundle, and a valid
    topology_override — the gate must be bypassed entirely.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    ev = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t4_override_bypass/class_a.py::ClassA",
        entity_type="class",
        evidence_category="structural_class",
        wing="t4ob-wing",
        room="t4ob-room",
        compartment="t4ob-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev],
                "is_new_wing": True,
                # No proof_bundle — override must bypass the gate
                "topology_override": {
                    "wing": "t4ob-wing",
                    "room": "t4ob-room",
                    "compartment": "t4ob-comp",
                    "override_reason": "test override bypass",
                    "applied_by": "test-agent",
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, (
        "topology_override with is_new_wing=True and no proof_bundle must succeed "
        f"(override bypasses proof-bundle gate); got error={manage.error!r}"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 5: Atomic persistence — simulated failure and success-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_atomic_rollback_on_db_failure(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Simulated DB failure during structural path leaves zero committed rows.

    ADR-013 Task 5: atomicity must be implementation-real, not documented-only.
    We inject a failure after the transaction opens (by replacing the backend's
    execute/query method to raise on the provenance INSERT) and assert that no
    claim, provenance, or provenance_evidence rows survive in the database.
    """
    from unittest.mock import patch

    from workflows_mcp.memory.memory_schema import MemoryRequest
    from workflows_mcp.memory.memory_service import MemoryContractError

    evidence_id = await _store_structural_evidence_and_get_id(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t5_structural_atomic.py::AtomicClass",
        wing="t5s-wing",
        room="t5s-room",
        compartment="t5s-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
            },
        }
    )

    # Track call count so we can fail on the provenance INSERT (2nd write query
    # after the claim SELECT/INSERT — makes the failure happen mid-transaction).
    original_query = knowledge_backend.query
    call_count = 0

    async def failing_query(sql: str, params: object = None) -> object:
        nonlocal call_count
        # Let the claim SELECT through, then fail on the provenance INSERT.
        if "knowledge_topology_provenance" in sql and "INSERT" in sql:
            raise RuntimeError("simulated DB failure during provenance insert")
        return await original_query(sql, params)

    # NOTE: knowledge_backend.query is patched rather than knowledge_backend.execute because
    # the current provenance INSERT (and provenance_evidence INSERT) are issued via
    # _backend.query (RETURNING clause required).  If production moves DML to
    # _backend.execute, the injection target below must change accordingly.
    with patch.object(knowledge_backend, "query", side_effect=failing_query):
        with pytest.raises((MemoryContractError, Exception)) as exc_info:
            await memory_service.execute(derive_req)  # type: ignore[union-attr]

    # Confirm the monkeypatch actually fired — the raised exception must carry the
    # injected text.  A pass without this assertion could mean the test succeeded
    # because of an unrelated validation or setup error, not the intended failure.
    exc_str = str(exc_info.value)
    assert "simulated DB failure" in exc_str, (
        f"Expected 'simulated DB failure' in exception message, got: {exc_str!r}. "
        "The monkeypatch may not have fired — check the injection target."
    )

    # Assert rollback: no committed rows for this evidence set.
    prov_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance ktp
          JOIN knowledge_topology_provenance_evidence kpe ON kpe.provenance_id = ktp.id
         WHERE ktp.palace = $1
           AND kpe.evidence_id = $2::uuid
        """,
        (PALACE, evidence_id),
    )
    assert prov_count.rows[0]["cnt"] == 0, (
        "knowledge_topology_provenance_evidence must have zero rows after structural "
        "path transaction rollback on simulated DB failure."
    )

    claim_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_claims
         WHERE palace = $1
           AND wing = 't5s-wing'
           AND room = 't5s-room'
        """,
        (PALACE,),
    )
    assert claim_count.rows[0]["cnt"] == 0, (
        "knowledge_semantic_claims must have zero rows after structural "
        "path transaction rollback on simulated DB failure."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_atomic_rollback_on_db_failure(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Simulated DB failure during override path leaves zero committed rows.

    ADR-013 Task 5: all five writes (claim, override, provenance, evidence links)
    must be committed atomically. A failure after the override INSERT but before
    commit must leave no claim, override, provenance, or provenance_evidence rows.
    """
    from unittest.mock import patch

    from workflows_mcp.memory.memory_schema import MemoryRequest
    from workflows_mcp.memory.memory_service import MemoryContractError

    evidence_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t5_override_atomic/class_a.py::ClassA",
        entity_type="class",
        evidence_category="structural_class",
        wing="t5o-wing",
        room="t5o-room",
        compartment="t5o-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                "topology_override": {
                    "wing": "t5o-wing",
                    "room": "t5o-room",
                    "compartment": "t5o-comp",
                    "override_reason": "atomic-test override",
                    "applied_by": "t5-atomic-agent",
                },
            },
        }
    )

    original_query = knowledge_backend.query

    async def failing_query(sql: str, params: object = None) -> object:
        # Fail on the provenance_evidence INSERT (last write before commit).
        if "knowledge_topology_provenance_evidence" in sql and "INSERT" in sql:
            raise RuntimeError("simulated DB failure during provenance_evidence insert")
        return await original_query(sql, params)

    # NOTE: knowledge_backend.query is patched rather than knowledge_backend.execute because
    # the current provenance_evidence INSERT uses RETURNING (issued via _backend.query).
    # If production moves DML to _backend.execute, the injection target below must change.
    with patch.object(knowledge_backend, "query", side_effect=failing_query):
        with pytest.raises((MemoryContractError, Exception)) as exc_info:
            await memory_service.execute(derive_req)  # type: ignore[union-attr]

    # Confirm the monkeypatch actually fired — the raised exception must carry the
    # injected text.  A pass without this assertion could mean the test succeeded
    # because of an unrelated validation or setup error, not the intended failure.
    exc_str = str(exc_info.value)
    assert "simulated DB failure" in exc_str, (
        f"Expected 'simulated DB failure' in exception message, got: {exc_str!r}. "
        "The monkeypatch may not have fired — check the injection target."
    )

    # Assert rollback: no surviving rows for this override attempt.
    prov_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance
         WHERE palace = $1 AND applied_by = 't5-atomic-agent'
        """,
        (PALACE,),
    )
    assert prov_count.rows[0]["cnt"] == 0, (
        "knowledge_topology_provenance must have zero rows after override "
        "path transaction rollback on simulated DB failure."
    )

    override_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_overrides so
          JOIN knowledge_semantic_claims sc ON sc.id = so.claim_id
         WHERE sc.palace = $1 AND so.applied_by = 't5-atomic-agent'
        """,
        (PALACE,),
    )
    assert override_count.rows[0]["cnt"] == 0, (
        "knowledge_semantic_overrides must have zero rows after override "
        "path transaction rollback on simulated DB failure."
    )

    claim_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_claims
         WHERE palace = $1 AND wing = 't5o-wing' AND room = 't5o-room'
        """,
        (PALACE,),
    )
    assert claim_count.rows[0]["cnt"] == 0, (
        "knowledge_semantic_claims must have zero rows after override "
        "path transaction rollback on simulated DB failure."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_all_four_tables_commit_atomically(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Success path: claim, override, provenance, and evidence link all commit together.

    ADR-013 Task 5: on a clean success, all four write tables must contain
    exactly one new row traceable back to the same derive_system1_topology call.
    This is the positive atomicity assertion — all-or-nothing in the success direction.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    evidence_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t5_success_atomic/class_b.py::ClassB",
        entity_type="class",
        evidence_category="structural_class",
        wing="t5a-wing",
        room="t5a-room",
        compartment="t5a-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                "topology_override": {
                    "wing": "t5a-wing",
                    "room": "t5a-room",
                    "compartment": "t5a-comp",
                    "override_reason": "all-four-tables test",
                    "applied_by": "t5-success-agent",
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success is True, (
        f"derive_system1_topology explicit override must succeed; error={manage.error!r}"
    )

    provenance_id = manage.provenance_id
    claim_id = manage.claim_id
    assert provenance_id, "provenance_id must be returned on success"
    assert claim_id, "claim_id must be returned on success"

    # 1. Semantic claim exists.
    claim_row = await knowledge_backend.query(
        "SELECT id FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert claim_row.rows, "knowledge_semantic_claims row must exist after successful commit"

    # 2. Semantic override exists linked to the claim.
    override_row = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_semantic_overrides
         WHERE claim_id = $1::uuid AND applied_by = 't5-success-agent'
        """,
        (claim_id,),
    )
    assert override_row.rows, "knowledge_semantic_overrides row must exist after successful commit"

    # 3. Topology provenance exists.
    prov_row = await knowledge_backend.query(
        "SELECT id FROM knowledge_topology_provenance WHERE id = $1::uuid",
        (provenance_id,),
    )
    assert prov_row.rows, "knowledge_topology_provenance row must exist after successful commit"

    # 4. Provenance evidence link exists.
    prov_ev_row = await knowledge_backend.query(
        """
        SELECT evidence_id FROM knowledge_topology_provenance_evidence
         WHERE provenance_id = $1::uuid AND evidence_id = $2::uuid
        """,
        (provenance_id, evidence_id),
    )
    assert prov_ev_row.rows, (
        "knowledge_topology_provenance_evidence row must exist after successful commit"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 5b: inline candidate passthrough + atomic store+derive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_inline_candidates_stored_and_derived_atomically(
    memory_service: Any,
    knowledge_backend: Any,
    clean_palace: None,
) -> None:
    """Inline candidates must be stored as knowledge_structural_evidence and derived atomically.

    ADR-013 Task 5b: one operation accepts inline candidates, stores them, and derives topology
    in a single transaction. On success, structural evidence rows must exist in the database.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "inline_candidates": [
                    {
                        "entity_stable_id": "src/t5b_inline/engine.py::EngineClass",
                        "entity_type": "class",
                        "evidence_category": "structural_class",
                        "evidence_data": {"source": "treesitter"},
                    },
                    {
                        "entity_stable_id": "src/t5b_inline/engine.py",
                        "entity_type": "module",
                        "evidence_category": "structural_module",
                        "evidence_data": {"source": "treesitter"},
                    },
                ],
                "topology_override": {
                    "wing": "t5b-wing",
                    "room": "t5b-room",
                    "compartment": "t5b-comp",
                    "override_reason": "inline candidate atomic test",
                    "applied_by": "t5b-agent",
                },
                "parser_metadata": {"language": "python", "file_path": "src/t5b_inline/engine.py"},
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success is True, (
        f"derive_system1_topology with inline_candidates must succeed; error={manage.error!r}"
    )

    provenance_id = manage.provenance_id
    claim_id = manage.claim_id
    assert provenance_id, "provenance_id must be returned on success"
    assert claim_id, "claim_id must be returned on success"

    # Inline candidates must have been stored as knowledge_structural_evidence rows.
    ev_rows = await knowledge_backend.query(
        "SELECT id FROM knowledge_structural_evidence"
        " WHERE palace = $1 AND entity_stable_id LIKE $2",
        (PALACE, "src/t5b_inline/%"),
    )
    assert len(ev_rows.rows) >= 2, (
        f"inline_candidates must be stored as knowledge_structural_evidence rows; "
        f"found {len(ev_rows.rows)}"
    )

    # Provenance evidence links must reference the stored inline candidate rows.
    prov_ev_rows = await knowledge_backend.query(
        "SELECT evidence_id FROM knowledge_topology_provenance_evidence"
        " WHERE provenance_id = $1::uuid",
        (provenance_id,),
    )
    assert prov_ev_rows.rows, (
        "knowledge_topology_provenance_evidence must link to inline candidate evidence rows"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_inline_candidates_insufficient_no_partial_write(
    memory_service: Any,
    knowledge_backend: Any,
    clean_palace: None,
) -> None:
    """Insufficient inline candidates without topology_override must fail closed with no write.

    ADR-013 Task 5b: when inline candidates cannot establish complete topology and no explicit
    override is provided, the operation must fail closed. No evidence rows, claim rows, or
    provenance rows must be written (atomic rollback).
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    # Single inline candidate without override — insufficient for complete topology.
    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "inline_candidates": [
                    {
                        "entity_stable_id": "src/t5b_insuff/lone.py::LoneFunc",
                        "entity_type": "function",
                        "evidence_category": "structural_function",
                        "evidence_data": {},
                    }
                ],
                # No topology_override — heuristic must fail closed for single-file evidence.
            },
        }
    )

    from workflows_mcp.memory.memory_service import MemoryContractError

    with pytest.raises(MemoryContractError) as exc_info:
        await memory_service.execute(derive_req)  # type: ignore[union-attr]

    error_code = exc_info.value.code
    assert "MEM_INSUFFICIENT" in error_code, (
        f"error must indicate insufficient evidence; got {error_code!r}"
    )

    # Rollback assertions: no rows must survive in the DB for this scope.
    # The inline candidate used entity_stable_id "src/t5b_insuff/lone.py::LoneFunc";
    # all four tables must be empty for this palace after the failed operation.
    evidence_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id LIKE 'src/t5b_insuff/%'
        """,
        (PALACE,),
    )
    assert evidence_count.rows[0]["cnt"] == 0, (
        "knowledge_structural_evidence must have zero rows for 't5b_insuff' scope "
        "after MEM_INSUFFICIENT failure — inline candidate evidence must not survive rollback."
    )

    prov_count = await knowledge_backend.query(
        "SELECT COUNT(*) AS cnt FROM knowledge_topology_provenance WHERE palace = $1",
        (PALACE,),
    )
    assert prov_count.rows[0]["cnt"] == 0, (
        "knowledge_topology_provenance must have zero rows after MEM_INSUFFICIENT failure — "
        "no provenance row must survive rollback."
    )

    prov_evidence_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance_evidence kpe
          JOIN knowledge_topology_provenance ktp ON ktp.id = kpe.provenance_id
         WHERE ktp.palace = $1
        """,
        (PALACE,),
    )
    assert prov_evidence_count.rows[0]["cnt"] == 0, (
        "knowledge_topology_provenance_evidence must have zero rows after MEM_INSUFFICIENT "
        "failure — no evidence link must survive rollback."
    )

    claim_count = await knowledge_backend.query(
        "SELECT COUNT(*) AS cnt FROM knowledge_semantic_claims WHERE palace = $1",
        (PALACE,),
    )
    assert claim_count.rows[0]["cnt"] == 0, (
        "knowledge_semantic_claims must have zero rows after MEM_INSUFFICIENT failure — "
        "no claim row must survive rollback."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_inline_candidates_wing_not_from_language(
    memory_service: Any,
    knowledge_backend: Any,
    clean_palace: None,
) -> None:
    """Derived wing must not equal parser_metadata language value.

    ADR-013 Task 5b: wing must not default to programming language.
    Parser metadata is evidence metadata only; language value must never appear as wing.
    """
    from workflows_mcp.memory.memory_schema import MemoryRequest

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "inline_candidates": [
                    {
                        "entity_stable_id": "src/t5b_lang/module.py::LangClass",
                        "entity_type": "class",
                        "evidence_category": "structural_class",
                        "evidence_data": {},
                    },
                    {
                        "entity_stable_id": "src/t5b_lang/module.py",
                        "entity_type": "module",
                        "evidence_category": "structural_module",
                        "evidence_data": {},
                    },
                ],
                "topology_override": {
                    "wing": "t5b-lang-wing",
                    "room": "t5b-lang-room",
                    "compartment": "t5b-lang-comp",
                    "override_reason": "language-not-wing test",
                    "applied_by": "t5b-agent",
                },
                "parser_metadata": {"language": "python"},
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success is True, (
        f"derive_system1_topology must succeed with override; error={manage.error!r}"
    )
    # Derived wing must come from override, not from parser_metadata.language.
    assert manage.derived_wing != "python", (
        "derived_wing must not equal parser_metadata.language ('python')"
    )
    assert manage.derived_wing == "t5b-lang-wing", (
        f"derived_wing must equal the override value; got {manage.derived_wing!r}"
    )
