"""Tests for the new-wing proof bundle gate (Task 9, ADR-013).

Gate rule: a semantic claim with is_new_wing=True must supply a proof bundle
containing at least two *distinct* evidence categories.  A single category
submitted twice must not satisfy the gate (duplicate-entry prevention).

Proof bundle references must be persisted to knowledge_wing_proof_bundles
so auditors can inspect the gate decision for any wing activation.

Non-new-wing derivations must continue to succeed without a proof bundle.

Error code for gate violations: MEM_INSUFFICIENT_EVIDENCE_BUNDLE
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

PALACE = "palace_wing_proof_test"
WING = "auth"
ROOM = "session"
COMPARTMENT = ""


def _scope() -> dict[str, str]:
    return {"palace": PALACE, "wing": WING, "room": ROOM}


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
    palace_pattern = f"{PALACE}%"

    async def _wipe() -> None:
        await knowledge_backend.execute(
            "DELETE FROM knowledge_wing_proof_bundles WHERE palace LIKE $1",
            (palace_pattern,),
        )
        # claim_evidence_links cascade from claims
        await knowledge_backend.execute(
            "DELETE FROM knowledge_semantic_claims WHERE palace LIKE $1",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_structural_evidence WHERE palace LIKE $1",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_verification_cycles WHERE palace LIKE $1",
            (palace_pattern,),
        )

    await _wipe()
    yield
    await _wipe()


async def _store_evidence(
    memory_service: Any,
    *,
    entity_stable_id: str,
    evidence_category: str,
) -> None:
    """Helper: store one structural evidence row for the test palace/wing/room."""
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
                            "entity_stable_id": entity_stable_id,
                            "entity_type": "class",
                            "evidence_category": evidence_category,
                            "evidence_data": {},
                        }
                    ],
                },
            }
        )
    )
    assert result.manage is not None and result.manage.success, (
        f"Failed to store evidence {entity_stable_id}/{evidence_category}: {result.manage}"
    )


# ---------------------------------------------------------------------------
# Gate rejection cases
# ---------------------------------------------------------------------------


async def test_new_wing_without_proof_bundle_is_rejected(
    memory_service: Any,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """is_new_wing=True with no proof_bundle supplied must be rejected."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    await _store_evidence(
        memory_service,
        entity_stable_id="src/auth.py::LoginService",
        evidence_category="structural_class",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "authentication",
                        "evidence_entity_stable_ids": ["src/auth.py::LoginService"],
                        "is_new_wing": True,
                        # no proof_bundle supplied
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success is False
    assert result.manage.error is not None
    assert "MEM_INSUFFICIENT_EVIDENCE_BUNDLE" in result.manage.error


async def test_new_wing_with_single_evidence_category_is_rejected(
    memory_service: Any,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """is_new_wing=True with only one distinct evidence category must be rejected."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    await _store_evidence(
        memory_service,
        entity_stable_id="src/auth.py::LoginService",
        evidence_category="structural_class",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "authentication",
                        "evidence_entity_stable_ids": ["src/auth.py::LoginService"],
                        "is_new_wing": True,
                        "proof_bundle": {
                            "evidence_categories": ["structural_class"],
                        },
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success is False
    assert result.manage.error is not None
    assert "MEM_INSUFFICIENT_EVIDENCE_BUNDLE" in result.manage.error


async def test_new_wing_with_duplicate_category_entries_is_rejected(
    memory_service: Any,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """is_new_wing=True with the same category listed twice must be rejected.

    Duplicate entries must not satisfy the >= 2 distinct categories gate.
    """
    from workflows_mcp.engine.memory_schema import MemoryRequest

    await _store_evidence(
        memory_service,
        entity_stable_id="src/auth.py::LoginService",
        evidence_category="structural_class",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "authentication",
                        "evidence_entity_stable_ids": ["src/auth.py::LoginService"],
                        "is_new_wing": True,
                        "proof_bundle": {
                            # same category duplicated — must NOT satisfy the gate
                            "evidence_categories": [
                                "structural_class",
                                "structural_class",
                            ],
                        },
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success is False
    assert result.manage.error is not None
    assert "MEM_INSUFFICIENT_EVIDENCE_BUNDLE" in result.manage.error


# ---------------------------------------------------------------------------
# Gate acceptance case
# ---------------------------------------------------------------------------


async def test_new_wing_with_two_distinct_evidence_categories_succeeds(
    memory_service: Any,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """is_new_wing=True with two distinct evidence categories must succeed
    and persist a proof bundle row in knowledge_wing_proof_bundles."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    await _store_evidence(
        memory_service,
        entity_stable_id="src/auth.py::LoginService",
        evidence_category="structural_class",
    )
    await _store_evidence(
        memory_service,
        entity_stable_id="src/auth_module",
        evidence_category="structural_module",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "authentication",
                        "evidence_entity_stable_ids": [
                            "src/auth.py::LoginService",
                            "src/auth_module",
                        ],
                        "is_new_wing": True,
                        "proof_bundle": {
                            "evidence_categories": [
                                "structural_class",
                                "structural_module",
                            ],
                        },
                    },
                },
            }
        )
    )
    assert result.manage is not None, "Expected manage envelope"
    assert result.manage.success is True, f"Gate should have passed: {result.manage.error}"
    assert result.manage.claim_ids, "Expected at least one claim_id"

    claim_id = result.manage.claim_ids[0]

    # Verify proof bundle row was persisted and linked to the claim.
    bundle_rows = await knowledge_backend.query(
        """
        SELECT palace, wing, activating_claim_id::text, evidence_categories, gate_satisfied
          FROM knowledge_wing_proof_bundles
         WHERE activating_claim_id = $1::uuid
        """,
        (claim_id,),
    )
    assert bundle_rows.rows, "Proof bundle row must be persisted in knowledge_wing_proof_bundles"
    bundle = bundle_rows.rows[0]
    assert bundle["palace"] == PALACE
    assert bundle["wing"] == WING
    assert bundle["gate_satisfied"] is True
    stored_cats = bundle["evidence_categories"]
    assert set(stored_cats) == {"structural_class", "structural_module"}


async def test_proof_bundle_row_links_claim_for_auditability(
    memory_service: Any,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """The proof bundle row must be queryable by claim_id for audit inspection."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    await _store_evidence(
        memory_service,
        entity_stable_id="src/auth.py::LoginService",
        evidence_category="structural_class",
    )
    await _store_evidence(
        memory_service,
        entity_stable_id="src/auth_test.py",
        evidence_category="structural_test",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "auth_services",
                        "evidence_entity_stable_ids": [
                            "src/auth.py::LoginService",
                            "src/auth_test.py",
                        ],
                        "is_new_wing": True,
                        "proof_bundle": {
                            "evidence_categories": [
                                "structural_class",
                                "structural_test",
                            ],
                        },
                    },
                },
            }
        )
    )
    assert result.manage is not None and result.manage.success
    claim_id = result.manage.claim_ids[0]

    idx_rows = await knowledge_backend.query(
        "SELECT id FROM knowledge_wing_proof_bundles WHERE activating_claim_id = $1::uuid",
        (claim_id,),
    )
    assert len(idx_rows.rows) == 1, "Exactly one proof bundle row expected per new-wing claim"


# ---------------------------------------------------------------------------
# Non-new-wing claims must not require proof bundle
# ---------------------------------------------------------------------------


async def test_non_new_wing_claim_succeeds_without_proof_bundle(
    memory_service: Any,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """derive_system2_semantic_claims with is_new_wing=False (default) must succeed
    without a proof_bundle, confirming the gate only applies to new-wing claims."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    await _store_evidence(
        memory_service,
        entity_stable_id="src/auth.py::LoginService",
        evidence_category="structural_class",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "login_handler",
                        "evidence_entity_stable_ids": ["src/auth.py::LoginService"],
                        "is_new_wing": False,
                        # no proof_bundle — must succeed
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success is True, (
        f"Non-new-wing claim must not require proof bundle: {result.manage.error}"
    )
    assert result.manage.claim_ids


async def test_non_new_wing_claim_with_one_evidence_category_succeeds(
    memory_service: Any,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Normal derivations with a single evidence category succeed — the >= 2
    categories gate is exclusive to is_new_wing=True."""
    from workflows_mcp.engine.memory_schema import MemoryRequest

    await _store_evidence(
        memory_service,
        entity_stable_id="src/session.py::SessionStore",
        evidence_category="structural_class",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "session_management",
                        "evidence_entity_stable_ids": ["src/session.py::SessionStore"],
                        # is_new_wing defaults to False
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success is True, f"Expected success: {result.manage.error}"
