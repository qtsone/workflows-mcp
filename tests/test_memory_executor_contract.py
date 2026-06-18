from __future__ import annotations

import pytest
from pydantic import ValidationError

from workflows_mcp.engine.executors_memory import MemoryInput
from workflows_mcp.engine.memory_schema import MemoryRequest


@pytest.mark.parametrize(
    "operation",
    [
        "derive_system2_semantic_claims",
        "apply_semantic_override",
        "reconcile_semantic_lifecycle",
    ],
)
def test_memory_executor_accepts_system2_operations(operation: str) -> None:
    parsed = MemoryInput.model_validate({"operation": operation})

    assert parsed.operation == operation


def test_memory_executor_rejects_unknown_operations() -> None:
    with pytest.raises(ValidationError):
        MemoryInput.model_validate({"operation": "semantic_magic"})


def test_system2_derivation_contract_accepts_wing_and_memory_claims() -> None:
    request = MemoryRequest.model_validate(
        {
            "operation": "derive_system2_semantic_claims",
            "scope": {
                "palace": "forge",
                "wing": "runtime",
                "room": "sync",
                "compartment": "runner",
            },
            "record": {
                "derivation": {
                    "wing_intent_label": "runtime orchestration",
                    "memory_claim_text": "Sync jobs are queued asynchronously.",
                    "evidence_entity_stable_ids": ["src/sync.py::sync_rebuild"],
                }
            },
        }
    )

    assert request.record is not None
    assert request.record.derivation is not None
    assert request.record.derivation.wing_intent_label == "runtime orchestration"
    assert request.record.derivation.memory_claim_text == "Sync jobs are queued asynchronously."
