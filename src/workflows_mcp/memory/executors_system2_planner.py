"""System 2 semantic derivation planner for ADR-013 project sync."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

from pydantic import Field

from workflows_mcp.engine.block import BlockInput, BlockOutput
from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)

from .memory_scope_resolver import scope_key


class System2PlannerInput(BlockInput):
    """Inputs for evidence-backed System 2 project planning."""

    palace: str
    default_wing: str
    default_room: str
    default_compartment: str = ""


class System2PlannerOutput(BlockOutput):
    """Planned System 2 workflow calls."""

    derivations: list[dict[str, Any]] = Field(default_factory=list)
    lifecycle: dict[str, Any] = Field(default_factory=dict)
    derivation_count: int = 0


class System2PlannerExecutor(BlockExecutor):
    """Build deterministic evidence-backed System 2 derivation requests."""

    type_name: ClassVar[str] = "System2Planner"
    input_type: ClassVar[type[BlockInput]] = System2PlannerInput
    output_type: ClassVar[type[BlockOutput]] = System2PlannerOutput
    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_network=True)

    async def execute(  # type: ignore[override]
        self,
        inputs: System2PlannerInput,
        context: Execution,
    ) -> System2PlannerOutput:
        execution_context = getattr(context, "execution_context", None)
        backend = getattr(execution_context, "memory_backend", None)
        if backend is None:
            raise RuntimeError("System 2 planner requires a connected memory backend")

        evidence_result = await backend.query(
            """
            SELECT wing, room, compartment, entity_stable_id, evidence_category
              FROM knowledge_structural_evidence
             WHERE palace = $1
               AND wing <> ''
               AND room <> ''
             ORDER BY wing, room, compartment, entity_stable_id
            """,
            (inputs.palace,),
        )
        existing_result = await backend.query(
            """
            SELECT wing, room, compartment, claim_type, claim_text
              FROM knowledge_semantic_claims
             WHERE palace = $1
               AND lifecycle_state <> 'archived'
            """,
            (inputs.palace,),
        )

        existing_claims = {
            (
                str(row["wing"]),
                str(row["room"]),
                str(row["compartment"] or ""),
                str(row["claim_type"]),
                str(row["claim_text"]),
            )
            for row in existing_result.rows
        }

        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in evidence_result.rows:
            wing = str(row["wing"])
            room = str(row["room"])
            compartment = str(row["compartment"] or "")
            grouped[(wing, room, compartment)].append(
                {
                    "entity_stable_id": str(row["entity_stable_id"]),
                    "evidence_category": str(row["evidence_category"]),
                }
            )

        derivations: list[dict[str, Any]] = []
        for (wing, room, compartment), rows in grouped.items():
            stable_ids = sorted({row["entity_stable_id"] for row in rows})
            evidence_categories = sorted({row["evidence_category"] for row in rows})
            base = {
                "palace": inputs.palace,
                "wing": wing,
                "room": room,
                "compartment": compartment,
                "evidence_entity_stable_ids": stable_ids,
                "wing_intent_label": "",
                "room_intent_label": "",
                "compartment_reasoning_unit": "",
                "memory_claim_text": "",
                "proof_bundle_evidence_categories": evidence_categories,
                "is_new_wing": False,
            }

            candidates: list[tuple[str, str, dict[str, Any]]] = [
                (
                    "wing_intent",
                    f"wing_intent:{wing}",
                    {
                        "wing_intent_label": wing,
                        "is_new_wing": len(evidence_categories) >= 2,
                    },
                ),
                ("room_intent", f"room_intent:{room}", {"room_intent_label": room}),
                (
                    "compartment_reasoning_unit",
                    f"compartment_reasoning_unit:{compartment or room}",
                    {"compartment_reasoning_unit": compartment or room},
                ),
                (
                    "memory_claim",
                    (
                        "memory_claim:"
                        f"Structural evidence identifies {wing}/{room}/{compartment or room} "
                        f"from {len(stable_ids)} source anchors."
                    ),
                    {
                        "memory_claim_text": (
                            f"Structural evidence identifies {wing}/{room}/"
                            f"{compartment or room} from {len(stable_ids)} source anchors."
                        )
                    },
                ),
            ]
            for claim_type, claim_text, payload in candidates:
                key = (wing, room, compartment, claim_type, claim_text)
                if key in existing_claims:
                    continue
                derivations.append({**base, **payload})

        lifecycle_scope = {
            "palace": inputs.palace,
            "wing": inputs.default_wing,
            "room": inputs.default_room,
            "compartment": inputs.default_compartment,
        }
        lifecycle = {
            "palace": inputs.palace,
            "wing": inputs.default_wing,
            "room": inputs.default_room,
            "compartment": inputs.default_compartment,
            "scope_key": scope_key(lifecycle_scope),
            "degrade_claim_ids": [],
            "force_archive_claim_ids": [],
            "absent_verification_cycle_ids": [],
        }
        return System2PlannerOutput(
            derivations=derivations,
            lifecycle=lifecycle,
            derivation_count=len(derivations),
        )
