"""Phase 3 invariant tests for graph ontology validation and onboard/sync wiring.

Covers:
- Required node types (GRAPH_MISSING_NODE_TYPE)
- Required corridor fields (GRAPH_MISSING_CORRIDOR_FIELD)
- Illegal same-level semantic links (GRAPH_ILLEGAL_LINK)
- Palace-level semantic edges beyond 'contains' (GRAPH_ILLEGAL_LINK)
- Orphan semantic nodes (GRAPH_ORPHAN_NODE)
- GRAPH_COMPLETENESS_FAILED summary envelope
- Valid graph passes validation
- GraphPayload builder utilities
- _validate_graph_step_payload integration helper
- scope_key included in checkpoint payloads
- sorted_scan_manifest ordering
"""

from __future__ import annotations

from typing import Any

from workflows_mcp.engine.memory_graph_builder import (
    REQUIRED_CORRIDOR_FIELDS,
    REQUIRED_NODE_TYPES,
    CorridorSemanticType,
    GraphCorridor,
    GraphNode,
    GraphPayload,
    NodeType,
    build_structural_graph,
)
from workflows_mcp.engine.memory_graph_validator import (
    GraphViolation,
    build_graph_error_envelope,
    validate_graph_payload,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _minimal_valid_payload() -> GraphPayload:
    """Return a minimal structurally-complete graph payload that passes validation."""
    return build_structural_graph(
        palace_id="p1",
        palace_label="MyOrg",
        wing_id="w1",
        wing_label="ServiceA",
        room_id="r1",
        room_label="API",
        compartment_id="c1",
        compartment_label="Auth",
        provenance="test",
        confidence=1.0,
    )


def _corridor(
    *,
    source_id: str = "src",
    target_id: str = "tgt",
    semantic_type: CorridorSemanticType = CorridorSemanticType.CONTAINS,
    confidence: float = 0.9,
    provenance: str = "test",
    evidence: list[str] | None = None,
) -> GraphCorridor:
    return GraphCorridor(
        source_id=source_id,
        target_id=target_id,
        semantic_type=semantic_type,
        confidence=confidence,
        provenance=provenance,
        evidence=evidence or ["ev-001"],
    )


# ===========================================================================
# 1. Valid payload passes
# ===========================================================================


class TestValidPayloadPasses:
    def test_minimal_complete_payload_is_valid(self) -> None:
        payload = _minimal_valid_payload()
        result = validate_graph_payload(payload)
        assert result.valid is True
        assert result.violations == []
        assert result.error_envelope is None

    def test_from_dict_roundtrip_is_valid(self) -> None:
        payload = _minimal_valid_payload()
        raw = payload.to_dict()
        result = validate_graph_payload(raw)
        assert result.valid is True


# ===========================================================================
# 2. Required node types
# ===========================================================================


class TestRequiredNodeTypes:
    def test_missing_palace_node_raises_violation(self) -> None:
        payload = _minimal_valid_payload()
        # Remove Palace nodes
        payload.nodes = [n for n in payload.nodes if n.node_type != NodeType.PALACE]
        result = validate_graph_payload(payload)
        assert result.valid is False
        codes = [v.code for v in result.violations]
        assert "GRAPH_MISSING_NODE_TYPE" in codes
        missing_types = [
            v.context.get("missing_node_type")
            for v in result.violations
            if v.code == "GRAPH_MISSING_NODE_TYPE"
        ]
        assert "Palace" in missing_types

    def test_missing_wing_node_raises_violation(self) -> None:
        payload = _minimal_valid_payload()
        payload.nodes = [n for n in payload.nodes if n.node_type != NodeType.WING]
        result = validate_graph_payload(payload)
        assert result.valid is False
        missing = [
            v.context["missing_node_type"]
            for v in result.violations
            if v.code == "GRAPH_MISSING_NODE_TYPE"
        ]
        assert "Wing" in missing

    def test_missing_room_node_raises_violation(self) -> None:
        payload = _minimal_valid_payload()
        payload.nodes = [n for n in payload.nodes if n.node_type != NodeType.ROOM]
        result = validate_graph_payload(payload)
        assert result.valid is False
        missing = [
            v.context["missing_node_type"]
            for v in result.violations
            if v.code == "GRAPH_MISSING_NODE_TYPE"
        ]
        assert "Room" in missing

    def test_missing_compartment_node_raises_violation(self) -> None:
        payload = _minimal_valid_payload()
        payload.nodes = [n for n in payload.nodes if n.node_type != NodeType.COMPARTMENT]
        result = validate_graph_payload(payload)
        assert result.valid is False
        missing = [
            v.context["missing_node_type"]
            for v in result.violations
            if v.code == "GRAPH_MISSING_NODE_TYPE"
        ]
        assert "Compartment" in missing

    def test_empty_nodes_list_misses_all_required_types(self) -> None:
        payload = GraphPayload(nodes=[], corridors=[])
        result = validate_graph_payload(payload)
        assert result.valid is False
        missing = [
            v.context["missing_node_type"]
            for v in result.violations
            if v.code == "GRAPH_MISSING_NODE_TYPE"
        ]
        assert set(missing) == {t.value for t in REQUIRED_NODE_TYPES}

    def test_all_required_types_present_no_missing_violation(self) -> None:
        payload = _minimal_valid_payload()
        result = validate_graph_payload(payload)
        missing_violations = [v for v in result.violations if v.code == "GRAPH_MISSING_NODE_TYPE"]
        assert missing_violations == []


# ===========================================================================
# 3. Required corridor fields
# ===========================================================================


class TestRequiredCorridorFields:
    def _payload_with_bare_corridor(self, **overrides: Any) -> GraphPayload:
        """Build a valid graph but replace corridor[0] with a corridor missing fields."""
        payload = _minimal_valid_payload()
        raw_corridor = payload.corridors[0].to_dict()
        raw_corridor.update(overrides)
        # Reconstruct manually to avoid Pydantic validation stripping empties
        payload.corridors[0] = GraphCorridor(
            source_id=raw_corridor.get("source_id", ""),
            target_id=raw_corridor.get("target_id", ""),
            semantic_type=CorridorSemanticType(raw_corridor.get("semantic_type", "contains")),
            confidence=raw_corridor.get("confidence", 0.0),
            provenance=raw_corridor.get("provenance", ""),
            evidence=raw_corridor.get("evidence", []),
        )
        return payload

    def test_missing_source_id_is_violation(self) -> None:
        payload = self._payload_with_bare_corridor(source_id="")
        result = validate_graph_payload(payload)
        fields = [
            v.context["missing_field"]
            for v in result.violations
            if v.code == "GRAPH_MISSING_CORRIDOR_FIELD"
        ]
        assert "source_id" in fields

    def test_missing_target_id_is_violation(self) -> None:
        payload = self._payload_with_bare_corridor(target_id="")
        result = validate_graph_payload(payload)
        fields = [
            v.context["missing_field"]
            for v in result.violations
            if v.code == "GRAPH_MISSING_CORRIDOR_FIELD"
        ]
        assert "target_id" in fields

    def test_missing_provenance_is_violation(self) -> None:
        payload = self._payload_with_bare_corridor(provenance="")
        result = validate_graph_payload(payload)
        fields = [
            v.context["missing_field"]
            for v in result.violations
            if v.code == "GRAPH_MISSING_CORRIDOR_FIELD"
        ]
        assert "provenance" in fields

    def test_missing_evidence_is_violation(self) -> None:
        # Evidence is required for non-structural (non-contains) corridors.
        # Use a DEPENDS_ON corridor between Wing and Room.
        payload = _minimal_valid_payload()
        # Replace one corridor with a non-structural one that has empty evidence
        payload.corridors[0] = GraphCorridor(
            source_id="w1",
            target_id="r1",
            semantic_type=CorridorSemanticType.DEPENDS_ON,
            confidence=0.9,
            provenance="test",
            evidence=[],  # Missing evidence for non-structural corridor
        )
        result = validate_graph_payload(payload)
        fields = [
            v.context["missing_field"]
            for v in result.violations
            if v.code == "GRAPH_MISSING_CORRIDOR_FIELD"
        ]
        assert "evidence" in fields

    def test_all_required_fields_present_no_corridor_violation(self) -> None:
        # Structural (contains) corridors from build_structural_graph are valid
        # without evidence — they are system-generated topology links.
        payload = _minimal_valid_payload()
        result = validate_graph_payload(payload)
        corridor_violations = [
            v for v in result.violations if v.code == "GRAPH_MISSING_CORRIDOR_FIELD"
        ]
        assert corridor_violations == []

    def test_required_corridor_fields_constant_is_complete(self) -> None:
        expected = frozenset(
            {"source_id", "target_id", "semantic_type", "confidence", "provenance", "evidence"}
        )
        assert frozenset(REQUIRED_CORRIDOR_FIELDS) == expected


# ===========================================================================
# 4. Illegal direct semantic links
# ===========================================================================


class TestIllegalDirectSemanticLinks:
    def _two_node_payload(
        self,
        src_type: NodeType,
        tgt_type: NodeType,
        semantic_type: CorridorSemanticType = CorridorSemanticType.RELATES_TO,
    ) -> GraphPayload:
        """Minimal 2-node payload for testing link rules (may fail node-type checks too)."""
        nodes = [
            GraphNode(node_id="src", node_type=src_type, label="Source"),
            GraphNode(node_id="tgt", node_type=tgt_type, label="Target"),
        ]
        corridors = [
            GraphCorridor(
                source_id="src",
                target_id="tgt",
                semantic_type=semantic_type,
                confidence=0.9,
                provenance="test",
                evidence=["ev-001"],
            )
        ]
        return GraphPayload(nodes=nodes, corridors=corridors)

    def test_compartment_to_compartment_is_illegal(self) -> None:
        payload = self._two_node_payload(NodeType.COMPARTMENT, NodeType.COMPARTMENT)
        result = validate_graph_payload(payload, strict_orphan_check=False)
        codes = [v.code for v in result.violations]
        assert "GRAPH_ILLEGAL_LINK" in codes

    def test_room_to_room_is_illegal(self) -> None:
        payload = self._two_node_payload(NodeType.ROOM, NodeType.ROOM)
        result = validate_graph_payload(payload, strict_orphan_check=False)
        codes = [v.code for v in result.violations]
        assert "GRAPH_ILLEGAL_LINK" in codes

    def test_wing_to_wing_is_illegal(self) -> None:
        payload = self._two_node_payload(NodeType.WING, NodeType.WING)
        result = validate_graph_payload(payload, strict_orphan_check=False)
        codes = [v.code for v in result.violations]
        assert "GRAPH_ILLEGAL_LINK" in codes

    def test_palace_to_palace_is_illegal(self) -> None:
        payload = self._two_node_payload(NodeType.PALACE, NodeType.PALACE)
        result = validate_graph_payload(payload, strict_orphan_check=False)
        codes = [v.code for v in result.violations]
        assert "GRAPH_ILLEGAL_LINK" in codes

    def test_palace_non_contains_semantic_type_is_illegal(self) -> None:
        """Palace nodes may only use 'contains' edges; any other type is forbidden."""
        payload = self._two_node_payload(
            NodeType.PALACE,
            NodeType.WING,
            semantic_type=CorridorSemanticType.DEPENDS_ON,
        )
        result = validate_graph_payload(payload, strict_orphan_check=False)
        codes = [v.code for v in result.violations]
        assert "GRAPH_ILLEGAL_LINK" in codes
        context = next(v.context for v in result.violations if v.code == "GRAPH_ILLEGAL_LINK")
        assert context.get("semantic_type") == "depends_on"

    def test_palace_contains_wing_is_legal(self) -> None:
        """Palace → Wing with 'contains' must not produce an illegal-link violation."""
        payload = self._two_node_payload(
            NodeType.PALACE,
            NodeType.WING,
            semantic_type=CorridorSemanticType.CONTAINS,
        )
        result = validate_graph_payload(payload, strict_orphan_check=False)
        illegal_violations = [v for v in result.violations if v.code == "GRAPH_ILLEGAL_LINK"]
        assert illegal_violations == []

    def test_wing_to_room_cross_level_is_legal(self) -> None:
        payload = self._two_node_payload(NodeType.WING, NodeType.ROOM)
        result = validate_graph_payload(payload, strict_orphan_check=False)
        illegal_violations = [v for v in result.violations if v.code == "GRAPH_ILLEGAL_LINK"]
        assert illegal_violations == []

    def test_illegal_link_context_contains_node_ids(self) -> None:
        payload = self._two_node_payload(NodeType.ROOM, NodeType.ROOM)
        result = validate_graph_payload(payload, strict_orphan_check=False)
        link_violation = next(v for v in result.violations if v.code == "GRAPH_ILLEGAL_LINK")
        assert "source_id" in link_violation.context
        assert "target_id" in link_violation.context
        assert "source_type" in link_violation.context
        assert "target_type" in link_violation.context


# ===========================================================================
# 5. Orphan semantic nodes
# ===========================================================================


class TestOrphanSemanticNodes:
    def test_orphan_node_with_strict_check_is_violation(self) -> None:
        payload = _minimal_valid_payload()
        # Add an isolated node with no corridor references
        orphan = GraphNode(node_id="orphan-99", node_type=NodeType.ROOM, label="Orphan Room")
        payload.nodes.append(orphan)
        result = validate_graph_payload(payload, strict_orphan_check=True)
        assert result.valid is False
        orphan_codes = [v.code for v in result.violations if v.code == "GRAPH_ORPHAN_NODE"]
        assert len(orphan_codes) == 1
        orphan_context = next(v.context for v in result.violations if v.code == "GRAPH_ORPHAN_NODE")
        assert orphan_context["node_id"] == "orphan-99"

    def test_orphan_node_skipped_when_strict_false(self) -> None:
        payload = _minimal_valid_payload()
        orphan = GraphNode(node_id="orphan-99", node_type=NodeType.ROOM, label="Orphan Room")
        payload.nodes.append(orphan)
        result = validate_graph_payload(payload, strict_orphan_check=False)
        orphan_violations = [v for v in result.violations if v.code == "GRAPH_ORPHAN_NODE"]
        assert orphan_violations == []

    def test_all_nodes_referenced_no_orphan_violation(self) -> None:
        payload = _minimal_valid_payload()
        result = validate_graph_payload(payload, strict_orphan_check=True)
        orphan_violations = [v for v in result.violations if v.code == "GRAPH_ORPHAN_NODE"]
        assert orphan_violations == []

    def test_orphan_context_includes_type_and_label(self) -> None:
        payload = _minimal_valid_payload()
        orphan = GraphNode(
            node_id="orphan-77",
            node_type=NodeType.COMPARTMENT,
            label="Orphan Compartment",
        )
        payload.nodes.append(orphan)
        result = validate_graph_payload(payload, strict_orphan_check=True)
        orphan_viol = next(v for v in result.violations if v.code == "GRAPH_ORPHAN_NODE")
        assert orphan_viol.context["node_type"] == "Compartment"
        assert orphan_viol.context["label"] == "Orphan Compartment"


# ===========================================================================
# 6. GRAPH_COMPLETENESS_FAILED envelope
# ===========================================================================


class TestCompletenessFailedEnvelope:
    def test_invalid_payload_produces_error_envelope(self) -> None:
        payload = GraphPayload(nodes=[], corridors=[])
        result = validate_graph_payload(payload)
        assert result.valid is False
        assert result.error_envelope is not None
        err = result.error_envelope.get("error", {})
        assert err.get("code") == "GRAPH_COMPLETENESS_FAILED"

    def test_envelope_has_required_fields(self) -> None:
        payload = GraphPayload(nodes=[], corridors=[])
        result = validate_graph_payload(payload)
        err = result.error_envelope["error"]
        assert "code" in err
        assert "message" in err
        assert "retryable" in err
        assert "stage" in err
        assert "actionable_fix" in err
        assert "violations" in err

    def test_envelope_code_is_string(self) -> None:
        payload = GraphPayload(nodes=[], corridors=[])
        result = validate_graph_payload(payload)
        assert isinstance(result.error_envelope["error"]["code"], str)

    def test_envelope_retryable_is_false(self) -> None:
        payload = GraphPayload(nodes=[], corridors=[])
        result = validate_graph_payload(payload)
        assert result.error_envelope["error"]["retryable"] is False

    def test_envelope_violations_list_non_empty(self) -> None:
        payload = GraphPayload(nodes=[], corridors=[])
        result = validate_graph_payload(payload)
        assert len(result.error_envelope["error"]["violations"]) > 0

    def test_envelope_stage_is_graph_validation(self) -> None:
        payload = GraphPayload(nodes=[], corridors=[])
        result = validate_graph_payload(payload)
        assert result.error_envelope["error"]["stage"] == "graph_validation"

    def test_valid_payload_produces_no_envelope(self) -> None:
        payload = _minimal_valid_payload()
        result = validate_graph_payload(payload)
        assert result.error_envelope is None

    def test_completeness_failed_count_in_message(self) -> None:
        """Message must include violation count."""
        payload = GraphPayload(nodes=[], corridors=[])
        result = validate_graph_payload(payload)
        message = result.error_envelope["error"]["message"]
        # Should mention a count > 0
        assert any(char.isdigit() for char in message)


# ===========================================================================
# 7. GraphPayload builder (build_structural_graph)
# ===========================================================================


class TestBuildStructuralGraph:
    def test_produces_four_nodes(self) -> None:
        payload = build_structural_graph(
            palace_id="p",
            palace_label="Org",
            wing_id="w",
            wing_label="Svc",
            room_id="r",
            room_label="API",
            compartment_id="c",
            compartment_label="Auth",
        )
        assert len(payload.nodes) == 4

    def test_produces_three_corridors(self) -> None:
        payload = build_structural_graph(
            palace_id="p",
            palace_label="Org",
            wing_id="w",
            wing_label="Svc",
            room_id="r",
            room_label="API",
            compartment_id="c",
            compartment_label="Auth",
        )
        assert len(payload.corridors) == 3

    def test_all_corridors_are_contains(self) -> None:
        payload = build_structural_graph(
            palace_id="p",
            palace_label="Org",
            wing_id="w",
            wing_label="Svc",
            room_id="r",
            room_label="API",
            compartment_id="c",
            compartment_label="Auth",
        )
        for corridor in payload.corridors:
            assert corridor.semantic_type == CorridorSemanticType.CONTAINS

    def test_corridor_chain_is_p_w_r_c(self) -> None:
        payload = build_structural_graph(
            palace_id="p",
            palace_label="Org",
            wing_id="w",
            wing_label="Svc",
            room_id="r",
            room_label="API",
            compartment_id="c",
            compartment_label="Auth",
        )
        links = [(c.source_id, c.target_id) for c in payload.corridors]
        assert ("p", "w") in links
        assert ("w", "r") in links
        assert ("r", "c") in links

    def test_node_types_match_ontology(self) -> None:
        payload = build_structural_graph(
            palace_id="p",
            palace_label="Org",
            wing_id="w",
            wing_label="Svc",
            room_id="r",
            room_label="API",
            compartment_id="c",
            compartment_label="Auth",
        )
        types = {n.node_type for n in payload.nodes}
        assert types == set(REQUIRED_NODE_TYPES)

    def test_to_dict_roundtrip(self) -> None:
        payload = build_structural_graph(
            palace_id="p",
            palace_label="Org",
            wing_id="w",
            wing_label="Svc",
            room_id="r",
            room_label="API",
            compartment_id="c",
            compartment_label="Auth",
        )
        d = payload.to_dict()
        assert "nodes" in d
        assert "corridors" in d
        assert len(d["nodes"]) == 4
        assert len(d["corridors"]) == 3


# ===========================================================================
# 8. _validate_graph_step_payload integration
# ===========================================================================


class TestValidateGraphStepPayload:
    def _import_helper(self) -> Any:
        from workflows_mcp.tools_memory import _validate_graph_step_payload

        return _validate_graph_step_payload

    def test_non_graph_payload_returns_none(self) -> None:
        fn = self._import_helper()
        result = fn({"format": "structured", "memories": []})
        assert result is None

    def test_valid_graph_payload_returns_none(self) -> None:
        fn = self._import_helper()
        payload = _minimal_valid_payload()
        raw = {"graph": payload.to_dict()}
        result = fn(raw)
        assert result is None

    def test_invalid_graph_payload_returns_error_envelope(self) -> None:
        fn = self._import_helper()
        raw = {"graph": {"nodes": [], "corridors": []}}
        result = fn(raw)
        assert result is not None
        assert "error" in result
        assert result["error"]["code"] == "GRAPH_COMPLETENESS_FAILED"

    def test_error_envelope_has_stage(self) -> None:
        fn = self._import_helper()
        raw = {"graph": {"nodes": [], "corridors": []}}
        result = fn(raw)
        assert "stage" in result["error"]

    def test_error_envelope_has_actionable_fix(self) -> None:
        fn = self._import_helper()
        raw = {"graph": {"nodes": [], "corridors": []}}
        result = fn(raw)
        assert "actionable_fix" in result["error"]

    def test_dict_without_nodes_corridors_keys_returns_none(self) -> None:
        """Dicts that don't look like graph payloads must be skipped."""
        fn = self._import_helper()
        result = fn({"format": "structured", "content": "hello"})
        assert result is None


# ===========================================================================
# 9. scope_key in checkpoint payloads (Phase 2 debt)
# ===========================================================================


class TestScopeKeyInCheckpoint:
    def _import_helper(self) -> Any:
        from workflows_mcp.tools_memory import _build_project_checkpoint_payload

        return _build_project_checkpoint_payload

    def test_checkpoint_contains_scope_key_field(self) -> None:
        fn = self._import_helper()
        scope = {"palace": "myorg", "wing": "svc", "room": "api", "compartment": "auth"}
        payload = fn(scope=scope, plan=[], next_index=0, completed=[])
        assert "scope_key" in payload

    def test_scope_key_is_string(self) -> None:
        fn = self._import_helper()
        scope = {"palace": "myorg", "wing": "svc", "room": "api", "compartment": "auth"}
        payload = fn(scope=scope, plan=[], next_index=0, completed=[])
        assert isinstance(payload["scope_key"], str)

    def test_scope_key_is_deterministic(self) -> None:
        fn = self._import_helper()
        scope = {"palace": "myorg", "wing": "svc", "room": "api", "compartment": "auth"}
        p1 = fn(scope=scope, plan=[], next_index=0, completed=[])
        p2 = fn(scope=scope, plan=[], next_index=0, completed=[])
        assert p1["scope_key"] == p2["scope_key"]

    def test_scope_key_changes_with_different_scope(self) -> None:
        fn = self._import_helper()
        scope_a = {"palace": "orgA", "wing": "svc"}
        scope_b = {"palace": "orgB", "wing": "svc"}
        p_a = fn(scope=scope_a, plan=[], next_index=0, completed=[])
        p_b = fn(scope=scope_b, plan=[], next_index=0, completed=[])
        assert p_a["scope_key"] != p_b["scope_key"]

    def test_scope_key_order_independent(self) -> None:
        fn = self._import_helper()
        scope1 = {"palace": "org", "wing": "svc", "room": "api", "compartment": "auth"}
        scope2 = {"compartment": "auth", "room": "api", "palace": "org", "wing": "svc"}
        p1 = fn(scope=scope1, plan=[], next_index=0, completed=[])
        p2 = fn(scope=scope2, plan=[], next_index=0, completed=[])
        assert p1["scope_key"] == p2["scope_key"]


# ===========================================================================
# 10. sorted_scan_manifest ordering
# ===========================================================================


class TestSortedScanManifest:
    def _import_helper(self) -> Any:
        from workflows_mcp.engine.memory_scope_resolver import sorted_scan_manifest

        return sorted_scan_manifest

    def test_empty_list_returns_empty(self) -> None:
        fn = self._import_helper()
        assert fn([]) == []

    def test_already_sorted_list_unchanged(self) -> None:
        fn = self._import_helper()
        entries = [{"path": "a.py"}, {"path": "b.py"}, {"path": "c.py"}]
        result = fn(entries)
        assert [e["path"] for e in result] == ["a.py", "b.py", "c.py"]

    def test_unsorted_list_is_sorted(self) -> None:
        fn = self._import_helper()
        entries = [{"path": "z.py"}, {"path": "a.py"}, {"path": "m.py"}]
        result = fn(entries)
        assert [e["path"] for e in result] == ["a.py", "m.py", "z.py"]

    def test_entries_missing_path_sort_first(self) -> None:
        fn = self._import_helper()
        entries = [{"path": "b.py"}, {"content": "no path"}, {"path": "a.py"}]
        result = fn(entries)
        # Entry without path sorts before others (treated as empty string)
        assert result[0].get("path") is None or result[0].get("content") == "no path"

    def test_sorting_is_stable_for_equal_paths(self) -> None:
        fn = self._import_helper()
        entries = [
            {"path": "same.py", "idx": 0},
            {"path": "same.py", "idx": 1},
        ]
        result = fn(entries)
        # Stable sort: original order preserved for equal keys
        assert result[0]["idx"] == 0
        assert result[1]["idx"] == 1

    def test_deep_paths_sorted_lexicographically(self) -> None:
        fn = self._import_helper()
        entries = [
            {"path": "src/z/foo.py"},
            {"path": "src/a/bar.py"},
            {"path": "docs/readme.md"},
        ]
        result = fn(entries)
        paths = [e["path"] for e in result]
        assert paths == sorted(paths)


# ===========================================================================
# 11. GraphViolation to_dict contract
# ===========================================================================


class TestGraphViolationToDict:
    def test_to_dict_contains_required_keys(self) -> None:
        v = GraphViolation(
            code="GRAPH_MISSING_NODE_TYPE",
            message="Test message",
            context={"missing_node_type": "Palace"},
        )
        d = v.to_dict()
        assert "code" in d
        assert "message" in d
        assert "context" in d

    def test_to_dict_code_matches(self) -> None:
        v = GraphViolation(code="GRAPH_ILLEGAL_LINK", message="msg")
        assert v.to_dict()["code"] == "GRAPH_ILLEGAL_LINK"


# ===========================================================================
# 12. build_graph_error_envelope
# ===========================================================================


class TestBuildGraphErrorEnvelope:
    def test_produces_error_envelope_shape(self) -> None:
        result = build_graph_error_envelope(
            code="GRAPH_COMPLETENESS_FAILED",
            message="Failed",
            stage="test_stage",
            actionable_fix="Fix it.",
        )
        assert "error" in result
        err = result["error"]
        assert err["code"] == "GRAPH_COMPLETENESS_FAILED"
        assert err["stage"] == "test_stage"
        assert err["actionable_fix"] == "Fix it."
        assert err["retryable"] is False

    def test_violations_included_when_provided(self) -> None:
        violations = [GraphViolation(code="GRAPH_MISSING_NODE_TYPE", message="missing Palace")]
        result = build_graph_error_envelope(
            code="GRAPH_COMPLETENESS_FAILED",
            message="Failed",
            stage="s",
            actionable_fix="fix",
            violations=violations,
        )
        assert "violations" in result["error"]
        assert len(result["error"]["violations"]) == 1

    def test_no_violations_key_when_none_provided(self) -> None:
        result = build_graph_error_envelope(
            code="GRAPH_COMPLETENESS_FAILED",
            message="Failed",
            stage="s",
            actionable_fix="fix",
        )
        assert "violations" not in result["error"]
