"""Contract tests for unified memory request envelope."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from workflows_mcp.engine.memory_service import MemoryRequest


def test_query_mode_search_accepts_radius_precision() -> None:
    req = MemoryRequest(
        operation="query",
        scope={"wing": "svc", "room": "comp", "hall": "topic"},
        query={"text": "incident", "mode": "search", "radius": 0, "precision": 0.7},
    )
    assert req.query is not None
    assert req.query.radius == 0
    assert req.query.precision == 0.7


def test_query_mode_graph_requires_text_but_graph_is_optional_payload() -> None:
    req = MemoryRequest(
        operation="query",
        query={
            "text": "dependency path",
            "mode": "graph",
            "graph": {"op": "path", "start": "a", "end": "b"},
        },
    )
    assert req.query is not None
    assert req.query.mode == "graph"
    assert req.query.graph.op == "path"


def test_query_accepts_from_to_interval_aliases() -> None:
    req = MemoryRequest(
        operation="query",
        query={
            "text": "incident timeline",
            "from": "2026-04-01T00:00:00Z",
            "to": "2026-04-30T23:59:59Z",
        },
    )
    assert req.query is not None
    assert req.query.from_ == "2026-04-01T00:00:00Z"
    assert req.query.to == "2026-04-30T23:59:59Z"


def test_query_rejects_as_of_with_interval_range() -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="query",
            query={
                "text": "incident timeline",
                "as_of": "2026-04-15T12:00:00Z",
                "from": "2026-04-01T00:00:00Z",
            },
        )
    assert "as_of cannot be combined with 'from'/'to'" in str(exc.value)


def test_query_rejects_unknown_temporal_fields_strictly() -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="query",
            query={
                "text": "incident timeline",
                "validity": {"as_of": "2026-04-15T12:00:00Z"},
            },
        )
    assert "Extra inputs are not permitted" in str(exc.value)


def test_precision_is_bounded() -> None:
    try:
        MemoryRequest(operation="query", query={"text": "x", "precision": 2.0})
    except ValidationError:
        return
    raise AssertionError("Expected ValidationError for precision > 1.0")


def test_ingest_raw_accepts_record_content() -> None:
    req = MemoryRequest(
        operation="ingest",
        scope={"wing": "svc", "room": "comp"},
        record={"format": "raw", "content": "stored memory"},
    )
    assert req.record is not None
    assert req.record.format == "raw"


def test_ingest_raw_accepts_temporal_validity_window_fields() -> None:
    req = MemoryRequest(
        operation="ingest",
        scope={"wing": "svc", "room": "comp"},
        record={
            "format": "raw",
            "content": "stored memory",
            "valid_from": "2026-04-01T00:00:00Z",
            "valid_to": "2026-04-30T23:59:59Z",
        },
    )
    assert req.record is not None
    assert req.record.format == "raw"
    assert req.record.valid_from == "2026-04-01T00:00:00Z"
    assert req.record.valid_to == "2026-04-30T23:59:59Z"


def test_ingest_raw_rejects_invalid_temporal_window_order() -> None:
    try:
        MemoryRequest(
            operation="ingest",
            scope={"wing": "svc", "room": "comp"},
            record={
                "format": "raw",
                "content": "stored memory",
                "valid_from": "2026-04-30T23:59:59Z",
                "valid_to": "2026-04-01T00:00:00Z",
            },
        )
    except ValidationError as exc:
        assert "valid_from must be less than or equal to valid_to" in str(exc)
        return
    raise AssertionError("Expected ValidationError for valid_from > valid_to")


def test_graph_upsert_link_accepts_from_to_aliases() -> None:
    req = MemoryRequest(
        operation="graph_upsert",
        graph={"kind": "link", "from": "a", "to": "b", "link_type": "depends_on"},
    )
    assert req.graph is not None
    assert req.graph.from_ref == "a"
    assert req.graph.to_ref == "b"


def test_archive_record_rejects_unknown_temporal_fields_strictly() -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="archive",
            record={
                "ids": ["11111111-1111-1111-1111-111111111111"],
                "until": "2026-04-30T23:59:59Z",
            },
        )
    assert "Extra inputs are not permitted" in str(exc.value)
