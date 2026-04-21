# Memory Tool Cheatsheet (Guide)

Purpose: practical usage guide for the unified MCP memory tool.

Active tool:

- `memory`

Legacy note: `query_memory` and `manage_memory` references are legacy-only and not the active public tool contract.

## 1) Canonical request envelope

```json
{
  "operation": "query|ingest|validate|supersede|archive|maintain|graph_upsert|graph_delete",
  "scope": {"wing": "string|null", "room": "string|null", "hall": "string|null"},
  "query": {},
  "record": {},
  "graph": {},
  "maintenance": {},
  "response": {"mode": "compact|evidence|graph", "debug": false, "include_candidates": false}
}
```

Populate only the sub-objects required by the selected `operation`.

## 2) Operation matrix

| Operation | Required payload |
| --- | --- |
| `query` | `query.text` |
| `ingest` | `record.format=raw` + `record.content` OR `record.format=structured` + `record.memories` |
| `validate` | `record.ids` |
| `supersede` | `record.ids` + `record.superseded_by` |
| `archive` | `record.ids` |
| `maintain` | optional `maintenance.mode` (default: `community_refresh`) |
| `graph_upsert` | `graph.kind=place` with `graph.place_name/place_type` OR `graph.kind=link` with `graph.from/graph.to/graph.link_type` |
| `graph_delete` | `graph.kind` + `graph.ids` |

## 3) Query behavior and scoping

`operation=query` supports:

- `query.mode=search`
- `query.mode=context`
- `query.mode=graph`

Scoping/filter controls apply consistently, including context mode:

- `scope.wing`, `scope.room`, `scope.hall`
- `query.source`, `query.categories`, `query.as_of`, `query.from`, `query.to`, `query.precision`
- `query.limits.tokens` for context budget

Temporal semantics:

- Use `query.as_of` for point-in-time lookup.
- Use `query.from`/`query.to` for interval overlap lookup.
- `query.as_of` cannot be combined with `query.from`/`query.to`.
- `query.mode=graph` supports `query.as_of` only; `query.from`/`query.to` are rejected.
- Archived records are excluded by default query behavior.

Graph extras (`query.mode=graph`):

- `query.graph.op = traverse|neighbors|path|stats`
- `query.graph.start`, `query.graph.end`, `query.graph.relation_types`
- `query.limits.hops`, `query.limits.nodes`

## 4) Category governance (ingest)

- `record.categories` must resolve to existing categories by default.
- Unknown categories fail ingest unless explicitly opted in.
- Use `record.allow_create_categories=true` to create missing categories intentionally.

## 5) Topology-scoped entity uniqueness (graph)

- Graph entity uniqueness is scoped by topology (`wing/room/hall`) plus entity type and name.
- The same entity name/type can exist in different topology scopes without collision.
- Name-based linking should use matching scope; prefer UUID references when names are ambiguous.

## 6) Migration and startup posture

- Memory schema compatibility is validated during server startup.
- Incompatible schema epochs fail fast.
- Apply documented migrations and restart the MCP server.
- No runtime destructive reset fallback is part of the supported behavior.

## 7) Lifecycle semantics and strictness

- `supersede` requires `record.superseded_by`.
- `archive` maps to forget semantics.
- Repeating `archive` on already archived records is safe/idempotent.
- Request payloads are strict (`extra=forbid`): unknown fields are rejected.

## 8) Temporal fields on ingest (raw)

- `operation=ingest` with `record.format=raw` supports `record.valid_from` and `record.valid_to`.
- Ordering is validated (`valid_from <= valid_to`).

## 9) Graph semantics

- `graph_upsert` with `graph.kind=link` is idempotent.
- `graph_delete` with `graph.kind=place` returns cascaded `deleted_relation_count`.

## 10) Direct-call JSON examples (valid and invalid)

Scoped context assembly:

```json
{
  "operation": "query",
  "scope": {"wing": "workflows", "room": "memory-engine"},
  "query": {
    "text": "Summarize current maintenance behavior",
    "mode": "context",
    "categories": ["memory"],
    "limits": {"tokens": 2000}
  }
}
```

Valid: ingest raw with temporal bounds:

```json
{
  "operation": "ingest",
  "scope": {"wing": "payments-service", "room": "ledger"},
  "record": {
    "format": "raw",
    "content": "Final RCA approved.",
    "valid_from": "2026-04-01T00:00:00Z",
    "valid_to": "2026-04-30T00:00:00Z"
  }
}
```

Valid: idempotent link upsert:

```json
{
  "operation": "graph_upsert",
  "scope": {"wing": "workflows", "room": "memory-engine"},
  "graph": {
    "kind": "link",
    "from": "11111111-1111-1111-1111-111111111111",
    "to": "22222222-2222-2222-2222-222222222222",
    "link_type": "depends_on"
  }
}
```

Valid: delete place request (response includes cascaded `deleted_relation_count`):

```json
{
  "operation": "graph_delete",
  "scope": {"wing": "workflows", "room": "memory-engine"},
  "graph": {
    "kind": "place",
    "ids": ["33333333-3333-3333-3333-333333333333"]
  }
}
```

Response excerpt:

```json
{
  "deleted_place_count": 1,
  "deleted_relation_count": 3
}
```

Invalid: mutually exclusive temporal query fields:

```json
{
  "operation": "query",
  "query": {
    "mode": "search",
    "text": "maintenance",
    "as_of": "2026-04-20T00:00:00Z",
    "from": "2026-04-01T00:00:00Z",
    "to": "2026-04-20T00:00:00Z"
  }
}
```

Invalid: graph query rejects `from`/`to`:

```json
{
  "operation": "query",
  "query": {
    "mode": "graph",
    "text": "service graph",
    "from": "2026-04-01T00:00:00Z",
    "to": "2026-04-20T00:00:00Z"
  }
}
```

Invalid: unknown/extra field rejected:

```json
{
  "operation": "query",
  "query": {
    "mode": "search",
    "text": "status",
    "unexpected": true
  }
}
```

Invalid: supersede missing required `record.superseded_by`:

```json
{
  "operation": "supersede",
  "record": {
    "ids": ["11111111-1111-1111-1111-111111111111"]
  }
}
```
