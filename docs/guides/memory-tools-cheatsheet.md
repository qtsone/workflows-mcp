# Memory Tools Cheatsheet (Current Memory Contract)

Practical guide for the current public memory contract.

Active MCP tools:

- `memory` (stable unified tool)
- `project_onboard`
- `project_sync`

Legacy tool names (`query_memory`, `manage_memory`) are not part of the active public contract.

## 1) Canonical request envelope (`memory`)

```json
{
  "operation": "query|ingest|validate|supersede|archive|maintain|graph_upsert|graph_delete",
  "scope": {
    "palace": "string|null",
    "wing": "string|null",
    "room": "string|null",
    "compartment": "string|null"
  },
  "scope_token": "string|null",
  "context_id": "string|null",
  "query": {},
  "record": {},
  "graph": {},
  "maintenance": {},
  "response": {"mode": "compact|evidence|graph", "debug": false, "include_candidates": false}
}
```

Populate only the sections required by the selected `operation`.

## 2) Taxonomy and scope key rules

- Current memory topology keys are: `palace`, `wing`, `room`, `compartment`.
- Legacy `hall` is rejected with `MEM_INVALID_TAXONOMY_KEY`.
- Unknown scope keys are rejected.

## 3) Context activation and scope defaulting

Scope resolution precedence is deterministic, field by field:

1. explicit `scope` in request
2. `scope_token` lookup from execution context (`memory_scope_tokens`)
3. `context_id` lookup from execution context (`memory_context_scopes`)

The tool returns:

- `resolved_scope` (effective values after merge)
- `scope_source` (`request|token|context` per field)

### Scope requirements by operation

| Operation | Scope requirement |
| --- | --- |
| `query` | all four fields must resolve (`palace/wing/room/compartment`) |
| `ingest` | all four fields must resolve (includes required direct-ingest `compartment`) |
| `graph_upsert` + `graph.kind=place` | all four fields must resolve |
| `validate`, `supersede`, `archive`, `maintain`, `graph_delete`, `graph_upsert` + `graph.kind=link` | no scope required |

If a required scope field cannot be resolved from `scope`/`scope_token`/`context_id`, request fails with `SCOPE_UNRESOLVED`, except direct ingest missing `compartment`, which returns `COMPARTMENT_REQUIRED`.

## 4) Direct vs derived memory rules and community semantics

- `ingest` is a direct-memory boundary: `record.memory_tier` must be `direct`.
- `record.memory_tier="derived"` is rejected for ingest (`MEM_BOUNDARY_VIOLATION`).
- Category governance for ingest is explicit:
  - Unknown `record.categories` fail deterministically with `MEM_UNKNOWN_CATEGORY` when `record.allow_create_categories=false`.
  - Set `record.allow_create_categories=true` to explicitly allow category creation and let ingest proceed when categories are otherwise valid.
- Derived community memories are created by maintenance flows (`operation=maintain`, mode `community_refresh`).
- `query.mode="communities"` maps to community retrieval strategy.
- `maintain` in `community_refresh` mode returns `communities_updated` in compact output; diagnostics are available when `response.debug=true` (and in mode-specific shapes such as `graph`).

Validation note (2026-04-21): this category behavior was live-validated via production-like direct MCP `memory` calls.

## 5) Operation matrix and minimum payloads

| Operation | Required section | Minimum payload |
| --- | --- | --- |
| `query` | `query` | `query.text` |
| `ingest` | `record` | `record.format=raw` + `record.content` (or structured payload) |
| `validate` | `record` | `record.ids` |
| `supersede` | `record` | `record.ids` + `record.superseded_by` |
| `archive` | `record` | `record.ids` |
| `maintain` | none (defaults apply) | optional `maintenance.mode` (default `community_refresh`) |
| `graph_upsert` | `graph` | `graph.kind=place` or `graph.kind=link` |
| `graph_delete` | `graph` | `graph.kind` + `graph.ids` |

## 6) Example payloads (`memory`) for all operations

### query

```json
{
  "operation": "query",
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory", "compartment": "contract-r2"},
  "query": {"text": "scope precedence", "mode": "search", "radius": 1}
}
```

### ingest

```json
{
  "operation": "ingest",
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory", "compartment": "contract-r2"},
  "record": {"format": "raw", "content": "Current memory contract enabled", "memory_tier": "direct"}
}
```

### validate

```json
{
  "operation": "validate",
  "record": {"ids": ["11111111-1111-1111-1111-111111111111"]}
}
```

### supersede

```json
{
  "operation": "supersede",
  "record": {
    "ids": ["11111111-1111-1111-1111-111111111111"],
    "superseded_by": "22222222-2222-2222-2222-222222222222",
    "reason": "Replaced by corrected incident summary"
  }
}
```

### archive

```json
{
  "operation": "archive",
  "record": {"ids": ["11111111-1111-1111-1111-111111111111"], "reason": "No longer relevant"}
}
```

### maintain

```json
{
  "operation": "maintain",
  "maintenance": {"mode": "community_refresh"},
  "response": {"mode": "compact", "debug": false}
}
```

### graph_upsert (place)

```json
{
  "operation": "graph_upsert",
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory", "compartment": "contract-r2"},
  "graph": {"kind": "place", "place_name": "Memory API (current)", "place_type": "feature"}
}
```

### graph_upsert (link)

```json
{
  "operation": "graph_upsert",
  "graph": {
    "kind": "link",
    "from": "11111111-1111-1111-1111-111111111111",
    "to": "22222222-2222-2222-2222-222222222222",
    "link_type": "depends_on"
  }
}
```

### graph_delete

```json
{
  "operation": "graph_delete",
  "graph": {"kind": "place", "ids": ["33333333-3333-3333-3333-333333333333"]}
}
```

## 7) Example payloads (Current memory project tools)

### project_onboard

```json
{
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory", "compartment": "contract-r2"},
  "ingest": {"format": "raw", "content": "Initial baseline", "memory_tier": "direct"},
  "supersede": {"ids": ["11111111-1111-1111-1111-111111111111"], "superseded_by": "22222222-2222-2222-2222-222222222222"},
  "archive": {"ids": ["33333333-3333-3333-3333-333333333333"]},
  "maintain": {"mode": "community_refresh"},
  "max_operations": 1
}
```

### project_sync (resume from checkpoint)

```json
{
  "checkpoint": {
    "version": "oss-r2",
    "scope": {"palace": "acme", "wing": "workflows", "room": "memory", "compartment": "contract-r2"},
    "plan": [{"operation": "ingest", "payload": {"format": "raw", "content": "Initial baseline", "memory_tier": "direct"}}],
    "next_index": 0,
    "completed": []
  },
  "max_operations": 3
}
```

## 8) Common invalid payloads

Invalid: legacy key `hall`.

```json
{
  "operation": "query",
  "scope": {"palace": "acme", "wing": "svc", "room": "comp", "hall": "legacy"},
  "query": {"text": "incident", "mode": "search"}
}
```

## 9) Additional query and lifecycle controls

- `query.mode` supports `search`, `context`, `graph`, and `communities`.
- Retrieval tuning:
  - `query.radius` controls scope expansion distance (`0` = exact container only).
  - `query.precision` controls semantic strictness (`0.0-1.0`, higher = stricter).
- Temporal controls:
  - Point-in-time: `query.as_of`
  - Interval overlap: `query.from`, `query.to` (either bound may be omitted)
  - `query.as_of` cannot be combined with `query.from`/`query.to`
- Graph query extras (`query.mode=graph`):
  - `query.graph.op = traverse|neighbors|path|stats`
  - `query.graph.start`, `query.graph.end`, `query.graph.relation_types`
  - `query.limits.hops`, `query.limits.nodes`
- Context query behavior (`query.mode=context`):
  - Uses the same scoping/filter controls as search mode.
  - Returns prompt-ready text in `query.context`, with token budgeting from `query.limits.tokens`.
- Lifecycle close semantics:
  - `supersede` and `archive` set `record.valid_to=NOW()` when `record.valid_to` is omitted.
  - If `record.valid_to` is provided, that explicit timestamp is honored.
- Strict request validation:
  - Unknown fields are rejected (`extra=forbid`) across the request envelope.
- Topology-scoped graph uniqueness:
  - Entities are unique per resolved topology scope + entity type + name.
  - The same entity name can exist in different scopes without collision.

Invalid: ingest with non-direct tier.

```json
{
  "operation": "ingest",
  "scope": {"palace": "acme", "wing": "svc", "room": "comp", "compartment": "topic"},
  "record": {"format": "raw", "content": "derived sample", "memory_tier": "derived"}
}
```

Invalid: mutually-exclusive temporal query fields.

```json
{
  "operation": "query",
  "scope": {"palace": "acme", "wing": "svc", "room": "comp", "compartment": "topic"},
  "query": {
    "mode": "search",
    "text": "maintenance",
    "as_of": "2026-04-20T00:00:00Z",
    "from": "2026-04-01T00:00:00Z",
    "to": "2026-04-20T00:00:00Z"
  }
}
```
