# Memory Tools Cheatsheet (Current Memory Contract)

Practical guide for the current public memory contract.

Active MCP tools:

- `memory` (stable unified tool)
- `onboard`
- `sync`

Legacy tool names are not part of the active public contract.

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
  "response": {"debug": false, "include_candidates": false}
}
```

Populate only the sections required by the selected `operation`.

## 2) Taxonomy and scope key rules

- Current memory topology keys are: `palace`, `wing`, `room`, `compartment`.
- Legacy `hall` is rejected with `MEM_INVALID_TAXONOMY_KEY`.
- Unknown scope keys are rejected.

## 3) Locality and scope

Each operation has its own locality contract. Providing less topology than the operation requires fails closed with `INSUFFICIENT_LOCALITY`, which includes the missing fields and retry guidance.

### Topology keys

Current memory topology keys, from broadest to narrowest: `palace`, `wing`, `room`, `compartment`.

- Legacy key `hall` is rejected with `MEM_INVALID_TAXONOMY_KEY`.
- The key `corridor` is rejected (`MEM_INVALID_TAXONOMY_KEY`) because it is an internal topology term.
- Unknown scope keys are rejected.

### Reference precedence

Scope fields are resolved field-by-field in this order:

1. explicit `scope` in the request
2. `scope_token` lookup from execution context (`memory_scope_tokens`)
3. `context_id` lookup from execution context (`memory_context_scopes`)

The response includes `resolved_scope` (effective merged values) and `scope_source` (origin per field: `request|token|context|active_context`). The `active_context` origin appears only for `query` when the field is filled from session active context fallback.

### Direct `memory()` active context

When using `memory()` directly, active project defaults and session active context apply as a fallback for `query` only. Standalone placement writes (`ingest`, `graph_upsert` with `kind=place`) do not use session active context as a fallback; they require an explicit locality reference.

### Operation locality matrix

| Operation | Locality requirement |
| --- | --- |
| `query` | `palace` minimum; `wing`, `room`, `compartment` narrow the search when present |
| `ingest` | complete topology (`palace/wing/room/compartment`) from explicit `scope`, `scope_token`, `context_id`, or operation-local `onboard`/`sync` data |
| `validate` | no topology required (ID-targeted) |
| `supersede` | no topology required; `record.superseded_by` (replacement ID) is required |
| `archive` | no topology required (ID-targeted) |
| `maintain` | current modes are global; no topology required |
| `graph_upsert` (`kind=place`) | complete topology required |
| `graph_upsert` (`kind=link`) with UUID refs | no topology required |
| `graph_upsert` (`kind=link`) with name refs | complete topology required for disambiguation |
| `graph_delete` with stable IDs | no topology required |
| `onboard` | `palace` minimum; derives lower topology from scanned structure |
| `sync` | uses session active context as continuation; supply narrowing `scope` to disambiguate multiple contexts |

> **Note:** For `ingest` and `graph_upsert` (`kind=place`), the `onboard`/`sync` locality source refers to operation-local checkpoint or continuation state produced by those tools — not the session active context fallback that applies to `query`. Standalone placement writes always require an explicit locality reference.

### Insufficient locality

When required topology fields cannot be resolved, the operation fails with `INSUFFICIENT_LOCALITY`. The error response includes the missing fields and retry guidance. Retry options:

- Supply a complete `scope` object.
- Supply a `scope_token` that resolves the missing fields.
- Supply a `context_id` that resolves the missing fields.
- Run an `onboard` or `sync` checkpoint flow that carries operation-local scope.

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
  "response": {"debug": false}
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

## 7) Example payloads (Project onboard/sync tools)

### onboard

```json
{
  "scope": {"palace": "acme"},
  "ingestion": {"mode": "programmatic"},
  "scan": {
    "patterns": ["src/**/*.py"],
    "root": "/path/to/repo",
    "max_size_kb": 512,
    "respect_gitignore": true
  }
}
```

Onboarding notes:

- `onboard` is compact by default. Completed steps strip heavy `plan[].payload.memories[].content` blobs.
- Pass root-level `debug=true` for full internals (`results[]`, full completed result details, full plan payloads).
- Programmatic onboarding accepts minimal scope with `scope.palace` and derives lower topology from scanned structure.
- Graph onboarding success requires a structurally complete graph payload: all four node types (Palace, Wing, Room, Compartment) with `contains` corridors.
- Ingestion mode: `programmatic` (default) is hash-based and deterministic. `llm` mode is optional with strict defaults.
- Binary and unsupported files produce metadata-only compartments (path, size, mime-type); no content is ingested.

### sync

```json
{
  "scope": {"palace": "acme"},
  "scan": {
    "patterns": ["src/**/*.py"],
    "root": "/path/to/repo"
  }
}
```

`sync({})` context behavior:

- `sync({})` (empty call) resolves context from successful onboard contexts in the current server session.
- Returns `status: "UNCHANGED"` when context resolves but no scan config is stored for automatic delta.
- Returns `status: "NO_CONTEXT"` when no context exists; call `onboard` first.
- Returns `status: "AMBIGUOUS_CONTEXT"` when multiple contexts match; supply a narrowing `scope`.

Completed-checkpoint semantics:

- If the checkpoint is already complete, `sync` returns a fast-path response with:
  - `from_checkpoint: true`
  - a `note` saying results came from checkpoint cache (no re-execution).
- Pass root-level `debug=true` to expand diagnostic detail.

Scan root policy:

- `scan.root` is validated against a single scan root.
- Default scan root is `/`.
- Override with environment variable `WORKFLOWS_SCAN_ROOT=/your/root`.
- For hardened deployments, do not leave `WORKFLOWS_SCAN_ROOT` at `/`; use a repo/workspace-specific path.
- Effective knowledge file access is the intersection of `WORKFLOWS_SCAN_ROOT` and project `fs_root`, plus explicitly approved extra allowlist roots.

## 8) Common invalid payloads

Invalid: legacy key `hall`.

```json
{
  "operation": "query",
  "scope": {"palace": "acme", "wing": "svc", "room": "comp", "hall": "legacy"},
  "query": {"text": "incident", "mode": "search"}
}
```

Invalid: placement write without complete locality (fails with `INSUFFICIENT_LOCALITY`).

```json
{
  "operation": "ingest",
  "scope": {"palace": "acme"},
  "record": {"format": "raw", "content": "Memory direct", "memory_tier": "direct"}
}
```

This fails because `ingest` requires complete topology (`palace/wing/room/compartment`). The error response includes the missing fields (`wing`, `room`, `compartment`) and retry guidance. To resolve:

- Add the missing `wing`, `room`, and `compartment` fields to `scope`.
- Supply a `scope_token` or `context_id` that resolves the missing fields.
- Run an `onboard` or `sync` checkpoint flow first; the resulting operation-local scope carries the complete topology needed for placement writes.

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
