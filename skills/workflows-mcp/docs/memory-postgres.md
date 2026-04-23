# Memory + PostgreSQL setup

Memory is optional but recommended. Enable it when you want persistent, queryable knowledge across sessions.

## 1) PostgreSQL prerequisites

- PostgreSQL 13+
- Reachable from MCP server process
- Credentials with access to target DB
- If using auto-create (`MEMORY_DB_AUTO_CREATE=true`), user needs `CREATE DATABASE` on admin DB

## 2) Required environment variables

Add to MCP server `env`:

```json
{
  "MEMORY_DB_HOST": "localhost",
  "MEMORY_DB_PORT": "5432",
  "MEMORY_DB_NAME": "memory_db",
  "MEMORY_DB_USER": "postgres",
  "MEMORY_DB_PASSWORD": "your-password",
  "MEMORY_DB_AUTO_CREATE": "true",
  "MEMORY_DB_ADMIN_DATABASE": "postgres"
}
```

Restart MCP client after config changes.

## 3) Verify memory is enabled

- Server starts cleanly.
- `memory` appears in available tools.
- Run a test ingest:

```json
{
  "operation": "ingest",
  "scope": {
    "palace": "acme",
    "wing": "platform",
    "room": "onboarding",
    "compartment": "smoke"
  },
  "record": {
    "format": "raw",
    "content": "Memory is active",
    "memory_tier": "direct"
  }
}
```

- Then run a scoped query to confirm retrieval:

```json
{
  "operation": "query",
  "scope": {
    "palace": "acme",
    "wing": "platform",
    "room": "onboarding",
    "compartment": "smoke"
  },
  "query": {
    "mode": "search",
    "text": "Memory is active"
  }
}
```

Expected cue: query response includes at least one matching record/content item.

## 4) `onboard` vs `sync`

- Use `onboard` to start a project onboarding session from scan input.
- Use `sync` to re-scan and reconcile changes after onboarding.
- Successful `onboard` sets the active context for the current MCP server session.
- Use `sync({})` (empty call) to operate on that active context.
- Use `select(scope={...})` to switch active context when multiple projects are onboarded.

Decision rule:
- Not onboarded yet -> `onboard`
- Context already onboarded and active -> `sync`
- Multiple onboarded contexts in same session -> `select(scope={...})`, then `sync({})`

### Example: start onboarding

```json
{
  "scope": {
    "palace": "acme"
  },
  "ingestion": {
    "mode": "programmatic"
  },
  "scan": {
    "patterns": ["src/**/*.py"],
    "root": "/path/to/repo",
    "max_files": 200,
    "max_size_kb": 512,
    "respect_gitignore": true
  }
}
```

Graph onboarding success criteria:

- The graph payload must include all four node types (Palace, Wing, Room, Compartment) with `contains` corridors connecting them.
- Ingestion mode: `programmatic` (default) is hash-based and deterministic. `llm` mode is optional with strict defaults.
- Binary and unsupported files produce metadata-only compartments (path, size, mime-type); no content is ingested.

### Example: sync current context

```json
{
  "scope": {"palace": "acme"},
  "scan": {
    "patterns": ["src/**/*.py"],
    "root": "/path/to/repo"
  }
}
```

### Compact/default vs debug responses

- `onboard` and `sync` return compact responses by default.
- Compact mode strips heavy payload internals for completed steps (for example large `plan[].payload.memories[].content` blobs).
- To inspect full internals for troubleshooting, pass root-level `debug=true`:

```json
{
  "debug": true
}
```

- If `sync` receives an already completed checkpoint, it returns a fast-path completed response with:
  - `from_checkpoint: true`
  - a note indicating results are from checkpoint cache and no operations were re-executed.

### `sync({})` context behavior

- `sync({})` resolves context from the active session context (set by `onboard` or `select`) without requiring an explicit checkpoint.
- Returns `status: "UNCHANGED"` when context resolves but no scan config is stored for automatic delta.
- Returns error code `MEM_NO_ACTIVE_CONTEXT` when no active context exists — call `onboard` or `select`, or pass explicit `scope`.
- Returns error code `AMBIGUOUS_CONTEXT` when context resolution is ambiguous — narrow with explicit `scope`.

### `memory(...)` with no explicit scope

- `memory` queries can omit `scope` and use the active session context.
- If no active context exists, the tool returns `MEM_NO_ACTIVE_CONTEXT` with an actionable fix.
- Explicit `scope` always overrides the active context for that call.

### Scan root policy

- `scan.root` must be inside the active scan root.
- Default scan root is `/`.
- Optional override: `WORKFLOWS_SCAN_ROOT=/your/root/path`.

### Scope prerequisite for onboarding

- `onboard` programmatic mode accepts minimal scope (`scope.palace`).
- Lower topology (`wing`/`room`/`compartment`) is derived deterministically from scanned structure.
- Direct `memory(operation="ingest")` still requires fully-resolved scope including `compartment`.

## 5) Memory usage patterns

### Ingest (store)

Use `operation="ingest"` with full scope (`palace/wing/room/compartment`) and `memory_tier="direct"`.

### Query (retrieve)

Use `operation="query"` + `query.mode`:
- `search` for scoped retrieval,
- `context` for prompt-ready context,
- `graph` for graph traversal/stats,
- `communities` for community-focused retrieval.

### Maintain/lifecycle

- `validate`: validate existing records
- `supersede`: mark old records superseded (`record.superseded_by` required)
- `archive`: close records without deletion
- `maintain`: run maintenance (`community_refresh`)

### Graph operations

- `graph_upsert` with `kind="place"` (scoped)
- `graph_upsert` with `kind="link"` (idempotent link upsert)
- `graph_delete` for place/link delete by ids

## Pitfalls

- Using legacy taxonomy keys such as `hall` (rejected).
- Missing `compartment` for direct ingest.
- Passing `memory_tier="derived"` to `ingest` (boundary violation).
- Combining `query.as_of` with `query.from/query.to` in one request.

## Troubleshooting (quick)

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `memory` tool is missing | Memory env vars are not set in MCP server config | Add values from `snippets/memory-env.json`, restart client, then re-check tool list. |
| Auth/connection error to Postgres | Host/port/user/password/db mismatch or network issue | Verify credentials and connectivity from server runtime; correct env vars and restart client. |
| `sync` rejects checkpoint | Invalid or stale checkpoint shape/version | Re-run `onboard` to create fresh context, then continue with `sync` (or pass a fresh checkpoint explicitly). |
| Ingest succeeds but query returns nothing | Scope mismatch between ingest and query | Reuse the exact same `scope` fields (`palace/wing/room/compartment`) in both calls. |
