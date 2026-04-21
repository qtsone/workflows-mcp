# Memory + PostgreSQL setup

Memory is optional. Enable it when you want persistent, queryable knowledge across sessions.

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

## 4) `project_onboard` vs `project_sync`

- Use `project_onboard` when you do **not** have a checkpoint yet.
- Use `project_sync` when you already have a checkpoint returned by `project_onboard` or a prior `project_sync` call.

Decision rule:
- No checkpoint in hand -> `project_onboard`
- Checkpoint in hand (`version: "oss-r2"`) -> `project_sync`

### Example: start onboarding

```json
{
  "scope": {
    "palace": "acme",
    "wing": "workflows",
    "room": "memory-engine",
    "compartment": "contract-r2"
  },
  "ingest": {
    "format": "raw",
    "content": "Initial baseline",
    "memory_tier": "direct"
  },
  "maintain": {"mode": "community_refresh"},
  "max_operations": 1
}
```

### Example: resume sync

```json
{
  "checkpoint": {
    "version": "oss-r2",
    "scope": {
      "palace": "acme",
      "wing": "workflows",
      "room": "memory-engine",
      "compartment": "contract-r2"
    },
    "plan": [
      {
        "operation": "ingest",
        "payload": {
          "format": "raw",
          "content": "Initial baseline",
          "memory_tier": "direct"
        }
      }
    ],
    "next_index": 0,
    "completed": []
  },
  "max_operations": 3
}
```

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
| `project_sync` rejects checkpoint | Invalid or stale checkpoint shape/version | Re-run `project_onboard` to create a fresh checkpoint, then continue with `project_sync`. |
| Ingest succeeds but query returns nothing | Scope mismatch between ingest and query | Reuse the exact same `scope` fields (`palace/wing/room/compartment`) in both calls. |
