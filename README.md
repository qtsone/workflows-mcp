# Workflows MCP

Run YAML workflows as MCP tools so agents can automate real tasks with one server.

## Why this project

`workflows-mcp` gives MCP clients a reusable automation layer:

- Define tasks once in YAML and run them from any MCP-compatible client.
- Orchestrate multi-step work with dependency-aware execution.
- Support interactive runs (`Prompt` blocks) with pause/resume.
- Keep secrets server-side via `WORKFLOW_SECRET_*` with redacted outputs.
- Run synchronous or async jobs with queue visibility and cancellation.

## Quickstart (5 minutes)

### 1) Install

Requires Python 3.12+.

```bash
uv pip install workflows-mcp
```

or:

```bash
pip install workflows-mcp
```

### 2) Add to your MCP client

Example (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": ["workflows-mcp"],
      "env": {
        "WORKFLOWS_TEMPLATE_PATHS": "/path/to/your/workflows",
        "WORKFLOWS_LOG_LEVEL": "INFO",
        "WORKFLOW_SECRET_API_KEY": "your-secret-value"
      }
    }
  }
}
```

If installed with `pip`:

```json
{
  "mcpServers": {
    "workflows": {
      "command": "workflows-mcp",
      "env": {
        "WORKFLOWS_TEMPLATE_PATHS": "/path/to/your/workflows",
        "WORKFLOWS_LOG_LEVEL": "INFO",
        "WORKFLOW_SECRET_API_KEY": "your-secret-value"
      }
    }
  }
}
```

### 3) Restart your MCP client and run first calls

1. `list_workflows`
2. `get_workflow_info`
3. `execute_workflow`


## Instructions for LLM Agents

Use this call order for reliable results:

1. **Discover**: call `list_workflows`.
2. **Inspect**: call `get_workflow_info` for required inputs.
3. **Execute**: call `execute_workflow`.
4. **Track async runs** (if `mode="async"`): call `get_job_status` (or `list_jobs`).
5. **Resume interactive workflows**: call `resume_workflow` only for `paused` jobs (typically from `Prompt` blocks).
6. **Reload definitions after YAML edits**: call `reload_workflows`.

When authoring workflows, validate first:

- `validate_workflow_yaml` before `execute_inline_workflow`.
- Use `get_workflow_schema` for current schema details.
- Use the block reference for exact field names and required inputs: `docs/llm/block-reference.md`.

Async mini-flow example:

```text
execute_workflow(workflow="python-ci-pipeline", inputs={...}, mode="async")
→ returns job_id
→ get_job_status(job_id="job_...") until completed/failed/paused
→ if paused, resume_workflow(job_id="job_...", response="...")
```

## Available MCP tools (catalog + call patterns)

### Workflow discovery and execution

| Tool | When to call | Typical call pattern |
| --- | --- | --- |
| `list_workflows` | List registered workflows | `list_workflows(tags=[], format="json")` |
| `get_workflow_info` | Exploratory to confirm inputs/outputs | `get_workflow_info(workflow="name", format="json")` |
| `execute_workflow` | Run a registered workflow | `execute_workflow(workflow="name", inputs={...}, mode="sync")` |
| `execute_inline_workflow` | Test one-off YAML without registering | `execute_inline_workflow(workflow_yaml="...", inputs={...})` |
| `reload_workflows` | After editing workflow YAML files on disk | `reload_workflows()` |

### Authoring and validation

| Tool | When to call | Typical call pattern |
| --- | --- | --- |
| `get_workflow_schema` | Debugging only. Retrieve full JSON schema for authoring | `get_workflow_schema()` |
| `validate_workflow_yaml` | Validate YAML before execution | `validate_workflow_yaml(yaml_content="...")` |

### Async, queue, and interactive control

| Tool | When to call | Typical call pattern |
| --- | --- | --- |
| `get_job_status` | Poll a specific async job | `get_job_status(job_id="job_...")` |
| `list_jobs` | Find jobs by status (especially paused) | `list_jobs(status="paused", limit=100)` |
| `cancel_job` | Stop queued/running jobs | `cancel_job(job_id="job_...")` |
| `get_queue_stats` | Monitor queue health/capacity | `get_queue_stats()` |
| `resume_workflow` | Continue paused `Prompt` workflows | `resume_workflow(job_id="job_...", response="...")` |

### Memory (conditional)

| Tool | When to call | Typical call pattern |
| --- | --- | --- |
| `memory` | Unified memory query/ingest/maintenance/graph operations | `memory(operation="query", scope={...}, query={...})` |
| `project_onboard` | Current memory onboarding flow with checkpoints | `project_onboard(scope={...}, ingest={...}, max_operations=1)` |
| `project_sync` | Current memory checkpoint resume/continuation | `project_sync(checkpoint={...}, max_operations=3)` |

IMPORTANT: The `memory` tool is registered only when memory DB setup is available and valid at startup (see [below](#memory))

Memory contract highlights:

- Unified envelope: `operation` + optional `scope/query/record/graph/maintenance/response`.
- Current memory taxonomy: `scope` accepts only `palace`, `wing`, `room`, `compartment`.
- Context activation and scope defaulting:
  - Resolution precedence is `scope` → `scope_token` → `context_id` (per-field merge).
  - `scope_token` resolves from execution context `memory_scope_tokens`; `context_id` resolves from `memory_context_scopes`.
  - Responses include `resolved_scope` and `scope_source` when available.
  - Required scope by operation:
    - `query`: all four fields must resolve.
    - `ingest`: all four fields must resolve (including `compartment`).
    - `graph_upsert` with `graph.kind="place"`: all four fields must resolve.
    - `validate|supersede|archive|maintain|graph_delete|graph_upsert(kind="link")`: scope is optional.
- Temporal query semantics:
  - `operation="query"` supports either `query.as_of` OR interval `query.from/query.to` (mutually exclusive).
  - `query.mode="graph"` supports `query.as_of` only; `query.from/query.to` are rejected.
  - `operation="ingest"` with `record.format="raw"` supports `record.valid_from` and `record.valid_to` with ordering validation (`valid_from <= valid_to`).
- Strict validation: unknown/extra fields are rejected.
- Direct vs derived boundaries:
  - `operation="ingest"` is direct-only (`record.memory_tier` must be `direct`).
  - Category governance for ingest is explicit:
    - Unknown `record.categories` fail deterministically with `MEM_UNKNOWN_CATEGORY` when `record.allow_create_categories=false` (default behavior).
    - Setting `record.allow_create_categories=true` opts into category creation and allows ingest to proceed when categories are otherwise valid.
  - Derived/community artifacts are produced by maintenance flows (for example `maintenance.mode="community_refresh"`).
  - `query.mode="communities"` uses the dedicated communities strategy.
- Lifecycle semantics:
  - Archived records are excluded by default query behavior.
  - `operation="supersede"` requires `record.superseded_by`.
  - `operation="archive"` maps to forget semantics; repeating archive on already archived records is safe/idempotent.
- Graph semantics:
  - `operation="graph_upsert"` with `graph.kind="link"` is idempotent.
  - `operation="graph_delete"` returns compact delete counters (`deleted_places` for `kind="place"`, `deleted_links` for `kind="link"`); optional debug output adds diagnostics metadata.

Validation note: this ingest category behavior (`MEM_UNKNOWN_CATEGORY` with create disabled, successful ingest with `allow_create_categories=true`) was live-validated on 2026-04-21 via production-like direct MCP `memory` calls.

Direct-call JSON examples:

`query`:

```json
{
  "operation": "query",
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory-engine", "compartment": "contract-r2"},
  "query": {"mode": "search", "text": "schema epoch", "as_of": "2026-04-20T00:00:00Z", "radius": 1}
}
```

`ingest`:

```json
{
  "operation": "ingest",
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory-engine", "compartment": "contract-r2"},
  "record": {"format": "raw", "content": "Memory active", "memory_tier": "direct"}
}
```

`validate`:

```json
{
  "operation": "validate",
  "record": {"ids": ["11111111-1111-1111-1111-111111111111"]}
}
```

`supersede`:

```json
{
  "operation": "supersede",
  "record": {
    "ids": ["11111111-1111-1111-1111-111111111111"],
    "superseded_by": "22222222-2222-2222-2222-222222222222"
  }
}
```

`archive`:

```json
{
  "operation": "archive",
  "record": {"ids": ["11111111-1111-1111-1111-111111111111"]}
}
```

`maintain`:

```json
{
  "operation": "maintain",
  "maintenance": {"mode": "community_refresh"}
}
```

`graph_upsert` (`kind=place`):

```json
{
  "operation": "graph_upsert",
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory-engine", "compartment": "contract-r2"},
  "graph": {"kind": "place", "place_name": "Memory API (current)", "place_type": "feature"}
}
```

`graph_upsert` (`kind=link`):

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

`graph_delete`:

```json
{
  "operation": "graph_delete",
  "graph": {"kind": "place", "ids": ["33333333-3333-3333-3333-333333333333"]}
}
```

`project_onboard`:

```json
{
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory-engine", "compartment": "contract-r2"},
  "ingest": {"format": "raw", "content": "Initial baseline", "memory_tier": "direct"},
  "supersede": {"ids": ["11111111-1111-1111-1111-111111111111"], "superseded_by": "22222222-2222-2222-2222-222222222222"},
  "archive": {"ids": ["33333333-3333-3333-3333-333333333333"]},
  "maintain": {"mode": "community_refresh"},
  "max_operations": 1
}
```

`project_sync`:

```json
{
  "checkpoint": {
    "version": "oss-r2",
    "scope": {"palace": "acme", "wing": "workflows", "room": "memory-engine", "compartment": "contract-r2"},
    "plan": [{"operation": "ingest", "payload": {"format": "raw", "content": "Initial baseline", "memory_tier": "direct"}}],
    "next_index": 0,
    "completed": []
  },
  "max_operations": 3
}
```

Invalid (mutually exclusive temporal filters):

```json
{
  "operation": "query",
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory-engine", "compartment": "contract-r2"},
  "query": {
    "mode": "search",
    "text": "maintenance",
    "as_of": "2026-04-20T00:00:00Z",
    "from": "2026-04-01T00:00:00Z",
    "to": "2026-04-20T00:00:00Z"
  }
}
```

Invalid (graph query rejects interval filters):

```json
{
  "operation": "query",
  "scope": {"palace": "acme", "wing": "workflows", "room": "memory-engine", "compartment": "contract-r2"},
  "query": {
    "mode": "graph",
    "text": "service graph",
    "from": "2026-04-01T00:00:00Z",
    "to": "2026-04-20T00:00:00Z"
  }
}
```

Invalid (legacy taxonomy key):

```json
{
  "operation": "query",
  "scope": {"palace": "acme", "wing": "svc", "room": "component", "hall": "legacy"},
  "query": {"text": "find this", "mode": "search"}
}
```

Invalid (supersede missing required `superseded_by`):

```json
{
  "operation": "supersede",
  "record": {
    "ids": ["11111111-1111-1111-1111-111111111111"]
  }
}
```

## Configuration

### Workflow loading and execution

- `WORKFLOWS_TEMPLATE_PATHS`: Comma-separated workflow directories to load.
- `WORKFLOWS_MAX_RECURSION_DEPTH`: Max workflow composition depth (default: `50`).
- `WORKFLOWS_LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`; default: `INFO`).

### Queue and async settings

- `WORKFLOWS_IO_QUEUE_ENABLED`: Enable serialized I/O queue (default: `true`).
- `WORKFLOWS_JOB_QUEUE_ENABLED`: Enable async job queue (default: `true`).
- `WORKFLOWS_JOB_QUEUE_WORKERS`: Queue workers (default: `3`).
- `WORKFLOWS_MAX_CONCURRENT_JOBS`: Max active + queued jobs (default: `500`).
- `WORKFLOWS_JOB_TIMEOUT`: Default async timeout in seconds (default: `3600`).
- `WORKFLOWS_JOB_HISTORY_MAX`: Max retained job records (default: `1000`).
- `WORKFLOWS_JOB_HISTORY_TTL`: Job retention TTL in seconds (default: `86400`).

### Secrets and LLM config

- `WORKFLOW_SECRET_<NAME>`: Secret value exposed as `{{secrets.NAME}}`.
- `WORKFLOWS_LLM_CONFIG`: Optional path override for LLM config.

## Memory

Memory is an optional persistent storage feature that lets agents and workflows record, retrieve, and organize information across sessions. It is backed by PostgreSQL and gives each agent a structured, scoped, and temporally-aware knowledge store.

### What memory provides

- **`memory` MCP tool** — direct call interface for LLM agents to store and query information without writing workflow YAML.
- **`project_onboard` / `project_sync` MCP tools** — Current memory contract helpers for checkpointed onboarding/sync sequences.
- **`Memory` workflow block** — use inside YAML workflows to automate memory operations as part of larger pipelines.
- **Memory topology scoping** (`palace` → `wing` → `room` → `compartment`) — current memory scope keys are strict and legacy keys (for example `hall`) are rejected. Scope can be supplied directly and/or resolved from context (`scope_token`, `context_id`) using deterministic precedence.
- **Temporal tracking** — records carry `valid_from` / `valid_to` timestamps supporting point-in-time and interval queries.
- **Knowledge graph** — link memories to places, entities, or concepts and query the resulting graph.
- **Lifecycle management** — archive or supersede records without deletion; archived records are excluded from default queries.

### Retrieval strategies

Current memory behavior exposes query modes that map to retrieval strategies internally:

| Query mode / trigger | Effective strategy | Behavior |
| --- | --- | --- |
| `query.mode="search"` + `radius=0` | `palace` | Strict scoped retrieval lane (no companion lane). |
| `query.mode="search"` + `radius>=1` | `auto` | Scoped lane + optional S2 companion lane (`s2_enabled=true` by default). |
| `query.mode="hybrid"` | `auto` | Same retrieval family as `auto` with fused ranking. |
| `query.mode="context"` | `context` | Context assembly retrieval path. |
| `query.mode="graph"` | `graph` | Graph traversal/path/stats retrieval. |
| `query.mode="communities"` | `communities` | Community-focused retrieval strategy. |

Every `query` call still requires a fully resolved current memory scope (`palace/wing/room/compartment`) resolved from request and/or context sources.

#### Scope call shape (with context activation)

Scope fields can be passed directly and/or resolved from `scope_token` / `context_id`:

```json
{
  "operation": "query",
  "scope": {"palace": "acme", "wing": "my-service"},
  "scope_token": "st_auth",
  "context_id": "ctx_default",
  "query": {"text": "refresh token lifetime", "mode": "context"}
}
```

Resolution precedence is `scope` → `scope_token` → `context_id` for each field.

### Prerequisites

- PostgreSQL 13 or later, network-accessible from the server process.
- A database user with `CREATE DATABASE` rights on the admin database (typically `postgres`) if auto-create is enabled, **or** a pre-existing database that the user can connect to.

### Configuration

| Variable | Default | Required | Description |
| --- | --- | --- | --- |
| `MEMORY_DB_HOST` | — | Yes | PostgreSQL hostname or IP. Setting this variable enables memory features. |
| `MEMORY_DB_PORT` | `5432` | No | PostgreSQL port. |
| `MEMORY_DB_NAME` | `memory_db` | No | Target database name. |
| `MEMORY_DB_USER` | — | Yes | Database username. |
| `MEMORY_DB_PASSWORD` | — | Yes | Database password. |
| `MEMORY_DB_AUTO_CREATE` | `true` | No | Auto-create the target database on first boot if it does not exist. Requires the user to have `CREATE DATABASE` rights on the admin database. |
| `MEMORY_DB_ADMIN_DATABASE` | `postgres` | No | Admin database used to issue the `CREATE DATABASE` statement when auto-create is enabled. |
| `AUDIT_FAIL_CLOSED` | `false` | No | When `true`, audit-logging failures abort the entire memory operation (compliance mode). Default is log-and-continue. |

### Enabling memory

Add the following variables to your MCP client config:

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": ["workflows-mcp"],
      "env": {
        "MEMORY_DB_HOST": "localhost",
        "MEMORY_DB_PORT": "5432",
        "MEMORY_DB_NAME": "memory_db",
        "MEMORY_DB_USER": "postgres",
        "MEMORY_DB_PASSWORD": "your-password"
      }
    }
  }
}
```

On first boot with `MEMORY_DB_AUTO_CREATE=true` (the default), the server creates the target database and applies the schema automatically. Restart your MCP client after adding the variables.

### Verifying memory is active

1. Restart your MCP client.
2. Call `list_workflows` — confirm the server started without errors in the client logs.
3. Check that the `memory` tool appears in the available tool list.
4. Run a test ingest:

```json
{
  "operation": "ingest",
  "scope": {"palace": "acme", "wing": "test", "room": "setup", "compartment": "smoke"},
  "record": {"format": "raw", "content": "Memory is working.", "memory_tier": "direct"}
}
```

If the tool is absent, check server logs for `MEMORY_DB_*` startup errors — the most common causes are an unreachable host, incorrect credentials, or a schema epoch mismatch.

### Schema compatibility

Memory schema versions are tracked by epoch. If the epoch in the database does not match the server version:

- Startup fails with an explicit error.
- Apply the documented migration and restart.
- No automatic destructive reset is performed. Set `MEMORY_SCHEMA_RESET_MODE` only if you intend a one-time destructive reset on a non-production instance.

---

## Workflows for users

Use registered workflows for repeatable automation, and inline workflows for experiments.

- **Registered workflows**: best for shared, reusable operations.
- **Inline workflows**: best for quick tests and prototyping.

Author workflow YAML with exact block input names from:

- `docs/llm/block-reference.md`

Example block families include `Shell`, `ReadFiles`, `HttpCall`, `LLMCall`, `Sql`, `Workflow`, `Prompt`, and `Memory`.

## Documentation map

- `README.md`: install, usage, and tool catalog.
- `docs/guides/memory-tools-cheatsheet.md`: Current memory contract quick reference, detailed guide, and examples.
- `docs/llm/block-reference.md`: exact block inputs/outputs for workflow authoring.
- `docs/TESTING.md`: test strategy and test commands.
- `ARCHITECTURE.md`: architecture overview.
- `docs/adr/`: design decisions and rationale.
- `CHANGELOG.md`: release history.

## Contributing

1. Fork the repository and create a focused branch.
2. Add or update tests with your change.
3. Run quality checks before opening a PR:

```bash
uv run pytest
uv run ruff check src/workflows_mcp/
uv run mypy src/workflows_mcp/
```

4. Describe behavior changes and config impact clearly in the PR.

## Support and community

- Issues and bug reports: https://github.com/qtsone/workflows-mcp/issues
- Project repository: https://github.com/qtsone/workflows-mcp

## License

AGPL-3.0-or-later. See [LICENSE](./LICENSE).
