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
| `list_workflows` | First step in most sessions | `list_workflows(tags=[], format="json")` |
| `get_workflow_info` | Before execution to confirm inputs/outputs | `get_workflow_info(workflow="name", format="json")` |
| `execute_workflow` | Run a registered workflow | `execute_workflow(workflow="name", inputs={...}, mode="sync")` |
| `execute_inline_workflow` | Test one-off YAML without registering | `execute_inline_workflow(workflow_yaml="...", inputs={...})` |
| `reload_workflows` | After editing YAML files on disk | `reload_workflows()` |

### Authoring and validation

| Tool | When to call | Typical call pattern |
| --- | --- | --- |
| `get_workflow_schema` | Need full JSON schema for authoring | `get_workflow_schema()` |
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

The `memory` tool is registered only when memory DB setup is available and valid at startup.

Memory contract highlights:

- Unified envelope: `operation` + optional `scope/query/record/graph/maintenance/response`.
- Temporal query semantics:
  - `operation="query"` supports either `query.as_of` OR interval `query.from/query.to` (mutually exclusive).
  - `query.mode="graph"` supports `query.as_of` only; `query.from/query.to` are rejected.
  - `operation="ingest"` with `record.format="raw"` supports `record.valid_from` and `record.valid_to` with ordering validation (`valid_from <= valid_to`).
- Strict validation: unknown/extra fields are rejected.
- Lifecycle semantics:
  - Archived records are excluded by default query behavior.
  - `operation="supersede"` requires `record.superseded_by`.
  - `operation="archive"` maps to forget semantics; repeating archive on already archived records is safe/idempotent.
- Graph semantics:
  - `operation="graph_upsert"` with `graph.kind="link"` is idempotent.
  - `operation="graph_delete"` with `graph.kind="place"` includes cascaded `deleted_relation_count` in the response.

Direct-call JSON examples:

Valid (point-in-time query):

```json
{
  "operation": "query",
  "scope": {"wing": "workflows", "room": "memory-engine"},
  "query": {"mode": "search", "text": "schema epoch", "as_of": "2026-04-20T00:00:00Z"}
}
```

Invalid (mutually exclusive temporal filters):

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

Invalid (graph query rejects interval filters):

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
- **`Memory` workflow block** — use inside YAML workflows to automate memory operations as part of larger pipelines.
- **Three-level memory palace scoping** (`wing` → `room` → `hall`) — a strict containment hierarchy: `wing` is a service/project, `room` is a component/module inside that wing, and `hall` is a topic lane inside that room. All three levels are optional and independently filterable. Providing only `wing` returns everything in that wing; adding `room` or `hall` narrows the scope further. `hall` is a sub-partition of a room, not a connection between rooms — cross-room recall is preserved separately by the global companion lane in `auto` queries.
- **Temporal tracking** — records carry `valid_from` / `valid_to` timestamps supporting point-in-time and interval queries.
- **Knowledge graph** — link memories to places, entities, or concepts and query the resulting graph.
- **Lifecycle management** — archive or supersede records without deletion; archived records are excluded from default queries.

### Retrieval strategies

Two strategies are available via the `memory` tool. The correct one is selected automatically based on the call:

| Strategy | When it triggers | Scope behavior |
| --- | --- | --- |
| `auto` | Default for most queries (any `radius ≥ 1` or unscoped) | Runs a scoped lane (filtered by `wing`/`room`/`hall`) **plus** a global companion lane of up to 20 items; results are fused via RRF. Cross-scope recall is preserved. |
| `palace` | When `radius == 0` **and** at least one scope field is set | Strict scoped retrieval only — no global companion lane. Requires at least `wing` or `room` to be set; returns an error if both are absent. Use when you want hard isolation. |

`graph` mode (`query.mode="graph"`) bypasses scope filtering and traverses the entity graph by ID.

#### Scope call shape

All three scope levels are passed under the `scope` key:

```json
{
  "operation": "query",
  "scope": {"wing": "my-service", "room": "auth", "hall": "tokens"},
  "query": {"text": "refresh token lifetime"}
}
```

Any of the three fields can be omitted. Filters that are present are applied with `AND` semantics.

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
  "scope": {"wing": "test", "room": "setup"},
  "record": {"format": "raw", "content": "Memory is working."}
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
