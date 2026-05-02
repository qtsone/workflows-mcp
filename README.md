# Workflows MCP

Run YAML workflows as MCP tools so agents can automate real tasks with one server.

## Why this project

`workflows-mcp` gives MCP clients a reusable automation layer:

- Define tasks once in YAML and run them from any MCP-compatible client.
- Orchestrate multi-step work with dependency-aware execution.
- Support interactive runs (`Prompt` blocks) with pause/resume.
- Keep secrets server-side via `WORKFLOW_SECRET_*` with redacted outputs.
- Run synchronous or async jobs with queue visibility and cancellation.

## Running workflows-mcp

`workflows-mcp` is a long-lived HTTP service with a web UI and MCP endpoint.

### 1) Install

Requires Python 3.12+.

```bash
uv pip install workflows-mcp
```

or:

```bash
pip install workflows-mcp
```

### 2) One-time bootstrap (admin setup)

Initialize server state before the first run:

```bash
workflows-mcp bootstrap --config-dir ~/.workflows
```

By default, the config directory is `~/.workflows` (or `WORKFLOWS_CONFIG_DIR` if set).

Bootstrap is idempotent: if state already exists, `workflows-mcp bootstrap` reports that
the instance is already initialized and exits without rewriting credentials or settings.

To intentionally change existing bootstrap state, pass `--reconfigure`.

On first bootstrap, the CLI prompts for:

- `Host [127.0.0.1]:`
- `Port [8000]:`
- `Admin password:`

Press Enter on host/port to accept the shown defaults. Admin password is required on first bootstrap.

For existing state, run `workflows-mcp bootstrap --reconfigure` to prompt through the same
fields using persisted defaults. Press Enter to keep current host/port, and press Enter at
the password prompt to keep the current password.

Flags remain script-friendly overrides:

- `--host` skips host prompt
- `--port` skips port prompt
- `--admin-password` skips password prompt and updates/sets the password

In non-interactive environments, host/port prompt EOF falls back to displayed defaults.
First-time password prompt EOF fails with a friendly error; pass `--admin-password` explicitly.

Bootstrap writes:

- `~/.workflows/server.db` (SQLite metadata/control-plane database)
- `~/.workflows/secrets.key` (server-side key material)

You can override the default config directory with `WORKFLOWS_CONFIG_DIR`.

Reconfigure examples:

```bash
# Change admin password for an existing bootstrap state
workflows-mcp bootstrap --reconfigure --admin-password "new-strong-password"

# Change host/port explicitly (other settings remain unchanged)
workflows-mcp bootstrap --reconfigure --host 0.0.0.0 --port 8080
```

### 3) Start the server

Set a bootstrap token for current legacy startup/config guards, then start:

```bash
WORKFLOWS_BOOTSTRAP_TOKEN="replace-with-a-secure-32-byte-or-longer-token" workflows-mcp
```

No-arg `workflows-mcp` starts the HTTP service. By default it binds to
`http://127.0.0.1:8000`; override with `WORKFLOWS_BIND_HOST` and `WORKFLOWS_PORT`.

### 4) Open the UI and create an MCP token

1. Open `http://127.0.0.1:8000/` and sign in at `/login` with the admin password set during bootstrap.
2. Create a project and generate an MCP bearer token in the UI.

### 5) Connect your MCP client

Configure your MCP client to connect over Streamable HTTP with the UI-generated MCP bearer token:

```json
{
  "mcpServers": {
    "workflows": {
      "transport": "streamable-http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": {
        "Authorization": "Bearer <mcp-token-from-ui>"
      }
    }
  }
}
```

Use your server host in the URL if you bind to a non-default interface.

### 6) Service endpoints and auth boundaries

| Endpoint | Auth required | Description |
| --- | --- | --- |
| `http://127.0.0.1:8000/` | No | Web UI entry point |
| `http://127.0.0.1:8000/login` | No | Admin login page |
| `http://127.0.0.1:8000/api/public/v1/*` | No | Public API surface |
| `http://127.0.0.1:8000/api/events/v1/*` | UI session cookie | UI event/session APIs |
| `http://127.0.0.1:8000/api/admin/v1/*` | UI session cookie (+ CSRF on mutating routes) | Admin control plane (projects, tokens, workflow sources, reload/validate) |
| `http://127.0.0.1:8000/mcp` | MCP bearer token | MCP transport endpoint |
| `http://127.0.0.1:8000/docs` | No | Swagger UI (browser-accessible) |
| `http://127.0.0.1:8000/openapi.json` | No | OpenAPI schema |
| `http://127.0.0.1:8000/health` | No | Liveness check |
| `http://127.0.0.1:8000/ready` | No | Readiness check |
| `http://127.0.0.1:8000/config/*` | Legacy bearer token flow | Legacy compatibility endpoints (not primary admin surface) |

Auth model at a glance:

```text
- UI admin: password login -> session cookie; mutating /api/admin/v1/* calls also require X-CSRF-Token.
- MCP clients: bearer token generated in the UI and sent to /mcp.
- WORKFLOWS_BOOTSTRAP_TOKEN: current server startup requirement and legacy `/config/*` bearer credential; not the UI admin password and not the token users paste into MCP clients.
```

### 7) Run first MCP calls

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
| `onboard` | Start project memory onboarding for a repo scan | `onboard(scope={"palace":"my-project"}, ingestion={"mode":"programmatic"}, scan={...})` |
| `sync` | Re-scan the current onboard context for changes | `sync({})` |

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
- Query request shape:
  - `operation="query"` requires `query` to be an object (for example `{"text": "...", "mode": "search"}`).
  - Passing `query` as a plain string is rejected by request validation.
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
- Project scan root policy:
  - `scan.root` is validated against a single allowed root.
  - Default allowed root is `/`.
  - Override with `WORKFLOWS_SCAN_ROOT=/your/root`.
  - Hardened deployments should not leave this at `/`; set a repo/workspace-specific boundary.
  - Recommended hardened values:
    - Local development: `WORKFLOWS_SCAN_ROOT=/absolute/path/to/your/workflows-mcp-repo`
    - CI/release runners: `WORKFLOWS_SCAN_ROOT=$CI_PROJECT_DIR` (or runner workspace root for this repository only)
  - Effective knowledge file access is an intersection: `WORKFLOWS_SCAN_ROOT ∩ fs_root`, plus explicitly approved extra allowlist roots.
- Project flow response compactness:
  - `onboard`/`sync` are compact by default.
  - Completed checkpoint steps strip large `plan[].payload.memories[].content` blobs in compact mode.
  - Pass root-level `debug=true` to expand the response with full diagnostics (equivalent to `response={"debug": true}` in the previous contract).
  - Completed-checkpoint `sync` fast-path returns `from_checkpoint=true` with a note indicating no re-execution.

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

`onboard` (programmatic mode):

```json
{
  "scope": {"palace": "acme"},
  "ingestion": {"mode": "programmatic"},
  "scan": {
    "patterns": ["src/**/*.py"],
    "root": "/path/to/repo",
    "max_files": 200,
    "max_size_kb": 512,
    "respect_gitignore": true
  }
}
```

Onboarding success criteria:

- A structurally complete graph payload (Palace → Wing → Room → Compartment with `contains` corridors) is required for graph operations.
- Ingestion mode: `programmatic` is the default (deterministic, hash-based file scan). `llm` mode is optional and uses strict defaults when enabled.
- Binary and unsupported files produce metadata-only compartments (path, size, mime-type) with no content ingested.

Compact-by-default note:

- `onboard` defaults to compact responses.
- To receive full `results[]` and full checkpoint internals, pass root-level `debug=true`:

```json
{
  "debug": true
}
```

`onboard` (llm mode):

```json
{
  "scope": {"palace": "acme"},
  "ingestion": {
    "mode": "llm",
    "llm_profile": "knowledge-ingest",
    "reproducibility": "strict"
  },
  "scan": {
    "patterns": ["src/**/*.py"],
    "root": "/path/to/repo"
  },
  "debug": true
}
```

`sync`:

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
- Returns `status: "UNCHANGED"` when context resolves but no scan config is stored for automatic delta execution.
- Returns `status: "NO_CONTEXT"` when no onboard context is available; call `onboard` first.
- Returns `status: "AMBIGUOUS_CONTEXT"` when multiple contexts match; narrow with explicit `scope`.

Completed-checkpoint fast-path cue:

- When a checkpoint is already complete, `sync` returns:
  - `status: "completed"`
  - `from_checkpoint: true`
  - a note clarifying results are from checkpoint cache and no operations were re-executed.

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

### HTTP service and auth

- `WORKFLOWS_CONFIG_DIR`: Config directory for bootstrap/runtime metadata (default: `~/.workflows`).
- `WORKFLOWS_BOOTSTRAP_TOKEN`: Current server startup requirement and legacy `/config/*` bearer credential. Must be at least 32 bytes. Not used as the UI admin password and not the token users paste into MCP clients.
- `WORKFLOWS_BIND_HOST`: Host to bind the HTTP server (default: `127.0.0.1`).
- `WORKFLOWS_PORT`: Port to bind the HTTP server (default: `8000`).
- `WORKFLOWS_LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`; default: `INFO`).

### Workflow loading and execution

- `WORKFLOWS_MAX_RECURSION_DEPTH`: Max workflow composition depth (default: `50`).
- Workflow sources for day-to-day operations are managed in SQLite via the web UI / `/api/admin/v1/workflows/sources` (including reload/validate actions).
- `WORKFLOWS_TEMPLATE_PATHS` remains available for template loading behavior, but it is not the primary live control-plane for HTTP/web management.

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
- LLM providers and profiles are managed in SQLite via the web UI / `/api/admin/v1/llm/config`; YAML is available only as explicit import/export data.

## Memory

Memory is an optional persistent storage feature that lets agents and workflows record, retrieve, and organize information across sessions. The HTTP/web-management control plane uses SQLite metadata as source of truth; PostgreSQL (optionally with pgvector) is used for memory/knowledge data.

### What memory provides

- **`memory` MCP tool** — direct call interface for LLM agents to store and query information without writing workflow YAML.
- **`onboard` / `sync` MCP tools** — project onboarding and incremental project-sync tools. `onboard` builds project memory from scan input; `sync({})` auto-resolves context from successful onboard contexts in the current server session, returning `NO_CONTEXT` or `AMBIGUOUS_CONTEXT` when context is insufficient.
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

Set the following environment variables when launching the server process:

```bash
MEMORY_DB_HOST=localhost \
MEMORY_DB_PORT=5432 \
MEMORY_DB_NAME=memory_db \
MEMORY_DB_USER=postgres \
MEMORY_DB_PASSWORD=your-password \
WORKFLOWS_BOOTSTRAP_TOKEN="replace-with-a-secure-32-byte-or-longer-token" \
uv run workflows-mcp
```

On first boot with `MEMORY_DB_AUTO_CREATE=true` (the default), the server creates the target database and applies the schema automatically. Restart the server process after adding the variables.

### Verifying memory is active

1. Restart the server process.
2. Check that the `memory` tool appears in the available tool list (call `list_workflows` via MCP).
3. Run a test ingest:

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
