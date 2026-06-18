# Workflows MCP Domain Context

This glossary records durable project language for architecture discussions and
implementation planning. It is a vocabulary, not a design document: each term is
defined by what it *is* in the product, not by how a given module implements it.

The product has three subsystems, mirrored by the package layout (ADR-015):

- **Workflow engine** (`engine/`) — runs YAML-defined workflows as DAGs.
- **Memory platform** (`memory/`, with `memory/knowledge/`) — the Memory Palace
  and its ingestion/retrieval paths.
- **Control plane / security** (`http/`, `security/`, `engine/secrets/`) — the
  admin API, session/CSRF protection, and server-side secrets.

---

## Workflow engine

### DAG

A workflow is a **DAG** (directed acyclic graph) of blocks connected by
`depends_on` edges. The engine resolves the DAG to an execution order before
running anything; a cycle is a definition error, not a runtime one. See ADR-006.

### Wave

A **wave** is the set of blocks the DAG resolver schedules to run together —
every block whose dependencies are already satisfied at that point. Blocks in one
wave run in parallel; waves run in sequence. Wave membership is a property of the
dependency graph, not of authoring order. See ADR-006.

### BlockExecutor

A **BlockExecutor** is the unit that runs one block type (Shell, HttpCall, Memory,
Workflow, and so on). It is stateless and shared across executions: it takes
validated inputs and the current execution context, returns outputs, and raises an
exception to signal failure rather than returning an error wrapper. A nested
workflow is itself a block, so the same executor contract applies at every level.
See ADR-001, ADR-004, ADR-006.

### Block-status tiers

**Block-status tiers** are the three progressively precise ways a workflow author
references whether a block ran and whether its operation succeeded:

- **Tier 1 — boolean shortcuts:** `succeeded` / `failed` / `skipped`.
- **Tier 2 — status string:** did the executor run? (`completed`, `failed`,
  `skipped`, `paused`, …).
- **Tier 3 — outcome string:** did the operation succeed? (`success`, `failure`,
  `n/a`).

The tiers exist because "the executor ran" and "the operation succeeded" are
distinct facts — a block can complete and still report a failed operation. See
ADR-005, ADR-007.

### Variable namespaces

The engine resolves `{{...}}` references against a fixed set of **namespaces**,
each a distinct source of values:

- `inputs` — the workflow's runtime parameters.
- `metadata` — workflow-level metadata (name and similar).
- `blocks` — other blocks' outputs and status.
- `secrets` — server-side secret values (resolved server-side, never echoed; see
  the control-plane section).
- `__internal__` — orchestration state, deliberately *not* reachable from
  workflow text; an attempt to read it is denied.

Keeping these separate is the engine's security and clarity boundary: a workflow
can only read what a namespace deliberately exposes. See ADR-006, ADR-008.

### Job

A **job** is one tracked execution of a workflow — the durable record under which
a run is enqueued, executed, and looked up, whether it finished synchronously, is
running asynchronously, or is paused awaiting input. The job is the unit the
control plane lists and inspects.

### Pause / resume

A workflow **pauses** when a block needs information the server does not have
(typically interactive input). Because MCP has no back-channel mid-call, a pause
ends the call and returns a handle; the caller later **resumes** with a response,
and execution continues from where it stopped. Pause is modelled as control flow,
not as an error, and a pause naturally propagates outward through nested
workflows. The state that survives a pause is a *checkpoint* (see ADR-002). See
ADR-002, ADR-006, ADR-010.

---

## Memory platform

The memory platform is the **Memory Palace**: a structured store of what the
system knows about a project, organised so that reads and writes target a precise
location. Its ontology and ownership rules are fixed by ADR-013; the terms below
are the working language layered on top of it.

### Topology scope

**Topology scope** is the four-field memory location: `palace`, `wing`, `room`,
and `compartment` — and only this containment chain (ADR-013). Broad read
operations can use partial topology scope. Placement writes need complete topology
scope unless another operation-local target identity is sufficient.

### Corridor

A **corridor** is a directed, typed edge between any two nodes in the palace graph
(for example `CALLS`, `IMPORTS`, `CONTAINS`). A corridor is *relation* semantics,
not a topology level: it expresses how nodes relate, not where they sit in the
`palace → wing → room → compartment` containment chain. See ADR-013.

### Memory locality

**Memory locality** is the target context that makes a memory operation safe and
unambiguous. It can come from topology scope, stable IDs, operation-local
checkpoint data, or session continuation state, depending on the operation.

### Operation locality contract

An **operation locality contract** declares what locality data a memory operation
needs, which sources may provide it, and when the operation must fail closed. The
contract is evaluated per operation instead of applying one blanket
explicit-or-implicit scope rule.

### Placement write

A **placement write** creates or places memory data into topology. Direct `ingest`
and graph place upsert are placement writes. Placement writes need complete,
provable locality.

### Broad read

A **broad read** can intentionally query a wider memory area. `query` is a broad
read and may use `palace` as its minimum locality, with `wing`, `room`, and
`compartment` acting as optional narrowing fields.

### ID-targeted mutation

An **ID-targeted mutation** changes or validates existing memory data by globally
unique identifiers. Archive, supersede, validate, and UUID-based graph deletion
are ID-targeted when they operate only on IDs.

### System 1 / System 2

**System 1** is the source of truth for live structural evidence — the graph the
watcher keeps in step with the actual code (containment, calls, imports). **System
2** is the *derived, interpretive* layer that labels that structure by intent
(naming rooms, refining compartments, proposing semantic corridors). System 1 is
authoritative and current; System 2 may become stale and degrade or archive when
its evidence weakens. The split keeps "what the code structurally is" separate
from "what we think it means." See ADR-013.

### Evidence category

An **evidence category** is one of the three independent kinds of proof a semantic
claim can draw on: naming/path evidence, graph/cluster evidence, and
memory/docs/docstring evidence. Evidence is existence-based — clear inspectable
relational proof, not LLM prose — and one artifact satisfies at most one category.
A new wing requires support from at least two categories; lighter claim types
require evidence appropriate to their kind. See ADR-013.

### Project flow

A **project flow** is a resumable, multi-step memory operation plan — the onboard
and sync paths — that executes ingest, supersede, archive, and maintain steps in
order, pausing between steps and resuming on a later call. It is owned by the
stateless `ProjectFlowService`, which advances one or more steps and returns the
next state; the single impure dependency (running one memory operation) is supplied
as an injected **operation executor (port)**. See ADR-014.

### Operation executor (port)

The **operation executor** is the port the project flow depends on to perform a
single memory operation: a plain async callable `(operation, scope, payload) ->
result`. It is the *only* impure dependency of the project flow, injected rather
than imported, so the flow's resume logic is testable with a pure fake and the
flow itself imports nothing from the memory subsystem. See ADR-014.

### Project checkpoint

A **project checkpoint** is the serialisable dict that carries all project-flow
state across calls: the resolved scope, the ordered plan, the next-step index, the
completed steps, and any scan snapshot. The checkpoint *is* the state — there is no
server-side flow state — so it round-trips through the client and keeps the flow
concurrency-safe under many agent sessions against one central server. See ADR-014.
(The engine has its own, separate notion of a *checkpoint* for workflow
pause/resume; see ADR-002. The two are distinct.)

### Continuation flow

A **continuation flow** continues a previously established memory operation
context. `sync({})` can use session active context when that context is the
relevant result of a prior onboard or sync operation.

### Scope candidate

A **scope candidate** is stored session context that can contain `scope`,
`scope_key_value`, checkpoint data, and source metadata. A complete scope candidate
can support continuation flows and reads, but does not automatically authorize
standalone placement writes.

### Knowledge module

**Knowledge** is the data layer beneath Memory — schema, hybrid (vector +
full-text) search and RRF fusion, embedding helpers, graph traversal, and
token-budgeted context assembly — living at `memory/knowledge/`. The relationship
is deliberate: *Memory* is the product-facing subsystem and its ontology; *Knowledge*
is the retrieval/storage machinery it is built on. The name predates the Memory
Palace framing and is retained for the storage layer rather than the user-facing
concept. See ADR-012, ADR-013, ADR-016.

### Memory → Postgres fast-path invariant

The memory subsystem requires a PostgreSQL backend with the `pgvector` extension.
Unlike the SQL block — which is multi-DB and selects its backend through the
`DatabaseBackend` dialect-selection protocol — memory binds `PostgresBackend` /
`DatabaseEngine.POSTGRESQL` directly at every production site and emits
Postgres-only SQL (pgvector ANN search, full-text search, RRF) with no SQLite path.
This asymmetry is intentional and documented as a fast-path invariant: memory is
single-DB by design; only the SQL block is portable. See ADR-016.

---

## Control plane / security

The control plane is the operator-facing surface around the engine and memory: an
admin API and the security mechanisms that protect it and the workflows it runs.

### Admin API

The **admin API** is the HTTP surface (`http/routes/admin_v1/`) operators use to
manage the running server — projects, workflow runs, watchers, secrets, LLM
configuration, database state, and memory sync. It is distinct from the MCP
transport that agents speak: the admin API is for humans/operators, the MCP
endpoint is for agents. The session/CSRF terms below govern the admin API
specifically.

> No dedicated ADR governs the admin-API session/CSRF mechanism yet; it is
> implemented under `security/` and `http/`. ADR-003 (executor security
> classification) and ADR-008 (secrets) cover adjacent, not identical, concerns.

### Session

A **session** is an authenticated admin login, tracked by a server-side record and
a `workflows_admin_session` cookie. Sessions enforce both an idle timeout and an
absolute lifetime, so an admin login does not stay valid indefinitely. Sessions
authenticate operators of the admin API; they are separate from per-agent MCP
transport sessions and from memory session context.

### CSRF protection

**CSRF protection** binds a per-session token (sent in the `X-CSRF-Token` header)
to state-changing admin requests, so a third-party site cannot drive the admin API
using an operator's session cookie. It guards the admin API specifically, because
that surface is cookie-authenticated and browser-reachable.

### Secret / redaction

A **secret** is a sensitive value (token, password, connection string) the server
holds and resolves *server-side* via the `{{secrets.*}}` namespace, so it never
enters the workflow definition or the agent's context. **Redaction** is the
fail-safe companion: secret values are scrubbed from all outputs, metadata, and
error messages before anything is returned, so a secret cannot leak even if a
block echoes it. See ADR-008.
