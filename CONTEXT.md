# Workflows MCP Domain Context

This glossary records durable project language for architecture discussions and implementation planning.

## Memory locality

Memory locality is the target context that makes a memory operation safe and unambiguous. It can come from topology scope, stable IDs, operation-local checkpoint data, or session continuation state depending on the operation.

## Topology scope

Topology scope is the four-field memory location: `palace`, `wing`, `room`, and `compartment`. Broad read operations can use partial topology scope. Placement writes need complete topology scope unless another operation-local target identity is sufficient.

## Operation locality contract

An operation locality contract declares what locality data a memory operation needs, which sources may provide it, and when the operation must fail closed. The contract is evaluated per operation instead of applying one blanket explicit-or-implicit scope rule.

## Placement write

A placement write creates or places memory data into topology. Direct `ingest` and graph place upsert are placement writes. Placement writes need complete, provable locality.

## Broad read

A broad read can intentionally query a wider memory area. `query` is a broad read and may use `palace` as its minimum locality, with `wing`, `room`, and `compartment` acting as optional narrowing fields.

## ID-targeted mutation

An ID-targeted mutation changes or validates existing memory data by globally unique identifiers. Archive, supersede, validate, and UUID-based graph deletion are ID-targeted when they operate only on IDs.

## Continuation flow

A continuation flow continues a previously established memory operation context. `sync({})` can use session active context when that context is the relevant result of a prior onboard or sync operation.

## Scope candidate

A scope candidate is stored session context that can contain `scope`, `scope_key_value`, checkpoint data, and source metadata. A complete scope candidate can support continuation flows and reads, but does not automatically authorize standalone placement writes.

## Memory→Postgres fast-path invariant

The memory subsystem requires a PostgreSQL backend with the `pgvector` extension. Unlike the SQL block — which is multi-DB and selects its backend through the `DatabaseBackend` dialect-selection Protocol — memory binds `PostgresBackend` / `DatabaseEngine.POSTGRESQL` directly at every production site and emits Postgres-only SQL (pgvector ANN search, full-text search, RRF) with no SQLite path. This asymmetry is intentional and documented as a fast-path invariant: memory is single-DB by design; only the SQL block is portable. See ADR-016.
