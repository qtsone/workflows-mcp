# Knowledge Ingestion Architecture

This document describes the ingestion path used by Track 4 structural extraction workflows and the Memory operations they call.

## 9. Workflow Loading and Runtime Placement

### 9.1 Built-in workflow directory

Packaged built-in workflows are loaded from `builtin_workflows/`.

Track 4 ingestion flows (including system1-scan) are referenced from that packaged directory at server runtime. The older `templates/builtin/` path is not used.

## 10. Structural Graph Persistence

### 10.2 `store_relations_by_qname`

`store_relations_by_qname` is the server-side relation persistence path for structural relations emitted by TreeSitter/system1-scan when workflows provide qualified names instead of entity UUIDs.

The workflow sends relation endpoints as `{source_qname, source_entity_type, target_qname, target_entity_type}` and the Memory service resolves these endpoints inside the requested palace before inserting relations.

Contract and behavior:

- Allowed relation types are `CONTAINS`, `INHERITS_FROM`, `IMPORTS`, and `CALLS`.
- Palace isolation is enforced: both resolved endpoint entities must exist in the same requesting palace; cross-palace links are rejected.
- Source entities are never auto-created by this operation.
- Target resolution uses `external_fallback`:
  - `module`: unresolved targets are materialized as `STRUCTURAL` `Module` entities marked external, then linked.
  - `reject`: unresolved targets are not inserted.
- Inserts are idempotent: the service performs a select-before-insert check using `(source_entity_id, target_entity_id, relation_type)` with metadata containment; existing matches are returned instead of duplicated.
- Result `relation_ids` preserves request order and is nullable per item:
  - non-null UUID for created or existing relations,
  - `null` for unresolved or rejected attempts.

Why workflows do not resolve UUIDs client-side:

- UUID resolution depends on palace-scoped entity state in Memory.
- Ambiguity handling and fallback policy (`module` or `reject`) are enforced centrally.
- Idempotency and palace boundary checks remain in one authoritative write path.
