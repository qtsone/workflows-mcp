# ADR-012: Store structural relations by qualified name

- Status: Accepted
- Date: 2026-05-05

## Context

Track 4 ingestion emits structural relations from TreeSitter/system1-scan using qualified names (`qname`) for endpoints. At emission time, workflows do not have stable knowledge entity UUIDs.

The system requires:

- palace-scoped relation writes,
- idempotent relation persistence,
- explicit handling for unresolved targets,
- no client-side duplication of graph resolution rules.

## Decision

The Memory service provides `store_relations_by_qname` as the canonical server-side operation for structural relation persistence.

`store_relations_by_qname`:

- accepts qname-keyed relation inputs,
- resolves endpoint entities within the requested palace,
- permits only structural relation types: `CONTAINS`, `INHERITS_FROM`, `IMPORTS`, `CALLS`,
- rejects cross-palace endpoint linking,
- never auto-creates source entities,
- supports target fallback modes:
  - `external_fallback=module`: create or reuse an external `STRUCTURAL` `Module` entity,
  - `external_fallback=reject`: reject unresolved targets,
- enforces idempotency by selecting existing matches before insert,
- returns ordered `relation_ids` with nullable entries for unresolved/rejected items.

## Consequences

Positive:

- workflows stay simple and avoid client-side UUID resolution,
- palace isolation is enforced in one write path,
- retries are safe because writes are idempotent,
- unresolved and ambiguous relations are observable without partial contract drift.

Trade-offs:

- relation resolution now depends on server-side state at write time,
- unresolved handling requires consumers to interpret nullable `relation_ids` and diagnostics.
