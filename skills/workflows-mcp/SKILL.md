---
name: workflows-mcp
description: Use when onboarding and operating workflows-mcp from an MCP client in any project, including install, client configuration, memory setup, onboard/sync usage, and workflow authoring/execution.
---

# workflows-mcp-user

User-facing skill for adopting `workflows-mcp` in your own project.

## When to use

Use this skill when you want to:
- install and run `workflows-mcp` from an MCP client,
- configure custom workflow directories and secrets,
- enable PostgreSQL-backed memory,
- use `onboard` / `sync` checkpoint flows,
- create, validate, and execute workflows.

## What you get

- Fast install paths (`uvx` or `pip`)
- Copy/paste MCP client config snippets
- Memory setup checklist (PostgreSQL + env vars)
- Practical memory operation patterns (`query`, `ingest`, lifecycle, graph)
- Workflow authoring + validation + execution loop
- Best practices, pitfalls, and troubleshooting steps

## Start here

1. Read `README.md` in this skill package.
2. Copy config from `snippets/` (`uvx` or `pip`).
3. Follow docs in order:
   - `docs/quickstart.md`
   - `docs/memory-postgres.md`
   - `docs/workflow-authoring.md`

## Ground rules

- Commands and payloads are aligned to current public docs and tool contracts.
- Use `reload_workflows` after editing workflow YAML files.
- Restart your MCP client after changing server env/config.
- Keep secrets in `WORKFLOW_SECRET_*` env vars (never inline in workflow files).

## Memory flow reminders

- For `memory(operation="query")`, pass `query` as an object (for example `{"text": "...", "mode": "search"}`), not a plain string.
- `onboard`/`sync` are compact by default as outputs can be very large; pass root-level `debug=true` for full output.
- Completed-checkpoint `sync` responses include `from_checkpoint: true` with a no-reexecution note.
- Successful `onboard` sets an active session context. Subsequent `memory`/`sync` calls can omit `scope` and use that active context.
- Use `select(scope={...})` to switch active context in sessions where multiple onboarded contexts exist.
- If no active context exists and no explicit scope is passed, tools return `MEM_NO_ACTIVE_CONTEXT` with actionable guidance.
- `sync({})` with a resolved context returns `UNCHANGED` when no scan config is stored (pass `scan` to run delta).
- `onboard` programmatic mode accepts minimal scope with `scope.palace`; lower topology is derived from scanned structure.
- Graph onboarding requires a complete structural graph: Palace → Wing → Room → Compartment with `contains` corridors.
- Binary and unsupported files produce metadata-only compartments; no content is ingested.
- Scan root policy uses a single root: default `/`, override with `WORKFLOWS_SCAN_ROOT`.
