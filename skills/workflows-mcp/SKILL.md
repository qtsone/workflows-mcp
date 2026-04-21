---
name: workflows-mcp
description: Use when onboarding and operating workflows-mcp from an MCP client in any project, including install, client configuration, memory setup, project_onboard/project_sync usage, and workflow authoring/execution.
---

# workflows-mcp-user

User-facing skill for adopting `workflows-mcp` in your own project.

## When to use

Use this skill when you want to:
- install and run `workflows-mcp` from an MCP client,
- configure custom workflow directories and secrets,
- enable PostgreSQL-backed memory,
- use `project_onboard` / `project_sync` checkpoint flows,
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
