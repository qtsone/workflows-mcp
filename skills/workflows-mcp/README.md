# Workflows MCP User Skill Package

This package is for **MCP users** who want to use `workflows-mcp` in their own projects.

It focuses on practical onboarding and operation (not repository contribution internals).

## Contents

- `SKILL.md` — trigger and usage scope
- `docs/quickstart.md` — install, client config, first calls
- `docs/memory-postgres.md` — PostgreSQL memory setup + tool patterns
- `docs/workflow-authoring.md` — creating, validating, and running workflows
- `snippets/` — copy/paste JSON/YAML examples

## Recommended path

1. Copy one server config snippet:
   - `snippets/mcp-config-uvx.json`
   - `snippets/mcp-config-pip.json`
2. Follow the **First 10 minutes** in `docs/quickstart.md`.
3. If you need persistent memory (highly recommended), enable PostgreSQL with `docs/memory-postgres.md`.
4. Create and run your first workflow with `docs/workflow-authoring.md`.

## Entrypoint order (copy/paste-first)

Use this order to avoid setup loops:

1. `docs/quickstart.md` (server visible + first successful execution)
2. `docs/memory-postgres.md` (only if you need memory tools)
3. `docs/workflow-authoring.md` (build and iterate on your own workflows)
