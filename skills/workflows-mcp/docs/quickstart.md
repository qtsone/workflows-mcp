# Quickstart (MCP users)

## Value proposition

Use `workflows-mcp` when you want repeatable automation behind one MCP server:
- define multi-step tasks once in YAML,
- execute them from any MCP-compatible client,
- support sync/async execution and interactive pause/resume.

## Install options

Python 3.12+ is required.

### Option A: `uvx` (no persistent environment required)

Use in MCP client config (or copy `snippets/mcp-config-uvx.json`):

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": ["workflows-mcp"],
      "env": {
        "WORKFLOWS_TEMPLATE_PATHS": "/absolute/path/to/workflows",
        "WORKFLOWS_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Option B: `pip` install + direct command

Install:

```bash
pip install workflows-mcp
```

Then MCP config:

(or copy `snippets/mcp-config-pip.json`)

```json
{
  "mcpServers": {
    "workflows": {
      "command": "workflows-mcp",
      "env": {
        "WORKFLOWS_TEMPLATE_PATHS": "/absolute/path/to/workflows",
        "WORKFLOWS_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## First-call flow (recommended)

After restarting your MCP client:

1. `list_workflows`
2. `get_workflow_info` (for required inputs)
3. `execute_workflow`

For async mode:
- call `execute_workflow(..., mode="async")`,
- poll with `get_job_status` (or `list_jobs`),
- call `resume_workflow` only when status is `paused`.

## First 10 minutes (external users)

Goal: confirm the server is reachable and can execute one workflow end-to-end.

1. Add one MCP config snippet (`uvx` or `pip`) and restart your MCP client.
2. Call `list_workflows`.
   - Expected cue: non-empty workflow list or a successful empty list response (not a tool error).
3. Pick one workflow name and call `get_workflow_info`.
   - Expected cue: schema with `inputs` and block details.
4. Call `execute_workflow` with required inputs.
   - Expected cue (sync): response includes execution status/result.
   - Expected cue (async): receive `job_id`; polling with `get_job_status` shows progress.
5. If you edit YAML files in your workflow directory, call `reload_workflows` and re-run step 2.

If step 2 fails, check Troubleshooting in this file before changing workflow YAML.

## Reload behavior

If you edit workflow YAML files on disk, call `reload_workflows` before re-running.

## Basic config notes

- `WORKFLOWS_TEMPLATE_PATHS`: comma-separated workflow directories.
- `WORKFLOW_SECRET_<NAME>`: exposes `{{secrets.NAME}}` to workflows.
- `WORKFLOWS_LOG_LEVEL`: `DEBUG|INFO|WARNING|ERROR|CRITICAL`.

## Troubleshooting (quick)

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `list_workflows` tool errors immediately | MCP server command/package not available in client runtime | For `uvx`, ensure `uv`/`uvx` is installed and accessible. For `pip`, ensure `workflows-mcp` is installed in the same environment used by your client. Restart client. |
| `list_workflows` returns empty list unexpectedly | `WORKFLOWS_TEMPLATE_PATHS` not set to a real directory with YAML workflows | Set absolute path(s), restart client, then call `reload_workflows`. |
| `execute_workflow` fails on missing input | Required inputs not provided | Run `get_workflow_info` first; provide required fields exactly. |
| Async run never completes | Job is paused waiting for input, or you are not polling | Poll `get_job_status`; if `paused`, call `resume_workflow` with required response. |
