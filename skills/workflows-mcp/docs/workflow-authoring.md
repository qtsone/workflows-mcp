# Create, validate, and execute workflows

## When to use registered vs inline workflows

- Registered (`execute_workflow`): reusable automation loaded from `WORKFLOWS_TEMPLATE_PATHS`.
- Inline (`execute_inline_workflow`): quick prototype or one-off validation.

## Minimal registered workflow example

Create a YAML file in your configured workflow directory.

```yaml
version: "1"
name: hello-workflow
description: Simple greeting workflow

inputs:
  name:
    type: string
    required: true

blocks:
  - id: greet
    type: Shell
    inputs:
      command: "echo Hello, {{inputs.name}}"
```

## Minimal inline workflow execution

```json
{
  "workflow_yaml": "version: \"1\"\nname: inline-hello\nblocks:\n  - id: hi\n    type: Shell\n    inputs:\n      command: \"echo inline hello\"",
  "inputs": {}
}
```

## Validation-first flow

1. `validate_workflow_yaml` (catch schema/input errors early)
2. `execute_inline_workflow` (optional quick check)
3. `reload_workflows` (after file edits)
4. `execute_workflow`

For exact required inputs on an existing workflow, call `get_workflow_info` and follow its input schema.

## Practical authoring rules

- Keep block IDs stable and descriptive.
- Model explicit dependencies with `depends_on`.
- Use `{{inputs.*}}` for external parameters.
- Use `{{blocks.<id>.outputs.*}}` when chaining outputs.
- Keep secrets in env and reference with `{{secrets.NAME}}`.

## Troubleshooting checklist

1. Workflow missing from list:
   - Verify `WORKFLOWS_TEMPLATE_PATHS` path(s)
   - Call `reload_workflows`
   - Restart MCP client if needed
2. Validation errors:
   - Check required block fields from block reference
   - Ensure YAML indentation and types are correct
3. Async workflow stalls:
   - Check `get_job_status`
   - If `paused`, call `resume_workflow`
4. Secret resolution failure:
   - Confirm `WORKFLOW_SECRET_<NAME>` exists in server env
   - Use matching `{{secrets.NAME}}` key

## Troubleshooting (quick)

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Workflow file exists but does not appear in `list_workflows` | Workflow directory is not included in `WORKFLOWS_TEMPLATE_PATHS` or not reloaded | Set correct absolute path, call `reload_workflows`, then re-check list. |
| `validate_workflow_yaml` fails with schema errors | Missing required fields or invalid block shape | Read the exact validation error, correct fields/indentation, and re-run validation before execution. |
| Variable resolution fails at runtime | Wrong namespace reference (e.g. input/output path typo) | Use `{{inputs.*}}` for runtime inputs and `{{blocks.<id>.outputs.*}}` for block outputs. |
| Secret placeholder resolves empty/error | Env name and template key do not match | Set `WORKFLOW_SECRET_<NAME>` and reference `{{secrets.NAME}}` with same `<NAME>`. |
