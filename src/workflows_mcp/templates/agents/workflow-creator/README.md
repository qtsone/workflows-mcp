# Workflow Creator v2 - Optimized Modular Design

## Architecture Overview

**Modular Composition with Recursion and for_each**

```text
┌─────────────────────────────────────────────────────────────────┐
│                 MAIN: workflow-creator.yaml                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            SUB-WORKFLOW: workflow-creator-bootstrap.yaml                 │
│  • git-checkout workflows-mcp repo (sparse)                     │
│  • ReadFiles: schema, executors, examples, templates            │
│  • LLMCall: Generate compressed executor reference              │
│  Returns: {schema, executor_docs, examples, templates}          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         MAIN: Requirements Gathering (for_each optimized)       │
│  • Filter prompts (skip if input provided)                      │
│  • for_each prompt in filtered_questions:                       │
│    - Ask question sequentially                                  │
│  • Structure responses with LLMCall + schema                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│       SUB-WORKFLOW: design-approval-loop.yaml (RECURSIVE)       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ 1. Generate workflow diagram (LLMCall)               │       │
│  │ 2. Present to user (Prompt: approve/refine/cancel)   │       │
│  │ 3. If "refine":                                      │       │
│  │    - Gather feedback                                 │       │
│  │    - RECURSIVE CALL to self with:                    │       │
│  │      * iteration + 1                                 │       │
│  │      * previous_diagram                              │       │
│  │      * feedback                                      │ ◄─────┤
│  │ 4. If "approve": return final_diagram                │       │
│  └──────────────────────────────────────────────────────┘       │
│  Max iterations enforced in condition                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              MAIN: Generate Initial Workflow                    │
│  • LLMCall with approved diagram + full context                 │
│  Returns: workflow_yaml                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│    SUB-WORKFLOW: validation-fix-attempt.yaml (for_each loop)    │
│  for_each attempt in [1, 2, 3]:                                 │
│    ┌─────────────────────────────────────────────────┐          │
│    │ 1. Save YAML to temp file                       │          │
│    │ 2. Validate with canonical validate.py          │          │
│    │ 3. If invalid:                                  │          │
│    │    - LLMCall to fix based on errors             │          │
│    │    - Return fixed YAML                          │          │
│    │ 4. If valid: return success + YAML              │          │
│    └─────────────────────────────────────────────────┘          │
│  Sequential execution, continue_on_error                        │
│  First successful attempt wins                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           SUB-WORKFLOW: finalize-and-save.yaml                  │
│  • Save final workflow to output path                           │
│  • Final validation check                                       │
│  • Generate summary report                                      │
│  • Display next steps + MCP restart requirement                 │
│  • Cleanup temp files (repo, drafts, state)                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Optimizations

### 1. for_each for Requirements Gathering

**Before (verbose):**

```yaml
- id: prompt_1
  type: Prompt
  inputs: {prompt: "Question 1"}

- id: prompt_2
  type: Prompt
  inputs: {prompt: "Question 2"}
  depends_on: [prompt_1]

# ... 3 more blocks
```

**After (optimized):**

```yaml
- id: gather_requirements
  type: Prompt
  for_each: "{{blocks.filter_prompts.outputs.filtered_questions}}"
  for_each_mode: sequential
  inputs:
    prompt: "Step {{each.value.step}}/5: {{each.value.question}}"
```

### 2. Recursive Sub-Workflow for Design Loop

**Self-calling workflow with iteration control:**

```yaml
- id: recurse
  type: Workflow
  condition: >-
    {{blocks.request_approval.outputs.response == 'refine' and
     inputs.iteration < inputs.max_iterations - 1}}
  inputs:
    workflow: design-approval-loop  # Calls itself!
    inputs:
      iteration: "{{inputs.iteration + 1}}"
      previous_diagram: "{{blocks.generate_diagram.outputs.response}}"
      feedback: "{{blocks.gather_feedback.outputs.response}}"
```

### 3. for_each for Validation Attempts

**Iterate until success or max attempts (file-based approach):**

```yaml
# Save workflow to temp file first
- id: save_draft_workflow
  type: CreateFile
  inputs:
    path: "{{tmp}}/draft-workflow-attempt-{{each.value}}.yaml"
    content: |
      {% if each.index == 0 %}
      {{blocks.generate_workflow.outputs.response}}
      {% else %}
      {{blocks.validation_loop[each.index - 1].outputs.yaml_content}}
      {% endif %}
    overwrite: true

# Validate using file path (not inline content)
- id: validation_loop
  type: Workflow
  for_each: [1, 2, 3]
  for_each_mode: sequential
  continue_on_error: true
  inputs:
    workflow: validation-fix-attempt
    inputs:
      workflow_path: "{{blocks.save_draft_workflow.outputs.path}}"
      schema_path: "{{blocks.bootstrap_context.outputs.llm_schema_path}}"
      executor_docs: "{{blocks.bootstrap_context.outputs.executor_docs}}"
      try: "{{each.value}}"
      workflows_mcp_path: "{{blocks.bootstrap_context.outputs.tmp_dir}}"
  depends_on: [save_draft_workflow]
```

**Note:** Large content (YAML, schemas) is passed via file paths for performance and reliability.

## File Structure

```text
src/workflows_mcp/templates/agents/
├── workflow-creator.yaml           # Main orchestrator (~300 lines)
├── workflow-creator-bootstrap.yaml          # Fetch context (~100 lines)
├── design-approval-loop.yaml       # Recursive design loop (~150 lines)
├── validation-fix-attempt.yaml     # Single validation attempt (~80 lines)
└── finalize-and-save.yaml          # Save and cleanup (~100 lines)

Total: ~730 lines across 5 focused files
vs ~1500 lines in monolithic approach
```

## Benefits of Modular Design

✅ **Focused** - Each workflow has single responsibility
✅ **Testable** - Test each component independently
✅ **Maintainable** - Easy to modify individual parts
✅ **Reusable** - Sub-workflows can be used elsewhere
✅ **Portable** - git-checkout fetches canonical sources
✅ **Efficient** - for_each reduces repetition by 60%
✅ **Reliable** - Recursive design + auto-fix validation
✅ **Clear** - Easy to understand flow

## Implementation Priorities

1. **workflow-creator-bootstrap.yaml** - Foundation (fetch all context)
2. **design-approval-loop.yaml** - Critical loop (recursive design)
3. **validation-fix-attempt.yaml** - Quality gate (validation + fix)
4. **finalize-and-save.yaml** - Completion (save + cleanup)
5. **workflow-creator.yaml** - Orchestrator (compose all)

## git-checkout Configuration

**Correct inputs for git-checkout workflow:**

```yaml
- id: checkout_repo
  type: Workflow
  inputs:
    workflow: git-checkout
    inputs:
      repo: "https://github.com/qtsone/workflows-mcp.git"
      path: "{{tmp}}/workflows-mcp-src"
      ref: "main"  # Optional, defaults to main
      sparse_checkout: "schema.json,src/workflows_mcp/engine,tests/workflows,src/workflows_mcp/templates"
```

**Notes:**

- `repo` and `path` are required
- `ref` defaults to "main"
- `sparse_checkout` for faster cloning (comma-separated paths)
- `fetch_depth` defaults to 1 (shallow clone)

## for_each Access Patterns

**Accessing for_each iteration outputs:**

```yaml
# In subsequent blocks:
- id: use_responses
  type: Shell
  inputs:
    command: |
      # Access by index (0-based)
      echo "Answer 1: {{blocks.gather_requirements['0'].outputs.response}}"
      echo "Answer 2: {{blocks.gather_requirements['1'].outputs.response}}"

# In validation loop (referencing previous iteration):
yaml_content: |
  {% if each.index == 0 %}
  {{blocks.generate_workflow.outputs.response.content}}
  {% else %}
  {{blocks.validation_loop[each.index - 1].outputs.yaml_content}}
  {% endif %}
```

## Testing Limitation

**Critical Note:**

```text
⚠️  Cannot test generated workflows in the same MCP session
⚠️  MCP server must be restarted to load new workflows
⚠️  Workflow-creator must acknowledge this limitation clearly
```

**Workaround:**

```yaml
- id: final_summary
  inputs:
    command: |
      echo "IMPORTANT: Testing Limitation"
      echo "⚠️  The workflow has been saved but cannot be tested now."
      echo "⚠️  You MUST restart the MCP server to load it."
      echo ""
      echo "After restart, test with:"
      echo "  execute_workflow(workflow='{{workflow_name}}', inputs={...})"
```

## Next Steps

Ready to implement these 5 workflows in order:

1. Start with `workflow-creator-bootstrap.yaml`
2. Then `design-approval-loop.yaml`
3. Then `validation-fix-attempt.yaml`
4. Then `finalize-and-save.yaml`
5. Finally `workflow-creator.yaml` (orchestrator)

Shall we proceed with implementation?
