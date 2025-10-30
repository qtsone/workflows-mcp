# GitHub Workflows

Interactive workflows for GitHub repository management using the GitHub CLI (`gh`).

## Available Workflows

### `github-create-issue`

Interactive workflow that leverages LLM intelligence to create well-structured GitHub issues with context awareness.

**Tags**: `github`, `interactive`, `issues`

#### Features

- 🧠 **LLM-Powered**: Generates professional title and body based on your description
- 🎯 **Context-Aware**: Analyzes repository patterns, labels, and recent issues
- 🛡️ **Safe**: Prevents shell injection via environment variables
- 👀 **Preview**: Shows complete issue before creation with confirmation step
- ⚡ **Fast**: Parallel context gathering for optimal performance

#### Prerequisites

1. **GitHub CLI** installed and authenticated:
   ```bash
   # Install GitHub CLI
   # macOS: brew install gh
   # Linux: See https://cli.github.com/

   # Authenticate
   gh auth login
   ```

2. **Git repository** with GitHub remote
3. **jq** installed (for JSON parsing):
   ```bash
   # macOS: brew install jq
   # Linux: apt-get install jq / yum install jq
   ```

#### Usage

**Basic Usage** (with confirmation):
```bash
# Execute workflow via MCP tool
execute_workflow(
  workflow="github-create-issue",
  inputs={"repo_path": "."}
)
```

**Auto-confirm Mode** (skip preview):
```bash
execute_workflow(
  workflow="github-create-issue",
  inputs={
    "repo_path": ".",
    "auto_confirm": true
  }
)
```

#### Interaction Flow

The workflow follows a structured 6-phase approach:

##### Phase 1: Prerequisite Validation
- Verifies `gh` CLI is installed
- Confirms you're in a git repository
- Checks GitHub authentication

##### Phase 2: Context Gathering (Parallel Execution)
- Fetches repository name, description, and URL
- Lists available labels with descriptions
- Retrieves recent issues for pattern reference

##### Phase 3: Issue Description
**[INTERACTIVE PAUSE]**

You'll be prompted:
```bash
# 🐛 GitHub Issue Creator

**Repository**: workflows-mcp
**Description**: DAG-based workflow execution MCP server

## Describe the issue you want to create

Please provide a comprehensive description including:
- **Problem or feature request**: What is the issue about?
- **Context**: When does this occur? What triggers it?
- **Expected behavior**: What should happen?
- **Actual behavior** (for bugs): What actually happens?
- **Additional context**: Any relevant details, logs, or examples

**Your description**:
```

**Example response**:
```bash
Bug: When executing workflows with Prompt blocks, the workflow doesn't properly
save the user's response to the checkpoint state. This causes issues when
resuming the workflow after a pause.

Context: This happens every time a Prompt block is executed in an interactive workflow.

Expected: The response should be saved to the checkpoint and available via
{{blocks.prompt_id.outputs.response}} in subsequent blocks.

Actual: The outputs field is empty after resume, causing variable resolution errors.
```

##### Phase 4: LLM Content Generation

The LLM automatically generates:

1. **Title** - Concise, follows repository patterns (max 80 chars)
2. **Body** - Well-structured Markdown with appropriate sections
3. **Labels** - Selects 1-3 relevant labels from available options

##### Phase 5: Preview and Confirmation
**[INTERACTIVE PAUSE]** (skipped if `auto_confirm=true`)

```bash
# 📝 GitHub Issue Preview

Ready to create the following issue in **workflows-mcp**:

---

## Title
Fix: Prompt block response not saved in checkpoint state

## Labels
bug, interactive

## Body
```markdown
## Description
When executing workflows containing Prompt blocks, the user's response is not
properly persisted to the checkpoint state...

## Steps to Reproduce
1. Create workflow with Prompt block
2. Execute workflow
3. Provide response when prompted
4. Check checkpoint state
5. Resume workflow

## Expected Behavior
The response should be saved to checkpoint.context and accessible via
{{blocks.prompt_id.outputs.response}} in subsequent blocks.

## Actual Behavior
The outputs field is empty after resume, causing variable resolution to fail.
```markdown
```

---

**Create this issue?**

Respond:
- **'yes'** to create the issue
- **'no'** to cancel
```bash

##### Phase 6: Issue Creation

If confirmed (or auto-confirmed), creates the issue and returns:

```json
{
  "status": "success",
  "outputs": {
    "created": true,
    "issue_url": "https://github.com/user/repo/issues/123",
    "title": "Fix: Prompt block response not saved in checkpoint state",
    "labels": "bug,interactive"
  }
}
```

#### Tips for Best Results

**For Bug Reports**:
- Describe the exact steps to reproduce
- Include error messages or logs
- Mention expected vs actual behavior
- Provide environment details if relevant

**For Feature Requests**:
- Explain the use case and motivation
- Describe the desired behavior clearly
- Mention any alternatives you've considered
- Explain how this benefits the project

**For Questions/Discussion**:
- Be specific about what you're trying to understand
- Provide context about your use case
- Reference relevant documentation you've already read

**General Tips**:
- More detail = better generated content
- The LLM learns from recent issues (check existing patterns)
- You can always say 'no' at the preview step if the generated content isn't right
- Use descriptive, specific language in your description

#### Workflow Architecture

```text
Prerequisite Checks
   ↓
Repository Context (Parallel)
   ├─ repo info
   ├─ labels
   └─ recent issues
   ↓
[PAUSE] User Description
   ↓
LLM Generation
   ├─ Title
   ├─ Body
   └─ Labels
   ↓
[PAUSE] Preview & Confirm
   ↓
GitHub Issue Creation
   ↓
Return issue URL
```

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `repo_path` | str | `"."` | Path to git repository |
| `auto_confirm` | bool | `false` | Skip confirmation prompt |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `created` | bool | Whether issue was created successfully |
| `issue_url` | str | URL of the created GitHub issue |
| `title` | str | Generated issue title |
| `labels` | str | Applied labels (comma-separated) |

#### Error Handling

The workflow includes robust error handling:

- **Missing `gh` CLI**: Clear message with installation link
- **Not in git repo**: Helpful error directing to repository
- **Not authenticated**: Instructions to run `gh auth login`
- **Issue creation fails**: Exit code 1 with error details

All errors fail fast with actionable error messages.

#### Security Considerations

This workflow follows security best practices:

✅ **Shell Injection Prevention**: All user input passed via environment variables, not command interpolation

✅ **File-based Body**: Issue body written to file, not passed as command argument (handles special characters safely)

✅ **No Code Execution**: User descriptions are never evaluated as code

✅ **GitHub CLI Auth**: Relies on `gh` CLI's secure authentication

#### Future Enhancements (v2)

Planned features for future versions:

- **Mode Selection**: Quick mode vs Detailed mode (step-by-step)
- **Issue Templates**: Detect and use `.github/ISSUE_TEMPLATE/` templates
- **Duplicate Detection**: Search for similar issues before creating
- **Assignee Selection**: Assign to team members
- **Milestone/Project**: Link to milestones and project boards
- **Edit Support**: Allow editing generated content before creation
- **Multi-repository**: Create issues across multiple repositories

#### Troubleshooting

**"GitHub CLI (gh) is not installed"**
- Install from: https://cli.github.com/
- macOS: `brew install gh`
- Verify: `gh --version`

**"Not inside a git repository"**
- Navigate to your git repository directory
- Or specify path: `{"repo_path": "/path/to/repo"}`

**"GitHub CLI is not authenticated"**
- Run: `gh auth login`
- Follow authentication prompts
- Verify: `gh auth status`

**"jq: command not found"**
- Install jq: `brew install jq` (macOS) or `apt-get install jq` (Linux)
- Required for JSON parsing in context gathering phase

**"Issue creation failed" after confirmation**
- Check `gh` CLI is authenticated: `gh auth status`
- Verify repository access: `gh repo view`
- Check label names exist: `gh label list`

#### Examples

**Example 1: Bug Report**

Description:
```bash
Bug: The workflow executor crashes when a block returns None instead of a Result object.

Context: This happens when a custom executor doesn't properly handle edge cases.

Expected: Should raise a clear validation error during workflow validation.

Actual: RuntimeError with cryptic traceback during execution.

Additional: Happens with custom executor blocks, built-in blocks work fine.
```

Generated Issue:
- **Title**: "Fix: Workflow executor crashes on None return from block"
- **Labels**: "bug", "executor"
- **Body**: Well-structured markdown with all sections

**Example 2: Feature Request**

Description:
```bash
Feature request: Add support for conditional workflow execution based on git branch.

Context: I want workflows to behave differently on main vs feature branches.

Expected: Add {{metadata.git_branch}} variable that can be used in conditions.

Use case: Run full CI on main, fast checks on feature branches.
```

Generated Issue:
- **Title**: "Add git branch context to workflow metadata"
- **Labels**: "enhancement", "workflows"
- **Body**: Clear feature description with use cases

## Related Documentation

- [Workflows MCP Documentation](../../README.md)
- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [Interactive Workflows Guide](../../../docs/interactive-workflows.md)
- [Prompt Block Reference](../../../docs/block-executors.md#prompt)
