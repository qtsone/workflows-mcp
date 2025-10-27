# Workflow Templates Directory

This directory contains pre-packaged YAML workflow templates that are automatically loaded by the MCP Workflows server and exposed as MCP tools to Claude Code.

## Directory Structure

```text
templates/
├── python/         # Python development workflows
├── node/           # Node.js development workflows
├── git/            # Git operations workflows
├── quality/        # QA and testing workflows
├── examples/       # Example/tutorial workflows
└── README.md       # This file
```

## Category Organization

### `python/`

Workflows for Python development tasks:

- Python project initialization
- Virtual environment management
- Dependency management (pip, uv, poetry)
- Python testing and quality checks
- Package building and distribution

### `node/`

Workflows for Node.js development tasks:
- npm/yarn/pnpm project initialization
- Dependency installation and updates
- Testing and linting
- Building and packaging

### `git/`

Git operation workflows:

- Repository initialization
- Branch management
- Worktree creation and cleanup
- Commit creation (conventional commits)
- Pull request workflows

### `quality/`

QA and testing workflows:

- Test execution (unit, integration, E2E)
- Code quality checks (linting, formatting, type checking)
- Security scanning
- Coverage reporting

### `examples/`

Tutorial and demonstration workflows:

- Simple hello-world workflow
- Multi-step workflow examples
- Dependency chain examples
- Variable substitution examples

## YAML Workflow Format

All workflows in this directory follow the YAML workflow specification. For detailed format documentation, see:

**[docs/YAML_WORKFLOW_GUIDE.md](../../docs/YAML_WORKFLOW_GUIDE.md)**

### Minimal Workflow Example

```yaml
name: example-workflow
description: Example workflow for demonstration
version: 1.0.0

inputs:
  - name: message
    type: string
    description: Message to display
    required: true

blocks:
  - id: display
    type: Shell
    inputs:
      command: echo "{{inputs.message}}"
```

## How Workflows Are Loaded

The MCP server automatically discovers and loads workflows from this directory:

1. **Startup**: Server scans all category subdirectories for `.yaml` files
2. **Validation**: Each workflow is validated against the schema (`WorkflowDefinition`)
3. **Registration**: Valid workflows are registered in the `WorkflowRegistry`
4. **Tool Exposure**: Each workflow becomes an MCP tool available to Claude Code

### Workflow Naming Convention

**File naming**: `workflow-name.yaml` (lowercase, hyphen-separated)
**Tool naming**: `execute_workflow_<workflow-name>` (e.g., `execute_workflow_example`)
**Workflow ID**: Derived from filename (e.g., `example-workflow` from `example-workflow.yaml`)

## Adding Custom Workflows

### Step 1: Create YAML File

Create a new `.yaml` file in the appropriate category directory:

```bash
# Example: Add a Python linting workflow
touch templates/python/python-lint.yaml
```

### Step 2: Define Workflow

Follow the YAML workflow format (see [docs/YAML_WORKFLOW_GUIDE.md](../docs/YAML_WORKFLOW_GUIDE.md)):

```yaml
name: python-lint
description: Run Python linting tools
version: 1.0.0

inputs:
  - name: path
    type: string
    description: Path to Python code to lint
    required: true
    default: "."

blocks:
  - id: ruff_check
    type: Shell
    inputs:
      command: ruff check {{inputs.path}}
      working_dir: {{inputs.path}}
```

### Step 3: Restart MCP Server

The server loads workflows at startup. Restart to pick up new workflows:

```bash
# If running in development mode
uv run mcp dev src/workflows_mcp/server.py

# If installed in Claude Desktop
# Restart Claude Desktop to reload the MCP server
```

### Step 4: Verify in Claude Code

The workflow is now available as an MCP tool. In Claude Code:

```text
Please run the python-lint workflow on ./src
```

Claude will discover and execute: `execute_workflow_python_lint(path="./src")`

## Workflow Development Best Practices

### 1. Clear Naming

- **Workflow name**: Descriptive, hyphen-separated (e.g., `python-test-unit`)
- **Block IDs**: Semantic, snake_case (e.g., `run_tests`, `generate_report`)

### 2. Input Validation

- Always specify `type`, `description`, and `required` for inputs
- Provide sensible `default` values where appropriate
- Use validation in blocks to fail fast on invalid inputs

### 3. Documentation

- Write clear `description` for workflow and each input
- Add comments in YAML for complex logic
- Include usage examples in workflow description

### 4. Dependency Management

- Explicitly declare `depends_on` for blocks with dependencies
- Avoid circular dependencies (will fail validation)
- Keep dependency chains shallow for better parallelization

### 5. Error Handling

- Design workflows to fail gracefully
- Provide meaningful error messages in block outputs
- Consider idempotency (safe to re-run)

### 6. Variable Substitution

- Use `{{inputs.name}}` for workflow inputs
- Use `{{block_id.field}}` for cross-block references
- Reference previous block outputs via `{{block_id.output_field}}`

## Workflow Schema Reference

### Top-Level Fields

```yaml
name: string              # Required: Unique workflow identifier
description: string       # Required: Human-readable description
version: string          # Required: Semantic version (e.g., "1.0.0")
inputs: []               # Optional: List of workflow inputs
blocks: []               # Required: List of execution blocks
```

### Input Definition

```yaml
- name: string           # Required: Input identifier
  type: string           # Required: Data type (string, integer, boolean, etc.)
  description: string    # Required: Human-readable description
  required: boolean      # Optional: Default false
  default: any           # Optional: Default value if not provided
```

### Block Definition

```yaml
- id: string             # Required: Unique block identifier
  type: string           # Required: Block type (e.g., Shell)
  inputs: {}             # Required: Block-specific inputs
  depends_on: []         # Optional: List of block IDs this block depends on
```

## Examples and Tutorials

See the `examples/` directory for complete workflow examples:

- `examples/hello-world.yaml` - Minimal workflow
- `examples/multi-step.yaml` - Sequential multi-step workflow
- `examples/parallel.yaml` - Parallel execution with dependencies
- `examples/variables.yaml` - Variable substitution examples

## Troubleshooting

### Workflow Not Loading

**Problem**: Created a workflow but it's not available as an MCP tool.

**Solutions**:

1. Check YAML syntax: `python -m yaml templates/category/workflow.yaml`
2. Verify workflow follows schema (see [docs/YAML_WORKFLOW_GUIDE.md](../docs/YAML_WORKFLOW_GUIDE.md))
3. Check server logs for validation errors
4. Ensure file has `.yaml` extension (not `.yml`)
5. Restart the MCP server

### Validation Errors

**Problem**: Workflow fails validation during loading.

**Solutions**:

1. Review error message for specific validation failure
2. Check required fields: `name`, `description`, `version`, `blocks`
3. Verify all block IDs are unique
4. Ensure `depends_on` references valid block IDs
5. Check input definitions have required fields

### Variable Substitution Failures

**Problem**: Variables not resolving correctly during execution.

**Solutions**:

1. Use correct syntax: `{{inputs.name}}` or `{{block_id.field}}`
2. Ensure referenced block has executed before current block
3. Check field exists in referenced block's output
4. Verify no circular dependencies in variable references

## Contributing Workflows

When contributing new workflows to this project:

1. **Choose appropriate category** - Place in correct subdirectory
2. **Follow naming conventions** - Lowercase, hyphen-separated
3. **Provide complete documentation** - Clear description and input docs
4. **Test thoroughly** - Verify workflow executes successfully
5. **Add examples** - Include usage examples in description
6. **Keep it simple** - Follow YAGNI principle, avoid over-engineering

## Further Reading

- **[YAML Workflow Guide](../docs/YAML_WORKFLOW_GUIDE.md)** - Complete YAML format specification
- **[Architecture Documentation](../ARCHITECTURE.md)** - System architecture and design decisions
- **[CLAUDE.md](../CLAUDE.md)** - Development guide for Claude Code
- **[MCP Specification](https://modelcontextprotocol.io/)** - Model Context Protocol documentation

---

**Ready to create workflows?** Start by exploring the `examples/` directory and referencing the YAML Workflow Guide!
