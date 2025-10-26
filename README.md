# Workflows MCP

**A Model Context Protocol (MCP) server that turns complex automation into simple workflow definitions.**

Think of it as your personal automation assistant—define what you want done in YAML, and this MCP server handles the execution. Perfect for CI/CD pipelines, Python development workflows, git operations, or any multi-step automation you can dream up.

## What's This All About?

Workflows MCP is an MCP server that exposes workflow execution capabilities to AI assistants like Claude. Instead of writing the same bash scripts over and over, you define workflows once and execute them with a single tool call.

Each workflow is a DAG (Directed Acyclic Graph) of tasks that can run in parallel or sequence, with smart variable substitution, conditionals, and the ability to compose workflows together. It's like having GitHub Actions, but integrated directly into your AI assistant.

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install workflows-mcp

# Or with pip
pip install workflows-mcp
```

### Running the Server

```bash
# Directly
workflows-mcp

# Or via Python module
python -m workflows_mcp
```

### Configure in Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": [
        "--from", "workflows-mcp",
        "workflows-mcp"
      ],
      "env": {
        "WORKFLOWS_LOG_LEVEL": "INFO",
        "WORKFLOWS_TEMPLATE_PATHS": "~/my-workflows,/opt/team-workflows",
        "WORKFLOWS_MAX_RECURSION_DEPTH": "50"
      }
    }
  }
}
```

All `env` variables are optional. Omit the entire `env` section for default settings.

Restart Claude Desktop, and you're ready to go!

## What Can You Do With It?

### Built-in Workflows

The server comes with ready-to-use workflows for common tasks:

**Python Development:**

- `python-ci-pipeline` - Complete CI pipeline (setup, lint, test)
- `setup-python-env` - Set up Python environment with dependencies
- `lint-python` - Run ruff and mypy
- `run-pytest` - Execute tests with coverage

**Git Operations:**

- `git-checkout-branch` - Create and checkout branches
- `git-commit` - Stage and commit changes
- `git-status` - Check repository status

**File Processing:**

- `generate-readme` - Create README files from templates
- `process-config` - Transform configuration files

**Examples & Tutorials:**

- `hello-world` - The simplest possible workflow
- `parallel-processing` - Run tasks in parallel
- `conditional-pipeline` - Use conditions to control flow

### Available Tools

Claude can use these MCP tools to work with workflows:

- **execute_workflow** - Run a workflow with inputs
- **execute_inline_workflow** - Execute YAML directly without registration
- **list_workflows** - See all available workflows (optionally filter by tags)
- **get_workflow_info** - Get detailed info about a workflow
- **validate_workflow_yaml** - Validate workflow definitions before running
- **get_workflow_schema** - Get the complete JSON schema

**Checkpoint Management:**

- **resume_workflow** - Resume paused workflows
- **list_checkpoints** - See all saved checkpoints
- **get_checkpoint_info** - Inspect checkpoint details
- **delete_checkpoint** - Clean up old checkpoints

## Creating Your Own Workflows

Workflows are defined in YAML. Here's the simplest example:

```yaml
name: hello-world
description: Simple hello world workflow
tags: [example, basic]

inputs:
  name:
    type: str
    description: Name to greet
    default: "World"

blocks:
  - id: greet
    type: EchoBlock
    inputs:
      message: "Hello, ${inputs.name}!"

outputs:
  greeting: "${blocks.greet.outputs.echoed}"
```

### Key Features

**Variable Substitution:**

Reference inputs, block outputs, and metadata anywhere in your workflow:

```yaml
# From inputs
message: "${inputs.project_name}"

# From block outputs
path: "${blocks.setup.outputs.venv_path}"

# From metadata
timestamp: "${metadata.start_time}"
```

**Conditionals:**

Run blocks only when conditions are met:

```yaml
- id: deploy
  type: Shell
  inputs:
    command: ./deploy.sh
  condition: "${blocks.run_tests.succeeded}"
  depends_on: [run_tests]
```

**Parallel Execution:**

Tasks with no dependencies run in parallel automatically:

```yaml
blocks:
  - id: lint
    type: Workflow
    inputs:
      workflow: "lint-python"
    depends_on: [setup]

  - id: test
    type: Workflow
    inputs:
      workflow: "run-pytest"
    depends_on: [setup]  # Both run in parallel after setup!
```

**Workflow Composition:**

Workflows can call other workflows:

```yaml
- id: ci_pipeline
  type: Workflow
  inputs:
    workflow: "python-ci-pipeline"
    inputs:
      project_path: "${inputs.workspace}"
```

### Available Block Types

- **Shell** - Execute shell commands
- **Workflow** - Call another workflow
- **EchoBlock** - Simple echo for testing
- **CreateFile** - Create files with content
- **ReadFile** - Read file contents
- **RenderTemplate** - Process Jinja2 templates
- **Prompt** - Interactive user prompts (with pause/resume)
- **LoadState** / **SaveState** - Manage JSON state

## Custom Workflow Templates

Want to add your own workflows? Set the `WORKFLOWS_TEMPLATE_PATHS` environment variable:

```bash
export WORKFLOWS_TEMPLATE_PATHS="~/my-workflows,/opt/company-workflows"
```

Or in your MCP configuration:

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": ["--from", "workflows-mcp", "workflows-mcp"],
      "env": {
        "WORKFLOWS_TEMPLATE_PATHS": "~/my-workflows,/opt/company-workflows"
      }
    }
  }
}
```

Your custom workflows override built-in ones if they have the same name. Later directories in the path override earlier ones.

## Configuration Options

### Environment Variables

- **WORKFLOWS_TEMPLATE_PATHS** - Comma-separated list of additional template directories
- **WORKFLOWS_MAX_RECURSION_DEPTH** - Maximum workflow recursion depth (default: 50, range: 1-10000)
- **WORKFLOWS_LOG_LEVEL** - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Example Usage with Claude

Once configured, you can ask Claude things like:

> "Run the Python CI pipeline on my project"
>
> "List all available workflows tagged with 'python'"
>
> "Execute the hello-world workflow with name='Claude'"
>
> "Show me what the python-ci-pipeline workflow does"

Claude will use the appropriate MCP tools to execute workflows, check their status, and report results.

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=workflows_mcp --cov-report=term-missing
```

### Code Quality

```bash
# Type checking
uv run mypy src/workflows_mcp/

# Linting
uv run ruff check src/workflows_mcp/

# Formatting
uv run ruff format src/workflows_mcp/
```

## Architecture

The server uses a **fractal execution model** where workflows and blocks share the same execution context structure. This enables clean composition and recursive workflows.

**Key Components:**

- **WorkflowRunner** - Orchestrates workflow execution
- **BlockOrchestrator** - Executes individual blocks with error handling
- **DAGResolver** - Resolves dependencies and computes parallel execution waves
- **Variable Resolution** - Four-namespace variable system (inputs, blocks, metadata, internal)
- **Checkpoint System** - Pause/resume support for interactive workflows

Workflows execute in **waves**—blocks with no dependencies or whose dependencies are satisfied run in parallel within each wave, maximizing efficiency.

## Why Use This?

**For AI Assistants:**

- Consistent, reliable automation without reinventing the wheel
- Complex operations become simple tool calls
- Built-in error handling and validation

**For Developers:**

- Define workflows once, use everywhere
- Compose complex pipelines from simple building blocks
- YAML definitions are easy to read and maintain
- Parallel execution out of the box

**For Teams:**

- Share common workflows across projects
- Custom templates for company-specific processes
- Version control your automation

## License

[AGPL-3.0-or-later](./LICENSE)

## Links

- **GitHub**: [github.com/qtsone/workflows-mcp](https://github.com/qtsone/workflows-mcp)
- **Issues**: [github.com/qtsone/workflows-mcp/issues](https://github.com/qtsone/workflows-mcp/issues)
- **MCP Protocol**: [modelcontextprotocol.io](https://modelcontextprotocol.io/)
