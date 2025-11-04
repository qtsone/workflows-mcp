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

### Configuration

**Option 1: Install via QTS Marketplace (Claude Desktop & Claude Code)**

Install the `workflows` plugin from the [qtsone marketplace](https://github.com/qtsone/marketplace) which includes:
- 🤖 **workflows-specialist agent** - Dedicated agent for workflow orchestration
- 📚 **workflows-expert skill** - Comprehensive knowledge base and best practices
- ⚙️ **MCP auto-configuration** - Automatic workflows-mcp server setup

```bash
# Add the marketplace (one-time setup)
/plugin marketplace add qtsone/marketplace

# Install the workflows plugin
/plugin install workflows@qtsone
```

The plugin automatically configures the MCP server with custom workflow directories: `~/.workflows` and `./.workflows`

**Option 2: Manual MCP Configuration (Any MCP-Compatible LLM)**

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": [
        "workflows-mcp",
        "--refresh"
      ],
      "env": {
        "WORKFLOWS_TEMPLATE_PATHS": "~/.workflows,./.workflows",
        "WORKFLOWS_LOG_LEVEL": "INFO",
        "WORKFLOWS_MAX_RECURSION_DEPTH": "50"
      }
    }
  }
}
```

All `env` variables are optional. The `--refresh` flag is recommended to ensure `uvx` always fetches the latest version of `workflows-mcp`. For Gemini CLI, this configuration would be in `~/.gemini/settings.json`.

Restart your LLM client, and you're ready to go!

## ✨ Features

- **DAG-Based Workflows**: Automatic dependency resolution and parallel execution
- **🔐 Secrets Management**: Server-side credential handling with automatic redaction (v5.0.0+)
- **🤖 LLM Integration**: Native LLM API calls with schema validation and retry logic
- **Workflow Composition**: Reusable workflows via Workflow blocks
- **Conditional Execution**: Boolean expressions for dynamic control flow
- **Variable Resolution**: Five-namespace system (inputs, metadata, blocks, secrets, __internal__)
- **File Operations**: CreateFile, ReadFile, RenderTemplate with Jinja2
- **HTTP Integration**: HttpCall for REST API interactions
- **Interactive Workflows**: Prompt blocks for user input
- **MCP Integration**: Exposed as tools for LLM agents via Model Context Protocol

### 🔐 Secrets Management

Securely manage API keys, passwords, and tokens in workflows:

```yaml
blocks:
  - id: call_github_api
    type: HttpCall
    inputs:
      url: "https://api.github.com/user"
      headers:
        Authorization: "Bearer {{secrets.GITHUB_TOKEN}}"
```

**Key Features:**

- ✅ **Server-side resolution** - Secrets never reach LLM context
- ✅ **Automatic redaction** - Secret values sanitized from all outputs
- ✅ **Fail-fast behavior** - Missing secrets cause immediate workflow failure
- ✅ **Audit logging** - Comprehensive tracking for compliance

**Configuration:**

```json
{
  "mcpServers": {
    "workflows": {
      "env": {
        "WORKFLOW_SECRET_GITHUB_TOKEN": "ghp_xxx",
        "WORKFLOW_SECRET_OPENAI_API_KEY": "sk-xxx"
      }
    }
  }
}
```

### 🤖 LLM Integration

Call LLM APIs directly from workflows with automatic retry logic and JSON schema validation:

```yaml
blocks:
  - id: extract_data
    type: LLMCall
    inputs:
      provider: openai
      model: gpt-4o-mini
      api_key: "{{secrets.OPENAI_API_KEY}}"
      prompt: "Extract key information from: {{inputs.text}}"
      response_schema:
        type: object
        required: [summary, confidence]
        properties:
          summary: {type: string}
          confidence: {type: number, minimum: 0, maximum: 1}
      max_retries: 3
```

**Supported Providers:**

- **OpenAI** - API access with native Structured Outputs
- **Anthropic** - API access for Claude models
- **Gemini** - API access for Gemini models
- **Ollama** - Local Ollama server API
- **OpenAI-compatible** - LM Studio, vLLM, and other compatible endpoints

**Key Features:**

- ✅ **Native schema validation** - Send JSON Schema to API for guaranteed structure
- ✅ **Automatic retry** - Exponential backoff with validation feedback loop
- ✅ **Token tracking** - Monitor usage and costs
- ✅ **Secrets integration** - Secure API key management

**For Local CLI Tools:**

To use local Claude CLI or Gemini CLI instead of APIs, use the `llm-process` workflow template which wraps CLI commands.

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
    type: Shell
    inputs:
      command: printf "Hello, {{inputs.name}}!"

outputs:
  greeting:
    value: "{{blocks.greet.outputs.stdout}}"
    type: str
    description: "The greeting message"
```

### Workflow Outputs

Workflow outputs support automatic type coercion, allowing you to declare the expected type and get properly typed values:

```yaml
outputs:
  output_name:
    value: "{{blocks.block_id.outputs.field}}"  # Variable expression
    type: str                                     # Type declaration (optional)
    description: "Human-readable description"    # Documentation (optional)
```

**Supported Types:**

- **str** - Text values (default if type not specified)
- **num** - Numeric values (integer or float - JSON doesn't distinguish)
- **bool** - Boolean values (true/false)
- **list** - List/array values
- **dict** - Dictionary/object values
- **json** - Parse JSON string into dict/list/str/num/bool/None

### Key Features

**Variable Substitution:**

Reference inputs, block outputs, and metadata anywhere in your workflow.

Examples:

- [Using workflow inputs](tests/workflows/core/variable-resolution/inputs.yaml) - `{{inputs.field_name}}`
- [Using block outputs](tests/workflows/core/variable-resolution/block-outputs.yaml) - `{{blocks.block_id.outputs.field}}`
- [Using metadata](tests/workflows/core/variable-resolution/metadata.yaml) - `{{metadata.workflow_name}}`
- [Variable shortcuts](tests/workflows/core/variable-resolution/shortcuts.yaml) - convenient shorthand syntax

**Conditionals:**

Run blocks only when conditions are met.

Examples:

- [Input-based conditions](tests/workflows/core/conditionals/input-based.yaml) - conditions using workflow inputs
- [Block status conditions](tests/workflows/core/conditionals/block-status.yaml) - conditions based on block execution

**Block Status Shortcuts:**

Use simple shortcuts to check if blocks succeeded, failed, or were skipped.

Examples:

- [Success detection](tests/workflows/core/block-status/success-detection.yaml) - `{{blocks.id.succeeded}}`
- [Failure detection](tests/workflows/core/block-status/failure-detection.yaml) - `{{blocks.id.failed}}`
- [Skip detection](tests/workflows/core/block-status/skip-detection.yaml) - `{{blocks.id.skipped}}`

**Parallel Execution:**

Tasks with no dependencies run in parallel automatically.

Example: [parallel-execution.yaml](tests/workflows/core/dag-execution/parallel-execution.yaml) - blocks with same dependencies execute concurrently

**Workflow Composition & Recursion:**

Workflows can call other workflows, including themselves (recursion supported with depth limits).

Examples:

- [Workflow composition](tests/workflows/core/composition/) - nested workflow patterns
- [Recursive workflows](tests/workflows/core/composition/recursion.yaml) - self-calling workflows

Control recursion depth with `WORKFLOWS_MAX_RECURSION_DEPTH` (default: 50, max: 10000).

### Available Block Types

- **Shell** - Execute shell commands
- **Workflow** - Call another workflow (enables composition and recursion)
- **CreateFile** - Create files with content
- **ReadFile** - Read file contents
- **RenderTemplate** - Process Jinja2 templates
- **HttpCall** - Make HTTP/REST API calls with environment variable substitution
- **Prompt** - Interactive user prompts (with pause/resume)
- **ReadJSONState** / **WriteJSONState** / **MergeJSONState** - Manage JSON state files

## Custom Workflow Templates

Want to add your own workflows? Set the `WORKFLOWS_TEMPLATE_PATHS` environment variable in your Claude Desktop configuration (see [Configure in Claude Desktop](#configure-in-claude-desktop) above).

Your custom workflows override built-in ones if they have the same name. Later directories in the path override earlier ones.

## Configuration Options

### Environment Variables

- **WORKFLOWS_TEMPLATE_PATHS** - Comma-separated list of additional template directories
- **WORKFLOWS_MAX_RECURSION_DEPTH** - Maximum workflow recursion depth (default: 50, range: 1-10000)
- **WORKFLOWS_LOG_LEVEL** - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **WORKFLOW_SECRET_<NAME>** - Define secrets for workflow execution (e.g., `WORKFLOW_SECRET_GITHUB_TOKEN`)

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

### Testing the MCP Server

For interactive testing and debugging, create a `.mcp.json` config file:

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uv",
      "args": ["run", "workflows-mcp"],
      "env": {
        "WORKFLOWS_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

Then use the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector --config .mcp.json --server workflows
```

This opens a web interface where you can test tool calls, inspect responses, and debug workflow execution.

## Architecture

The server uses a **fractal execution model** where workflows and blocks share the same execution context structure. This enables clean composition and recursive workflows.

**Key Components:**

- **WorkflowRunner** - Orchestrates workflow execution
- **BlockOrchestrator** - Executes individual blocks with error handling
- **DAGResolver** - Resolves dependencies and computes parallel execution waves
- **Variable Resolution** - Five-namespace variable system (inputs, blocks, metadata, secrets, __internal__)
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
