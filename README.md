# Workflows MCP

**Automate anything with simple YAML workflows for your AI assistant.**

Workflows MCP is a [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that lets you define powerful, reusable automation workflows in YAML and execute them through AI assistants like Claude. Think of it as GitHub Actions for your AI assistant—define your automation once, run it anywhere.

---

## Table of Contents

- [What Does This Give Me?](#what-does-this-give-me)
- [Why Should I Use This?](#why-should-i-use-this)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [What Can I Build?](#what-can-i-build)
- [Creating Your First Workflow](#creating-your-first-workflow)
- [Key Features](#key-features)
- [Built-in Workflows](#built-in-workflows)
- [Available MCP Tools](#available-mcp-tools)
- [Configuration Reference](#configuration-reference)
- [Examples](#examples)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What Does This Give Me?

Workflows MCP transforms your AI assistant into an automation powerhouse. Instead of manually running commands or writing repetitive scripts, you define workflows in YAML, and your AI assistant executes them for you.

**Real-world example:**
```text
You: "Run the Python CI pipeline on my project"
Claude: *Executes workflow that sets up environment, runs linting, and runs tests*
Claude: "✓ All checks passed! Linting: ✓, Tests: ✓, Coverage: 92%"
```

---

## Why Should I Use This?

### For Non-Technical Users
- **No coding required** - Define automation in simple YAML
- **Reusable templates** - Use pre-built workflows for common tasks
- **AI-powered execution** - Just ask your AI assistant in plain English

### For Developers
- **DRY principle** - Define once, use everywhere
- **Parallel execution** - Automatic optimization of independent tasks
- **Type-safe** - Validated inputs and outputs
- **Composable** - Build complex workflows from simple building blocks

### For Teams
- **Shared automation** - Version control your workflows
- **Consistent processes** - Everyone uses the same tested workflows
- **Custom templates** - Build company-specific automation libraries

---

## Quick Start

### Step 1: Install

Choose one of these installation methods:

**Option A: Using `uv` (Recommended - Faster)**
```bash
uv pip install workflows-mcp
```

**Option B: Using `pip`**
```bash
pip install workflows-mcp
```

**Requirements:**
- Python 3.12 or higher
- That's it!

### Step 2: Configure Your AI Assistant

#### For Claude Desktop or Claude Code

**Method 1: QTS Marketplace (Easiest)**

Install the complete workflows plugin with agent, skills, and auto-configuration:

```bash
# Add the marketplace (one-time setup)
/plugin marketplace add qtsone/marketplace

# Install the workflows plugin
/plugin install workflows@qtsone
```

This automatically configures the MCP server to look for custom workflows in `~/.workflows` and `./.workflows` (optional directories you can create if you want to add custom workflows).

**Method 2: Manual Configuration**

Add this to your Claude Desktop config file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": ["workflows-mcp", "--refresh"],
      "env": {
        "WORKFLOWS_TEMPLATE_PATHS": "~/.workflows,./.workflows"
      }
    }
  }
}
```

**Note:** The `WORKFLOWS_TEMPLATE_PATHS` directories are optional. Only create them if you want to add custom workflows. The server works perfectly fine with just the built-in workflows. We recommend using `~/.workflows` as it's also the default location for optional LLM configuration (`llm-config.yml`).

#### For Other MCP-Compatible AI Assistants

The configuration is similar. For example, Gemini CLI users would add this to `~/.gemini/settings.json` with the same structure.

### Step 3: Restart and Test

1. Restart your AI assistant (e.g., Claude Desktop)
2. Try it out:

```text
You: "List all available workflows"
```

Your AI assistant will show you all the built-in workflows ready to use!

---

## How It Works

Workflows MCP operates on three simple concepts:

### 1. **Workflows** (The What)
YAML files that define what you want to automate. Each workflow has:
- **Inputs** - Parameters users can customize
- **Blocks** - Individual tasks to execute
- **Outputs** - Results returned to the user

### 2. **Blocks** (The How)
Individual tasks within a workflow. Available block types:
- `Shell` - Run shell commands
- `LLMCall` - Call AI/LLM APIs
- `HttpCall` - Make HTTP requests
- `CreateFile`, `ReadFile`, `RenderTemplate` - File operations
- `Workflow` - Call other workflows (composition)
- `Prompt` - Interactive user prompts
- `ReadJSONState`, `WriteJSONState`, `MergeJSONState` - State management

### 3. **Execution** (The Magic)
The server automatically:
- Analyzes dependencies between blocks
- Runs independent blocks in parallel
- Handles errors gracefully
- Substitutes variables dynamically

**Example Flow:**
```yaml
# Python CI Pipeline
setup_env → run_linting ↘
                         → validate_results
setup_env → run_tests   ↗
```

Tasks run in parallel when possible, saving time!

---

## What Can I Build?

The possibilities are endless. Here are some examples:

### Development Automation
- **CI/CD Pipelines** - Automated testing, linting, building
- **Code Quality Checks** - Run multiple linters and formatters in parallel
- **Deployment Workflows** - Build, test, and deploy applications

### Git Operations
- **Smart Branch Management** - Create feature branches with proper naming
- **Automated Commits** - Stage files and commit with generated messages
- **Repository Analysis** - Analyze changes, detect patterns

### Data Processing
- **File Transformations** - Process and transform files in batch
- **API Orchestration** - Chain multiple API calls together
- **Report Generation** - Generate reports from templates

### AI-Powered Tasks
- **Content Analysis** - Use LLMs to analyze and extract insights
- **Code Generation** - Generate code based on specifications
- **Automated Review** - Review code, documents, or data

---

## Creating Your First Workflow

Let's create a simple workflow that greets a user:

### 1. Create a YAML file

First, create the workflows directory (if it doesn't exist):
```bash
mkdir -p ~/.workflows
```

Then save this as `~/.workflows/greet-user.yaml`:

```yaml
name: greet-user
description: A friendly greeting workflow
tags: [example, greeting]

inputs:
  name:
    type: str
    description: Name of the person to greet
    default: "World"

  language:
    type: str
    description: Language for greeting (en, es, fr)
    default: "en"

blocks:
  - id: create_greeting
    type: Shell
    inputs:
      command: |
        case "{{inputs.language}}" in
          es) echo "¡Hola, {{inputs.name}}!" ;;
          fr) echo "Bonjour, {{inputs.name}}!" ;;
          *) echo "Hello, {{inputs.name}}!" ;;
        esac

outputs:
  greeting:
    value: "{{blocks.create_greeting.outputs.stdout}}"
    type: str
    description: The personalized greeting
```

### 2. Restart your AI assistant

The workflow is automatically discovered from `~/.workflows/`

### 3. Use it!

```text
You: "Run the greet-user workflow with name=Alice and language=es"
Claude: *Executes workflow*
Claude: "¡Hola, Alice!"
```

### Understanding the Workflow

- **inputs** - Define customizable parameters
- **blocks** - Each block is a task (here, running a shell command)
- **{{inputs.name}}** - Variable substitution (replaced at runtime)
- **outputs** - What gets returned to the user

---

## Key Features

### 🚀 Smart Parallel Execution

The server automatically detects which tasks can run in parallel:

```yaml
blocks:
  - id: setup
    type: Shell
    inputs:
      command: "npm install"

  # These run in PARALLEL after setup
  - id: lint
    type: Shell
    depends_on: [setup]
    inputs:
      command: "npm run lint"

  - id: test
    type: Shell
    depends_on: [setup]
    inputs:
      command: "npm test"

  # This waits for both
  - id: report
    type: Shell
    depends_on: [lint, test]
    inputs:
      command: "echo 'All checks passed!'"
```

### 🔐 Secure Secrets Management

Store sensitive data like API keys securely:

```json
{
  "mcpServers": {
    "workflows": {
      "env": {
        "WORKFLOW_SECRET_GITHUB_TOKEN": "ghp_your_token_here",
        "WORKFLOW_SECRET_OPENAI_API_KEY": "sk-your_key_here"
      }
    }
  }
}
```

Use in workflows:

```yaml
blocks:
  - id: call_api
    type: HttpCall
    inputs:
      url: "https://api.github.com/user"
      headers:
        Authorization: "Bearer {{secrets.GITHUB_TOKEN}}"
```

**Security features:**
- ✅ Secrets never appear in LLM context
- ✅ Automatic redaction in all outputs
- ✅ Server-side resolution only
- ✅ Fail-fast on missing secrets

### 🤖 LLM Integration

Call AI models directly from workflows with automatic retry and validation:

**Profile-based (Recommended):**

Optionally create `~/.workflows/llm-config.yml` (you'll need to create the directory first if it doesn't exist):

```yaml
version: "1.0"

providers:
  openai:
    type: openai
    api_key_secret: "OPENAI_API_KEY"  # References WORKFLOW_SECRET_OPENAI_API_KEY env var

  local:
    type: openai
    api_url: "http://localhost:1234/v1/chat/completions"

profiles:
  default:
    provider: openai
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 4000

default_profile: default
```

**Important:** The `api_key_secret` value (e.g., `"OPENAI_API_KEY"`) is **not** your actual API key. It's the name that references your environment variable `WORKFLOW_SECRET_OPENAI_API_KEY` in your MCP server configuration. The actual key value should be set as an environment variable (see [Secure Secrets Management](#secure-secrets-management)).

Use in workflows:

```yaml
blocks:
  - id: analyze_code
    type: LLMCall
    inputs:
      profile: default
      prompt: "Analyze this code and suggest improvements: {{inputs.code}}"
      response_schema:
        type: object
        required: [summary, suggestions]
        properties:
          summary: {type: string}
          suggestions: {type: array, items: {type: string}}
```

**Supported providers:** OpenAI, Anthropic, Gemini, Ollama, OpenAI-compatible (LM Studio, vLLM)

### 🔁 Universal Iteration (for_each)

Iterate over collections with ANY block type:

```yaml
blocks:
  - id: process_files
    type: Shell
    for_each:
      file1: {path: "src/main.py", lines: 150}
      file2: {path: "src/utils.py", lines: 80}
    for_each_mode: parallel  # or sequential
    max_parallel: 3
    continue_on_error: true
    inputs:
      command: "echo Processing {{each.key}}: {{each.value.path}}"
```

**Iteration variables:**
- `{{each.key}}` - Current key ("file1", "file2")
- `{{each.value}}` - Current value
- `{{each.index}}` - Zero-based position (0, 1, 2...)
- `{{each.count}}` - Total iterations

**Access results:**
```yaml
# Use bracket notation
{{blocks.process_files["file1"].outputs.stdout}}

# Block-level aggregations
{{blocks.process_files.succeeded}}  # All succeeded?
{{blocks.process_files.metadata.count}}  # Total count
```

### 🔄 Workflow Composition

Build complex workflows from simple reusable pieces:

```yaml
name: full-ci-pipeline
blocks:
  - id: setup
    type: Workflow
    inputs:
      workflow: setup-python-env
      inputs:
        python_version: "3.12"

  - id: lint
    type: Workflow
    depends_on: [setup]
    inputs:
      workflow: lint-python

  - id: test
    type: Workflow
    depends_on: [setup]
    inputs:
      workflow: run-pytest
```

**Supports recursion** with configurable depth limits!

### 📝 Conditional Execution

Run blocks only when conditions are met:

```yaml
blocks:
  - id: check_env
    type: Shell
    inputs:
      command: "echo $ENVIRONMENT"

  - id: deploy_prod
    type: Shell
    condition: "{{blocks.check_env.outputs.stdout}} == 'production'"
    depends_on: [check_env]
    inputs:
      command: "./deploy.sh production"

  - id: deploy_staging
    type: Shell
    condition: "{{blocks.check_env.outputs.stdout}} != 'production'"
    depends_on: [check_env]
    inputs:
      command: "./deploy.sh staging"
```

### 💬 Interactive Workflows

Pause workflows to get user input:

```yaml
blocks:
  - id: ask_confirmation
    type: Prompt
    inputs:
      prompt: "Deploy to production? (yes/no)"

  - id: deploy
    type: Shell
    condition: "{{blocks.ask_confirmation.outputs.response}} == 'yes'"
    depends_on: [ask_confirmation]
    inputs:
      command: "./deploy.sh"
```

Workflows pause when they reach a `Prompt` block. Use `resume_workflow(job_id, response)` to continue execution with the user's input.

**Example:**
```text
1. Execute workflow → Pauses at Prompt block, returns job_id
2. Check status: get_job_status(job_id="job_abc123")
3. Resume: resume_workflow(job_id="job_abc123", response="yes")
```

### ⚡ Async Execution

Execute long-running workflows without blocking:

```yaml
# Start async workflow (returns immediately)
execute_workflow(
    workflow="long-running-deployment",
    inputs={...},
    mode="async",
    timeout=3600  # 1 hour timeout (optional)
)
# Returns: {"job_id": "job_abc123", "status": "queued"}

# Check progress
get_job_status(job_id="job_abc123")
# Returns: {"status": "running", ...}

# Cancel if needed
cancel_job(job_id="job_abc123")

# List all jobs
list_jobs(status="running")

# Monitor queue health
get_queue_stats()
```

**Use cases:**
- Long CI/CD pipelines
- Large-scale data processing
- Multi-stage deployments
- Resource-intensive analysis

---

## Built-in Workflows

The server includes many ready-to-use workflows:

### 📋 A Note on Quality & Testing

**Core workflows** (like `python-ci-pipeline`, `lint-python`, `run-pytest`, `git-checkout-branch`) are actively used and maintained. However, not all built-in workflows have comprehensive automated tests yet.

**Why?** Some workflows interact with external systems (GitHub API, package managers, deployment services) which makes them difficult to test in CI without real credentials and infrastructure. We're actively working on expanding test coverage and will be refining the workflow library over time.

**Our commitment:** The workflow library is evolving. We're focusing on testing and validating each workflow thoroughly before recommending them for production use.

**How to stay safe:**
- ✅ **Inspect before running** - Use `get_workflow_info` to see exactly what a workflow does
- ✅ **Start with simple workflows** - Try core Python/Git workflows first
- ✅ **Test in safe environments** - Run workflows on test projects before production
- ✅ **Create your own** - The safest workflows are ones you write and understand yourself
- ✅ **Check the source** - Workflows are just YAML files in `src/workflows_mcp/templates/`

**Looking for battle-tested examples?** Check out the workflows in `tests/workflows/` - these are thoroughly tested in CI on every PR and demonstrate all the core features reliably.

### Discovering Workflows

The best way to see what workflows are available is to ask your AI assistant:

**List all workflows:**
```text
You: "List all available workflows"
```

Your AI will show you all currently available workflows with their descriptions.

**Get detailed information:**
```text
You: "Show me details about the python-ci-pipeline workflow"
```

This shows inputs, outputs, and what the workflow does - inspect before running!

**Filter by category:**
```text
You: "List workflows tagged with 'python'"
You: "Show me all git workflows"
```

**Popular workflows include:** Python CI pipelines, Git operations (checkout, commit, status), linting tools, test runners, and file operations. The library evolves over time, so use the discovery tools above for the current list.

---

## Available MCP Tools

When you configure workflows-mcp, your AI assistant gets these tools:

### Workflow Execution

- **execute_workflow** - Run a registered workflow by name
  ```bash
  Usage: "Run the python-ci-pipeline workflow on ./my-project"
  Parameters:
    - mode: "sync" (default) or "async"
    - timeout: Optional timeout in seconds for async mode (default: 3600, max: 86400)

  Sync mode: Returns results immediately
  Async mode: Returns job_id for tracking via get_job_status
  ```

- **execute_inline_workflow** - Execute YAML directly without registration
  ```text
  Usage: "Execute this workflow: [YAML content]"
  ```

### Workflow Discovery

- **list_workflows** - List all available workflows (optional tag filtering)
  ```text
  Usage: "List workflows tagged with 'python'"
  ```

- **get_workflow_info** - Get detailed information about a workflow
  ```text
  Usage: "Show me the python-ci-pipeline workflow details"
  ```

### Workflow Validation

- **validate_workflow_yaml** - Validate YAML before execution
  ```text
  Usage: "Validate this workflow YAML: [content]"
  ```

- **get_workflow_schema** - Get the complete JSON schema
  ```text
  Usage: "Show me the workflow schema"
  ```

### Job Management

- **get_job_status** - Get status and outputs of a workflow job
  ```bash
  Usage: "Get status of job_abc123"
  Parameters: job_id (from execute_workflow in async mode or paused workflows)
  ```

- **cancel_job** - Cancel a pending or running job
  ```text
  Usage: "Cancel job_abc123"
  Parameters: job_id
  ```

- **list_jobs** - List workflow jobs with optional filtering
  ```text
  Usage: "List all running jobs" or "List completed jobs"
  Parameters:
    - status: Optional filter (queued, running, completed, failed, cancelled, paused)
    - limit: Maximum results (default: 100)
  ```

- **get_queue_stats** - Get queue statistics for monitoring
  ```text
  Usage: "Show queue statistics"
  Returns: IO queue and job queue metrics
  ```

- **resume_workflow** - Resume a paused workflow
  ```text
  Usage: "Resume job_abc123 with response 'yes'"
  Parameters:
    - job_id: ID of paused workflow job
    - response: User's response to the pause prompt
  ```

---

## Configuration Reference

### Environment Variables

Configure the server behavior with these environment variables:

| Variable | Description | Default | Range |
|----------|-------------|---------|-------|
| `WORKFLOWS_TEMPLATE_PATHS` | Comma-separated workflow directories | *(none)* | Valid paths |
| `WORKFLOWS_MAX_RECURSION_DEPTH` | Maximum workflow recursion depth | `50` | `1-10000` |
| `WORKFLOWS_LOG_LEVEL` | Logging verbosity | `INFO` | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `WORKFLOW_SECRET_<NAME>` | Secret value (e.g., `WORKFLOW_SECRET_API_KEY`) | *(none)* | Any string |
| `WORKFLOWS_IO_QUEUE_ENABLED` | Enable serialized I/O operations | `true` | true, false |
| `WORKFLOWS_JOB_QUEUE_ENABLED` | Enable async workflow execution | `true` | true, false |
| `WORKFLOWS_JOB_QUEUE_WORKERS` | Worker pool size for async jobs | `3` | 1-100 |
| `WORKFLOWS_MAX_CONCURRENT_JOBS` | Maximum concurrent jobs | `100` | 1-10000 |

### Example Configuration

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": ["workflows-mcp", "--refresh"],
      "env": {
        "WORKFLOWS_TEMPLATE_PATHS": "~/.workflows,./project-workflows",
        "WORKFLOWS_LOG_LEVEL": "DEBUG",
        "WORKFLOWS_MAX_RECURSION_DEPTH": "100",
        "WORKFLOWS_IO_QUEUE_ENABLED": "true",
        "WORKFLOWS_JOB_QUEUE_ENABLED": "true",
        "WORKFLOWS_JOB_QUEUE_WORKERS": "5",
        "WORKFLOWS_MAX_CONCURRENT_JOBS": "200",
        "WORKFLOW_SECRET_GITHUB_TOKEN": "ghp_xxxxx",
        "WORKFLOW_SECRET_OPENAI_API_KEY": "sk-xxxxx"
      }
    }
  }
}
```

### Custom Workflow Directories

The server loads workflows from:
1. Built-in templates (always loaded)
2. Custom directories (specified in `WORKFLOWS_TEMPLATE_PATHS`, optional)

**Note:** Custom workflow directories are not created automatically. You need to create them manually if you want to add your own workflows. The server works fine without them using only built-in workflows.

**Load order priority:** Later directories override earlier ones by workflow name.

**Example:**
```bash
WORKFLOWS_TEMPLATE_PATHS="~/.workflows,./project-workflows"

# Load order:
# 1. Built-in templates
# 2. ~/.workflows (overrides built-in by name)
# 3. ./project-workflows (overrides both by name)
```

---

## Examples

### Example 1: Simple Shell Command

```yaml
name: disk-usage
description: Check disk usage
tags: [utility, system]

blocks:
  - id: check_disk
    type: Shell
    inputs:
      command: "df -h"

outputs:
  disk_info:
    value: "{{blocks.check_disk.outputs.stdout}}"
    type: str
```

### Example 2: Parallel Processing

```yaml
name: multi-lint
description: Run multiple linters in parallel
tags: [python, linting]

inputs:
  project_path:
    type: str
    default: "."

blocks:
  - id: ruff
    type: Shell
    inputs:
      command: "ruff check {{inputs.project_path}}"

  - id: mypy
    type: Shell
    inputs:
      command: "mypy {{inputs.project_path}}"

  - id: black
    type: Shell
    inputs:
      command: "black --check {{inputs.project_path}}"

  - id: summary
    type: Shell
    depends_on: [ruff, mypy, black]
    inputs:
      command: |
        echo "Linting Results:"
        echo "  Ruff: {{blocks.ruff.succeeded}}"
        echo "  Mypy: {{blocks.mypy.succeeded}}"
        echo "  Black: {{blocks.black.succeeded}}"

outputs:
  all_passed:
    value: "{{blocks.ruff.succeeded}} and {{blocks.mypy.succeeded}} and {{blocks.black.succeeded}}"
    type: bool
```

### Example 3: API Integration with LLM

```yaml
name: analyze-github-repo
description: Analyze a GitHub repository using AI
tags: [github, ai, analysis]

inputs:
  repo_url:
    type: str
    description: GitHub repository URL
    required: true

blocks:
  - id: fetch_readme
    type: HttpCall
    inputs:
      url: "{{inputs.repo_url}}/raw/main/README.md"
      method: GET

  - id: analyze
    type: LLMCall
    depends_on: [fetch_readme]
    inputs:
      profile: default
      prompt: |
        Analyze this GitHub repository README and provide:
        1. Main purpose
        2. Key technologies used
        3. Installation complexity (1-10)

        README:
        {{blocks.fetch_readme.outputs.response_body}}
      response_schema:
        type: object
        required: [purpose, technologies, complexity]
        properties:
          purpose: {type: string}
          technologies: {type: array, items: {type: string}}
          complexity: {type: number, minimum: 1, maximum: 10}

outputs:
  analysis:
    value: "{{blocks.analyze.outputs.response}}"
    type: dict
```

### Example 4: Conditional Deployment

```yaml
name: smart-deploy
description: Deploy to staging or production based on branch
tags: [deployment, git]

blocks:
  - id: get_branch
    type: Shell
    inputs:
      command: "git branch --show-current"

  - id: run_tests
    type: Shell
    inputs:
      command: "npm test"

  - id: deploy_production
    type: Shell
    condition: "{{blocks.get_branch.outputs.stdout}} == 'main' and {{blocks.run_tests.succeeded}}"
    depends_on: [get_branch, run_tests]
    inputs:
      command: "./deploy.sh production"

  - id: deploy_staging
    type: Shell
    condition: "{{blocks.get_branch.outputs.stdout}} != 'main' and {{blocks.run_tests.succeeded}}"
    depends_on: [get_branch, run_tests]
    inputs:
      command: "./deploy.sh staging"

outputs:
  deployed_to:
    value: "{{blocks.deploy_production.succeeded}} ? 'production' : 'staging'"
    type: str
```

---

## Development

### Running Tests

```bash
# Install development dependencies
uv sync --all-extras

# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=workflows_mcp --cov-report=term-missing

# Run specific test
uv run pytest tests/test_mcp_client.py -v
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

For interactive testing and debugging:

**1. Create `.mcp.json`:**
```json
{
  "mcpServers": {
    "workflows": {
      "command": "uv",
      "args": ["run", "workflows-mcp"],
      "env": {
        "WORKFLOWS_LOG_LEVEL": "DEBUG",
        "WORKFLOWS_TEMPLATE_PATHS": "~/.workflows"
      }
    }
  }
}
```

**2. Run MCP Inspector:**
```bash
npx @modelcontextprotocol/inspector --config .mcp.json --server workflows
```

This opens a web interface for testing tool calls and debugging workflow execution.

### Project Structure

```bash
workflows-mcp/
├── src/workflows_mcp/          # Main source code
│   ├── engine/                  # Workflow execution engine
│   │   ├── executor_base.py     # Base executor class
│   │   ├── executors_core.py    # Shell, Workflow executors
│   │   ├── executors_file.py    # File operation executors
│   │   ├── executors_http.py    # HTTP call executor
│   │   ├── executors_llm.py     # LLM call executor
│   │   ├── executors_state.py   # State management executors
│   │   ├── workflow_runner.py   # Main workflow orchestrator
│   │   ├── dag.py               # DAG resolution
│   │   └── secrets/             # Secrets management
│   ├── templates/               # Built-in workflow templates
│   │   ├── python/              # Python workflows
│   │   ├── git/                 # Git workflows
│   │   ├── node/                # Node.js workflows
│   │   └── ...
│   ├── server.py                # MCP server setup
│   ├── tools.py                 # MCP tool implementations
│   └── __main__.py              # Entry point
├── tests/                       # Test suite
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

---

## Troubleshooting

### Installation Issues

**Problem:** `command not found: workflows-mcp`

**Solution:**
```bash
# Ensure Python 3.12+ is installed
python --version  # Should be 3.12 or higher

# Reinstall with uv
uv pip install --force-reinstall workflows-mcp

# Or verify installation
pip show workflows-mcp
```

**Problem:** Python version too old

**Solution:**
```bash
# Install Python 3.12+ using your package manager
# macOS (Homebrew)
brew install python@3.12

# Ubuntu/Debian
sudo apt install python3.12

# Update uv to use Python 3.12
uv venv --python 3.12
```

### Configuration Issues

**Problem:** Workflows not loading in Claude

**Solution:**
1. Verify config file location (see [Quick Start](#quick-start))
2. Check JSON syntax with a validator
3. Restart Claude Desktop completely
4. Check Claude Desktop logs:
   - macOS: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\logs\`

**Problem:** Custom workflows not found

**Solution:**
```bash
# First, make sure you created the directory
mkdir -p ~/.workflows

# Verify WORKFLOWS_TEMPLATE_PATHS is correct
# Paths should exist and contain .yaml files

# Check directory exists and contains workflows
ls ~/.workflows/

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('~/.workflows/my-workflow.yaml'))"
```

### Workflow Execution Issues

**Problem:** Workflow fails with "not found" error

**Solution:**
```text
You: "List all workflows"
# This shows exact workflow names
# Use the exact name from the list
```

**Problem:** Variables not substituting

**Solution:**
- Check syntax: `{{inputs.name}}` not `{inputs.name}`
- Ensure input is defined in `inputs:` section
- For block outputs: `{{blocks.block_id.outputs.field_name}}`

**Problem:** Secrets not working

**Solution:**
1. Check environment variable name: `WORKFLOW_SECRET_<NAME>`
2. Reference in workflow: `{{secrets.NAME}}` (without prefix)
3. Verify secrets are in MCP server config, not workflow YAML
4. Restart MCP server after adding secrets

### Performance Issues

**Problem:** Workflows running slowly

**Solution:**
- Check if tasks can run in parallel (remove unnecessary `depends_on`)
- Enable debug logging to see execution waves:
  ```text
  You: "Run workflow X with debug=true"
  ```

- Review task dependencies—too many serialized tasks slow execution

**Problem:** Shell commands timing out

**Solution:**
```yaml
blocks:
  - id: long_task
    type: Shell
    inputs:
      command: "./long-script.sh"
      timeout: 600  # Increase timeout (default: 120 seconds)
```

### Debug Mode

Enable detailed logging for troubleshooting:

**Method 1: Environment variable**
```json
{
  "mcpServers": {
    "workflows": {
      "env": {
        "WORKFLOWS_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

**Method 2: Per-execution debug**
```text
You: "Run python-ci-pipeline with debug=true"
```

Debug logs are written to `/tmp/<workflow>-<timestamp>.json` with:
- Block execution details
- Variable resolution steps
- DAG wave analysis
- Timing information

---

## Architecture

Workflows MCP uses a **fractal execution model** where workflows and blocks share the same execution context structure. This enables clean composition and recursive workflows.

### Key Components

- **WorkflowRunner** - Orchestrates workflow execution
- **BlockOrchestrator** - Executes individual blocks with error handling
- **DAGResolver** - Resolves dependencies and computes parallel execution waves
- **Variable Resolution** - Five-namespace variable system:
  - `inputs` - Workflow runtime inputs
  - `blocks` - Block outputs and metadata
  - `metadata` - Workflow metadata
  - `secrets` - Server-side secrets (never exposed to LLM)
  - `__internal__` - Internal execution state

### Execution Model

Workflows execute in **waves**—groups of blocks that can run in parallel:

```text
Wave 1: [setup]
Wave 2: [lint, test]      ← Parallel execution
Wave 3: [validate]
```

This maximizes efficiency by running independent tasks concurrently.

---

## Contributing

We welcome contributions! Here's how you can help:

### Report Issues
- [GitHub Issues](https://github.com/qtsone/workflows-mcp/issues)

### Contribute Workflows
1. Create a new workflow in appropriate category
2. Test thoroughly
3. Submit a pull request

### Improve Documentation
- Fix typos or unclear explanations
- Add examples
- Improve troubleshooting guides

### Code Contributions
- Follow existing code style
- Add tests for new features
- Update documentation

---

## Links

- **GitHub**: [github.com/qtsone/workflows-mcp](https://github.com/qtsone/workflows-mcp)
- **Issues**: [github.com/qtsone/workflows-mcp/issues](https://github.com/qtsone/workflows-mcp/issues)
- **Changelog**: [CHANGELOG.md](./CHANGELOG.md)
- **MCP Protocol**: [modelcontextprotocol.io](https://modelcontextprotocol.io/)

---

## License

[AGPL-3.0-or-later](./LICENSE)

---

## FAQ

### Do I need to know Python to use this?

No! You only need to:
1. Install the package (one command)
2. Configure your AI assistant (copy-paste JSON)
3. Write simple YAML workflows (or use built-in ones)

### Do I need to know what MCP is?

No! Just think of it as a way for your AI assistant to run workflows. The technical details are handled for you.

### Can I use this without Claude?

Yes! Any MCP-compatible AI assistant can use workflows-mcp. The configuration is similar across different assistants.

### Are workflows secure?

Yes! The server includes:
- Server-side secret resolution (secrets never reach the AI)
- Automatic redaction of sensitive data
- Sandboxed execution contexts
- Audit logging

### Are all built-in workflows production-ready?

The core workflows (Python CI, Git operations, basic file operations) are actively used and reliable. Some advanced workflows (GitHub integration, TDD orchestration) are still being refined and tested.

**Best practice:** Always inspect a workflow before using it (`get_workflow_info` tool) and test on non-production systems first. The workflows in `tests/workflows/` are thoroughly tested in CI and are great examples to learn from.

We're actively expanding test coverage and refining the workflow library. Think of it as an evolving collection - contributions welcome!

### Can I share workflows with my team?

Absolutely! Workflows are just YAML files. You can:
- Commit them to version control
- Share them in a company repository
- Publish them as packages

### What's the performance like?

Excellent! The DAG-based execution model automatically parallelizes independent tasks. Many users see 2-3x speedup compared to sequential execution.

### Can workflows call other workflows?

Yes! Use the `Workflow` block type to compose workflows. Recursion is supported with configurable depth limits.

### How do I get help?

1. Check [Troubleshooting](#troubleshooting)
2. Search [GitHub Issues](https://github.com/qtsone/workflows-mcp/issues)
3. Open a new issue with details

---

**Ready to automate?** Install workflows-mcp and start building powerful automation workflows today! 🚀
