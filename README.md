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
- `CreateFile`, `ReadFiles`, `EditFile` - File operations
- `Workflow` - Call other workflows (composition)
- `Prompt` - Interactive user prompts
- `ImageGen` - Generate and edit images (DALL-E, Stable Diffusion)
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

The server automatically detects which tasks can run in parallel. Use `depends_on` to specify dependencies—independent tasks run concurrently for maximum efficiency.

**See examples:** `tests/workflows/core/dag-execution/parallel-execution.yaml`

### 🔐 Secure Secrets Management

Store sensitive data like API keys securely using environment variables. Secrets are resolved server-side and never exposed to the LLM context.

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

Use in workflows: `{{secrets.GITHUB_TOKEN}}`

**Security features:**
- ✅ Secrets never appear in LLM context
- ✅ Automatic redaction in all outputs
- ✅ Server-side resolution only
- ✅ Fail-fast on missing secrets

**See examples:** `tests/workflows/core/secrets/`

### 🎨 Full Jinja2 Template Support

All workflow fields are Jinja2 templates with support for:
- **Variable expressions**: `{{inputs.name}}`, `{{blocks.test.outputs.result}}`
- **Control structures**: `{% if condition %}...{% endif %}`, `{% for item in list %}...{% endfor %}`
- **Custom filters**: `quote`, `prettyjson`, `b64encode`, `hash`, `trim`, `upper`, `lower`, `replace`
- **Global functions**: `now()`, `render()`, `get()`, `len()`, `range()`
- **Safe accessor**: `get(obj, 'path.to.key', default)` - dotted paths, JSON auto-parse, never throws
- **Filter chaining**: `{{inputs.text | trim | lower | replace(' ', '_')}}`

**See examples:** `tests/workflows/core/filters/filters-chaining.yaml`

### 📂 ReadFiles Block with Outline Extraction

Read files with glob patterns, multiple output modes, and automatic outline extraction:

**Features:**
- Glob pattern support (`*.py`, `**/*.ts`)
- Three output modes:
  - `full` - Complete file content (with optional line-range slicing)
  - `outline` - Structural outline + structured sections tree (90-97% context reduction)
  - `summary` - Outline + docstrings/comments
- **Structured sections tree** — Outline mode returns a nested `sections` tree with line ranges, suitable for recursive section-by-section processing
- **Markdown intelligence** (outline/summary mode for `.md` files):
  - **Frontmatter extraction** — Parses YAML frontmatter into a dict
  - **Reference detection** — Extracts wikilinks (`[[target]]`) and file paths in backticks
  - **Code block identification** — Locates fenced code blocks with language tags
  - **Per-section token estimation** — `own_tokens` and `subtree_tokens` on every section node
- **Line-range reading** — `line_start`/`line_end` parameters in full mode to read specific portions of a file
- Gitignore integration and file filtering
- Size limits and file count limits
- Multi-file reading in single block
- Smart output format:
  - Single file: Direct content (string)
  - Multiple files: YAML-formatted structure
  - No files: Empty string

**Outline mode outputs:**

| Output | Description |
|--------|-------------|
| `content` | Display outline string |
| `sections` | Nested section tree with `id`, `heading`, `path`, `level`, `line_start`, `line_end`, `own_start`, `own_end`, `is_leaf`, `children`, `own_tokens`, `subtree_tokens` |
| `max_depth` | Maximum heading depth in document structure |
| `total_sections` | Total number of sections across all levels |
| `frontmatter` | Parsed YAML frontmatter dict (None if absent) |
| `references` | List of `{type, target}` dicts — wikilinks and file paths |
| `code_blocks` | List of `{lang, start_line, end_line}` dicts |

**Example — Structured sections:**
```yaml
- id: read_outline
  type: ReadFiles
  inputs:
    path: "/path/to/document.md"
    mode: outline

- id: show_structure
  type: Shell
  depends_on: [read_outline]
  inputs:
    command: echo "Found {{blocks.read_outline.outputs.total_sections}} sections"
```

**Example — Line-range reading:**
```yaml
- id: read_section
  type: ReadFiles
  inputs:
    path: "/path/to/document.md"
    mode: full
    line_start: 10
    line_end: 25
```

**See examples:** `tests/workflows/core/file-operations/readfiles-test.yaml`

### ✏️ Deterministic File Editing

The `EditFile` block provides powerful deterministic file editing with multiple operation strategies:

**Features:**
- 6 operation types (replace_text, replace_lines, insert_lines, delete_lines, patch, regex_replace)
- Atomic transactions (all-or-nothing by default)
- Automatic backup creation before editing
- Dry-run mode for previewing changes
- Comprehensive diff generation
- Path traversal protection

**See examples:** `tests/workflows/core/file-operations/editfile-operations-test.yaml`

### 🤖 LLM Integration

Call AI models directly from workflows with automatic retry and validation. Configure providers using `~/.workflows/llm-config.yml` (optional).

**Supported providers:** OpenAI, Anthropic, Gemini, Ollama, OpenAI-compatible (LM Studio, vLLM)

**Example:**
```yaml
- id: analyze
  type: LLMCall
  inputs:
    profile: default  # Existing profile from ~/.workflows/llm-config.yml
    prompt: "Analyze this text: {{inputs.text}}"
    response_schema:
      type: object
      required: [sentiment, summary]
      properties:
        sentiment:
          type: string
          enum: [positive, negative, neutral]
        summary:
          type: string
```

**Profile Fallback for Portable Workflows:**

When a workflow requests a profile that doesn't exist in your config, the system automatically falls back to `default_profile` with a warning. This enables **workflow portability** - authors can write workflows with semantic profile names (like `cloud-mini`, `cloud-thinking`, `local`) without requiring specific user configurations.

```yaml
# ~/.workflows/llm-config.yml
profiles:
  my-model:
    provider: openai-cloud
    model: gpt-4o
    max_tokens: 4000

default_profile: my-model
```

### 🎨 Image Generation

Generate, edit, and create variations of images using OpenAI DALL-E or compatible providers (like local Stable Diffusion via OpenAI-compatible API).

**Key Features:**
- **Model Capability System**: Automatic validation of operations and parameters based on model support
- **Profile Support**: Use `~/.workflows/llm-config.yml` to manage providers and models (same as LLMCall)
- **Direct File Saving**: Save generated images directly to disk with `output_file`
- **Pluggable Providers**: Support for OpenAI (DALL-E 2/3) and any OpenAI-compatible image API

**Model Compatibility:**

The executor validates operations and parameters based on model capabilities:

| Model | Generate | Edit | Variation | Optional Parameters |
|-------|----------|------|-----------|---------------------|
| `dall-e-3` | ✓ | ✗ | ✗ | `response_format`, `quality`, `style` |
| `dall-e-2` | ✓ | ✓ | ✓ | `response_format` |
| `gpt-image-1` | ✓ | ✓ | ✗ | None |
| `gpt-image-*` | ✓ | ✓ | ✗ | None |

The executor automatically filters parameters and provides clear error messages when operations are not supported.

**Configuration:**

| Input | Description | Default |
|-------|-------------|---------|
| `prompt` | Text description (required for generate/edit) | - |
| `profile` | Profile name from config (recommended) | `default` |
| `operation` | `generate`, `edit`, or `variation` | `generate` |
| `model` | Model name (e.g., `dall-e-3`) | `dall-e-3` |
| `size` | Image dimensions (e.g., `1024x1024`) | `1024x1024` |
| `quality` | `standard` or `hd` (dall-e-3 only) | `standard` |
| `style` | `vivid` or `natural` (dall-e-3 only) | `vivid` |
| `response_format` | `url` or `b64_json` | `url` |
| `n` | Number of images to generate | `1` |
| `output_file` | Path to save image (e.g., `{{tmp}}/img.png`) | - |
| `image` | Path to base image (for edit/variation) | - |
| `mask` | Path to mask image (transparent areas define where to edit) | - |

**Examples:**

**1. Basic Generation (using profile):**
```yaml
- id: generate_art
  type: ImageGen
  inputs:
    prompt: "A cyberpunk city at night"
    profile: default
    size: "1024x1024"
    output_file: "{{tmp}}/city.png"
```

**2. Image Editing with DALL-E 2:**
```yaml
- id: edit_photo
  type: ImageGen
  inputs:
    operation: edit
    model: dall-e-2  # DALL-E 3 does not support edit
    prompt: "Add a red hat to the person"
    image: "/path/to/photo.png"
    mask: "/path/to/mask.png"
    output_file: "{{tmp}}/edited.png"
```

**3. Using Custom Provider (e.g., Local SD):**
```yaml
- id: local_gen
  type: ImageGen
  inputs:
    prompt: "A medieval castle"
    provider: openai_compatible
    api_url: "http://localhost:8000/v1"
    model: "sd-xl"
    output_file: "{{tmp}}/castle.png"
```

### 🧠 Knowledge Store

Store and retrieve knowledge propositions using PostgreSQL with pgvector for semantic search. The `Knowledge` block type provides hybrid search (vector + full-text), storage with auto-computed embeddings, and token-budgeted context assembly for LLM prompts.

**Requirements:**
- PostgreSQL with `pgvector` extension
- `asyncpg` dependency: `pip install workflows-mcp[postgresql]`
- Embedding profile in `~/.workflows/llm-config.yml`

**Configuration:**

Set connection details once using environment variables in your MCP server config:

```json
{
  "mcpServers": {
    "workflows": {
      "env": {
        "KNOWLEDGE_DB_HOST": "localhost",
        "KNOWLEDGE_DB_PORT": "5432",
        "KNOWLEDGE_DB_NAME": "knowledge_db",
        "KNOWLEDGE_DB_USER": "postgres",
        "KNOWLEDGE_DB_PASSWORD": "your_password",
        "KNOWLEDGE_ORG_ID": "your-org-uuid"
      }
    }
  }
}
```

Tables and the pgvector extension are created automatically on first use.

| Input | Description | Default |
|-------|-------------|---------|
| `op` | Operation: `search`, `store`, `recall`, `forget`, `context` | Required |
| `query` | Search text (required for search/context) | - |
| `content` | Text to store (required for store) | - |
| `source` | Filter by source name (exact or prefix with `*`) | - |
| `categories` | Filter by category UUIDs | - |
| `min_confidence` | Minimum confidence threshold | `0.3` |
| `limit` | Maximum results | `10` |
| `max_tokens` | Token budget for context assembly | `4000` |
| `diversity` | Use MMR for diverse context results | `false` |
| `where` | Filter dict for recall/forget (`source_name`, `lifecycle_state`, `category`, `min_confidence`) | - |
| `order` | Order by fields (e.g., `["relevance_score:desc"]`) | `["created_at:desc"]` |
| `created_after` | Filter: propositions created after this ISO date (recall/forget) | - |
| `created_before` | Filter: propositions created before this ISO date (recall/forget) | - |
| `proposition_ids` | UUIDs of propositions to archive (forget) | - |
| `embedding_profile` | LLM config profile for embeddings | `embedding` |

**Operations:**

**1. Search — Hybrid vector + full-text with RRF fusion:**
```yaml
- id: search_docs
  type: Knowledge
  inputs:
    op: search
    query: "deployment strategies for microservices"
    source: "engineering-docs"
    limit: 10
```

**2. Store — Persist with auto-computed embedding:**
```yaml
- id: save_finding
  type: Knowledge
  inputs:
    op: store
    content: "Redis connection pooling reduces latency by 40% under load"
    source: "performance-tests"
    confidence: 0.85
```

**3. Context — Token-budgeted assembly for LLM prompts:**
```yaml
- id: gather_context
  type: Knowledge
  inputs:
    op: context
    query: "database optimization techniques"
    max_tokens: 2000
    diversity: true
```
Returns clean content only (no metadata) in `context_text`, ready for LLM prompt injection.

**4. Recall — Filtered retrieval with source prefix, confidence, and date ranges:**
```yaml
- id: recent_items
  type: Knowledge
  inputs:
    op: recall
    where:
      source_name: "daily-reports"       # exact match
      # source_name: "workflow:*"         # or prefix match
      # min_confidence: 0.7              # confidence threshold
      # lifecycle_state: active
      # category: "cat-uuid"
    created_after: "2026-01-01"          # ISO date range
    order: ["created_at:desc"]
    limit: 5
```

**5. Forget — Archive propositions by IDs or filter:**
```yaml
# By explicit IDs
- id: cleanup_ids
  type: Knowledge
  inputs:
    op: forget
    proposition_ids:
      - "uuid-1"
      - "uuid-2"

# By filter (archive all from a source)
- id: cleanup_source
  type: Knowledge
  inputs:
    op: forget
    where:
      source_name: "deprecated-docs"
    created_before: "2025-01-01"
```
Returns `archived_count` and `skipped_count` in the output.

**6. Chaining Operations — Store, search, and cleanup in one workflow:**
```yaml
blocks:
  - id: store_finding
    type: Knowledge
    inputs:
      op: store
      content: "Redis connection pooling reduces latency by 40% under load"
      source: "perf-tests"
      confidence: 0.85

  - id: search_related
    type: Knowledge
    depends_on: [store_finding]
    inputs:
      op: search
      query: "database performance optimization"
      limit: 5

  - id: build_context
    type: Knowledge
    depends_on: [store_finding]
    inputs:
      op: context
      query: "performance tuning recommendations"
      max_tokens: 2000

  - id: use_results
    type: Shell
    depends_on: [search_related, build_context]
    inputs:
      command: |
        echo "Found: {{blocks.search_related.outputs.row_count}} results"
        echo "IDs: {{blocks.search_related.outputs.rows | map(attribute='id') | list}}"
        echo "Context ({{blocks.build_context.outputs.tokens_used}} tokens):"
        echo "{{blocks.build_context.outputs.context_text}}"
```

Use `map(attribute='id') | list` to extract values from result rows — this works with any attribute (`id`, `content`, `confidence`, etc.).

**Embedding Configuration:**

Add an `embedding` profile to `~/.workflows/llm-config.yml`:

```yaml
profiles:
  embedding:
    provider: openai-cloud
    model: text-embedding-3-small
    api_key: sk-your-key
```

Supports any OpenAI-compatible embedding API (OpenAI, Ollama, vLLM, etc.).

### 🔁 Universal Iteration (for_each)

Iterate over collections with ANY block type using `for_each`. Supports parallel and sequential execution modes with error handling.

**Iteration variables:**
- `{{each.key}}` - Current key
- `{{each.value}}` - Current value
- `{{each.index}}` - Zero-based position
- `{{each.count}}` - Total iterations

**See examples:** `tests/workflows/core/for_each/for-each-comprehensive.yaml`

### 🔄 Workflow Composition

Build complex workflows from simple reusable pieces using the `Workflow` block type. Supports recursion with configurable depth limits.

**See examples:** `tests/workflows/core/composition/`

### 📝 Conditional Execution

Run blocks only when conditions are met using the `condition` field. Conditions are evaluated as Jinja2 expressions.

**See examples:** `tests/workflows/core/conditionals/`

### 💬 Interactive Workflows

Pause workflows to get user input using the `Prompt` block. Use `resume_workflow(job_id, response)` to continue execution with the user's input.

**See examples:** `tests/workflows/interactive-simple-approval.yaml`

### ⚡ Async Execution

Execute long-running workflows without blocking using `mode="async"`. Track progress with `get_job_status(job_id)` and cancel with `cancel_job(job_id)`.

**Use cases:**
- Long CI/CD pipelines
- Large-scale data processing
- Multi-stage deployments
- Resource-intensive analysis

---

## Built-in Workflows

The server includes many ready-to-use workflows for common tasks.

### 📋 Quality & Testing

**Core workflows** (Python CI, Git operations, file operations) are actively used and thoroughly tested. Some advanced workflows are still being refined.

**Best practice:** Always inspect a workflow before using it (`get_workflow_info` tool) and test on non-production systems first.

**Battle-tested examples:** The workflows in `tests/workflows/core/` are comprehensively tested in CI and demonstrate all core features reliably.

### Discovering Workflows

**List all workflows:**
```text
You: "List all available workflows"
```

**Get detailed information:**
```text
You: "Show me details about the python-ci-pipeline workflow"
```

**Filter by category:**
```text
You: "List workflows tagged with 'python'"
You: "Show me all git workflows"
```

**Popular workflows include:** Python CI pipelines, Git operations (checkout, commit, status), linting tools, test runners, and file operations.

---

## Available MCP Tools

When you configure workflows-mcp, your AI assistant gets these tools:

### Workflow Execution

- **execute_workflow** - Run a registered workflow by name
  - `mode`: "sync" (default) or "async"
  - `timeout`: Optional timeout in seconds for async mode (default: 3600, max: 86400)

- **execute_inline_workflow** - Execute YAML directly without registration

### Workflow Discovery

- **list_workflows** - List all available workflows (optional tag filtering)
- **get_workflow_info** - Get detailed information about a workflow

### Workflow Validation

- **validate_workflow_yaml** - Validate YAML before execution
- **get_workflow_schema** - Get the complete JSON schema

### Job Management

- **get_job_status** - Get status and outputs of a workflow job
- **cancel_job** - Cancel a pending or running job
- **list_jobs** - List workflow jobs with optional filtering
- **get_queue_stats** - Get queue statistics for monitoring
- **resume_workflow** - Resume a paused workflow

### Knowledge Tools

- **search_knowledge** - Hybrid search (vector + full-text + RRF fusion)
  - `query`, `source`, `categories`, `min_confidence`, `limit`

- **store_knowledge** - Persist a new fact with auto-computed embedding
  - `content`, `source`, `confidence` (default 0.8), `categories`

- **recall_knowledge** - Filter-based retrieval (no semantic search)
  - `source`, `categories`, `lifecycle_state`, `min_confidence`, `limit`, `order`

- **forget_knowledge** - Archive propositions (transition to ARCHIVED state)
  - `proposition_ids`, `reason`

- **knowledge_context** - Token-budgeted context assembly for LLM prompts
  - `query`, `source`, `categories`, `max_tokens`, `diversity`

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
| `KNOWLEDGE_DB_HOST` | Knowledge DB PostgreSQL host | `localhost` | Hostname/IP |
| `KNOWLEDGE_DB_PORT` | Knowledge DB PostgreSQL port | `5432` | 1-65535 |
| `KNOWLEDGE_DB_NAME` | Knowledge DB database name | `knowledge_db` | Valid DB name |
| `KNOWLEDGE_DB_USER` | Knowledge DB username | *(none)* | Any string |
| `KNOWLEDGE_DB_PASSWORD` | Knowledge DB password | *(none)* | Any string |
| `KNOWLEDGE_ORG_ID` | Organization ID for knowledge scoping | *(none)* | UUID string |

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

---

## Examples

For comprehensive examples demonstrating all features, see the test workflows in `tests/workflows/core/`:

- **File operations**: `tests/workflows/core/file-operations/`
- **Parallel execution**: `tests/workflows/core/dag-execution/`
- **Conditionals**: `tests/workflows/core/conditionals/`
- **Composition**: `tests/workflows/core/composition/`
- **Secrets**: `tests/workflows/core/secrets/`
- **Filters**: `tests/workflows/core/filters/`
- **Iteration**: `tests/workflows/core/for_each/`

### Quick Example: Simple Shell Command

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
│   │   ├── executors_file.py    # File operation executors (CreateFile, ReadFiles, EditFile)
│   │   ├── file_outline.py      # File outline extraction utilities
│   │   ├── executors_http.py    # HTTP call executor
│   │   ├── executors_llm.py     # LLM call executor
│   │   ├── executors_knowledge.py # Knowledge store executor
│   │   ├── knowledge/            # Knowledge subpackage
│   │   │   ├── constants.py      # Enums and defaults
│   │   │   ├── schema.py         # Idempotent DDL
│   │   │   ├── search.py         # Hybrid search + RRF
│   │   │   └── context.py        # Token-budgeted assembly
│   │   ├── executors_state.py   # State management executors
│   │   ├── workflow_runner.py   # Main workflow orchestrator
│   │   ├── dag.py               # DAG resolution
│   │   ├── resolver/            # Unified variable resolver (Jinja2)
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
- **UnifiedVariableResolver** - Jinja2-based variable resolution with four namespaces:
  - `inputs` - Workflow runtime inputs
  - `blocks` - Block outputs and metadata
  - `metadata` - Workflow metadata
  - `secrets` - Server-side secrets (never exposed to LLM)

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

### Workflow Files and Your IP

**YAML workflow files that use the workflows-mcp engine are NOT considered derivative works under AGPL-3.0.** Users retain full ownership and may license their workflow files under any terms they choose.

This clarification applies to:
- `.yaml`/`.yml` workflow definition files
- Configuration files (e.g., `llm-config.yml`)
- Workflow documentation and examples you create

This does NOT apply to:
- Modifications to the engine source code
- Custom block executors (Python code)
- Forks of the engine itself

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

The core workflows (Python CI, Git operations, basic file operations) are actively used and reliable. Some advanced workflows are still being refined and tested.

**Best practice:** Always inspect a workflow before using it (`get_workflow_info` tool) and test on non-production systems first. The workflows in `tests/workflows/` are thoroughly tested in CI and are great examples to learn from.

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
