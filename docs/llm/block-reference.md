# Block Executor Reference

> **This file is auto-generated.** Do not edit manually.
> Run `python scripts/generate_block_reference.py` to regenerate.

This reference is programmatically extracted from Pydantic executor models.
Field names are **exact** - use them precisely in your workflows.

## CRITICAL: Required Input Field Names

| Block Type | Required Fields |
|------------|-----------------|
| Shell | command |
| Workflow | workflow |
| CreateFile | path, content |
| EditFile | path, operations |
| ReadFiles | (none) |
| HttpCall | url |
| LLMCall | prompt; **profile OR provider** |
| Embedding | text |
| ImageGen | **profile OR provider** |
| Prompt | prompt |
| ReadJSONState | path |
| WriteJSONState | path, data |
| MergeJSONState | path, updates |
| Sql | engine |

### Common Field Name Mistakes

| Wrong | Correct |
|-------|---------|
| LLMCall with `command` | LLMCall with `prompt` |
| Prompt with `message` | Prompt with `prompt` |
| Prompt with `command` | Prompt with `prompt` |
| CreateFile with `command` | CreateFile with `path` and `content` |

---

## Shell

**Description**: Shell command executor.

### Required Inputs

- **`command`** (string): Shell command to execute

### Optional Inputs

- **`working_dir`** (string) *(default: ``)*: Working directory (empty = current dir)
- **`timeout`** (any) *(default: `120`)*: Timeout in seconds (or interpolation string)
- **`env`** (object): Environment variables
- **`capture_output`** (boolean) *(default: `True`)*: Capture stdout/stderr
- **`shell`** (boolean) *(default: `True`)*: Execute via shell

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`exit_code`** (integer): Process exit code (0 if command crashed before execution)
- **`stdout`** (string): Standard output (empty if command crashed)
- **`stderr`** (string): Standard error (empty if command crashed)

### Example

```yaml
- id: run-command
  type: Shell
  inputs:
    command: echo "Hello World"
```

---

## Workflow

**Description**: Workflow composition executor (fractal pattern).

### Required Inputs

- **`workflow`** (string): Workflow name to execute

### Optional Inputs

- **`inputs`** (any): Inputs to pass to child workflow (variables resolved in parent context)
- **`timeout_ms`** (any): Optional timeout for child execution in milliseconds

### Example

```yaml
- id: run-child
  type: Workflow
  inputs:
    workflow: child-workflow-name
    inputs:
      param1: "{{inputs.value}}"
```

---

## CreateFile

**Description**: File creation executor.

### Required Inputs

- **`path`** (string): File path (absolute or relative)
- **`content`** (string): File content to write

### Optional Inputs

- **`encoding`** (string) *(default: `utf-8`)*: Text encoding
- **`mode`** (any): File permissions (Unix only, e.g., 0o644, 644, or '644')
- **`overwrite`** (any) *(default: `True`)*: Whether to overwrite existing file (or interpolation string)
- **`create_parents`** (any) *(default: `True`)*: Create parent directories if missing (or interpolation string)

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`path`** (string): Absolute path to created file (empty string if failed)
- **`size_bytes`** (integer): File size in bytes (0 if failed)
- **`created`** (boolean): True if file was created, False if overwritten or failed
- **`content`** (string): Content written to the file (empty string if failed)

### Example

```yaml
- id: write-output
  type: CreateFile
  inputs:
    path: "{{tmp}}/output.txt"
    content: "{{blocks.previous.outputs.result}}"
```

---

## EditFile

**Description**: File editing executor with multiple strategies.

### Required Inputs

- **`path`** (string): Path to file to edit (relative or absolute)
- **`operations`** (any): List of edit operations to apply sequentially

### Optional Inputs

- **`encoding`** (string) *(default: `utf-8`)*: File encoding
- **`create_if_missing`** (any) *(default: `False`)*: Create file if it doesn't exist (or interpolation string)
- **`backup`** (any) *(default: `True`)*: Create .bak backup before editing (or interpolation string)
- **`dry_run`** (any) *(default: `False`)*: Preview changes without applying (returns diff) (or interpolation string)
- **`atomic`** (any) *(default: `True`)*: All operations succeed or none applied (or interpolation string)

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`operations_applied`** (integer): Number of operations successfully applied
- **`lines_added`** (integer): Number of lines added
- **`lines_removed`** (integer): Number of lines removed
- **`lines_modified`** (integer): Number of lines modified
- **`diff`** (any): Unified diff of changes (always provided)
- **`backup_path`** (any): Path to backup file (if backup=true)
- **`success`** (boolean): True if all operations succeeded
- **`errors`** (array): Error messages if atomic=false and some operations failed

### Example

```yaml
- id: update-config
  type: EditFile
  inputs:
    path: "./config.json"
    operations:
      - type: regex_replace
        pattern: '"version": ".*"'
        replacement: '"version": "2.0.0"'
```

---

## ReadFiles

**Description**: File reading executor with multi-file and outline support.

### Optional Inputs

- **`path`** (any): Single file path to read (absolute or relative). Mutually exclusive with patterns. Use for single-file reads.
- **`patterns`** (any): Glob patterns for files to read (e.g., ['*.py', '**/*.ts', 'docs/**/*.md'])
- **`base_path`** (any) *(default: `.`)*: Base directory to search from (relative or absolute). Used with patterns.
- **`mode`** (any) *(default: `full`)*: Output mode: 'full' (complete content), 'outline' (symbol tree with line ranges), 'summary' (outline + docstrings)
- **`exclude_patterns`** (array): Additional patterns to exclude beyond defaults (e.g., ['*test*', '*.min.js'])
- **`max_files`** (any) *(default: `20`)*: Maximum number of files to read (1-100, supports interpolation)
- **`max_file_size_kb`** (any) *(default: `100`)*: Maximum individual file size in KB (supports interpolation)
- **`respect_gitignore`** (any) *(default: `True`)*: Whether to respect .gitignore patterns (supports interpolation)
- **`encoding`** (string) *(default: `utf-8`)*: Text encoding for reading files

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`files`** (array): List of successfully processed files with content
- **`total_files`** (integer): Number of files successfully processed
- **`total_size_kb`** (integer): Total size in KB of all processed files
- **`skipped_files`** (array): Files that were skipped (too large, binary, excluded, etc.)
- **`patterns_matched`** (integer): Total number of files matching patterns before filtering

### Example

```yaml
- id: read-sources
  type: ReadFiles
  inputs:
    patterns: ["src/**/*.py"]
    mode: outline
```

---

## HttpCall

**Description**: HTTP/REST API call executor.

### Required Inputs

- **`url`** (string): Request URL (supports ${ENV_VAR} substitution)

### Optional Inputs

- **`method`** (string) *(default: `POST`)*: HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
- **`headers`** (object): HTTP headers (supports ${ENV_VAR} substitution in values)
- **`json`** (any): JSON request body (mutually exclusive with content). Use 'json' in YAML for httpx compatibility.
- **`content`** (any): Text or binary request body (mutually exclusive with json). Matches httpx parameter name.
- **`timeout`** (any) *(default: `30`)*: Request timeout in seconds (or interpolation string)
- **`follow_redirects`** (any) *(default: `True`)*: Whether to follow HTTP redirects (or interpolation string)
- **`verify_ssl`** (any) *(default: `True`)*: Whether to verify SSL certificates (or interpolation string)

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`status_code`** (integer): HTTP response status code (0 if request failed before receiving response)
- **`response_body`** (string): Response body as text (empty string if request failed)
- **`response_json`** (any): Parsed JSON response (None if not valid JSON or request failed)
- **`headers`** (object): Response headers (empty dict if request failed)
- **`success`** (boolean): True if status code is 2xx, False otherwise or if request failed

### Example

```yaml
- id: call-api
  type: HttpCall
  inputs:
    url: "https://api.example.com/data"
    method: GET
    headers:
      Authorization: "Bearer {{secrets.API_KEY}}"
```

---

## LLMCall

**Description**: Executor for LLMCall blocks.

### ⚠️ Configuration Requirement

LLMCall blocks REQUIRE exactly ONE of:
  - `profile`: Load config from ~/.workflows/llm-config.yml
  - `provider` + optionally `model`: Specify inline
Without this, execution fails with 'LLM configuration required' error.

### Required Inputs

- **`prompt`** (string): User prompt to send to the LLM

### Optional Inputs

- **`profile`** (any): Profile name from ~/.workflows/llm-config.yml (e.g., 'cloud', 'local', 'default'). If specified, provider/model are loaded from config. Mutually exclusive with direct provider/model specification.
- **`provider`** (any): LLM provider (enum or interpolation string). Required if profile not specified. Ignored if profile specified.
- **`model`** (any): Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022, gemini-2.0-flash-exp). Required if profile not specified. Can override profile model if both specified.
- **`system_instructions`** (any): System instructions (optional)
- **`api_key`** (any): API key (pre-resolved from {{secrets.PROVIDER_API_KEY}})
- **`api_url`** (any): Custom API endpoint URL (optional, for custom deployments)
- **`response_schema`** (any): JSON Schema for expected response structure (dict or JSON string, enables validation and retry)
- **`max_retries`** (any) *(default: `3`)*: Maximum number of retry attempts (or interpolation string)
- **`retry_delay`** (any) *(default: `2.0`)*: Initial retry delay in seconds (exponential backoff, or interpolation string)
- **`timeout`** (any) *(default: `60`)*: Request timeout in seconds (or interpolation string)
- **`temperature`** (any): Sampling temperature 0.0-2.0 (or interpolation string)
- **`max_tokens`** (any): Maximum tokens to generate (or interpolation string)
- **`validation_prompt_template`** (string) *(default: `Your previous response failed JSON schema validation.

Error: {validation_error}

Expected schema:
{schema}

Please provide a valid response that conforms to the schema.`)*: Template for validation retry prompt

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`response`** (object): Response dictionary. Contains validated JSON structure if schema provided and validation succeeded. Contains {'content': 'raw text'} if no schema or validation failed. Empty dict {} if request failed completely.
- **`success`** (boolean): True if LLM API call succeeded (response received from provider). False if request failed (network error, timeout, API error, etc.). Independent of schema validation - check metadata.validation_failed for validation status.
- **`metadata`** (object): Execution metadata including: attempts (int), validation_failed (bool, if schema validation failed), validation_error (str, error message if validation_failed=true), model (str), usage (dict), finish_reason (str), etc. Empty dict if request failed before execution.

### Example

```yaml
- id: summarize
  type: LLMCall
  inputs:
    profile: default
    prompt: "Summarize this text: {{inputs.text}}"
```

---

## Embedding

**Description**: Executor for generating text embeddings using OpenAI-compatible API.

### Required Inputs

- **`text`** (string): Text to generate embedding for

### Optional Inputs

- **`profile`** (string) *(default: `embedding`)*: Profile name from ~/.workflows/llm-config.yml (defaults to 'embedding')
- **`model`** (any): Override embedding model (uses profile model if not specified)
- **`api_key`** (any): Override API key (uses profile api_key_secret if not specified)
- **`api_url`** (any): Override API endpoint URL (uses profile api_url if not specified)
- **`timeout`** (any) *(default: `30`)*: Request timeout in seconds

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`embedding`** (array): Embedding vector (list of floats)
- **`dimensions`** (integer): Number of dimensions in the embedding
- **`success`** (boolean): Whether the embedding generation succeeded
- **`metadata`** (object): Execution metadata (model, usage, etc.)

### Example

```yaml
- id: embed_text
  type: Embedding
  inputs:
    text: "Search for authentication bugs"
```

---

## ImageGen

**Description**: Executor for Image Generation.

### ⚠️ Configuration Requirement

ImageGen blocks REQUIRE exactly ONE of:
  - `profile`: Load config from ~/.workflows/llm-config.yml
  - `provider`: Specify inline (openai, openai_compatible)
Without this, execution fails with configuration error.

### Optional Inputs

- **`prompt`** (any): Text prompt (required for generate/edit, not used for variation)
- **`profile`** (any): Profile name from ~/.workflows/llm-config.yml. If specified, provider/model are loaded from config. Mutually exclusive with direct provider/model specification.
- **`provider`** (any): Image provider (openai, openai_compatible). Required if profile not specified.
- **`model`** (any) *(default: `dall-e-3`)*: Model to use (dall-e-3, dall-e-2, or custom model name)
- **`api_url`** (any): Custom API endpoint URL (required for openai_compatible)
- **`api_key`** (any): API key (pre-resolved from {{secrets.OPENAI_API_KEY}})
- **`operation`** (string) *(default: `generate`)*: Operation to perform
- **`size`** (string) *(default: `1024x1024`)*: Image size (e.g., 1024x1024, 256x256, 512x512)
- **`quality`** (any) *(default: `standard`)*: Image quality (dall-e-3 only)
- **`style`** (any) *(default: `vivid`)*: Image style (dall-e-3 only)
- **`response_format`** (string) *(default: `url`)*: Format of the response
- **`n`** (any) *(default: `1`)*: Number of images to generate (supports interpolation)
- **`image`** (any): Path to base image (required for edit/variation)
- **`mask`** (any): Path to mask image (optional for edit)
- **`output_file`** (any): Path to save the generated image(s). If n>1, appends index.

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`urls`** (array): List of image URLs
- **`b64_json`** (array): List of base64 encoded image data
- **`revised_prompts`** (array): List of revised prompts (dall-e-3)
- **`saved_files`** (array): List of paths where images were saved
- **`success`** (boolean): Whether the operation succeeded
- **`provider_metadata`** (object): Metadata from the provider response

### Example

```yaml
- id: generate-image
  type: ImageGen
  inputs:
    profile: default
    operation: generate
    prompt: "A beautiful sunset over mountains"
```

---

## Prompt

**Description**: Interactive prompt executor - pauses workflow for LLM input.

### Required Inputs

- **`prompt`** (string): Prompt/question to display to LLM. The LLM will provide a response.

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`response`** (string): Raw LLM response to the prompt (empty string if failed or crashed)

### Example

```yaml
- id: ask-user
  type: Prompt
  inputs:
    prompt: "Do you approve this change? (yes/no)"
```

---

## ReadJSONState

**Description**: Read JSON state file executor.

### Required Inputs

- **`path`** (string): Path to JSON file

### Optional Inputs

- **`required`** (any) *(default: `False`)*: Whether file must exist (False returns empty dict, or interpolation string)

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`data`** (object): JSON data from file (empty dict if failed or not found)
- **`found`** (boolean): Whether file was found (False if failed or not found)
- **`path`** (string): Absolute path to file (empty string if failed)

### Example

```yaml
- id: load-state
  type: ReadJSONState
  inputs:
    path: "{{tmp}}/state.json"
```

---

## WriteJSONState

**Description**: Write JSON state file executor.

### Required Inputs

- **`path`** (string): Path to JSON file
- **`data`** (object): JSON data to write

### Optional Inputs

- **`create_parents`** (any) *(default: `True`)*: Create parent directories if missing (or interpolation string)

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`path`** (string): Absolute path to file (empty string if failed)
- **`size_bytes`** (integer): Size of written file in bytes (0 if failed)

### Example

```yaml
- id: save-state
  type: WriteJSONState
  inputs:
    path: "{{tmp}}/state.json"
    data:
      status: completed
      result: "{{blocks.process.outputs.value}}"
```

---

## MergeJSONState

**Description**: Merge JSON state file executor.

### Required Inputs

- **`path`** (string): Path to JSON file
- **`updates`** (object): Updates to merge

### Optional Inputs

- **`create_if_missing`** (any) *(default: `True`)*: Create file if it doesn't exist (or interpolation string)
- **`create_parents`** (any) *(default: `True`)*: Create parent directories if missing (or interpolation string)

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`path`** (string): Absolute path to file (empty string if failed)
- **`created`** (boolean): Whether file was created (vs updated), False if failed
- **`merged_data`** (object): Result after merge (empty dict if failed)

### Example

```yaml
- id: update-state
  type: MergeJSONState
  inputs:
    path: "{{tmp}}/state.json"
    updates:
      last_updated: "{{now()}}"
```

---

## Sql

**Description**: SQL executor for database operations.

### Required Inputs

- **`engine`** (string): Database engine. Required.

### Optional Inputs

- **`path`** (any): SQLite: Database file path. Use ':memory:' for in-memory DB.
- **`host`** (any): Database host
- **`port`** (any): Database port (default: 5432 for PostgreSQL, 3306 for MariaDB)
- **`database`** (any): Database name
- **`username`** (any): Database username
- **`password`** (any): Database password. Use {{secrets.DB_PASSWORD}} for security.
- **`sql`** (any): 
        SQL statement(s) to execute (Raw SQL mode).
        - Use ? for positional params (SQLite) or $1, $2 for PostgreSQL
        - MariaDB uses %s for positional params
        - Multi-statement scripts: separate with semicolons
        Mutually exclusive with 'model' field.
        
- **`params`** (any): 
        Query parameters for raw SQL (prevents SQL injection).
        - List for positional: [value1, value2]
        - Dict for named: {"name": value} (PostgreSQL/MariaDB)
        
- **`model`** (any): 
        Model schema for CRUD operations (Model mode).
        Defines table structure with columns, types, indexes.
        Mutually exclusive with 'sql' field.
        Example:
          model:
            table: tasks
            columns:
              id: {type: text, primary: true, auto: uuid}
              name: {type: text, required: true}
            indexes:
              - columns: [name]
        
- **`op`** (any): 
        CRUD operation (required when using model mode).
        - schema: Create table + indexes
        - insert: Insert row (requires data)
        - select: Query rows (optional where, order, limit, offset)
        - update: Update rows (requires where and data)
        - delete: Delete rows (requires where)
        - upsert: Insert or update on conflict (requires data and conflict)
        
- **`data`** (any): Row data for insert/update/upsert operations.
- **`where`** (any): 
        Filter conditions for select/update/delete.
        - Simple equality: {status: running}
        - Operators: {priority: {">": 5}}
        - IN: {type: {in: [a, b, c]}}
        - IS NULL: {deleted_at: {is: null}}
        
- **`order`** (any): Sort order for select. Format: ["column:asc", "column:desc"]
- **`limit`** (any): Maximum rows to return (select).
- **`offset`** (any): Rows to skip (select).
- **`columns`** (any): Specific columns to select (default: all columns).
- **`conflict`** (any): Conflict columns for upsert (usually primary key).
- **`init_sql`** (any): 
        DDL to execute before the main operation (idempotent).
        Use CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS, etc.
        
- **`isolation_level`** (any): 
        Transaction isolation level.
        - PostgreSQL/MariaDB: read_uncommitted, read_committed, repeatable_read, serializable
        - SQLite: immediate (recommended for writes), exclusive, or default (deferred)
        
- **`ssl`** (any) *(default: `False`)*: Enable SSL/TLS. Boolean or sslmode string (require, verify-ca, verify-full)
- **`timeout`** (any) *(default: `30`)*: Query execution timeout in seconds
- **`connect_timeout`** (any) *(default: `10`)*: Connection establishment timeout in seconds
- **`pool_size`** (any) *(default: `5`)*: Connection pool size (PostgreSQL/MariaDB only)
- **`sqlite_pragmas`** (any): 
        SQLite PRAGMA settings applied on connection.
        Defaults: journal_mode=WAL, busy_timeout=30000, synchronous=NORMAL, foreign_keys=ON
        

### Outputs

- **`meta`** (object): Executor-specific metadata fields (exit_code, tokens_used, etc.)
- **`rows`** (array): Result rows as list of dicts
- **`columns`** (array): Column names from result set
- **`row_count`** (integer): Number of rows returned (select) or affected (insert/update/delete)
- **`affected_rows`** (integer): Rows affected by INSERT/UPDATE/DELETE
- **`last_insert_id`** (any): Last inserted row ID (auto-increment)
- **`success`** (boolean): Operation completed successfully
- **`engine`** (string): Database engine used
- **`execution_time_ms`** (number): Query execution time in milliseconds

### Example

```yaml
# Raw SQL mode - SQLite query
- id: get_users
  type: Sql
  inputs:
    engine: sqlite
    path: "/data/app.db"
    sql: "SELECT * FROM users WHERE status = ?"
    params: ["active"]

# Model mode - Create table and insert
- id: create_task
  type: Sql
  inputs:
    engine: sqlite
    path: "{{state.db_path}}"
    model:
      table: tasks
      columns:
        task_id: {type: text, primary: true, auto: uuid}
        name: {type: text, required: true}
        status: {type: text, default: pending}
        created_at: {type: timestamp, auto: created}
      indexes:
        - columns: [status]
    op: insert
    data:
      name: "My Task"

# Model mode - Select with filters
- id: find_tasks
  type: Sql
  inputs:
    engine: sqlite
    path: "{{state.db_path}}"
    model: "{{inputs.models.task}}"
    op: select
    where:
      status: running
    order: [created_at:desc]
    limit: 10
```

---

## Variable Access Patterns

Use double braces `{{ }}` for variable interpolation in YAML:

| Pattern | Description |
|---------|-------------|
| `{{inputs.param}}` | Access workflow input parameters |
| `{{blocks.id.outputs.field}}` | Access another block's output |
| `{{blocks.id.succeeded}}` | Check if block succeeded (boolean) |
| `{{blocks.id.failed}}` | Check if block failed (boolean) |
| `{{blocks.id.skipped}}` | Check if block was skipped (boolean) |
| `{{secrets.NAME}}` | Access secrets (e.g., API keys) |
| `{{tmp}}` | Temporary directory path |
| `{{now()}}` | Current timestamp |

## Control Flow

| Field | Description |
|-------|-------------|
| `depends_on: [block_id]` | Generate DAG. Expects parent blocks to succeed |
| `condition: "{{expression}}"` | Conditional execution |
| `continue_on_error: true` | Only used in sequential for_each |
| `for_each: "{{list}}"` | Iterate over list items |
| `for_each_mode: parallel` | Run iterations in parallel (default) |
| `for_each_mode: sequential` | Run iterations sequentially |

## Jinja2 Filters

Common filters for transforming values:

| Filter | Description |
|--------|-------------|
| `trim` | Remove leading/trailing whitespace |
| `lower` | Convert to lowercase |
| `upper` | Convert to uppercase |
| `replace(old, new)` | Replace text |
| `length` | Get length of string or list |
| `tojson` | Convert to JSON string |
| `fromjson` | Parse JSON string |
| `toyaml` | Convert to YAML string |

## Workflow Composition

Use the `Workflow` block type to call other workflows (composition pattern):

```yaml
- id: process_data
  type: Workflow
  inputs:
    workflow: "child-workflow-name"
    inputs:
      param1: "{{inputs.value}}"
      param2: "{{blocks.previous.outputs.result}}"
```

### Accessing Child Workflow Outputs

```yaml
{{blocks.process_data.outputs.result}}  # Access child output
{{blocks.process_data.succeeded}}       # Check success
```

## Recursive Workflows

A workflow can call itself for iterative processing:

```yaml
name: recursive-processor
inputs:
  count: {type: num, default: 0}
  max: {type: num, default: 5}

blocks:
  - id: process
    type: Shell
    inputs:
      command: echo "Processing {{inputs.count}}"

  - id: recurse
    type: Workflow
    depends_on: [process]
    condition: "{{inputs.count < inputs.max}}"  # Termination condition
    inputs:
      workflow: recursive-processor  # Self-reference
      inputs:
        count: "{{inputs.count + 1}}"
        max: "{{inputs.max}}"
```

**Important**: Always include a termination condition to prevent infinite loops.

## for_each Iteration

Any block can iterate over collections using `for_each`:

```yaml
- id: process_items
  type: Shell
  for_each: "{{inputs.items}}"  # List or dict
  for_each_mode: parallel       # parallel (default) or sequential
  inputs:
    command: "process {{each.value}}"
```

### Iteration Variables

| Variable | Description |
|----------|-------------|
| `{{each.key}}` | Current key (index for lists) |
| `{{each.value}}` | Current value |
| `{{each.index}}` | Zero-based position |
| `{{each.count}}` | Total iterations |

### Accessing Iteration Results (Bracket Notation)

```yaml
{{blocks.process_items["0"].outputs.stdout}}       # By index
{{blocks.process_items["key1"].outputs.result}}    # By key
{{blocks.process_items.metadata.count}}            # Total count
{{blocks.process_items.metadata.count_failed}}     # Failed count
```

### Dynamic Iteration (Generating Lists)

**IMPORTANT**: `range()` is NOT valid in for_each expressions.
To iterate a dynamic number of times, generate a list first:

```yaml
# Step 1: Generate list of indices
- id: generate_indices
  type: Shell
  inputs:
    command: |
      python3 -c "import json; print(json.dumps(list(range({{inputs.count}}))))"

# Step 2: Use the generated list in for_each
- id: process_each
  type: Shell
  depends_on: [generate_indices]
  for_each: "{{blocks.generate_indices.outputs.stdout | fromjson}}"
  inputs:
    command: "echo Processing item {{each.value}}"
```

### Common for_each Mistakes

| Wrong | Correct |
|-------|---------|
| `for_each: "{{range(inputs.n)}}"` | Use Shell block to generate list |
| `{{loop.index}}` | `{{each.index}}` |
| `{{item}}` | `{{each.value}}` |
| `blocks.id.outputs.result` (for_each block) | `blocks.id["0"].outputs.result` |

### Nested Iteration (via Composition)

For nested loops, use workflow composition:

```yaml
# Parent: iterate regions
- id: process_regions
  type: Workflow
  for_each: "{{inputs.regions}}"
  inputs:
    workflow: process-servers
    inputs:
      region: "{{each.key}}"
      servers: "{{each.value.servers}}"

# Child workflow: process-servers.yaml
# Iterates over servers within each region
```
