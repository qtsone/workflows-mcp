# Recursive Workflows with State Management

Complete guide to building recursive workflow patterns with hierarchical state tracking in workflows-mcp.

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [State Management Basics](#state-management-basics)
4. [Workflow Patterns](#workflow-patterns)
5. [Safety and Depth Tracking](#safety-and-depth-tracking)
6. [Debugging and Visualization](#debugging-and-visualization)
7. [Examples](#examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Introduction

Recursive workflows enable complex, multi-level task execution with built-in state tracking and progress monitoring. The system provides hierarchical task trees that persist across workflow calls, enabling sophisticated patterns like iterative investigation, batch processing, and tree traversal.

### Key Features

**Hierarchical State Tracking**: Build multi-level task trees with parent-child relationships, enabling complex workflow decomposition and progress tracking.

**Persistent State**: Task state persists across workflow calls in SQLite databases, enabling pause/resume, progress monitoring, and audit trails with concurrent access support.

**Progress Aggregation**: Automatic rollup of child task status, including completion percentages, pending tasks, and failure detection.

**Visual Debugging**: ASCII tree visualization shows task hierarchy, status, durations, and custom data fields for easy debugging.

**Depth Safety**: Built-in recursion depth tracking and configurable limits prevent infinite loops and runaway execution.

**Flexible Patterns**: Supports single-level (parent + phases), multi-level (parent + phases + sub-tasks), and fully recursive (self-calling) workflows.

### Use Cases

**Agent Workflows**: LLM-driven agents that investigate multiple files, make iterative decisions, and track progress hierarchically (e.g., PR review with per-file analysis).

**Batch Processing**: Process large datasets by recursively splitting work into manageable chunks, tracking progress for each item.

**Tree Traversal**: Navigate hierarchical data structures (filesystems, dependency graphs) with automatic depth tracking and termination.

**Multi-Phase Execution**: Break complex workflows into phases (context gathering, analysis, synthesis, action) with state tracking at each level.

### When to Use Recursive Workflows

Use recursive patterns when:
- Tasks naturally decompose into hierarchical subtasks
- Progress tracking across multiple levels is required
- Iterative refinement with LLM decision-making is needed
- You need to process variable-length lists with state persistence
- Debugging complex multi-step processes requires visibility

Do not use when:
- Simple linear workflows suffice (use regular workflows)
- No state persistence is needed between calls
- Task decomposition is shallow (1-2 levels) and fixed

---

## Quick Start

### 1. Create a Simple Recursive Workflow

```yaml
name: batch-processor
description: Process items recursively with state tracking

inputs:
  items:
    type: list
    required: true
  state:
    type: str
    required: false
    default: ""

blocks:
  # Create root task on first call
  - id: init_state
    type: Workflow
    condition: "{{ inputs.state == '' }}"
    inputs:
      workflow: agent-state-management
      inputs:
        task: "Batch Processing"
        caller: "batch-processor"
        status: "in-progress"

  # Process first item
  - id: process_item
    type: Shell
    condition: "{{ inputs.items | length > 0 }}"
    depends_on:
      - block: init_state
        required: false
    inputs:
      command: |
        echo "Processing: {{ inputs.items[0] }}"

  # Track completion of this item
  - id: track_item
    type: Workflow
    condition: "{{ inputs.items | length > 0 }}"
    depends_on: [process_item]
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"
        parent_id: "{{ blocks.init_state.outputs.task_id | default('', true) }}"
        task: "Item: {{ inputs.items[0] }}"
        status: "done"
        caller: "batch-processor"

  # Recurse for remaining items
  - id: process_remaining
    type: Workflow
    condition: "{{ inputs.items | length > 1 }}"
    depends_on: [track_item]
    inputs:
      workflow: batch-processor
      inputs:
        items: "{{ inputs.items[1:] }}"
        state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"

  # Mark root task complete when all items processed
  - id: finalize
    type: Workflow
    condition: "{{ inputs.items | length == 0 and inputs.state != '' }}"
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        status: "done"
        caller: "batch-processor"

outputs:
  state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"
  processed_count: "{{ blocks.track_item.outputs.children_summary.done | default(0) }}"
```

### 2. Execute the Workflow

```python
# Call via MCP
result = await execute_workflow(
    workflow="batch-processor",
    inputs={
        "items": ["file1.py", "file2.py", "file3.py"]
    }
)

# State database created at: ~/.workflows/tasks/<trace_id>.db
print(f"State: {result['state']}")
print(f"Processed: {result['processed_count']} items")
```

### 3. Visualize the State

```python
# Visualize task tree
await execute_workflow(
    workflow="agent-state-visualize",
    inputs={
        "state": result["state"]
    }
)

# Output:
# Batch Processing [task-abc123] ✓ done (5.2s)
# ├─ Item: file1.py ✓ 1.5s
# ├─ Item: file2.py ✓ 2.1s
# └─ Item: file3.py ✓ 1.6s
```

---

## State Management Basics

### State Database Structure

The state management system persists task hierarchies in SQLite databases at `~/.workflows/tasks/<trace_id>.db`.

**Database Schema:**

```sql
-- Task hierarchy with parent-child relationships
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES tasks(task_id),
    task TEXT NOT NULL,
    task_type TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    data JSON DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Key-value memory storage for workflow state
CREATE TABLE memory (
    key TEXT PRIMARY KEY,
    value JSON,
    updated_at TEXT NOT NULL
);

-- Audit trail for all state operations
CREATE TABLE audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    task_id TEXT,
    caller TEXT NOT NULL,
    action TEXT NOT NULL,
    description TEXT,
    changes JSON,
    parent_id TEXT
);

-- Iteration tracking for loops
CREATE TABLE iterations (
    task_id TEXT PRIMARY KEY,
    current INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    cap INTEGER DEFAULT 100,
    started_at TEXT,
    completed_at TEXT,
    checkpoints JSON DEFAULT '[]'
);
```

**Example task data (logical structure):**

```json
{
  "task_id": "task-abc123",
  "task": "PR Review",
  "task_type": "pr-review",
  "status": "in-progress",
  "parent_id": null,
  "data": {
    "platform": "github",
    "files_changed": 19
  },
  "created_at": "2025-12-12T10:30:00.123Z",
  "updated_at": "2025-12-12T10:35:45.456Z"
}
```

### Task Status Values

| Status | Description | Terminal |
|--------|-------------|----------|
| `pending` | Task created but not started | No |
| `in-progress` | Task actively executing | No |
| `done` | Task completed successfully | Yes |
| `failed` | Task failed with error | Yes |
| `blocked` | Task waiting on dependency | No |

### Key Outputs from State Management

The `agent-state-management` workflow provides comprehensive outputs for building recursive patterns:

#### Core Identifiers
- **`state`**: Path to SQLite database (`~/.workflows/tasks/<trace_id>.db`)
- **`task_id`**: ID of created/updated task
- **`root_task_id`**: ID of the root task in the tree
- **`trace_id`**: Execution trace ID (filename without extension)

#### Task Information
- **`status`**: Current task status (`pending`, `in-progress`, `done`, `failed`, `blocked`)
- **`parent_id`**: Parent task ID (null for root tasks)
- **`task_data`**: Custom data dictionary attached to the task

#### Child Task Management
- **`children`**: List of all child task IDs
- **`pending_children`**: Child task IDs with status != `done`/`failed`
- **`all_children_done`**: Boolean - true if all children are `done` (gates to next phase)
- **`any_child_failed`**: Boolean - true if any child is `failed` (error handling)

#### Progress Tracking
- **`progress_pct`**: Completion percentage (0-100) based on children
- **`children_summary`**: Status counts by category:
  ```json
  {
    "total": 10,
    "done": 7,
    "failed": 1,
    "in_progress": 2,
    "pending": 0
  }
  ```

### Audit Trail

Each state operation is logged to the `audit` table in the same SQLite database. Query with SQL:

```sql
SELECT timestamp, task_id, caller, action, description, changes
FROM audit
ORDER BY timestamp;
```

**Example audit entries (logical structure):**

```json
[
  {
    "timestamp": "2025-12-12T10:30:00.123Z",
    "task_id": "task-abc123",
    "caller": "pr-review",
    "action": "create_root",
    "description": "Created root task: PR Review"
  },
  {
    "timestamp": "2025-12-12T10:30:01.234Z",
    "task_id": "task-def456",
    "parent_id": "task-abc123",
    "caller": "pr-review:context-gathering",
    "action": "create_subtask",
    "description": "Created sub-task: Phase: Context Gathering"
  },
  {
    "timestamp": "2025-12-12T10:30:11.234Z",
    "task_id": "task-def456",
    "caller": "pr-review:context-gathering",
    "action": "task_completed",
    "description": "Updated: ['status']",
    "changes": {"status": ["in-progress", "done"]}
  }
]
```

---

## Workflow Patterns

### Pattern 1: Single-Level (Parent + Phases)

**Use Case**: Multi-phase workflows where each phase is a distinct step executed sequentially.

**Structure**: One root task with multiple child tasks representing phases.

**Example**: PR review with context gathering → assessment → synthesis → action phases.

```yaml
name: single-level-workflow
description: Parent task with sequential phases

inputs:
  data:
    type: dict
    required: true

blocks:
  # Create root task
  - id: init_root
    type: Workflow
    inputs:
      workflow: agent-state-management
      inputs:
        task: "Multi-Phase Processing"
        caller: "single-level"
        status: "in-progress"

  # Phase 1: Setup
  - id: phase1
    type: Workflow
    depends_on: [init_root]
    inputs:
      workflow: phase-setup
      inputs:
        state: "{{ blocks.init_root.outputs.state }}"
        parent_task_id: "{{ blocks.init_root.outputs.task_id }}"
        data: "{{ inputs.data }}"

  # Phase 2: Processing (waits for phase 1)
  - id: phase2
    type: Workflow
    depends_on: [phase1]
    inputs:
      workflow: phase-processing
      inputs:
        state: "{{ blocks.init_root.outputs.state }}"
        parent_task_id: "{{ blocks.init_root.outputs.task_id }}"
        phase1_output: "{{ blocks.phase1.outputs.result }}"

  # Phase 3: Finalization
  - id: phase3
    type: Workflow
    depends_on: [phase2]
    inputs:
      workflow: phase-finalize
      inputs:
        state: "{{ blocks.init_root.outputs.state }}"
        parent_task_id: "{{ blocks.init_root.outputs.task_id }}"
        phase2_output: "{{ blocks.phase2.outputs.result }}"

  # Mark root complete
  - id: finalize
    type: Workflow
    depends_on: [phase3]
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ blocks.init_root.outputs.state }}"
        task_id: "{{ blocks.init_root.outputs.task_id }}"
        status: "done"
        caller: "single-level"

outputs:
  state: "{{ blocks.init_root.outputs.state }}"
  result: "{{ blocks.phase3.outputs.result }}"
```

**State Tree**:
```bash
Multi-Phase Processing [task-abc] ✓ done
├─ Phase: Setup ✓
├─ Phase: Processing ✓
└─ Phase: Finalization ✓
```

**Phase Workflow Template**:
```yaml
name: phase-setup
description: Individual phase with state tracking

inputs:
  state:
    type: str
    required: true
  parent_task_id:
    type: str
    required: true
  data:
    type: dict
    required: true

blocks:
  # Track phase start
  - id: track_start
    type: Workflow
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        parent_id: "{{ inputs.parent_task_id }}"
        task: "Phase: Setup"
        task_type: "phase"
        status: "in-progress"
        caller: "phase-setup"

  # Do actual work
  - id: work
    type: Shell
    depends_on: [track_start]
    inputs:
      command: |
        echo "Processing data: {{ inputs.data | tojson }}"

  # Track phase completion
  - id: track_done
    type: Workflow
    depends_on: [work]
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        task_id: "{{ blocks.track_start.outputs.task_id }}"
        status: "done"
        caller: "phase-setup"
        data:
          result_count: "{{ blocks.work.outputs.stdout | length }}"

outputs:
  result: "{{ blocks.work.outputs.stdout }}"
```

### Pattern 2: Multi-Level (Parent + Phases + Sub-Tasks)

**Use Case**: Workflows where phases contain variable numbers of sub-tasks, often driven by LLM decisions.

**Structure**: Root task → Phase tasks → Individual item tasks (per file, per record, etc.).

**Example**: PR review with investigation phase that analyzes multiple files individually.

```yaml
name: multi-level-workflow
description: Parent with phases containing sub-tasks

inputs:
  files:
    type: list
    required: true

blocks:
  # Root task
  - id: init_root
    type: Workflow
    inputs:
      workflow: agent-state-management
      inputs:
        task: "File Analysis"
        caller: "multi-level"
        status: "in-progress"

  # Phase: Investigation (creates sub-tasks per file)
  - id: investigation_phase
    type: Workflow
    depends_on: [init_root]
    inputs:
      workflow: investigation-phase
      inputs:
        state: "{{ blocks.init_root.outputs.state }}"
        parent_task_id: "{{ blocks.init_root.outputs.task_id }}"
        files: "{{ inputs.files }}"

  # Mark root complete
  - id: finalize
    type: Workflow
    depends_on: [investigation_phase]
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ blocks.init_root.outputs.state }}"
        task_id: "{{ blocks.init_root.outputs.task_id }}"
        status: "done"
        caller: "multi-level"

outputs:
  state: "{{ blocks.init_root.outputs.state }}"
  results: "{{ blocks.investigation_phase.outputs.results }}"
```

**Investigation Phase with For-Each**:
```yaml
name: investigation-phase
description: Phase that creates sub-tasks per file

inputs:
  state:
    type: str
    required: true
  parent_task_id:
    type: str
    required: true
  files:
    type: list
    required: true

blocks:
  # Track phase start
  - id: track_phase_start
    type: Workflow
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        parent_id: "{{ inputs.parent_task_id }}"
        task: "Phase: Investigation"
        task_type: "phase"
        status: "in-progress"
        caller: "investigation-phase"
        data:
          file_count: "{{ inputs.files | length }}"

  # Process each file (creates sub-task per file)
  - id: process_files
    type: Workflow
    depends_on: [track_phase_start]
    for_each: "{{ inputs.files }}"
    for_each_mode: sequential
    inputs:
      workflow: file-investigation
      inputs:
        state: "{{ inputs.state }}"
        parent_task_id: "{{ blocks.track_phase_start.outputs.task_id }}"
        file_path: "{{ each.value }}"

  # Track phase completion
  - id: track_phase_done
    type: Workflow
    depends_on: [process_files]
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        task_id: "{{ blocks.track_phase_start.outputs.task_id }}"
        status: "done"
        caller: "investigation-phase"
        data:
          files_processed: "{{ inputs.files | length }}"

outputs:
  results: "{{ blocks.process_files.outputs }}"
```

**File Investigation Sub-Workflow**:
```yaml
name: file-investigation
description: Investigate a single file with state tracking

inputs:
  state:
    type: str
    required: true
  parent_task_id:
    type: str
    required: true
  file_path:
    type: str
    required: true

blocks:
  # Track file task start
  - id: track_file_start
    type: Workflow
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        parent_id: "{{ inputs.parent_task_id }}"
        task: "Investigate: {{ inputs.file_path }}"
        task_type: "file-investigation"
        status: "in-progress"
        caller: "file-investigation"
        data:
          path: "{{ inputs.file_path }}"

  # Analyze file with LLM
  - id: analyze
    type: LLMCall
    depends_on: [track_file_start]
    inputs:
      profile: default
      prompt: |
        Analyze this file: {{ inputs.file_path }}
        Provide findings and severity.
      response_schema:
        type: object
        properties:
          severity:
            type: string
            enum: [critical, high, medium, low, info]
          findings:
            type: array
            items:
              type: string

  # Track file task completion
  - id: track_file_done
    type: Workflow
    depends_on: [analyze]
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        task_id: "{{ blocks.track_file_start.outputs.task_id }}"
        status: "done"
        caller: "file-investigation"
        data:
          severity: "{{ blocks.analyze.outputs.response.severity }}"
          findings_count: "{{ blocks.analyze.outputs.response.findings | length }}"

outputs:
  result: "{{ blocks.analyze.outputs.response }}"
```

**State Tree**:
```bash
File Analysis [task-abc] ✓ done (15.3s)
└─ Phase: Investigation ✓ 14.8s
   ├─ Investigate: file1.py ✓ 4.2s (severity: high)
   ├─ Investigate: file2.py ✓ 5.1s (severity: medium)
   └─ Investigate: file3.py ✓ 5.5s (severity: low)
```

### Pattern 3: Fully Recursive (Self-Calling)

**Use Case**: Workflows that need to call themselves with modified inputs until a termination condition is met.

**Structure**: Workflow calls itself recursively, tracking depth and checking termination conditions.

**Example**: Tree traversal, iterative refinement with LLM, batch processing with chunking.

```yaml
name: recursive-tree-traversal
description: Traverse directory tree recursively

inputs:
  path:
    type: str
    required: true
  state:
    type: str
    required: false
    default: ""
  parent_task_id:
    type: str
    required: false
    default: ""
  depth:
    type: num
    required: false
    default: 0
  max_depth:
    type: num
    required: false
    default: 5

blocks:
  # Create root task on first call
  - id: init_state
    type: Workflow
    condition: "{{ inputs.state == '' }}"
    inputs:
      workflow: agent-state-management
      inputs:
        task: "Traverse: {{ inputs.path }}"
        caller: "recursive-tree-traversal"
        status: "in-progress"
        data:
          depth: "{{ inputs.depth }}"

  # Create sub-task for non-root calls
  - id: track_subtask
    type: Workflow
    condition: "{{ inputs.state != '' }}"
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        parent_id: "{{ inputs.parent_task_id }}"
        task: "Traverse: {{ inputs.path }}"
        caller: "recursive-tree-traversal"
        status: "in-progress"
        data:
          depth: "{{ inputs.depth }}"

  # Get current state path
  - id: get_state
    type: Shell
    depends_on:
      - block: init_state
        required: false
      - block: track_subtask
        required: false
    inputs:
      command: |
        echo "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"

  # Process current directory
  - id: process_dir
    type: Shell
    depends_on: [get_state]
    inputs:
      command: |
        echo "Processing: {{ inputs.path }}"
        # Do actual work here

  # List subdirectories
  - id: list_subdirs
    type: Shell
    condition: "{{ inputs.depth < inputs.max_depth }}"
    depends_on: [process_dir]
    inputs:
      command: |
        find "{{ inputs.path }}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | \
          python3 -c "import sys, json; print(json.dumps([line.strip() for line in sys.stdin]))"

  # Recurse into each subdirectory
  - id: recurse_subdirs
    type: Workflow
    condition: "{{ inputs.depth < inputs.max_depth and blocks.list_subdirs.succeeded }}"
    depends_on: [list_subdirs]
    for_each: "{{ blocks.list_subdirs.outputs.stdout | fromjson }}"
    for_each_mode: sequential
    inputs:
      workflow: recursive-tree-traversal
      inputs:
        path: "{{ each.value }}"
        state: "{{ blocks.get_state.outputs.stdout | trim }}"
        parent_task_id: "{{ blocks.track_subtask.outputs.task_id | default(blocks.init_state.outputs.task_id, true) }}"
        depth: "{{ inputs.depth + 1 }}"
        max_depth: "{{ inputs.max_depth }}"

  # Mark task complete
  - id: finalize
    type: Workflow
    depends_on:
      - process_dir
      - block: recurse_subdirs
        required: false
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ blocks.get_state.outputs.stdout | trim }}"
        task_id: "{{ blocks.track_subtask.outputs.task_id | default(blocks.init_state.outputs.task_id, true) }}"
        status: "done"
        caller: "recursive-tree-traversal"

outputs:
  state: "{{ blocks.get_state.outputs.stdout | trim }}"
  depth_reached: "{{ inputs.depth }}"
```

**State Tree**:
```bash
Traverse: /project [task-abc] ✓ done (8.5s)
├─ Traverse: /project/src ✓ 3.2s (depth: 1)
│  ├─ Traverse: /project/src/components ✓ 1.1s (depth: 2)
│  └─ Traverse: /project/src/utils ✓ 0.8s (depth: 2)
├─ Traverse: /project/tests ✓ 2.1s (depth: 1)
└─ Traverse: /project/docs ✓ 1.5s (depth: 1)
```

**Iterative Refinement Example**:
```yaml
name: iterative-refinement
description: LLM-driven iterative refinement with depth tracking

inputs:
  initial_prompt:
    type: str
    required: true
  state:
    type: str
    required: false
    default: ""
  iteration:
    type: num
    required: false
    default: 0
  max_iterations:
    type: num
    required: false
    default: 5
  confidence_threshold:
    type: num
    required: false
    default: 0.9

blocks:
  # Create root task
  - id: init_state
    type: Workflow
    condition: "{{ inputs.state == '' }}"
    inputs:
      workflow: agent-state-management
      inputs:
        task: "Iterative Refinement"
        caller: "iterative-refinement"
        status: "in-progress"

  # LLM refinement step
  - id: refine
    type: LLMCall
    depends_on:
      - block: init_state
        required: false
    inputs:
      profile: default
      prompt: |
        {% if inputs.iteration == 0 %}
        Initial request: {{ inputs.initial_prompt }}
        {% else %}
        Iteration {{ inputs.iteration }}: Refine previous response.
        {% endif %}

        Provide: response, confidence (0.0-1.0), needs_refinement (bool)
      response_schema:
        type: object
        properties:
          response:
            type: string
          confidence:
            type: number
            minimum: 0
            maximum: 1
          needs_refinement:
            type: boolean
        required: [response, confidence, needs_refinement]

  # Track iteration
  - id: track_iteration
    type: Workflow
    depends_on: [refine]
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"
        parent_id: "{{ blocks.init_state.outputs.task_id if inputs.iteration == 0 else '' }}"
        task: "Iteration {{ inputs.iteration }}"
        status: "done"
        caller: "iterative-refinement"
        data:
          confidence: "{{ blocks.refine.outputs.response.confidence }}"
          needs_refinement: "{{ blocks.refine.outputs.response.needs_refinement }}"

  # Decide whether to continue
  - id: should_continue
    type: Shell
    depends_on: [refine]
    inputs:
      command: |
        python3 -c "
        import json
        confidence = {{ blocks.refine.outputs.response.confidence }}
        needs_refinement = {{ blocks.refine.outputs.response.needs_refinement | tojson }}
        iteration = {{ inputs.iteration }}
        max_iter = {{ inputs.max_iterations }}
        threshold = {{ inputs.confidence_threshold }}

        should_continue = (
            needs_refinement and
            confidence < threshold and
            iteration < (max_iter - 1)
        )

        print(json.dumps({'continue': should_continue}))
        "

  # Recurse if needed
  - id: recurse
    type: Workflow
    condition: "{{ (blocks.should_continue.outputs.stdout | fromjson).continue }}"
    depends_on: [track_iteration, should_continue]
    inputs:
      workflow: iterative-refinement
      inputs:
        initial_prompt: "{{ inputs.initial_prompt }}"
        state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"
        iteration: "{{ inputs.iteration + 1 }}"
        max_iterations: "{{ inputs.max_iterations }}"
        confidence_threshold: "{{ inputs.confidence_threshold }}"

  # Finalize
  - id: finalize
    type: Workflow
    depends_on:
      - track_iteration
      - block: recurse
        required: false
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"
        task_id: "{{ blocks.init_state.outputs.task_id if inputs.iteration == 0 else blocks.track_iteration.outputs.task_id }}"
        status: "done"
        caller: "iterative-refinement"

outputs:
  state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"
  final_response: "{{ blocks.refine.outputs.response.response }}"
  final_confidence: "{{ blocks.refine.outputs.response.confidence }}"
  iterations_completed: "{{ inputs.iteration + 1 }}"
```

---

## Safety and Depth Tracking

### Preventing Infinite Recursion

**Always include depth tracking** in recursive workflows:

```yaml
inputs:
  depth:
    type: num
    required: false
    default: 0
  max_depth:
    type: num
    required: false
    default: 10

blocks:
  # Check depth limit BEFORE recursing
  - id: check_depth
    type: Shell
    inputs:
      command: |
        python3 -c "
        depth = {{ inputs.depth }}
        max_depth = {{ inputs.max_depth }}
        if depth >= max_depth:
            print('Max depth reached')
            exit(1)
        print('OK')
        "

  # Recurse only if depth check passed
  - id: recurse
    type: Workflow
    condition: "{{ blocks.check_depth.succeeded }}"
    depends_on: [check_depth]
    inputs:
      workflow: my-recursive-workflow
      inputs:
        depth: "{{ inputs.depth + 1 }}"
        max_depth: "{{ inputs.max_depth }}"
```

### Environment-Based Limits

Configure global recursion limits via environment variables:

```bash
# In Claude Desktop config or shell
export WORKFLOWS_MAX_RECURSION_DEPTH=50  # Default limit
```

The system enforces this limit automatically, failing workflows that exceed the configured depth.

### Termination Conditions

**Always provide multiple termination conditions**:

```yaml
# Terminate when:
# 1. Depth limit reached
condition: "{{ inputs.depth < inputs.max_depth }}"

# 2. No more work to do
condition: "{{ inputs.items | length > 0 }}"

# 3. Confidence threshold met
condition: "{{ blocks.llm_call.outputs.response.confidence < 0.9 }}"

# 4. Iteration limit reached
condition: "{{ inputs.iteration < inputs.max_iterations }}"

# 5. All children complete
condition: "{{ not blocks.state_mgmt.outputs.all_children_done }}"
```

### Graceful Degradation

Handle depth limits gracefully:

```yaml
- id: process_with_limit
  type: Shell
  condition: "{{ inputs.depth < inputs.max_depth }}"
  inputs:
    command: |
      # Full processing
      echo "Processing at depth {{ inputs.depth }}"

- id: process_shallow
  type: Shell
  condition: "{{ inputs.depth >= inputs.max_depth }}"
  inputs:
    command: |
      # Fallback processing (no recursion)
      echo "Depth limit reached, shallow processing only"
```

---

## Debugging and Visualization

### Visualizing Task Trees

Use the `agent-state-visualize` workflow to inspect task hierarchies:

```python
# Execute visualization
result = await execute_workflow(
    workflow="agent-state-visualize",
    inputs={
        "state": "/Users/user/.workflows/tasks/exec-abc123.db",
        "show_data": True,
        "max_depth": 0  # 0 = unlimited
    }
)

print(result["tree"])
```

**Example Output**:
```bash
PR review [task-c4087afc] ● in-progress (1m6s)
├─ Phase: Context Gathering ✓ 9.8s (platform: github, files: 19)
├─ Phase: Initial Assessment ✓ 22.0s (risk: medium, focus: general)
├─ Phase: Investigation Loop ● 28.5s
│  ├─ Investigate: src/api.py ✓ 4.2s (severity: high)
│  ├─ Investigate: src/auth.py ✓ 5.1s (severity: medium)
│  ├─ Investigate: tests/test_api.py ✓ 3.5s (severity: low)
│  └─ Investigate: README.md ○ pending
├─ Phase: Synthesis ○ pending
└─ Phase: Action ○ pending
```

### Status Icons

| Icon | Status | Meaning |
|------|--------|---------|
| ✓ | `done` | Completed successfully |
| ✗ | `failed` | Failed with error |
| ● | `in-progress` | Currently executing |
| ○ | `pending` | Not yet started |
| ⚠ | `blocked` | Waiting on dependency |

### Visualization Options

```yaml
inputs:
  show_data:
    type: bool
    default: true
    description: Include task data fields in output

  max_depth:
    type: num
    default: 0
    description: Limit tree depth (0 = unlimited)

  task_id:
    type: str
    default: ""
    description: Visualize subtree starting from this task
```

### Querying State Directly

State is stored in SQLite with queryable tables:

```bash
# Open database
sqlite3 ~/.workflows/tasks/exec-abc123.db

# List all tasks
SELECT task_id, status, task FROM tasks;

# Get task details
SELECT * FROM tasks WHERE task_id = 'task-abc123';

# Check children status
SELECT task_id, status FROM tasks WHERE parent_id = 'task-abc123';

# Task tree with hierarchy
SELECT t.task_id, t.status, t.task, p.task_id as parent
FROM tasks t
LEFT JOIN tasks p ON t.parent_id = p.task_id;
```

### Audit Trail Analysis

Review audit logs to understand workflow execution:

```bash
# Open database
sqlite3 ~/.workflows/tasks/exec-abc123.db

# View audit log
SELECT * FROM audit ORDER BY timestamp;

# Filter by caller
SELECT * FROM audit WHERE caller LIKE 'pr-review:investigation-loop%';

# Timeline of events
SELECT timestamp, task_id, action, description FROM audit ORDER BY timestamp;
```

---

## Examples

### Example 1: PR Review Agent (Real-World)

Complete multi-level recursive workflow for autonomous PR review:

```yaml
# Main workflow: agent-pr-review
name: agent-pr-review
description: AI autonomous agent for in-depth PR/MR review

inputs:
  url:
    type: str
    required: false
  repo_path:
    type: str
    required: false
  threshold:
    type: str
    default: low
  iteration_cap:
    type: num
    default: 10

blocks:
  # Root task
  - id: state_management
    type: Workflow
    inputs:
      workflow: agent-state-management
      inputs:
        task: "PR review"
        caller: "pr-review"
        data: "{{ inputs }}"

  # Phase 1: Context Gathering
  - id: context_gathering
    type: Workflow
    depends_on: [state_management]
    inputs:
      workflow: context-gathering
      inputs:
        url: "{{ inputs.url }}"
        repo_path: "{{ inputs.repo_path }}"
        state: "{{ blocks.state_management.outputs.state }}"
        parent_task_id: "{{ blocks.state_management.outputs.task_id }}"

  # Phase 2: Initial Assessment
  - id: initial_assessment
    type: Workflow
    depends_on: [context_gathering]
    inputs:
      workflow: initial-assessment
      inputs:
        diff: "{{ blocks.context_gathering.outputs.diff }}"
        state: "{{ blocks.state_management.outputs.state }}"
        parent_task_id: "{{ blocks.state_management.outputs.task_id }}"

  # Phase 3: Investigation Loop (Multi-level recursion)
  - id: agent_investigation_loop
    type: Workflow
    depends_on: [initial_assessment]
    inputs:
      workflow: agent-investigation-loop
      inputs:
        investigation_targets: "{{ blocks.initial_assessment.outputs.investigation_targets }}"
        repo_path: "{{ blocks.context_gathering.outputs.repo_path }}"
        iteration_cap: "{{ inputs.iteration_cap }}"
        state: "{{ blocks.state_management.outputs.state }}"
        parent_task_id: "{{ blocks.state_management.outputs.task_id }}"

  # Phase 4: Synthesis
  - id: synthesis
    type: Workflow
    depends_on: [agent_investigation_loop]
    inputs:
      workflow: synthesis
      inputs:
        investigation_results: "{{ blocks.agent_investigation_loop.outputs.investigation_results }}"
        threshold: "{{ inputs.threshold }}"
        state: "{{ blocks.state_management.outputs.state }}"
        parent_task_id: "{{ blocks.state_management.outputs.task_id }}"

  # Finalize root task
  - id: finalize
    type: Workflow
    depends_on: [synthesis]
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ blocks.state_management.outputs.state }}"
        task_id: "{{ blocks.state_management.outputs.task_id }}"
        status: "done"
        caller: "pr-review"

outputs:
  report: "{{ blocks.synthesis.outputs.report }}"
  approve: "{{ blocks.synthesis.outputs.approve }}"
  state: "{{ blocks.state_management.outputs.state }}"
```

**Investigation Loop with Per-File Sub-Tasks**:

```yaml
# Investigation phase: agent-investigation-loop
name: agent-investigation-loop
description: Iteratively investigate PR with per-file sub-tasks

inputs:
  investigation_targets:
    type: list
    required: true
  repo_path:
    type: str
    required: true
  iteration_cap:
    type: num
    default: 10
  state:
    type: str
    required: true
  parent_task_id:
    type: str
    required: true

blocks:
  # Create phase task
  - id: track_start
    type: Workflow
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        parent_id: "{{ inputs.parent_task_id }}"
        task: "Phase: Investigation Loop"
        task_type: "phase"
        status: "in-progress"
        caller: "pr-review:investigation-loop"

  # Process each file (creates sub-task per file)
  - id: process_targets
    type: Workflow
    depends_on: [track_start]
    for_each: "{{ inputs.investigation_targets }}"
    for_each_mode: sequential
    inputs:
      workflow: investigation-sub-workflow
      inputs:
        target: "{{ each.value }}"
        repo_path: "{{ inputs.repo_path }}"
        state: "{{ inputs.state }}"
        parent_task_id: "{{ blocks.track_start.outputs.task_id }}"

  # Mark phase complete
  - id: track_done
    type: Workflow
    depends_on: [process_targets]
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        task_id: "{{ blocks.track_start.outputs.task_id }}"
        status: "done"
        caller: "pr-review:investigation-loop"

outputs:
  investigation_results: "{{ blocks.process_targets.outputs }}"
```

**File Investigation Sub-Workflow with Recursive Related Files**:

```yaml
# Per-file investigation: investigation-sub-workflow
name: investigation-sub-workflow
description: Investigate single file, recursively check related files

inputs:
  target:
    type: dict
    required: true
  repo_path:
    type: str
    required: true
  state:
    type: str
    required: true
  parent_task_id:
    type: str
    required: true
  investigation_depth:
    type: num
    default: 0
  max_investigation_depth:
    type: num
    default: 2

blocks:
  # Create file task
  - id: track_file_start
    type: Workflow
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        parent_id: "{{ inputs.parent_task_id }}"
        task: "Investigate: {{ inputs.target.path }}"
        task_type: "file-investigation"
        status: "in-progress"
        caller: "investigation-sub-workflow"

  # Read file and analyze
  - id: read_file
    type: Shell
    depends_on: [track_file_start]
    inputs:
      working_dir: "{{ inputs.repo_path }}"
      command: |
        cat "{{ inputs.target.path }}" | head -n 1000 | cat -n

  - id: analyze_file
    type: LLMCall
    depends_on: [read_file]
    inputs:
      profile: default
      prompt: |
        Analyze this file for issues:
        {{ blocks.read_file.outputs.stdout }}
      response_schema:
        type: object
        properties:
          severity:
            type: string
            enum: [critical, high, medium, low, info]
          issues:
            type: array
            items:
              type: object
          needs_further_investigation:
            type: boolean
          related_files:
            type: array
            items:
              type: string

  # Recursively investigate related files (depth-limited)
  - id: investigate_related
    type: Workflow
    condition: >-
      {{ blocks.analyze_file.outputs.response.needs_further_investigation
         and blocks.analyze_file.outputs.response.related_files | length > 0
         and inputs.investigation_depth < inputs.max_investigation_depth }}
    depends_on: [analyze_file]
    for_each: "{{ blocks.analyze_file.outputs.response.related_files }}"
    for_each_mode: sequential
    inputs:
      workflow: investigation-sub-workflow
      inputs:
        target:
          path: "{{ each.value }}"
          reason: "Related to {{ inputs.target.path }}"
        repo_path: "{{ inputs.repo_path }}"
        state: "{{ inputs.state }}"
        parent_task_id: "{{ blocks.track_file_start.outputs.task_id }}"
        investigation_depth: "{{ inputs.investigation_depth + 1 }}"
        max_investigation_depth: "{{ inputs.max_investigation_depth }}"

  # Mark file complete
  - id: track_file_done
    type: Workflow
    depends_on:
      - analyze_file
      - block: investigate_related
        required: false
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ inputs.state }}"
        task_id: "{{ blocks.track_file_start.outputs.task_id }}"
        status: "done"
        caller: "investigation-sub-workflow"
        data:
          severity: "{{ blocks.analyze_file.outputs.response.severity }}"
          issues_count: "{{ blocks.analyze_file.outputs.response.issues | length }}"

outputs:
  investigation_result: "{{ blocks.analyze_file.outputs.response }}"
```

**Resulting State Tree**:
```bash
PR review [task-fa67c4a3] ✓ done (1m45s) (platform: github, files: 19)
├─ Phase: Context Gathering ✓ 9.8s
├─ Phase: Initial Assessment ✓ 22.0s (risk: medium)
├─ Phase: Investigation Loop ✓ 58.3s
│  ├─ Investigate: src/api.py ✓ 12.5s (severity: high, issues: 3)
│  │  ├─ Investigate: src/auth.py ✓ 4.1s (severity: medium, issues: 1)
│  │  └─ Investigate: src/models.py ✓ 3.2s (severity: low, issues: 0)
│  ├─ Investigate: tests/test_api.py ✓ 8.7s (severity: low, issues: 2)
│  └─ Investigate: README.md ✓ 5.3s (severity: info, issues: 0)
├─ Phase: Synthesis ✓ 11.2s
└─ Phase: Action ✓ 3.7s (approve: false)
```

### Example 2: Batch File Processor

Process large file batches with chunking and state tracking:

```yaml
name: batch-file-processor
description: Process files in batches with progress tracking

inputs:
  files:
    type: list
    required: true
  batch_size:
    type: num
    default: 10
  state:
    type: str
    required: false
    default: ""

blocks:
  # Create root task
  - id: init_state
    type: Workflow
    condition: "{{ inputs.state == '' }}"
    inputs:
      workflow: agent-state-management
      inputs:
        task: "Batch Processing ({{ inputs.files | length }} files)"
        caller: "batch-file-processor"
        status: "in-progress"
        data:
          total_files: "{{ inputs.files | length }}"
          batch_size: "{{ inputs.batch_size }}"

  # Get current batch
  - id: get_batch
    type: Shell
    depends_on:
      - block: init_state
        required: false
    inputs:
      command: |
        python3 -c "
        import json
        files = {{ inputs.files | tojson }}
        batch_size = {{ inputs.batch_size }}
        batch = files[:batch_size]
        remaining = files[batch_size:]
        print(json.dumps({'batch': batch, 'remaining': remaining}))
        "

  # Process current batch
  - id: process_batch
    type: Workflow
    depends_on: [get_batch]
    for_each: "{{ (blocks.get_batch.outputs.stdout | fromjson).batch }}"
    for_each_mode: parallel  # Process batch in parallel
    inputs:
      workflow: process-single-file
      inputs:
        file_path: "{{ each.value }}"
        state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"
        parent_task_id: "{{ blocks.init_state.outputs.task_id }}"

  # Check if more batches remain
  - id: has_remaining
    type: Shell
    depends_on: [process_batch]
    inputs:
      command: |
        python3 -c "
        import json
        remaining = {{ (blocks.get_batch.outputs.stdout | fromjson).remaining | tojson }}
        print(json.dumps({'has_remaining': len(remaining) > 0, 'count': len(remaining)}))
        "

  # Recurse for next batch
  - id: process_next_batch
    type: Workflow
    condition: "{{ (blocks.has_remaining.outputs.stdout | fromjson).has_remaining }}"
    depends_on: [has_remaining]
    inputs:
      workflow: batch-file-processor
      inputs:
        files: "{{ (blocks.get_batch.outputs.stdout | fromjson).remaining }}"
        batch_size: "{{ inputs.batch_size }}"
        state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"

  # Mark complete
  - id: finalize
    type: Workflow
    depends_on:
      - process_batch
      - block: process_next_batch
        required: false
    condition: "{{ not (blocks.has_remaining.outputs.stdout | fromjson).has_remaining }}"
    inputs:
      workflow: agent-state-management
      inputs:
        state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"
        task_id: "{{ blocks.init_state.outputs.task_id }}"
        status: "done"
        caller: "batch-file-processor"

outputs:
  state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"
  progress: "{{ blocks.finalize.outputs.progress_pct | default(0) }}"
```

---

## Best Practices

### 1. Always Use State Management for Recursive Workflows

**✅ Good** (state tracked):
```yaml
- id: init_state
  type: Workflow
  condition: "{{ inputs.state == '' }}"
  inputs:
    workflow: agent-state-management
    inputs:
      task: "{{ inputs.task_name }}"
      caller: "my-workflow"
```

**❌ Bad** (no state tracking):
```yaml
- id: recurse
  type: Workflow
  inputs:
    workflow: my-workflow
    inputs:
      items: "{{ inputs.items[1:] }}"
  # No state tracking = no visibility into progress
```

### 2. Track Phase Boundaries Explicitly

Mark phase start and completion:

```yaml
# Phase start
- id: track_phase_start
  type: Workflow
  inputs:
    workflow: agent-state-management
    inputs:
      state: "{{ inputs.state }}"
      parent_id: "{{ inputs.parent_task_id }}"
      task: "Phase: {{ inputs.phase_name }}"
      status: "in-progress"
      caller: "my-workflow:{{ inputs.phase_name }}"

# ... do work ...

# Phase completion
- id: track_phase_done
  type: Workflow
  depends_on: [work]
  inputs:
    workflow: agent-state-management
    inputs:
      state: "{{ inputs.state }}"
      task_id: "{{ blocks.track_phase_start.outputs.task_id }}"
      status: "done"
      caller: "my-workflow:{{ inputs.phase_name }}"
```

### 3. Use Descriptive Task Names and Data

```yaml
# ✅ Good: Descriptive names and meaningful data
- id: track_file
  type: Workflow
  inputs:
    workflow: agent-state-management
    inputs:
      task: "Investigate: src/api.py"
      data:
        path: "src/api.py"
        severity: "high"
        issues_count: 3

# ❌ Bad: Generic names, no data
- id: track_task
  type: Workflow
  inputs:
    workflow: agent-state-management
    inputs:
      task: "Task 1"
```

### 4. Gate Phases on Child Completion

Use `all_children_done` to wait for sub-tasks:

```yaml
# Phase that creates sub-tasks
- id: investigation_phase
  type: Workflow
  inputs:
    workflow: investigation-phase
    # ... creates children ...

# Wait for all children to complete
- id: synthesis_phase
  type: Workflow
  depends_on: [investigation_phase]
  # This phase waits because depends_on ensures investigation_phase
  # has completed, which includes all its children
```

### 5. Handle Errors Gracefully

Check for failed children:

```yaml
- id: check_failures
  type: Shell
  depends_on: [phase_with_children]
  inputs:
    command: |
      python3 -c "
      import json, sys
      any_failed = {{ blocks.phase_with_children.outputs.any_child_failed | tojson }}
      if any_failed:
          print('ERROR: Some children failed')
          sys.exit(1)
      print('All children succeeded')
      "

- id: handle_success
  type: Workflow
  condition: "{{ blocks.check_failures.succeeded }}"
  depends_on: [check_failures]
  # ... continue normally ...

- id: handle_failure
  type: Workflow
  condition: "{{ blocks.check_failures.failed }}"
  depends_on: [check_failures]
  # ... error recovery ...
```

### 6. Limit Recursion Depth

Always provide termination conditions:

```yaml
inputs:
  depth:
    type: num
    default: 0
  max_depth:
    type: num
    default: 10

blocks:
  - id: recurse
    type: Workflow
    # Multiple termination conditions
    condition: >-
      {{ inputs.depth < inputs.max_depth
         and inputs.items | length > 0
         and not blocks.state.outputs.all_children_done }}
    inputs:
      workflow: my-recursive-workflow
      inputs:
        depth: "{{ inputs.depth + 1 }}"
        max_depth: "{{ inputs.max_depth }}"
```

### 7. Use Meaningful Caller Names

Caller names appear in audit logs:

```yaml
# ✅ Good: Hierarchical caller names
caller: "pr-review:investigation-loop:file-analysis"

# ❌ Bad: Generic caller
caller: "workflow"
```

### 8. Visualize During Development

Use visualization to debug:

```bash
# After running recursive workflow
execute_workflow(
    workflow="agent-state-visualize",
    inputs={"state": result["state"]}
)
```

---

## Troubleshooting

### Error: "Maximum recursion depth exceeded"

**Cause**: Workflow exceeded `WORKFLOWS_MAX_RECURSION_DEPTH` limit (default: 50).

**Solution**:

1. **Check termination conditions**:
   ```yaml
   # Ensure conditions prevent infinite loops
   condition: "{{ inputs.depth < inputs.max_depth }}"
   ```

2. **Increase limit** (if legitimately needed):
   ```bash
   export WORKFLOWS_MAX_RECURSION_DEPTH=100
   ```

3. **Verify iteration counter increments**:
   ```yaml
   # Make sure depth actually increases
   depth: "{{ inputs.depth + 1 }}"
   ```

### State File Not Found

**Cause**: State file path not persisted between recursive calls.

**Solution**:

```yaml
# ✅ Always pass state through recursion
- id: recurse
  type: Workflow
  inputs:
    workflow: my-workflow
    inputs:
      state: "{{ blocks.init_state.outputs.state | default(inputs.state, true) }}"

# ❌ Don't lose state reference
- id: recurse
  type: Workflow
  inputs:
    workflow: my-workflow
    # Missing state parameter!
```

### Children Not Showing in State Tree

**Cause**: Sub-tasks not linked to parent via `parent_id`.

**Solution**:

```yaml
# ✅ Provide parent_id when creating sub-tasks
- id: create_subtask
  type: Workflow
  inputs:
    workflow: agent-state-management
    inputs:
      state: "{{ inputs.state }}"
      parent_id: "{{ inputs.parent_task_id }}"  # Link to parent
      task: "Sub-task name"
      caller: "my-workflow"
```

### Tasks Stuck in "in-progress"

**Cause**: Workflow failed before marking task complete.

**Solution**:

1. **Check workflow execution logs** for errors
2. **Always finalize tasks**:
   ```yaml
   - id: finalize
     type: Workflow
     depends_on: [work]
     inputs:
       workflow: agent-state-management
       inputs:
         state: "{{ inputs.state }}"
         task_id: "{{ blocks.track_start.outputs.task_id }}"
         status: "done"  # or "failed" on error
         caller: "my-workflow"
   ```

3. **Use error handling blocks**:
   ```yaml
   - id: finalize_on_error
     type: Workflow
     condition: "{{ blocks.work.failed }}"
     depends_on: [work]
     inputs:
       workflow: agent-state-management
       inputs:
         status: "failed"
   ```

### Visualization Shows Wrong Tree Structure

**Cause**: Incorrect `parent_id` references.

**Diagnosis**:

```bash
# Check parent-child relationships
sqlite3 ~/.workflows/tasks/exec-abc123.db \
  "SELECT task_id, parent_id FROM tasks"
```

**Solution**: Ensure `parent_id` matches the actual parent's `task_id`.

---

## FAQ

### Q: Can I pause and resume recursive workflows?

**A**: Yes, the SQLite database persists between calls. You can stop execution and resume later by passing the same `state` path:

```yaml
# First execution
result1 = execute_workflow(workflow="my-workflow", inputs={"items": [...], "state": ""})

# Resume later with same state
result2 = execute_workflow(workflow="my-workflow", inputs={"items": [...], "state": result1["state"]})
```

### Q: How deep can recursion go?

**A**: Default limit is 50 levels (configurable via `WORKFLOWS_MAX_RECURSION_DEPTH`). Maximum: 10,000 levels.

### Q: Can I run child tasks in parallel?

**A**: Yes, use `for_each_mode: parallel`:

```yaml
- id: process_files
  type: Workflow
  for_each: "{{ inputs.files }}"
  for_each_mode: parallel  # Process all files concurrently
```

### Q: How do I delete old state databases?

**A**: State databases accumulate in `~/.workflows/tasks/`. Clean up manually:

```bash
# Delete state databases older than 7 days
find ~/.workflows/tasks -name "*.db" -mtime +7 -delete
```

### Q: Can I use state management without recursion?

**A**: Yes, use state tracking for any multi-phase or multi-task workflow, even without recursion.

### Q: What's the performance impact of state tracking?

**A**: Minimal - SQLite operations are fast with WAL mode. Typical overhead: < 100ms per task. Concurrent writes are handled automatically.

### Q: Can I share state between different workflows?

**A**: Yes, pass the `state` path between workflows:

```yaml
- id: workflow1
  type: Workflow
  inputs:
    workflow: first-workflow
    inputs:
      state: ""

- id: workflow2
  type: Workflow
  depends_on: [workflow1]
  inputs:
    workflow: second-workflow
    inputs:
      state: "{{ blocks.workflow1.outputs.state }}"
```

### Q: How do I test recursive workflows?

**A**: Use small inputs and low depth limits during testing:

```yaml
# Test with minimal recursion
inputs:
  max_depth: 2  # instead of 10
  items: ["item1", "item2"]  # instead of large list
```

---

## Conclusion

Recursive workflows with state management enable sophisticated, multi-level task execution with full visibility and control. By following the patterns and best practices in this guide, you can build complex agent workflows, batch processors, and tree traversal systems that are maintainable, debuggable, and production-ready.

**Key Takeaways**:
- ✅ Use state management for all recursive patterns
- ✅ Track phase boundaries explicitly with state transitions
- ✅ Provide multiple termination conditions for safety
- ✅ Use visualization for debugging complex hierarchies
- ✅ Attach meaningful data to tasks for observability
- ✅ Gate phases on child completion using `all_children_done`

**Next Steps**:
- Study the PR review agent example (real-world multi-level pattern)
- Build a simple recursive workflow with state tracking
- Visualize your task trees to understand execution flow
- Explore the `agent-state-management` and `agent-state-visualize` workflows

**Resources**:
- [State Management Workflow](../../src/workflows_mcp/templates/agents/state-management/main.yaml)
- [State Visualization Workflow](../../src/workflows_mcp/templates/agents/state-visualize/main.yaml)
- [PR Review Agent Example](../../src/workflows_mcp/templates/agents/pr-review/)
- [ADR-009: Unified Job Architecture](../adr/ADR-009-unified-job-architecture.md)

For questions or issues, please file a GitHub issue or consult the documentation.
