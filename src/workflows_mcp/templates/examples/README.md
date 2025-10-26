# Example Workflows

This directory contains example YAML workflows that demonstrate the workflow system capabilities.

## Overview

These workflows showcase different features and patterns for building DAG-based workflows:

1. **hello-world.yaml** - Simplest possible workflow
2. **sequential-echo.yaml** - Sequential execution with dependencies
3. **parallel-echo.yaml** - Parallel execution (diamond pattern)
4. **input-substitution.yaml** - Variable substitution patterns
5. **complex-workflow.yaml** - Realistic multi-stage workflow

All examples use the `EchoBlock` for demonstration purposes.

## Workflow Descriptions

### 1. hello-world.yaml

**Purpose**: Simplest possible workflow demonstrating basic structure

**Features**:
- Single block execution
- Input parameter with default value
- Variable substitution from inputs
- Output mapping

**Usage**:
```python
from workflows_mcp.engine.executor import WorkflowExecutor

executor = WorkflowExecutor()
result = await executor.execute_workflow("hello-world")
# Output: {"greeting": "Echo: Hello, World!"}

result = await executor.execute_workflow("hello-world", {"name": "Claude"})
# Output: {"greeting": "Echo: Hello, Claude!"}
```

**Blocks**: 1
**Inputs**: 1 (name: string)
**Outputs**: 1 (greeting)

---

### 2. sequential-echo.yaml

**Purpose**: Demonstrate sequential execution with dependency chains

**Features**:
- Multiple blocks executing in order (block1 → block2 → block3)
- Explicit dependency declaration with `depends_on`
- Variable substitution from previous block outputs
- DAG resolution for sequential execution
- Output chaining through variables

**Execution Flow**:
```text
echo_first (no dependencies)
    ↓
echo_second (depends on echo_first)
    ↓
echo_third (depends on echo_second)
```

**Usage**:
```python
result = await executor.execute_workflow("sequential-echo")
# Outputs: first_output, second_output, final_output, total_time_ms

result = await executor.execute_workflow("sequential-echo", {
    "initial_message": "Custom start"
})
```

**Blocks**: 3
**Inputs**: 1 (initial_message: string)
**Outputs**: 4 (first_output, second_output, final_output, total_time_ms)

---

### 3. parallel-echo.yaml

**Purpose**: Demonstrate parallel execution capabilities (diamond DAG pattern)

**Features**:
- Diamond DAG pattern: start → (parallel_a, parallel_b) → final
- Parallel block execution (parallel_a and parallel_b run simultaneously)
- Multiple dependencies (final depends on both parallel blocks)
- Wave detection and concurrent execution
- Efficient resource utilization through parallelism

**Execution Flow**:
```bash
Wave 1: start_block
         ↓
Wave 2: parallel_a, parallel_b (execute simultaneously)
         ↓
Wave 3: final_merge (waits for both)
```

**Usage**:
```python
result = await executor.execute_workflow("parallel-echo")
# Outputs: start_output, branch_a_output, branch_b_output, final_output, merge_time_ms
```

**Blocks**: 4
**Inputs**: 1 (start_message: string)
**Outputs**: 5 (start_output, branch_a_output, branch_b_output, final_output, merge_time_ms)

**Performance Note**: Total execution time is less than sum of all delays due to parallel execution.

---

### 4. input-substitution.yaml

**Purpose**: Comprehensive demonstration of variable substitution patterns

**Features**:
- `${input_name}` - Direct input variable substitution
- `${block_id.field}` - Block output field substitution
- Multiple input types (string, integer, boolean)
- Input defaults for optional parameters
- Complex variable resolution chains
- Combining inputs and block outputs in messages

**Variable Substitution Rules**:
1. Input variables: `${input_name}` (no dot notation)
2. Block outputs: `${block_id.field_name}` (with dot notation)
3. Only works in string values (not integers/booleans)
4. Variables are resolved at execution time

**Usage**:
```python
# Use defaults
result = await executor.execute_workflow("input-substitution")

# Override inputs
result = await executor.execute_workflow("input-substitution", {
    "user_name": "Alice",
    "project_name": "awesome-app",
    "iterations": 5,
    "verbose": True
})
```

**Blocks**: 6
**Inputs**: 4 (user_name, project_name, iterations, verbose)
**Outputs**: 6 (greeting, configuration, combined, chained, metrics, final_execution_time)

---

### 5. complex-workflow.yaml

**Purpose**: Realistic multi-stage workflow demonstrating real-world patterns

**Features**:
- Multiple execution stages (initialize → process → validate → finalize)
- Mixed sequential and parallel execution patterns
- Complex dependency graph (8 blocks)
- Multiple inputs with various types
- Real-world workflow structure
- Wave-based execution optimization
- Comprehensive output mapping

**Workflow Structure**:
```text
Wave 1: initialize
         ↓
Wave 2: fetch_config, fetch_data (parallel)
         ↓
Wave 3: validate_config, validate_data (parallel)
         ↓
Wave 4: merge_results
         ↓
Wave 5: transform_output
         ↓
Wave 6: finalize
```

**Usage**:
```python
# Development environment
result = await executor.execute_workflow("complex-workflow")

# Production environment with custom settings
result = await executor.execute_workflow("complex-workflow", {
    "environment": "production",
    "data_source": "backup-db",
    "max_retries": 5,
    "enable_validation": True,
    "output_format": "json"
})
```

**Blocks**: 8
**Inputs**: 5 (environment, data_source, max_retries, enable_validation, output_format)
**Outputs**: 12 (initialization, config_status, data_status, merged, transformed, final_result, environment, output_format, validation_enabled, total_execution_time_ms, fetch_config_time_ms, fetch_data_time_ms)

---

## Testing

### Validate All Workflows

```bash
python test_example_workflows.py
```

This script:
- Validates YAML schema for all workflows
- Loads workflows via loader
- Displays workflow structure and metadata
- Reports pass/fail status for each workflow

### Load via WorkflowRegistry

```bash
python test_registry_load.py
```

This script:
- Loads workflows into WorkflowRegistry
- Lists all registered workflows
- Tests category filtering
- Displays workflow metadata

### Execute a Workflow

```python
from workflows_mcp.engine.executor import WorkflowExecutor
from workflows_mcp.engine.registry import WorkflowRegistry

# Initialize registry and load workflows
registry = WorkflowRegistry()
registry.load_from_directory("templates/examples")

# Get workflow definition
workflow_def = registry.get("hello-world")

# Execute workflow
executor = WorkflowExecutor()
executor.load_workflow(workflow_def)

result = await executor.execute_workflow("hello-world", {"name": "World"})

if result.is_success:
    print(f"Success! Outputs: {result.value}")
else:
    print(f"Execution failed: {result.error}")
```

## Learning Path

Recommended order for learning the workflow system:

1. **Start with hello-world.yaml**: Understand basic structure
2. **Study sequential-echo.yaml**: Learn about dependencies
3. **Explore parallel-echo.yaml**: Understand parallel execution
4. **Review input-substitution.yaml**: Master variable substitution
5. **Analyze complex-workflow.yaml**: See realistic patterns

## Key Concepts Demonstrated

### 1. Workflow Structure
- **Metadata**: name, description, category, version, author, tags
- **Inputs**: Typed parameters with defaults
- **Blocks**: Workflow steps with dependencies
- **Outputs**: Result mapping via variable substitution

### 2. Dependency Management
- **No dependencies**: Blocks execute in Wave 1
- **Sequential dependencies**: `depends_on: [previous_block]`
- **Multiple dependencies**: `depends_on: [block_a, block_b]`
- **DAG resolution**: Automatic topological sort

### 3. Variable Substitution
- **Input variables**: `${input_name}`
- **Block outputs**: `${block_id.field_name}`
- **Resolution order**: Inputs first, then block outputs
- **Validation**: Checked at load time

### 4. Execution Waves
- **Wave detection**: Automatic parallel execution grouping
- **Concurrent execution**: Blocks in same wave run simultaneously
- **Performance optimization**: Reduced total execution time

## Common Patterns

### Single Block Workflow
```yaml
blocks:
  - id: task
    type: EchoBlock
    inputs:
      message: "${input_message}"
```

### Sequential Chain
```yaml
blocks:
  - id: step1
    type: EchoBlock
  - id: step2
    type: EchoBlock
    depends_on: [step1]
  - id: step3
    type: EchoBlock
    depends_on: [step2]
```

### Parallel Execution
```yaml
blocks:
  - id: start
    type: EchoBlock
  - id: parallel_a
    type: EchoBlock
    depends_on: [start]
  - id: parallel_b
    type: EchoBlock
    depends_on: [start]
  - id: merge
    type: EchoBlock
    depends_on: [parallel_a, parallel_b]
```

### Variable Chaining
```yaml
blocks:
  - id: block1
    type: EchoBlock
    inputs:
      message: "${input_value}"
  - id: block2
    type: EchoBlock
    inputs:
      message: "Previous: ${block1.echoed}"
    depends_on: [block1]
```

## Next Steps

After understanding these examples:

1. **Create custom workflows**: Apply patterns to your use cases
2. **Build custom blocks**: Extend with new block types
3. **Integrate with MCP**: Expose workflows as MCP tools
4. **Explore templates**: Check other template categories (git, python, node, quality)

## Resources

- **YAML Workflow Guide**: `/docs/YAML_WORKFLOW_GUIDE.md`
- **Phase 1 Implementation**: [Phase 1 Implementation](../../docs/phase-1/PHASE1_IMPLEMENTATION.md)
- **Architecture Overview**: `/ARCHITECTURE.md`
- **Block Development**: `/src/workflows_mcp/engine/block.py`
