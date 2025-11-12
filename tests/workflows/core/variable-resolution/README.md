# Variable Resolution Test Suite

Comprehensive test suite for the four-namespace variable system in workflows-mcp.

## Test Coverage

### 1. **inputs.yaml** - Input Namespace Testing
**Enhanced with type validation**

Tests variable resolution from the `inputs` namespace with Python-based type validation:

- ✅ String inputs with type validation
- ✅ Numeric inputs with arithmetic operations
- ✅ Boolean inputs with type checking
- ✅ Multiple inputs in single expression
- ✅ Inputs in environment variables

**Example:**
```python
python3 -c "
value = {{inputs.number_input}}
assert isinstance(value, (int, float))
assert value == 42
result = value + 8
assert result == 50
"
```

### 2. **block-outputs.yaml** - Block Outputs Namespace Testing
**Enhanced with comprehensive output validation**

Tests variable resolution from the `blocks` namespace with validation of all output types:

- ✅ Standard outputs (stdout, stderr, exit_code)
- ✅ Chained variable resolution across blocks
- ✅ Shortcut syntax vs explicit paths
- ✅ Type validation for each output type

**Example:**
```python
# Shortcut syntax (auto-expands to outputs.stdout)
shortcut = '{{blocks.producer.stdout}}'
# Full path syntax
fullpath = '{{blocks.producer.outputs.stdout}}'
assert shortcut == fullpath
```

### 3. **custom-outputs.yaml** - Shell Custom Outputs Testing
**New test for file-based custom outputs**

Tests the Shell block's custom file-based outputs feature:

- ✅ String custom outputs from files
- ✅ Numeric custom outputs with type parsing
- ✅ Boolean custom outputs with type conversion
- ✅ JSON custom outputs with structure validation
- ✅ Explicit vs shortcut access patterns
- ✅ Complex expressions mixing standard and custom outputs

**Example:**
```yaml
blocks:
  - id: producer
    type: Shell
    inputs:
      command: |
        echo "42" > "$SCRATCH/number.txt"
        echo '{"count": 5}' > "$SCRATCH/data.json"
    outputs:
      number:
        type: num
        path: "$SCRATCH/number.txt"
      data:
        type: json
        path: "$SCRATCH/data.json"
```

### 4. **cross-namespace.yaml** - Multi-Namespace Testing
**New test for cross-namespace variable access**

Tests mixing variables from multiple namespaces in single expressions:

- ✅ `inputs` + `metadata` combination
- ✅ `inputs` + `blocks` combination
- ✅ `metadata` + `blocks` combination
- ✅ All three namespaces together
- ✅ Nested access patterns

**Example:**
```python
# Mix all three namespaces
base = {{inputs.base_value}}           # inputs namespace
workflow = '{{metadata.workflow_name}}' # metadata namespace
number = {{blocks.producer.number}}     # blocks namespace (custom output)
exit_code = {{blocks.producer.exit_code}} # blocks namespace (standard)
```

### 5. **chained-flow.yaml** - Data Pipeline Testing
**New test for variable flow through pipelines**

Tests how variables flow through multi-stage data transformation pipelines:

- ✅ Sequential data transformations (6 stages)
- ✅ Numeric pipeline (input → stage1 → stage2 → stage3 → stage4)
- ✅ String transformations
- ✅ JSON aggregation from multiple stages
- ✅ Accessing the entire chain from any block
- ✅ Nested JSON field access

**Pipeline Flow:**
```text
Initial: 5
├─ Stage 1: 5 * 2 = 10
├─ Stage 2: 10 + 15 = 25
├─ Stage 3: 10 + 25 = 35
├─ Stage 4: 35 * 2 = 70
├─ Stage 5: format("value_70")
└─ Stage 6: JSON aggregation of all stages
```

### 6. **metadata.yaml** - Metadata Namespace Testing
**Existing test (unchanged)**

Tests variable resolution from the `metadata` namespace:
- ✅ Workflow name access
- ✅ Timestamp access

### 7. **shortcuts.yaml** - Shortcut Syntax Testing
**Existing test (unchanged)**

Tests shortcut syntax for accessing block outputs:
- ✅ Status shortcuts (`blocks.id.status`)
- ✅ Boolean shortcuts (`blocks.id.succeeded`)
- ✅ Full path equivalence

## Key Testing Patterns

### Type Validation Pattern
All enhanced tests use Python one-liners for robust type validation:

```python
python3 -c "
value = {{variable}}
assert isinstance(value, expected_type)
assert value == expected_value
print(f'Validation passed: {value}')
"
```

### Benefits:
- **Portable**: Works on any platform with Python 3
- **Clear**: Type errors are explicit
- **Testable**: Exit code 0 = success, non-zero = failure
- **Debuggable**: Assertions provide clear error messages

## Four-Namespace System

The test suite validates all four namespaces:

1. **`inputs`** - Workflow input parameters
2. **`metadata`** - Workflow execution metadata
3. **`blocks`** - Block execution results (outputs + metadata)
4. **`__internal__`** - System state (blocked by security boundary)

## Test Execution

Run all variable resolution tests:
```bash
uv run pytest tests/test_mcp_client.py::TestWorkflowSnapshots -k "variable-resolution" -v
```

Run specific test:
```bash
uv run pytest tests/test_mcp_client.py::TestWorkflowSnapshots::test_workflow_execution_matches_snapshot -k "custom-outputs" -v
```

## Coverage Summary

| Feature | Tested | Location |
|---------|--------|----------|
| Input types (str, num, bool) | ✅ | inputs.yaml |
| Standard outputs | ✅ | block-outputs.yaml |
| Custom file outputs | ✅ | custom-outputs.yaml |
| Exit codes | ✅ | block-outputs.yaml |
| Metadata access | ✅ | metadata.yaml |
| Shortcut syntax | ✅ | shortcuts.yaml |
| Cross-namespace mixing | ✅ | cross-namespace.yaml |
| Data pipelines | ✅ | chained-flow.yaml |
| Type validation | ✅ | All tests |
| Environment variables | ✅ | inputs.yaml |
| JSON outputs | ✅ | custom-outputs.yaml, chained-flow.yaml |

## Schema Improvements

This test suite also validates the schema fix for Shell block `outputs` field:

**Before:** `outputs` was incorrectly named `custom_outputs` in the `inputs` object
**After:** `outputs` is correctly defined at the block level (sibling to `inputs`)

```yaml
blocks:
  - id: example
    type: Shell
    inputs:
      command: echo "test"
    outputs:          # ← Block-level field (not in inputs)
      custom_field:
        type: str
        path: "$SCRATCH/file.txt"
```
