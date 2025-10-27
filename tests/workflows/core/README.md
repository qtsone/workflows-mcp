# Core Test Workflows

This directory contains focused, atomic test workflows that validate individual features of the workflow engine. Each test has a single, clear purpose and is designed to be easily understood and debugged.

## Test Organization

Tests are organized by feature category, with each test validating a specific capability:

### Variable Resolution (4 tests)

Tests for the variable resolution system across all namespaces:

- **[inputs.yaml](variable-resolution/inputs.yaml)**: Tests `{{inputs.*}}` variable resolution
- **[metadata.yaml](variable-resolution/metadata.yaml)**: Tests `{{metadata.*}}` variable resolution
- **[block-outputs.yaml](variable-resolution/block-outputs.yaml)**: Tests `{{blocks.*.outputs.*}}` variable resolution
- **[shortcuts.yaml](variable-resolution/shortcuts.yaml)**: Tests shortcut syntax `{{blocks.*.field}}` (ADR-007)

### Conditional Execution (3 tests)

Tests for conditional block execution with boolean expressions:

- **[input-based.yaml](conditionals/input-based.yaml)**: Tests conditions based on `{{inputs.*}}`
- **[block-status.yaml](conditionals/block-status.yaml)**: Tests conditions based on block status (`succeeded`, `failed`)
- **[complex-expressions.yaml](conditionals/complex-expressions.yaml)**: Tests complex boolean expressions (AND, OR, NOT)

### Block Status Detection (3 tests) ⭐ NEW

Tests for ADR-007 block status shortcuts - **previously missing coverage**:

- **[success-detection.yaml](block-status/success-detection.yaml)**: Tests `.succeeded` shortcut
- **[failure-detection.yaml](block-status/failure-detection.yaml)**: Tests `.failed` shortcut
- **[skip-detection.yaml](block-status/skip-detection.yaml)**: Tests `.skipped` shortcut

### File Operations (3 tests)

Tests for file manipulation executors:

- **[create-file.yaml](file-operations/create-file.yaml)**: Tests CreateFile with encoding, permissions
- **[read-file.yaml](file-operations/read-file.yaml)**: Tests ReadFile with encoding, optional files
- **[template-render.yaml](file-operations/template-render.yaml)**: Tests RenderTemplate with Jinja2 templates

### State Management (3 tests)

Tests for JSON state management executors:

- **[write-state.yaml](state-management/write-state.yaml)**: Tests WriteJSONState with various data types
- **[read-state.yaml](state-management/read-state.yaml)**: Tests ReadJSONState with nested field access
- **[merge-state.yaml](state-management/merge-state.yaml)**: Tests MergeJSONState for updating existing state

### DAG Execution (3 tests)

Tests for workflow execution order and parallelization:

- **[parallel-execution.yaml](dag-execution/parallel-execution.yaml)**: Tests parallel execution of independent blocks
- **[dependency-order.yaml](dag-execution/dependency-order.yaml)**: Tests sequential dependency execution ⭐ NEW
- **[optional-deps.yaml](dag-execution/optional-deps.yaml)**: Tests optional dependencies (`required: false`)

### Workflow Composition (4 tests)

Tests for workflow composition and recursion:

- **[_child-calculator.yaml](composition/_child-calculator.yaml)**: Helper workflow for composition tests
- **[workflow-call.yaml](composition/workflow-call.yaml)**: Tests calling child workflows
- **[output-passing.yaml](composition/output-passing.yaml)**: Tests passing outputs between workflows ⭐ NEW
- **[recursion.yaml](composition/recursion.yaml)**: Tests recursive workflow patterns

## Test Naming Convention

All test files follow this naming pattern:

```text
<category>-<specific-feature>.yaml
```

Examples:
- `variable-resolution-inputs.yaml` - Clear what's being tested
- `block-status-success-detection.yaml` - Specific feature tested
- `dag-execution-parallel.yaml` - Feature and behavior

## Test Principles

Each test follows these principles from MCP best practices:

### KISS (Keep It Simple, Stupid)
- Each test validates ONE feature
- Minimal number of blocks (typically 2-5)
- Clear, straightforward logic

### Single Responsibility
- Test name indicates exactly what's tested
- When test fails, immediately know which feature broke
- No mixing of multiple feature tests

### Clarity
- Descriptive names and descriptions
- Inline comments for complex logic
- Outputs validate expected behavior

### Diagnostic Value
- Test failure points to specific feature
- Easy to understand what broke and why
- Quick to fix or diagnose issues

## Running Tests

To run all core tests:

```bash
# Run all variable resolution tests
pytest tests/test_workflows/core/variable-resolution/

# Run specific test
pytest tests/test_workflows/core/variable-resolution/inputs.yaml

# Run all core tests
pytest tests/test_workflows/core/
```

## Test Coverage

These core tests provide:
- ✅ Complete coverage of variable resolution (4 namespaces)
- ✅ Complete coverage of conditional execution (3 types)
- ✅ Complete coverage of block status detection (3 shortcuts) ⭐ NEW
- ✅ Complete coverage of file operations (3 executors)
- ✅ Complete coverage of state management (3 executors)
- ✅ Complete coverage of DAG execution (parallel, sequential, optional)
- ✅ Complete coverage of workflow composition (call, outputs, recursion)

## Benefits Over Old Tests

### Clarity
- **Before**: "test-01-variables failed" - which namespace broke?
- **After**: "variable-resolution-shortcuts failed" - shortcut feature broke

### Maintainability
- **Before**: 74-line test with 6 features
- **After**: 6 tests, 10-25 lines each

### Diagnostic Value
- **Before**: Read entire test to understand failure
- **After**: Test name tells you exactly what broke

### Coverage
- **Before**: Missing ADR-007 shortcut tests
- **After**: Explicit tests for all documented features

## See Also

- [Integration Tests](../integration/) - End-to-end workflow patterns
- [Test Refactoring Plan](../../TEST_REFACTORING_PLAN.md) - Complete refactoring documentation
