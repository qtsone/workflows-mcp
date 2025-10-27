# Git Semantic Commit Workflow

Intelligent semantic commit message generation following [Conventional Commits](https://www.conventionalcommits.org/) specification.

## Overview

The `git-semantic-commit` workflow analyzes your staged git changes and automatically generates a semantic commit message with appropriate type, scope, and description.

**Two Generation Modes:**

1. **Bash Heuristics (Default)**: Fast pattern-matching analysis using sophisticated regex and file analysis (~100ms, no cost)
2. **LLM-Powered (Optional)**: Intelligent semantic understanding using Claude via MCP interactive blocks (~2-3s, leverages current MCP session)

## Features

- **Dual Generation Modes**: Choose between fast bash heuristics or intelligent LLM analysis
- **Intelligent Type Detection**: Automatically determines commit type (feat, fix, docs, test, refactor, perf, style, chore, ci, build, revert)
- **Scope Extraction**: Identifies the most relevant scope from changed files
- **Breaking Change Detection**: Detects breaking changes from diff content
- **Conventional Commits**: Follows the industry-standard conventional commits specification
- **Interactive Confirmation**: Review and approve messages before committing (optional)
- **Flexible Workflow**: Supports automatic commit, manual override, and LLM enhancement

## Commit Type Detection

The workflow uses sophisticated heuristics to determine the commit type:

| Type | Detection Criteria | Example |
|------|-------------------|---------|
| **feat** | New files added, "implement"/"introduce" in diff | `feat(auth): add user authentication` |
| **fix** | "fix"/"bug"/"issue"/"error" in diff | `fix(api): resolve connection timeout` |
| **docs** | Only documentation files changed | `docs(readme): update installation guide` |
| **test** | Only test files changed | `test(users): add integration tests` |
| **refactor** | "refactor"/"restructure" in diff | `refactor(core): simplify data flow` |
| **perf** | "performance"/"optimize" in diff | `perf(queries): optimize database queries` |
| **style** | Only style files or formatting keywords | `style(components): apply code formatting` |
| **chore** | Config/dependency changes | `chore(deps): update dependencies` |
| **ci** | CI/CD configuration changes | `ci(github): add automated tests` |
| **build** | Build system/dependency changes | `build(npm): upgrade to webpack 5` |
| **revert** | "revert" in diff | `revert: revert previous changes` |

## Usage

### Basic Usage (Review Before Committing)

```bash
# Stage your changes
git add file1.py file2.py

# Generate commit message (displays proposal without committing)
mcp execute_workflow git-semantic-commit
```

This will display:
```text
=== PROPOSED SEMANTIC COMMIT ===

Message: feat(core): add user authentication implementation

=== STAGED FILES ===
A	src/core/auth.py
M	src/core/models.py

=== DIFF SUMMARY ===
 src/core/auth.py   | 42 ++++++++++++++++++++++++++++++++++++++++++
 src/core/models.py | 15 +++++++++++++--
 2 files changed, 55 insertions(+), 2 deletions(-)
```

**To actually commit**, run again with `override_message` after reviewing:

```bash
mcp execute_workflow git-semantic-commit \
  --inputs '{"override_message": "feat(core): add user authentication implementation"}'
```

### Auto-Commit Mode (Skip Review)

```bash
# Stage changes and commit immediately with generated message
mcp execute_workflow git-semantic-commit \
  --inputs '{"auto_commit": true}'
```

### Stage All Files First

```bash
# Stage all modified files and generate commit
mcp execute_workflow git-semantic-commit \
  --inputs '{"stage_all": true, "auto_commit": true}'
```

### Manual Message Override

```bash
# Provide your own commit message
mcp execute_workflow git-semantic-commit \
  --inputs '{"override_message": "fix(api)!: breaking change to authentication", "auto_commit": true}'
```

### Custom Repository Path

```bash
# Work with a different repository
mcp execute_workflow git-semantic-commit \
  --inputs '{"repository_path": "/path/to/other/repo", "auto_commit": true}'
```

## LLM-Powered Commit Messages

For complex changes requiring deeper semantic understanding, enable LLM-powered generation:

### Enable LLM Analysis

```bash
# Use Claude to analyze changes and generate commit message
git add src/complex-refactor.py

mcp execute_workflow git-semantic-commit \
  --inputs '{"use_llm": true, "auto_commit": true}'
```

**What happens:**
1. Workflow collects diff, file status, and statistics
2. Pauses execution and presents context to Claude (via Prompt interactive block)
3. Claude analyzes the semantic meaning of changes
4. Generates conventional commit message with proper type, scope, and description
5. Workflow resumes and commits with the LLM-generated message

### LLM vs Bash Heuristics Comparison

| Aspect | Bash Heuristics | LLM-Powered |
|--------|-----------------|-------------|
| **Speed** | ~100ms | ~2-3 seconds |
| **Cost** | Free | Uses current MCP session (no additional API cost) |
| **Accuracy** | Good for standard patterns | Excellent for complex/ambiguous changes |
| **Best For** | - Simple changes<br>- Clear patterns<br>- Standard file types | - Complex refactoring<br>- Multi-purpose changes<br>- Subtle semantic changes |
| **Dependencies** | None | Requires MCP interactive block support |

### When to Use LLM Generation

**Use LLM (`use_llm: true`) when:**
- Changes span multiple concerns (refactor + fix + optimization)
- Diff patterns are ambiguous or misleading
- You want natural language understanding of code intent
- Breaking changes need careful description
- Complex architectural changes

**Use Bash (default) when:**
- Changes are straightforward and clear
- Speed is critical (CI/CD pipelines)
- Offline/air-gapped environments
- Simple file additions or documentation updates

### Example: Complex Refactoring

```bash
# Complex change: refactored authentication + added caching + fixed bug
git add src/auth/ src/cache/ src/models/

# LLM understands the primary intent despite mixed signals
mcp execute_workflow git-semantic-commit \
  --inputs '{"use_llm": true, "auto_commit": true}'

# LLM might generate:
# "refactor(auth): improve authentication flow with caching layer"
#
# Bash might generate:
# "feat(auth): add new functionality across 3 files"
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stage_all` | boolean | `false` | Stage all modified files before analyzing |
| `repository_path` | string | `"."` | Path to the git repository |
| `auto_commit` | boolean | `true` | Skip review and commit immediately |
| `use_llm` | boolean | `false` | Use LLM (Claude) for intelligent semantic analysis instead of bash heuristics |
| `override_message` | string | `""` | Manually provide commit message (skips generation) |

## Output Values

| Output | Type | Description |
|--------|------|-------------|
| `has_changes` | boolean | Whether there are staged changes |
| `generated_message` | string | The generated semantic commit message |
| `commit_created` | boolean | Whether a commit was created |
| `commit_hash` | string | The SHA hash of the created commit |
| `file_status` | string | Git status of staged files |
| `diff_summary` | string | Summary of changes |

## Examples

### Example 1: Feature Addition

**Changes**: Added new authentication module
```bash
git add src/auth/ tests/test_auth.py

mcp execute_workflow git-semantic-commit --inputs '{"auto_commit": true}'
```

**Generated**: `feat(auth): add user authentication implementation`

### Example 2: Bug Fix

**Changes**: Fixed validation error in user model
```bash
git add src/models/user.py

mcp execute_workflow git-semantic-commit --inputs '{"auto_commit": true}'
```

**Generated**: `fix(models): resolve issues in user`

### Example 3: Documentation Update

**Changes**: Updated README and added API docs
```bash
git add README.md docs/api.md

mcp execute_workflow git-semantic-commit --inputs '{"auto_commit": true}'
```

**Generated**: `docs: update documentation`

### Example 4: Performance Optimization

**Changes**: Optimized database queries
```bash
git add src/database/queries.py

# Diff contains "optimize" keyword
mcp execute_workflow git-semantic-commit --inputs '{"auto_commit": true}'
```

**Generated**: `perf(database): optimize queries performance`

### Example 5: Breaking Change

**Changes**: Changed API response format (breaking change)
```bash
git add src/api/responses.py

# Diff contains "BREAKING CHANGE" comment
mcp execute_workflow git-semantic-commit --inputs '{"auto_commit": true}'
```

**Generated**: `feat(api)!: add new functionality across 1 files`

## Scope Detection

The workflow automatically extracts scope from:

1. **Most common directory**: `src/auth/` → `auth`
2. **Component identification**: `src/components/Button.tsx` → `button`
3. **Module names**: `lib/database/` → `database`

Special cases:
- `src/` or `lib/` → looks one level deeper
- `tests/` → scope is `tests`
- `docs/` → scope is `docs`

## Breaking Changes

Breaking changes are detected when:
- Diff contains `BREAKING CHANGE:` or `BREAKING:`
- The `!` modifier is added: `feat(api)!: change response format`

## Integration with Development Workflow

### Workflow Composition

Combine with other workflows:

```yaml
- id: run_tests
  type: Workflow
  inputs:
    workflow: python-ci-pipeline

- id: commit_changes
  type: Workflow
  inputs:
    workflow: git-semantic-commit
    inputs:
      stage_all: true
      auto_commit: true
  depends_on: [run_tests]
  condition: "{{blocks.run_tests.succeeded}}"
```

### Git Hooks Integration

Use in pre-commit hooks:

```bash
#!/bin/bash
# .git/hooks/prepare-commit-msg

# Generate semantic message if none provided
if [ -z "$2" ]; then
  MSG=$(mcp execute_workflow git-semantic-commit --inputs '{"auto_commit": false}' | grep "Message:" | cut -d' ' -f2-)
  echo "$MSG" > "$1"
fi
```

## Technical Implementation

### Interactive Block Architecture

The LLM-powered generation uses MCP's `Prompt` interactive block, which:

1. **Pauses Workflow**: Suspends execution at the generation step
2. **Creates Context**: Packages diff, file status, and statistics into a structured prompt
3. **Invokes LLM**: Presents prompt to Claude (the MCP executor) with Conventional Commits guidelines
4. **Resumes Workflow**: Continues with LLM-generated commit message
5. **Validates Output**: Ensures proper format before committing

### Prompt Engineering

The LLM prompt includes:
- Complete Conventional Commits specification
- File change statistics and diff summary
- Type definitions with examples
- Scope extraction guidelines
- Breaking change patterns
- Output format constraints

This structured approach ensures consistent, high-quality commit messages.

## Best Practices

1. **Review Generated Messages**: Always review the proposed message before committing (unless using `auto_commit`)
2. **Stage Intentionally**: Stage only related changes together for accurate type detection
3. **Add Context**: For complex changes, use `override_message` to provide more detailed descriptions
4. **Breaking Changes**: Manually add `!` or `BREAKING CHANGE:` footer for breaking changes
5. **Atomic Commits**: Make small, focused commits for better semantic analysis

## Troubleshooting

### No Changes Detected

```text
Error: No staged changes to commit
```

**Solution**: Stage files first with `git add <files>` or use `stage_all: true`

### Incorrect Type Detection

**Solution**: Use `override_message` to manually specify the correct type:
```bash
mcp execute_workflow git-semantic-commit \
  --inputs '{"override_message": "refactor(core): restructure data layer", "auto_commit": true}'
```

### Breaking Changes Not Detected

**Solution**: Add `BREAKING CHANGE:` to commit message body or use `!` modifier manually:
```bash
mcp execute_workflow git-semantic-commit \
  --inputs '{"override_message": "feat(api)!: change authentication flow", "auto_commit": true}'
```

## Related Workflows

- `commit-and-push`: Commit and push to remote
- `git-checkout-branch`: Create feature branch with worktree
- `python-ci-pipeline`: Run tests before committing

## References

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Angular Commit Message Guidelines](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
- [Semantic Versioning](https://semver.org/)
