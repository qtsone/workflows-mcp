# Secrets Management Test Workflows

This directory contains comprehensive test workflows for the secrets management system.

## Test Coverage

### Basic Functionality Tests

1. **shell-basic.yaml**
   - Tests basic secret resolution in Shell blocks
   - Verifies secrets work as environment variables
   - Tests: `{{secrets.TEST_SECRET}}`

2. **http-auth.yaml**
   - Tests secret resolution in HttpCall headers
   - Verifies Authorization header with Bearer token
   - Tests: `{{secrets.API_TOKEN}}`
   - Uses: httpbin.org/bearer for validation

3. **file-content.yaml**
   - Tests secret resolution in CreateFile blocks
   - Verifies secrets in configuration file content
   - Tests: `{{secrets.API_KEY}}`, `{{secrets.DB_CONNECTION_STRING}}`
   - Creates: Config file with secrets

4. **template-render.yaml**
   - Tests secret resolution in RenderTemplate blocks
   - Verifies secrets in Jinja2 template variables
   - Tests: `{{secrets.API_KEY}}`, `{{secrets.DB_PASSWORD}}`
   - Creates: .env file from template

### Error Handling Tests

5. **missing-error.yaml**
   - Tests error handling for missing secrets
   - Verifies fail-fast behavior with clear error messages
   - Tests: `{{secrets.NONEXISTENT_SECRET}}`
   - Expected: Workflow fails with SecretNotFoundError

### Security Tests

6. **redaction.yaml**
   - Tests comprehensive output redaction
   - Verifies secrets are redacted in stdout, stderr, file content
   - Tests multiple redaction scenarios:
     - Direct echo
     - Variable assignment
     - JSON output
     - Mixed with other text
   - Tests: `{{secrets.REDACTION_TEST}}`

### Integration Tests

7. **multiple-blocks.yaml**
   - Tests secrets across all block types in one workflow
   - Verifies: Shell, HttpCall, CreateFile, RenderTemplate
   - Tests: `{{secrets.MULTI_SECRET}}`
   - Comprehensive end-to-end test

8. **audit-tracking.yaml**
   - Tests secret access audit logging
   - Verifies secret access events are tracked
   - Tests: `{{secrets.AUDIT_SECRET_1}}`, `{{secrets.AUDIT_SECRET_2}}`
   - Validates: secret_access_count in metadata

## Required Environment Variables

To run these tests, set the following environment variables:

```bash
# Basic tests
export WORKFLOW_SECRET_TEST_SECRET="test-value-123"
export WORKFLOW_SECRET_API_TOKEN="valid-bearer-token"
export WORKFLOW_SECRET_API_KEY="sk-test-key-456"

# Database tests
export WORKFLOW_SECRET_DB_CONNECTION_STRING="postgresql://user:pass@localhost/db"
export WORKFLOW_SECRET_DB_PASSWORD="db-secure-password"

# Integration tests
export WORKFLOW_SECRET_MULTI_SECRET="multi-block-test-value"

# Audit tests
export WORKFLOW_SECRET_AUDIT_SECRET_1="audit-test-1"
export WORKFLOW_SECRET_AUDIT_SECRET_2="audit-test-2"

# Redaction tests
export WORKFLOW_SECRET_REDACTION_TEST="secret-to-be-redacted"
```

**Note:** The `missing-error.yaml` test intentionally references a non-existent secret to test error handling.

## Running Tests

### Generate Snapshots

```bash
# From project root
python tests/generate_snapshots.py
```

This will:
1. Execute all workflows tagged with `test` (including secrets tests)
2. Generate snapshot files in `tests/snapshots/`
3. Use detailed response format to capture full execution details

### Validate Snapshots

```bash
# Run the full test suite
uv run pytest tests/test_mcp_client.py -v
```

This compares actual execution against saved snapshots.

### Manual Testing

```bash
# Test individual workflow
uv run mcp dev src/workflows_mcp/server.py

# Then use MCP Inspector to execute:
execute_workflow(workflow="secrets-shell-basic", inputs={})
```

## Expected Behavior

### Successful Tests
- `shell-basic.yaml`: All blocks succeed, secret resolved
- `http-auth.yaml`: API call returns 200, auth succeeds
- `file-content.yaml`: Files created with redacted content
- `template-render.yaml`: Template rendered with redacted secrets
- `multiple-blocks.yaml`: All 4 block types succeed
- `audit-tracking.yaml`: All blocks succeed, metadata contains secret_access_count
- `redaction.yaml`: All outputs contain "***REDACTED***"

### Expected Failures
- `missing-error.yaml`: Workflow status = "failure", error contains "Secret not found: NONEXISTENT_SECRET"

## Security Validation

All test snapshots should verify:
1. **No Secret Leakage**: No actual secret values in outputs
2. **Redaction Markers**: Outputs contain "***REDACTED***" where secrets were used
3. **Audit Trail**: Metadata includes `secret_access_count`
4. **Fail-Fast**: Missing secrets cause immediate failure with clear error

## Notes

- Tests use `/tmp/test-secrets-*` directories for temporary files
- All tests include cleanup blocks to remove temporary files
- httpbin.org is used for HTTP tests (reliable, predictable responses)
- Secrets are resolved server-side and never reach LLM context
- Redaction is automatic and applied to all outputs
