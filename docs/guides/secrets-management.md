# Secrets Management User Guide

Complete guide to using secrets securely in workflows-mcp workflows.

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Security Best Practices](#security-best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Topics](#advanced-topics)
8. [FAQ](#faq)

---

## Introduction

Secrets management in workflows-mcp provides enterprise-grade security for handling sensitive credentials like API keys, passwords, and tokens. The system ensures that secrets never reach the LLM context while maintaining a clean and intuitive developer experience.

### Key Features

**Server-Side Resolution**: Secrets are resolved during workflow execution on the server, never transmitted to the LLM agent or visible in conversation history.

**Automatic Redaction**: All workflow outputs are automatically scanned for secret values, which are replaced with `***REDACTED***` before being returned to the LLM.

**Fail-Fast Behavior**: Missing or inaccessible secrets cause immediate workflow failure with clear error messages, preventing silent failures.

**Comprehensive Audit Trail**: All secret access is logged to stderr with structured event data for compliance and security monitoring.

**Universal Block Support**: Works seamlessly with all block types including Shell, HttpCall, CreateFile, RenderTemplate, and custom blocks.

### Security Model

The secrets management system follows defense-in-depth principles:

1. **Namespace Isolation**: Secrets live in dedicated `{{secrets.*}}` namespace, separate from other variables
2. **Provider Abstraction**: Pluggable backend support (environment variables, Vault, AWS Secrets Manager, etc.)
3. **Automatic Sanitization**: Multi-layer redaction prevents accidental secret exposure
4. **Audit Logging**: Complete tracking of all secret access for compliance and forensics
5. **Fail-Safe Design**: System fails closed when secrets are unavailable or compromised

### Comparison with Industry Standards

Workflows-mcp secrets management aligns with industry-leading platforms:

| Feature | workflows-mcp | GitHub Actions | Argo Workflows | Tekton |
|---------|--------------|----------------|----------------|--------|
| Server-side resolution | ✅ | ✅ | ✅ | ✅ |
| Automatic redaction | ✅ | ✅ | ✅ | ✅ |
| Audit logging | ✅ | ✅ | ✅ | ✅ |
| Provider abstraction | ✅ | ⚠️ | ✅ | ✅ |
| Environment variables | ✅ | ✅ | ✅ | ✅ |
| External secret managers | 🔄 Phase 2 | ✅ | ✅ | ✅ |

---

## Quick Start

### 1. Configure a Secret

Add secrets to your Claude Desktop configuration:

**File**: `~/.config/claude-desktop/claude_desktop_config.json` (Linux/macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": ["--from", "workflows-mcp", "workflows-mcp"],
      "env": {
        "WORKFLOW_SECRET_GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

### 2. Restart Claude Desktop

Secrets are loaded when the MCP server starts, so you must restart Claude Desktop after configuration changes.

### 3. Use the Secret in a Workflow

```yaml
name: test-github-api
description: Test GitHub API with secret token

blocks:
  - id: get_user_info
    type: HttpCall
    inputs:
      url: "https://api.github.com/user"
      headers:
        Authorization: "Bearer {{secrets.GITHUB_TOKEN}}"

outputs:
  username: "{{blocks.get_user_info.outputs.response_json.login}}"
```

### 4. Verify Security

- Secret value never appears in LLM conversation
- Workflow outputs show `***REDACTED***` for secret values
- Audit log (stderr) shows secret access events

---

## Configuration

### Environment Variable Configuration

Secrets are configured using environment variables with the `WORKFLOW_SECRET_` prefix.

#### Naming Convention

| Environment Variable | Workflow Reference | Example Value |
|---------------------|-------------------|---------------|
| `WORKFLOW_SECRET_GITHUB_TOKEN` | `{{secrets.GITHUB_TOKEN}}` | `ghp_1234567890abcdef` |
| `WORKFLOW_SECRET_OPENAI_API_KEY` | `{{secrets.OPENAI_API_KEY}}` | `sk-1234567890abcdef` |
| `WORKFLOW_SECRET_DATABASE_PASSWORD` | `{{secrets.DATABASE_PASSWORD}` | `supersecret123` |
| `WORKFLOW_SECRET_SLACK_WEBHOOK_URL` | `{{secrets.SLACK_WEBHOOK_URL}}` | `https://hooks.slack.com/...` |

**Rules**:
- Prefix: `WORKFLOW_SECRET_` (required)
- Key name: UPPERCASE_WITH_UNDERSCORES
- Workflow reference: Lowercase key name after prefix
- Case-insensitive matching (internally converted to uppercase)

#### Claude Desktop Configuration

**Location**:
- macOS/Linux: `~/.config/claude-desktop/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Example Configuration**:

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": ["--from", "workflows-mcp", "workflows-mcp"],
      "env": {
        "WORKFLOW_SECRET_GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "WORKFLOW_SECRET_OPENAI_API_KEY": "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "WORKFLOW_SECRET_DATABASE_PASSWORD": "db_password_here",
        "WORKFLOW_SECRET_SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
        "WORKFLOW_SECRET_NPM_TOKEN": "npm_xxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

**Security Note**: The Claude Desktop config file should have restricted permissions (readable only by your user account).

```bash
# Set secure permissions on macOS/Linux
chmod 600 ~/.config/claude-desktop/claude_desktop_config.json
```

#### Using .env Files (Recommended)

For development environments, you can load secrets from a `.env` file:

**File**: `.env` (in your workflow project directory)

```bash
# .env file (DO NOT commit to version control!)
WORKFLOW_SECRET_GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WORKFLOW_SECRET_OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WORKFLOW_SECRET_DATABASE_PASSWORD=supersecret123
```

**Claude Desktop Configuration**:

```json
{
  "mcpServers": {
    "workflows": {
      "command": "uvx",
      "args": [
        "--env-file",
        "/path/to/your/project/.env",
        "workflows-mcp"
      ]
    }
  }
}
```

**Important**: Add `.env` to your `.gitignore` to prevent committing secrets to version control.

### Validating Configuration

To verify your secrets are configured correctly:

```yaml
name: test-secrets-config
description: Validate secret configuration

blocks:
  - id: list_secrets
    type: Shell
    inputs:
      command: |
        env | grep WORKFLOW_SECRET_
outputs:
  summary:
    type: str
    value: "{{ blocks.list_secrets.outputs.stdout }}"
```

This will list all configured secret keys (values remain hidden).

---

## Usage Examples

### Example 1: GitHub API Authentication

Access GitHub's REST API using a personal access token
(the token requires user:read scope at least):

```yaml
name: github-user-info
description: Fetch authenticated user information from GitHub

blocks:
  - id: get_user
    type: HttpCall
    inputs:
      url: "https://api.github.com/user"
      method: "GET"
      headers:
        Authorization: "Bearer {{secrets.GITHUB_TOKEN}}"
        Accept: "application/vnd.github+json"

outputs:
  username:
    type: str
    value: "{{blocks.get_user.outputs.response_json.login}}"
  name:
    type: str
    value: "{{blocks.get_user.outputs.response_json.name}}"
  email:
    type: str
    value: "{{blocks.get_user.outputs.response_json.email}}"
```

**Configuration**:
```json
{
  "env": {
    "WORKFLOW_SECRET_GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  }
}
```

**Usage**:
```bash
# Execute workflow (token is resolved server-side)
execute_workflow(workflow="github-user-info")
```

### Example 2: OpenAI API Integration

Call OpenAI's API with your API key:

```yaml
name: openai-chat-completion
description: Generate text using OpenAI GPT-4

inputs:
  prompt:
    type: str
    required: true
    description: "User prompt for GPT-4"

blocks:
  - id: call_openai
    type: HttpCall
    inputs:
      url: "https://api.openai.com/v1/chat/completions"
      method: "POST"
      headers:
        Content-Type: "application/json"
        Authorization: "Bearer {{secrets.OPENAI_API_KEY}}"
      json:
        model: "gpt-4"
        messages:
          - role: "system"
            content: "You are a helpful assistant."
          - role: "user"
            content: "{{inputs.prompt}}"
        temperature: 0.7
        max_tokens: 500

outputs:
  response:
    type: str
    value: "{{blocks.call_openai.outputs.response_json.choices[0].message.content}}"
  tokens_used:
    type: num
    value: "{{blocks.call_openai.outputs.response_json.usage.total_tokens}}"
```

**Configuration**:
```json
{
  "env": {
    "WORKFLOW_SECRET_OPENAI_API_KEY": "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  }
}
```

### Example 3: Database Backup with Credentials

Backup a PostgreSQL database using secure credentials:

```yaml
name: postgres-backup
description: Backup PostgreSQL database securely

inputs:
  database_name:
    type: str
    required: true
  backup_path:
    type: str
    required: true

blocks:
  - id: create_backup
    type: Shell
    inputs:
      command: |
        # Set password from secret (not echoed to stdout)
        export PGPASSWORD="{{secrets.DATABASE_PASSWORD}}"

        # Create backup
        pg_dump \
          -h db.example.com \
          -U postgres \
          -d {{inputs.database_name}} \
          > {{inputs.backup_path}}

        # Verify backup (don't show password)
        echo "Backup created: $(ls -lh {{inputs.backup_path}})"
      working_dir: "/tmp"

  - id: compress_backup
    type: Shell
    inputs:
      command: |
        gzip {{inputs.backup_path}}
        echo "Compressed backup: $(ls -lh {{inputs.backup_path}}.gz)"
    depends_on: [create_backup]

outputs:
  backup_file:
    type: str
    value: "{{inputs.backup_path}}.gz"
  backup_succeeded:
    type: bool
    value: "{{blocks.compress_backup.succeeded}}"
```

**Configuration**:
```json
{
  "env": {
    "WORKFLOW_SECRET_DATABASE_PASSWORD": "your_secure_db_password"
  }
}
```

### Example 4: Multi-Service Deployment

Deploy application to multiple services with different credentials:

```yaml
name: multi-service-deployment
description: Deploy app to Docker, Kubernetes, and notify Slack

inputs:
  app_version:
    type: str
    required: true

blocks:
  # Build and push Docker image
  - id: docker_build_push
    type: Shell
    inputs:
      command: |
        docker build -t myapp:{{inputs.app_version}} .
        echo "{{secrets.DOCKER_PASSWORD}}" | docker login -u {{secrets.DOCKER_USERNAME}} --password-stdin
        docker push myapp:{{inputs.app_version}}

  # Deploy to Kubernetes
  - id: k8s_deploy
    type: Shell
    inputs:
      command: |
        kubectl set image deployment/myapp myapp=myapp:{{inputs.app_version}}
        kubectl rollout status deployment/myapp
      env:
        KUBECONFIG: "{{secrets.KUBECONFIG_PATH}}"
    depends_on: [docker_build_push]

  # Notify team on Slack
  - id: slack_notification
    type: HttpCall
    inputs:
      url: "{{secrets.SLACK_WEBHOOK_URL}}"
      method: "POST"
      headers:
        Content-Type: "application/json"
      json:
        text: "Deployed myapp:{{inputs.app_version}} to production"
        blocks:
          - type: "section"
            text:
              type: "mrkdwn"
              text: "*Deployment Status*: Success"
    condition: "{{blocks.k8s_deploy.succeeded}}"
    depends_on: [k8s_deploy]

outputs:
  deployment_succeeded:
    type: bool
    value: "{{blocks.k8s_deploy.succeeded}}"
  notification_sent:
    type: bool
    value: "{{blocks.slack_notification.succeeded}}"
```

**Configuration**:
```json
{
  "env": {
    "WORKFLOW_SECRET_DOCKER_USERNAME": "myuser",
    "WORKFLOW_SECRET_DOCKER_PASSWORD": "docker_password_here",
    "WORKFLOW_SECRET_KUBECONFIG_PATH": "/home/user/.kube/config",
    "WORKFLOW_SECRET_SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/XXX/YYY/ZZZ"
  }
}
```

### Example 5: Configuration File Generation

Create configuration files with embedded secrets:

```yaml
name: generate-app-config
description: Generate application configuration with secrets

inputs:
  environment:
    type: str
    required: true
  app_name:
    type: str
    required: true

blocks:
  - id: create_config
    type: CreateFile
    inputs:
      path: "/tmp/{{inputs.app_name}}-config.json"
      content: |
        {
          "environment": "{{inputs.environment}}",
          "application": {
            "name": "{{inputs.app_name}}",
            "version": "1.0.0"
          },
          "database": {
            "host": "db.example.com",
            "port": 5432,
            "password": "{{secrets.DATABASE_PASSWORD}}"
          },
          "apis": {
            "openai": {
              "key": "{{secrets.OPENAI_API_KEY}}",
              "endpoint": "https://api.openai.com/v1"
            },
            "github": {
              "token": "{{secrets.GITHUB_TOKEN}}",
              "api_url": "https://api.github.com"
            }
          },
          "notifications": {
            "slack_webhook": "{{secrets.SLACK_WEBHOOK_URL}}"
          }
        }
      mode: "0600"

outputs:
  config_path:
    type: str
    value: "/tmp/{{inputs.app_name}}-config.json"
  config_created:
    type: bool
    value: "{{blocks.create_config.succeeded}}"
```

---

## How Secrets Stay Secure

**Server-side resolution**: Secrets are resolved during workflow execution on the MCP server. The LLM never sees secret values - only `{{secrets.NAME}}` placeholders.

**Automatic redaction**: If a secret value appears in workflow outputs (stdout, HTTP responses, file content), it's replaced with `***REDACTED***` before returning to the LLM.

**Missing secrets resolve to empty string**: Referencing an undefined secret logs a warning and resolves to an empty string. Check MCP server logs if secrets aren't working as expected.

### Using Secrets in Shell Blocks

Use the `env` field to pass secrets to shell commands:

```yaml
- id: deploy
  type: Shell
  inputs:
    command: ./deploy.sh
    env:
      API_KEY: "{{secrets.API_KEY}}"
```

This keeps secrets out of the command string (which appears in process lists) and lets the script access them via environment variables.

---

## Troubleshooting

### Error: "Secret 'API_KEY' not found"

**Cause**: Secret not configured in environment variables.

**Solution**:

1. Add secret to Claude Desktop configuration:
   ```json
   {
     "env": {
       "WORKFLOW_SECRET_API_KEY": "your_api_key_here"
     }
   }
   ```

2. Verify environment variable name matches workflow reference:
   - Environment: `WORKFLOW_SECRET_API_KEY`
   - Workflow: `{{secrets.API_KEY}}`

3. Restart Claude Desktop to load new configuration.

4. Test workflow again.

### Secret Value Appears as "***REDACTED***"

**Cause**: Automatic output redaction detected and replaced secret value.

**Impact**: This is **working correctly** - the secret was prevented from leaking to LLM context.

**Review**: Check if workflow accidentally echoed secret to stdout:

```yaml
# ❌ This causes redaction in output
- id: problematic
  type: Shell
  inputs:
    command: |
      echo "API Key: {{secrets.API_KEY}}"

# ✅ This does not trigger redaction
- id: correct
  type: Shell
  inputs:
    command: |
      export API_KEY="{{secrets.API_KEY}}"
      ./use-api-key.sh
```

### Audit Log Shows Unexpected Secret Access

**Investigation Steps**:

1. Review audit log for workflow and block information:
   ```text
   2025-11-02T10:30:45Z [INFO] Secret access: workflow=unknown-workflow block=suspicious-block key=SENSITIVE_KEY
   ```

2. Identify the workflow:
   - Check workflow name in audit log
   - Search codebase for workflow definition

3. Review workflow YAML for secret usage:
   - Verify secret is used correctly
   - Check for unintended secret references

4. Implement access controls (future feature):
   - Role-based access control (RBAC)
   - Secret scope restrictions

### MCP Server Not Loading Secrets

**Symptoms**:
- Workflows fail with "Secret provider not configured"
- Audit logs show no secret access attempts

**Diagnosis**:

1. Verify MCP server configuration:
   ```bash
   # Check Claude Desktop logs
   # macOS: ~/Library/Logs/Claude/
   # Linux: ~/.config/Claude/logs/
   # Windows: %APPDATA%\Claude\logs\
   ```

2. Test environment variables in shell:
   ```bash
   env | grep WORKFLOW_SECRET_
   ```

3. Restart Claude Desktop completely (quit and relaunch).

4. Verify MCP server is running:
   - Claude Desktop shows "workflows" in server list
   - Test with simple workflow

### Performance Issues with Many Secrets

**Cause**: Redaction scans all outputs for every configured secret value.

**Solution**:

1. **Minimize configured secrets**: Only configure secrets actually used by workflows.

2. **Use specific secret names**: Avoid very short or common secret values that cause many false positives.

3. **Optimize workflow outputs**: Return only necessary data in block outputs.

4. **Monitor redaction performance** (future feature): Metrics for redaction overhead.

---

## Advanced Topics

### Custom Secret Providers (Future)

**Planned Support**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault

The provider classes exist but are not yet implemented (methods raise `NotImplementedError`).

**Example Configuration** (future):

```python
from workflows_mcp.engine.secrets import VaultSecretProvider

# Not yet implemented - will raise NotImplementedError
provider = VaultSecretProvider(
    url="https://vault.example.com:8200",
    token="s.xxxxxxxxxxxxxxxxxxxxxxxx",
    mount_point="secret"
)

# Future: Fetch secrets from Vault
secret_value = await provider.get_secret("database/password")
```

**Benefits** (when implemented):
- Centralized secret management
- Automatic secret rotation
- Access control policies
- Audit logging at provider level
- Secret versioning and rollback

### Programmatic Secret Access

Access secrets programmatically for testing or scripting:

```python
import asyncio
from workflows_mcp.engine.secrets import EnvVarSecretProvider

async def main():
    provider = EnvVarSecretProvider()

    # Get single secret
    api_key = await provider.get_secret("api_key")
    print(f"API Key retrieved: {api_key[:10]}...")

    # List all available secrets
    secret_keys = await provider.list_secret_keys()
    print(f"Available secrets: {secret_keys}")

asyncio.run(main())
```

### Testing with Secrets

**Test Secret Provider**:

```python
from workflows_mcp.engine.secrets import SecretProvider

class TestSecretProvider(SecretProvider):
    """Mock secret provider for testing."""

    def __init__(self, secrets: dict[str, str]):
        self._secrets = secrets

    async def get_secret(self, key: str) -> str:
        if key not in self._secrets:
            raise SecretNotFoundError(key=key)
        return self._secrets[key]

    async def list_secret_keys(self) -> list[str]:
        return list(self._secrets.keys())

# Use in tests
test_provider = TestSecretProvider({
    "test_api_key": "test_value_12345",
    "test_password": "test_password"
})
```

**Test Workflow with Secrets**:

```python
import pytest
from workflows_mcp.engine.workflow_runner import WorkflowRunner
from workflows_mcp.engine.loader import load_workflow_from_yaml

@pytest.mark.asyncio
async def test_workflow_with_secrets(monkeypatch):
    # Configure test secret
    monkeypatch.setenv("WORKFLOW_SECRET_TEST_KEY", "test_value")

    # Load workflow
    workflow_yaml = """
    name: test-workflow
    blocks:
      - id: use_secret
        type: Shell
        inputs:
          command: echo "Using secret"
          env:
            API_KEY: "{{secrets.TEST_KEY}}"
    """

    result = load_workflow_from_yaml(workflow_yaml)
    assert result.is_success

    # Execute workflow
    runner = WorkflowRunner()
    execution = await runner.execute(result.value, {})

    # Verify secret was used (but not exposed)
    assert execution.status == "success"
    # Secret value should be redacted in outputs
```

### Secret Rotation Workflows

Automate secret rotation using workflows:

```yaml
name: rotate-api-key
description: Rotate API key and update configuration

inputs:
  service_name:
    type: str
    required: true

blocks:
  # Generate new API key
  - id: generate_new_key
    type: HttpCall
    inputs:
      url: "https://api.example.com/keys"
      method: "POST"
      headers:
        Authorization: "Bearer {{secrets.ADMIN_TOKEN}}"
      json:
        name: "{{inputs.service_name}}-key-rotated"

  # Update secret in Vault (future)
  - id: update_vault
    type: Shell
    inputs:
      command: |
        # Future: Update secret in external secret manager
        vault kv put secret/{{inputs.service_name}}/api_key \
          value="${NEW_KEY}"
      env:
        NEW_KEY: "{{blocks.generate_new_key.outputs.response_json.key}}"
    depends_on: [generate_new_key]

  # Revoke old API key
  - id: revoke_old_key
    type: HttpCall
    inputs:
      url: "https://api.example.com/keys/{{secrets.OLD_API_KEY}}/revoke"
      method: "DELETE"
      headers:
        Authorization: "Bearer {{secrets.ADMIN_TOKEN}}"
    depends_on: [update_vault]

outputs:
  new_key_id: "{{blocks.generate_new_key.outputs.response_json.id}}"
  rotation_succeeded: "{{blocks.revoke_old_key.succeeded}}"
```

---

## FAQ

### Q: Are secrets encrypted in environment variables?

**A**: No, environment variables are stored in plain text by the operating system. For production deployments:

- **Phase 1** (current): Use OS-level security (file permissions, user isolation)
- **Phase 2** (future): Integrate external secret managers (Vault, AWS Secrets Manager) with encryption at rest

**Best Practice**: Use restricted file permissions on Claude Desktop config:
```bash
chmod 600 ~/.config/claude-desktop/claude_desktop_config.json
```

### Q: Can I use secrets in workflow outputs?

**A**: Secret values in workflow outputs are automatically redacted. The output structure is preserved, but secret values are replaced with `***REDACTED***`:

```yaml
outputs:
  api_key: "{{secrets.API_KEY}}"  # Output: "***REDACTED***"
  config:
    password: "{{secrets.PASSWORD}}"  # Output: { "password": "***REDACTED***" }
```

### Q: How do I share secrets between workflows?

**A**: Secrets are configured at the MCP server level, so all workflows have access to the same secrets. Use naming conventions to organize:

```json
{
  "env": {
    "WORKFLOW_SECRET_PROD_API_KEY": "production_key",
    "WORKFLOW_SECRET_DEV_API_KEY": "development_key",
    "WORKFLOW_SECRET_PROD_DATABASE_PASSWORD": "prod_password",
    "WORKFLOW_SECRET_DEV_DATABASE_PASSWORD": "dev_password"
  }
}
```

Then reference environment-specific secrets in workflows:

```yaml
- id: deploy
  type: Shell
  inputs:
    env:
      API_KEY: "{{secrets.PROD_API_KEY}}"  # or DEV_API_KEY for development
```

### Q: What happens if I forget to configure a required secret?

**A**: The workflow fails immediately when the block tries to resolve the secret, with a clear error message:

```text
Error: Secret 'DATABASE_PASSWORD' not found.
Set WORKFLOW_SECRET_DATABASE_PASSWORD environment variable in Claude Desktop configuration.
```

The workflow stops execution at the failing block, and the error is logged to the audit trail.

### Q: Can I use the same secret multiple times in a workflow?

**A**: Yes, secrets can be referenced multiple times across different blocks:

```yaml
blocks:
  - id: step1
    type: HttpCall
    inputs:
      headers:
        Authorization: "Bearer {{secrets.API_KEY}}"

  - id: step2
    type: Shell
    inputs:
      env:
        API_KEY: "{{secrets.API_KEY}}"

  - id: step3
    type: CreateFile
    inputs:
      content: "api_key: {{secrets.API_KEY}}"
```

Each access is logged separately in the audit trail for compliance tracking.

### Q: How do I debug secret-related issues?

**A**: Follow this debugging workflow:

1. **Check secret configuration**:
   ```bash
   # List configured secrets (values hidden)
   env | grep WORKFLOW_SECRET_ | cut -d= -f1
   ```

2. **Review audit logs** (stderr output) for secret access attempts

3. **Test secret resolution** with simple workflow:
   ```yaml
   name: test-secret
   blocks:
     - id: test
       type: Shell
       inputs:
         command: echo "Secret configured"
         env:
           TEST: "{{secrets.YOUR_SECRET}}"
   ```

4. **Verify naming convention**:
   - Environment: `WORKFLOW_SECRET_YOUR_SECRET`
   - Workflow: `{{secrets.YOUR_SECRET}}`
   - Case doesn't matter (internally uppercase)

### Q: Can secrets contain special characters?

**A**: Yes, secrets can contain any characters including special characters, whitespace, and newlines. Ensure proper quoting in YAML:

```yaml
# Literal block scalar for multi-line secrets
- id: use_certificate
  type: CreateFile
  inputs:
    content: |
      {{secrets.TLS_CERTIFICATE}}

# Double quotes for secrets with special characters
- id: use_password
  type: Shell
  inputs:
    env:
      PASSWORD: "{{secrets.DATABASE_PASSWORD}}"  # Handles special chars
```

### Q: How do I migrate from hardcoded credentials to secrets?

**A**: Follow this migration process:

1. **Identify hardcoded credentials** in workflows:
   ```bash
   # Search for potential hardcoded secrets
   grep -r "password\|api_key\|token" workflows/
   ```

2. **Configure secrets** in Claude Desktop config:
   ```json
   {
     "env": {
       "WORKFLOW_SECRET_API_KEY": "actual_api_key_value"
     }
   }
   ```

3. **Update workflows** to reference secrets:
   ```yaml
   # Before (hardcoded)
   headers:
     Authorization: "Bearer sk-1234567890abcdef"

   # After (secure)
   headers:
     Authorization: "Bearer {{secrets.OPENAI_API_KEY}}"
   ```

4. **Test workflows** to verify secret resolution

5. **Remove hardcoded values** from version control history (if committed)

### Q: What is the performance impact of secret redaction?

**A**: Redaction adds minimal overhead:

- **Secret loading**: One-time initialization when redactor starts
- **Output scanning**: O(n × m) where n = output size, m = number of secrets
- **Typical impact**: < 10ms for workflows with reasonable output sizes

**Optimization tips**:
- Configure only necessary secrets
- Return minimal data in block outputs
- Use specific secret names (avoid very short values)

---

## Conclusion

Secrets management in workflows-mcp provides enterprise-grade security while maintaining simplicity and developer experience. By following the best practices in this guide, you can safely integrate external services without exposing credentials to LLM context.

**Key Takeaways**:
- ✅ Secrets are resolved server-side, never transmitted to LLM
- ✅ Automatic redaction prevents accidental exposure
- ✅ Comprehensive audit logging supports compliance requirements
- ✅ Simple configuration via environment variables
- ✅ Universal support across all block types

**Next Steps**:
- Configure your first secret in Claude Desktop
- Test with example workflows (GitHub API, OpenAI API)
- Review audit logs for secret access patterns
- Explore advanced topics (custom providers, testing, rotation)

**Resources**:
- [ADR-008: Secrets Management Architecture](../adr/ADR-008-secrets-management.md)
- [CLAUDE.md Development Guide](../../CLAUDE.md#secrets-management-adr-008)
- [Example Workflows](../../src/workflows_mcp/templates/examples/)
- [GitHub Issues](https://github.com/qtsone/workflows-mcp/issues)

For questions or issues, please file a GitHub issue or consult the documentation.
