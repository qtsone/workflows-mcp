#!/usr/bin/env bash
# Regenerate test snapshots with correct test secrets
# Secret values match tests/workflows/core/secrets/README.md

set -e

# Basic tests
export WORKFLOW_SECRET_TEST_SECRET="test-value-123"
export WORKFLOW_SECRET_API_KEY="sk-test-key-456"
export WORKFLOW_SECRET_API_TOKEN="valid-bearer-token"

# Database tests
export WORKFLOW_SECRET_DB_PASSWORD="db-secure-password"
export WORKFLOW_SECRET_DB_CONNECTION_STRING="postgresql://user:pass@localhost/db"

# Integration tests
export WORKFLOW_SECRET_MULTI_SECRET="multi-block-test-value"

# Audit tests
export WORKFLOW_SECRET_AUDIT_SECRET_1="audit-test-1"
export WORKFLOW_SECRET_AUDIT_SECRET_2="audit-test-2"

# Redaction tests
export WORKFLOW_SECRET_REDACTION_TEST="secret-to-be-redacted"

echo "🔐 Test secrets configured"
echo "📸 Regenerating snapshots..."
cd "$(dirname "$0")"
uv run python generate_snapshots.py

echo ""
echo "✅ Snapshots regenerated successfully"
echo "📁 Location: tests/snapshots/"
