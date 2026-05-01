CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS server_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS admin_credentials (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    password_hash TEXT NOT NULL,
    password_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    palace TEXT NOT NULL UNIQUE,
    default_wing TEXT NOT NULL,
    default_room TEXT NOT NULL,
    fs_root TEXT NOT NULL,
    fs_allowlist_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS admin_sessions (
    session_hash TEXT PRIMARY KEY,
    csrf_token_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    idle_expires_at TEXT NOT NULL,
    absolute_expires_at TEXT NOT NULL,
    revoked_at TEXT
);

-- Reserved for future phase: CSRF/session secret references.
CREATE TABLE IF NOT EXISTS session_secrets (
    id TEXT PRIMARY KEY,
    secret_ref TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT
);

-- Reserved for future phase: encrypted secret metadata.
CREATE TABLE IF NOT EXISTS encrypted_secret_metadata (
    id TEXT PRIMARY KEY,
    secret_name TEXT NOT NULL UNIQUE,
    key_id TEXT,
    encrypted_payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Reserved for future phase: LLM providers and profiles.
CREATE TABLE IF NOT EXISTS llm_providers (
    provider_name TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_profiles (
    profile_name TEXT PRIMARY KEY,
    provider_name TEXT NOT NULL,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (provider_name) REFERENCES llm_providers(provider_name) ON DELETE CASCADE
);

-- Reserved for future phase: PostgreSQL settings.
CREATE TABLE IF NOT EXISTS postgresql_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    dsn_ref TEXT,
    enabled INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Reserved for future phase: MCP tokens.
CREATE TABLE IF NOT EXISTS mcp_tokens (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    token_hash TEXT NOT NULL UNIQUE,
    capabilities_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TEXT,
    revoked_at TEXT
);

-- Reserved for future phase: project-token bindings.
CREATE TABLE IF NOT EXISTS project_token_bindings (
    token_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (token_id, project_id),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE CASCADE
);

-- Reserved for future phase: watcher dirty queue/status.
CREATE TABLE IF NOT EXISTS watcher_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    path TEXT NOT NULL,
    event_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    enqueued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS watcher_status (
    project_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    last_event_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Reserved for future phase: workflow source/reload metadata.
CREATE TABLE IF NOT EXISTS workflow_sources (
    source_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    checksum TEXT,
    discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_sources_project_path_unique
ON workflow_sources(project_id, source_path);

CREATE TABLE IF NOT EXISTS workflow_reload_state (
    source_id TEXT PRIMARY KEY,
    last_loaded_at TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    FOREIGN KEY (source_id) REFERENCES workflow_sources(source_id) ON DELETE CASCADE
);

-- Reserved for future phase: run/job metadata.
CREATE TABLE IF NOT EXISTS job_runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT,
    token_id TEXT,
    workflow_name TEXT NOT NULL,
    status TEXT NOT NULL,
    cancellable INTEGER NOT NULL DEFAULT 0,
    result_summary TEXT,
    error_summary TEXT,
    execution_state_json TEXT,
    timeout_seconds INTEGER NOT NULL DEFAULT 3600,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    inputs_json TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL,
    FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_job_runs_started_at
ON job_runs(started_at ASC, run_id ASC);

CREATE INDEX IF NOT EXISTS idx_job_runs_status
ON job_runs(status ASC, started_at ASC, run_id ASC);

INSERT OR IGNORE INTO schema_migrations (version) VALUES (1);
