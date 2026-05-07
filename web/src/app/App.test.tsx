import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;
  readonly listeners = new Map<string, Set<EventListener>>();
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  close = vi.fn();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: EventListener): void {
    const existing = this.listeners.get(type) ?? new Set<EventListener>();
    existing.add(listener);
    this.listeners.set(type, existing);
  }

  emitError(): void {
    this.onerror?.(new Event("error"));
  }
}

const NAV_LINKS = [
  "/setup",
  "/projects",
  "/database",
  "/llm",
  "/secrets",
  "/watchers",
  "/sync",
  "/mcp-clients",
  "/workflows",
  "/runs",
] as const;

const TITLES_BY_PATH: Record<string, string> = {
  "/login": "Login",
  "/setup": "Setup",
  "/projects": "Projects",
  "/database": "Database",
  "/llm": "LLM",
  "/secrets": "Secrets",
  "/watchers": "Watchers",
  "/sync": "Sync",
  "/mcp-clients": "MCP Clients",
  "/workflows": "Workflows",
  "/runs": "Runs",
};

const renderAtPath = (path: string): void => {
  window.history.pushState({}, "", path);
  render(<App />);
};

const openNewProjectModal = async (): Promise<void> => {
  fireEvent.click(await screen.findByRole("button", { name: /^new$/i }));
  await screen.findByRole("heading", { name: /new project/i });
};

const openNewMcpClientModal = async (): Promise<void> => {
  fireEvent.click(await screen.findByRole("button", { name: /^new$/i }));
  await screen.findByRole("dialog", { name: /new mcp client/i });
};

const openMcpClientDetailModal = async (label = "ci-agent"): Promise<void> => {
  const table = await screen.findByRole("table", { name: /registered mcp clients/i });
  const row = within(table).getByRole("row", { name: new RegExp(`open mcp client ${label} details`, "i") });
  fireEvent.click(row);
  await screen.findByRole("dialog", { name: new RegExp(`mcp client ${label}`, "i") });
};

const buildLargeLlmConfig = (count: number): Record<string, unknown> => {
  const providers: Record<string, Record<string, unknown>> = {};
  const profiles: Record<string, Record<string, unknown>> = {};
  for (let index = 1; index <= count; index += 1) {
    const suffix = String(index).padStart(2, "0");
    providers[`provider-${suffix}`] = {
      type: "openai",
      api_url: `https://api-${suffix}.example.test/v1`,
      api_key_secret: `API_KEY_${suffix}`,
      model: `model-${suffix}`,
      timeout: 30,
      max_retries: 2,
      retry_delay: 1,
      extra_headers: {},
    };
    profiles[`profile-${suffix}`] = {
      provider: `provider-${suffix}`,
      model: `model-${suffix}`,
      temperature: 0.2,
      max_tokens: 1000 + index,
      description: `Profile ${suffix}`,
    };
  }
  return {
    version: "1.0",
    providers,
    profiles,
    default_profile: "profile-01",
  };
};

afterEach(() => {
  cleanup();
  document.body.style.overflow = "";
  window.history.pushState({}, "", "/");
  vi.restoreAllMocks();
  FakeEventSource.instances = [];
});

beforeEach(() => {
  vi.restoreAllMocks();
});

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });

const deferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const installApiMock = (
  {
    projectCount = 1,
    projects: customProjects,
    createMcpClientResponse,
    regenerateMcpClientResponse,
    mcpClients: customMcpClients,
    watchers: customWatchers,
    syncProjects: customSyncProjects,
    syncLogs: customSyncLogs,
    syncDetails: customSyncDetails,
    syncActionResponses: customSyncActionResponses,
    workflows: customWorkflows,
    workflowSources: customWorkflowSources,
    workflowSchema: customWorkflowSchema,
    workflowReloadStatus = 200,
    workflowReloadErrorMessage = "reload failed due to duplicate names",
    runs: customRuns,
    runDetailById: customRunDetailById,
    cancelRunStatus = 200,
    cancelRunErrorMessage = "cancel failed",
    resumeRunStatus = 200,
    resumeRunErrorMessage = "resume failed",
    filesystemListingByPath,
    filesystemErrorByPath,
    initialDatabaseSettings,
    llmConfig: initialLlmConfig,
    secrets: customSecrets,
    secretsListPromise,
    secretsListStatus = 200,
    secretsListErrorMessage = "Unable to list secrets",
  }: {
    projectCount?: number;
    projects?: Array<Record<string, unknown>>;
    createMcpClientResponse?: Record<string, unknown>;
    regenerateMcpClientResponse?: Record<string, unknown>;
    mcpClients?: Array<Record<string, unknown>>;
    watchers?: Array<Record<string, unknown>>;
    syncProjects?: Array<Record<string, unknown>>;
    syncLogs?: Record<string, Array<Record<string, unknown>>>;
    syncDetails?: Record<string, Record<string, unknown>>;
    syncActionResponses?: Record<string, Record<string, unknown>>;
    workflows?: Array<Record<string, unknown>>;
    workflowSources?: Array<Record<string, unknown>>;
    workflowSchema?: Record<string, unknown>;
    workflowReloadStatus?: number;
    workflowReloadErrorMessage?: string;
    runs?: Array<Record<string, unknown>>;
    runDetailById?: Record<string, Record<string, unknown>>;
    cancelRunStatus?: number;
    cancelRunErrorMessage?: string;
    resumeRunStatus?: number;
    resumeRunErrorMessage?: string;
    filesystemListingByPath?: Record<string, Record<string, unknown>>;
    filesystemErrorByPath?: Record<string, { status: number; body: Record<string, unknown> }>;
    initialDatabaseSettings?: Record<string, unknown>;
    llmConfig?: Record<string, unknown>;
    secrets?: Array<Record<string, unknown>>;
    secretsListPromise?: Promise<Response>;
    secretsListStatus?: number;
    secretsListErrorMessage?: string;
  } = {},
) => {
  const projects: Array<Record<string, unknown>> =
    customProjects ??
    Array.from({ length: projectCount }, (_, index) => ({
      id: `p${index + 1}`,
      name: `P${index + 1}`,
      slug: `p${index + 1}`,
      palace: "x",
      default_wing: "w",
      default_room: "r",
      fs_root: "/tmp",
      fs_allowlist: [],
      created_at: "",
      updated_at: "",
    }));
  const mcpClients: Array<Record<string, unknown>> = customMcpClients ?? [];
  const watchers: Array<Record<string, unknown>> =
    customWatchers ??
    [{ project_id: "p1", state: "enabled", dirty_count: 2, requires_reconciliation: true, last_event_at: null, updated_at: "2026-04-30T00:00:00Z" }];
  const syncProjects: Array<Record<string, unknown>> =
    customSyncProjects ?? [{ project_id: "p1", dirty_count: 2, requires_reconciliation: true }];
  const syncLogsByProjectId: Record<string, Array<Record<string, unknown>>> = customSyncLogs
    ? { ...customSyncLogs }
    : {};
  const syncDetailsByProjectId: Record<string, Record<string, unknown>> = customSyncDetails
    ? { ...customSyncDetails }
    : {};
  const syncActionResponses = customSyncActionResponses ?? {};
  const workflows: Array<Record<string, unknown>> =
    customWorkflows ??
    [
      {
        name: "python-ci-pipeline",
        description: "Runs lint and test checks",
        version: "1.2.0",
        tags: ["python", "ci"],
        source_path: "/workspace/workflows/python-ci.yaml",
      },
    ];
  const workflowSources: Array<Record<string, unknown>> =
    customWorkflowSources ??
    [
      {
        source_id: "src-1",
        project_id: "p1",
        source_path: "/workspace/workflows",
        checksum: null,
        discovered_at: "2026-04-30T00:00:00Z",
        last_loaded_at: "2026-04-30T00:10:00Z",
        status: "loaded",
        error_message: null,
      },
    ];
  const workflowSchema: Record<string, unknown> =
    customWorkflowSchema ?? { title: "WorkflowSchema", type: "object", properties: { blocks: { type: "array" } } };
  const runs: Array<Record<string, unknown>> =
    customRuns ??
    [
      {
        run_id: "run-1",
        job_id: "job-1",
        workflow_name: "python-ci-pipeline",
        status: "failed",
        execution_mode: "sync",
        created_at: "2026-04-30T00:00:00Z",
        started_at: "2026-04-30T00:01:00Z",
        finished_at: "2026-04-30T00:01:02Z",
        updated_at: "2026-04-30T00:02:00Z",
        duration_ms: 2200,
        cancellable: false,
        project_id: "p1",
        token_id: "t1",
      },
      {
        run_id: "run-2",
        job_id: "job-2",
        workflow_name: "deploy-gate",
        status: "paused",
        execution_mode: "async",
        created_at: "2026-04-30T01:00:00Z",
        started_at: "2026-04-30T01:01:00Z",
        finished_at: null,
        updated_at: "2026-04-30T01:02:00Z",
        duration_ms: null,
        cancellable: true,
        project_id: "p1",
        token_id: "t1",
      },
      {
        run_id: "run-3",
        job_id: "job-3",
        workflow_name: "node-ci-pipeline",
        status: "completed",
        execution_mode: "async",
        created_at: "2026-04-30T02:00:00Z",
        started_at: "2026-04-30T02:01:00Z",
        finished_at: "2026-04-30T02:04:00Z",
        updated_at: "2026-04-30T02:04:00Z",
        duration_ms: 180000,
        cancellable: false,
        project_id: null,
        token_id: null,
      },
    ];
  const runDetailById: Record<string, Record<string, unknown>> =
    customRunDetailById ?? {
      "run-1": {
        run_id: "run-1",
        job_id: "job-1",
        workflow_name: "python-ci-pipeline",
        status: "failed",
        execution_mode: "sync",
        created_at: "2026-04-30T00:00:00Z",
        started_at: "2026-04-30T00:01:00Z",
        finished_at: "2026-04-30T00:01:02Z",
        updated_at: "2026-04-30T00:02:00Z",
        duration_ms: 2200,
        cancellable: false,
        project_id: "p1",
        token_id: "t1",
        result_summary: "Workflow failed: missing input required_param",
        error_summary: "required_param is required",
        inputs: { project_path: "/workspace/app" },
        outputs: null,
        error: "required_param is required",
        metadata: { workflow_name: "python-ci-pipeline", execution_time_seconds: 2.2 },
        blocks: [
          {
            block_id: "validate_inputs",
            block_type: "Input",
            status: "failed",
            outcome: "failure",
            duration_ms: 18,
            message: "required_param is required",
            inputs: { project_path: "/workspace/app" },
            outputs: {},
            metadata: { field: "required_param" },
          },
        ],
        technical_json: {
          status: "failure",
          inputs: { project_path: "/workspace/app" },
          error: "required_param is required",
          metadata: { workflow_name: "python-ci-pipeline", execution_time_seconds: 2.2 },
          blocks: [
            {
              block_id: "validate_inputs",
              block_type: "Input",
              status: "failed",
              outcome: "failure",
              duration_ms: 18,
              message: "required_param is required",
            },
          ],
        },
      },
      "run-2": {
        run_id: "run-2",
        job_id: "job-2",
        workflow_name: "deploy-gate",
        status: "paused",
        execution_mode: "async",
        created_at: "2026-04-30T01:00:00Z",
        started_at: "2026-04-30T01:01:00Z",
        finished_at: null,
        updated_at: "2026-04-30T01:02:00Z",
        duration_ms: null,
        cancellable: true,
        project_id: "p1",
        token_id: "t1",
        result_summary: "Awaiting user input",
        error_summary: null,
        inputs: { environment: "prod" },
        outputs: null,
        error: null,
        metadata: { workflow_name: "deploy-gate", prompt: "Approve deployment?" },
        blocks: [
          {
            block_id: "approval_gate",
            block_type: "Prompt",
            status: "paused",
            outcome: "n/a",
            duration_ms: null,
            message: "Approve deployment?",
            inputs: { environment: "prod" },
            outputs: {},
            metadata: { gate: "prod" },
          },
        ],
        technical_json: {
          status: "paused",
          inputs: { environment: "prod" },
          metadata: { workflow_name: "deploy-gate", prompt: "Approve deployment?" },
          blocks: [
            {
              block_id: "approval_gate",
              block_type: "Prompt",
              status: "paused",
              message: "Approve deployment?",
            },
          ],
        },
      },
    };
  const pathEntriesListingByPath: Record<string, Record<string, unknown>> = filesystemListingByPath ?? {};
  const pathEntriesErrorByPath = filesystemErrorByPath ?? {};
  let databaseSettings: Record<string, unknown> = {
    configured: false,
    updated_at: "2026-04-29T00:00:00Z",
    host: "",
    port: 5432,
    database: "",
    username: "",
    password_configured: false,
    ssl_mode: "prefer",
    extra_params: "",
    container_name: "workflows-postgres",
    container_image: "pgvector/pgvector:pg17",
    container_host_port: 5432,
    volume_name: "workflows-postgres-data",
    dsn_import: null,
    ...initialDatabaseSettings,
  };
  let llmConfig: Record<string, unknown> = initialLlmConfig ?? {
    version: "1.0",
    providers: {},
    profiles: {},
    default_profile: null,
  };
  const secrets: Array<Record<string, unknown>> = customSecrets
    ? [...customSecrets]
    : [
        {
          name: "OPENAI_API_KEY",
          key_id: "openai-prod",
          created_at: "2026-04-30T00:00:00Z",
          updated_at: "2026-04-30T01:00:00Z",
        },
      ];

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const method = (init?.method ?? "GET").toUpperCase();
    const url = typeof input === "string" ? input : input.toString();

    if (url.endsWith("/api/admin/v1/auth/login") && method === "POST") {
      return jsonResponse({ ok: true });
    }
    if (url.endsWith("/api/admin/v1/auth/session") && method === "GET") {
      return jsonResponse({ authenticated: true });
    }
    if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
      return jsonResponse({ csrf_token: "csrf-test-token" });
    }
    if (url.endsWith("/api/admin/v1/database/setup") && method === "GET") {
      return jsonResponse({
        docker: "docker run workflows-postgres",
        image: "pgvector/pgvector:pg17",
        podman: "podman run workflows-postgres",
        notes: ["Use a unique password."],
      });
    }
    if (url.endsWith("/api/admin/v1/database/settings") && method === "GET") {
      return jsonResponse(databaseSettings);
    }
    if (url.endsWith("/api/admin/v1/database/settings") && method === "PUT") {
      const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      databaseSettings = {
        ...databaseSettings,
        ...payload,
        port: Number(payload.port ?? 5432),
        container_host_port: Number(payload.container_host_port ?? 5432),
        password_configured:
          payload.password_clear === true
            ? false
            : typeof payload.password === "string"
              ? payload.password.trim().length > 0
              : (databaseSettings.password_configured as boolean),
        configured: payload.password_clear !== true,
        updated_at: "2026-04-30T00:00:00Z",
      };
      return jsonResponse(databaseSettings);
    }
    if (url.endsWith("/api/admin/v1/database/connection-test") && method === "POST") {
      return jsonResponse({
        ok: false,
        status: "degraded",
        configured: true,
        blockers: ["postgresql_connectivity"],
        actionable: ["Verify host, port, credentials, and network reachability."],
      });
    }
    if (url.endsWith("/api/public/v1/system/status") && method === "GET") {
      return jsonResponse({ status: "ok" });
    }
    if (url.endsWith("/api/admin/v1/llm/config") && method === "GET") {
      return jsonResponse(llmConfig);
    }
    if (url.endsWith("/api/admin/v1/llm/config") && method === "PUT") {
      llmConfig = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      return jsonResponse(llmConfig);
    }
    if (url.endsWith("/api/admin/v1/llm/export") && method === "GET") {
      return jsonResponse({ raw_yaml: "version: '1.0'\nproviders:\n  openai:\n    type: openai\n" });
    }
    if (url.endsWith("/api/admin/v1/llm/preview") && method === "POST") {
      return jsonResponse({
        version: "1.0",
        providers: { anthropic: { type: "anthropic", model: "claude-sonnet" } },
        profiles: { review: { provider: "anthropic", model: "claude-sonnet", temperature: 0.2 } },
        default_profile: "review",
      });
    }
    if (url.endsWith("/api/admin/v1/llm/import") && method === "POST") {
      llmConfig = {
        version: "1.0",
        providers: { anthropic: { type: "anthropic", model: "claude-sonnet" } },
        profiles: { review: { provider: "anthropic", model: "claude-sonnet", temperature: 0.2 } },
        default_profile: "review",
      };
      return jsonResponse(llmConfig);
    }
    if (url.endsWith("/api/admin/v1/secrets") && method === "GET") {
      if (secretsListPromise) {
        return secretsListPromise;
      }
      if (secretsListStatus >= 400) {
        return jsonResponse({ message: secretsListErrorMessage }, secretsListStatus);
      }
      return jsonResponse({ secrets });
    }
    if (url.endsWith("/api/admin/v1/secrets") && method === "POST") {
      const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      const name = String(payload.name ?? "");
      const index = secrets.findIndex((secret) => secret.name === name);
      const saved = {
        name,
        key_id: typeof payload.key_id === "string" && payload.key_id.trim().length > 0 ? payload.key_id : null,
        created_at: index >= 0 ? secrets[index].created_at : "2026-04-30T02:00:00Z",
        updated_at: "2026-04-30T02:00:00Z",
      };
      if (index >= 0) {
        secrets[index] = saved;
      } else {
        secrets.push(saved);
      }
      return jsonResponse(saved);
    }
    if (/\/api\/admin\/v1\/secrets\/[^/]+$/.test(url) && method === "DELETE") {
      const secretName = decodeURIComponent(url.split("/").pop() ?? "");
      const index = secrets.findIndex((secret) => secret.name === secretName);
      if (index < 0) {
        return jsonResponse({ message: "Secret not found" }, 404);
      }
      secrets.splice(index, 1);
      return jsonResponse({ deleted: true });
    }
    if (url.endsWith("/api/admin/v1/projects") && method === "GET") {
      return jsonResponse({ projects });
    }
    if (url.includes("/api/admin/v1/filesystem/entries") && method === "GET") {
      const requestUrl = new URL(url, "https://admin.example.test");
      const path = requestUrl.searchParams.get("path") ?? "";
      const configuredError = pathEntriesErrorByPath[path];
      if (configuredError) {
        return jsonResponse(configuredError.body, configuredError.status);
      }
      const listing = pathEntriesListingByPath[path];
      if (listing) return jsonResponse(listing);
      return jsonResponse({ root: "/", path: "/", parent: null, can_go_up: false, entries: [] });
    }
    if (url.endsWith("/api/admin/v1/projects") && method === "POST") {
      const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      const created = {
        id: `p${projects.length + 1}`,
        name: String(payload.name ?? ""),
        slug: String(payload.slug ?? ""),
        palace: String(payload.palace ?? ""),
        default_wing: String(payload.default_wing ?? ""),
        default_room: String(payload.default_room ?? ""),
        fs_root: String(payload.fs_root ?? ""),
        fs_allowlist: Array.isArray(payload.fs_allowlist) ? payload.fs_allowlist : [],
        system1_enabled: true,
        system2_enabled: payload.system2_enabled === true,
        created_at: "2026-04-30T00:00:00Z",
        updated_at: "2026-04-30T00:00:00Z",
      };
      projects.push(created);
      return jsonResponse(created, 201);
    }
    if (url.includes("/api/admin/v1/projects/") && method === "PATCH") {
      const projectId = url.split("/").pop() ?? "";
      const index = projects.findIndex((project) => project.id === projectId);
      if (index < 0) {
        return jsonResponse({ message: "Project not found" }, 404);
      }
      const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      const updated = {
        ...projects[index],
        name: typeof payload.name === "string" ? payload.name : projects[index].name,
        slug: typeof payload.slug === "string" ? payload.slug : projects[index].slug,
        palace: typeof payload.palace === "string" ? payload.palace : projects[index].palace,
        default_wing: typeof payload.default_wing === "string" ? payload.default_wing : projects[index].default_wing,
        default_room: typeof payload.default_room === "string" ? payload.default_room : projects[index].default_room,
        fs_root: typeof payload.fs_root === "string" ? payload.fs_root : projects[index].fs_root,
        fs_allowlist: Array.isArray(payload.fs_allowlist) ? payload.fs_allowlist : projects[index].fs_allowlist,
        system1_enabled: true,
        system2_enabled: typeof payload.system2_enabled === "boolean" ? payload.system2_enabled : projects[index].system2_enabled,
        updated_at: "2026-04-30T00:00:01Z",
      };
      projects[index] = updated;
      return jsonResponse(updated);
    }
    if (url.includes("/api/admin/v1/projects/") && method === "DELETE") {
      const projectId = url.split("/").pop() ?? "";
      const index = projects.findIndex((project) => project.id === projectId);
      if (index < 0) {
        return jsonResponse({ message: "Project not found" }, 404);
      }
      projects.splice(index, 1);
      return jsonResponse({ deleted: true });
    }
    if (url.endsWith("/api/admin/v1/watchers") && method === "GET") {
      return jsonResponse({ watchers });
    }
    if (/\/api\/admin\/v1\/watchers\/[^/]+\/pause$/.test(url) && method === "POST") {
      const projectId = url.split("/").at(-2) ?? "";
      const target = watchers.find((watcher) => watcher.project_id === projectId);
      if (!target) return jsonResponse({ message: "Watcher not found" }, 404);
      target.state = "paused";
      target.updated_at = "2026-04-30T00:10:00Z";
      return jsonResponse(target);
    }
    if (/\/api\/admin\/v1\/watchers\/[^/]+\/resume$/.test(url) && method === "POST") {
      const projectId = url.split("/").at(-2) ?? "";
      const target = watchers.find((watcher) => watcher.project_id === projectId);
      if (!target) return jsonResponse({ message: "Watcher not found" }, 404);
      target.state = "enabled";
      target.updated_at = "2026-04-30T00:11:00Z";
      return jsonResponse(target);
    }
    if (/\/api\/admin\/v1\/watchers\/[^/]+\/disable$/.test(url) && method === "POST") {
      const projectId = url.split("/").at(-2) ?? "";
      const target = watchers.find((watcher) => watcher.project_id === projectId);
      if (!target) return jsonResponse({ message: "Watcher not found" }, 404);
      target.state = "disabled";
      target.updated_at = "2026-04-30T00:12:00Z";
      return jsonResponse(target);
    }
    if (url.endsWith("/api/events/v1/watchers/state") && method === "GET") {
      return jsonResponse({ version: 1, items: watchers });
    }
    if (url.endsWith("/api/admin/v1/sync") && method === "GET") {
      return jsonResponse({ projects: syncProjects });
    }
    if (/\/api\/admin\/v1\/sync\/[^/]+\/logs(?:\?|$)/.test(url) && method === "GET") {
      const projectId = url.split("?")[0].split("/").at(-2) ?? "";
      return jsonResponse({ project_id: projectId, entries: syncLogsByProjectId[projectId] ?? [] });
    }
    if (/\/api\/admin\/v1\/sync\/[^/]+\/details$/.test(url) && method === "GET") {
      const projectId = url.split("/").at(-2) ?? "";
      return jsonResponse(
        syncDetailsByProjectId[projectId] ?? {
          project_id: projectId,
          system1_enabled: true,
          system1_state: "queued",
          system2_enabled: false,
          system2_state: "disabled",
          embedding_profile_required: false,
          embedding_profile: "embedding",
          memory_backend_ready: false,
          counts: {
            source_items: 0,
            structural_evidence: 0,
            verification_cycles: 0,
            wings: 0,
            rooms: 0,
            compartments: 0,
            semantic_claims: 0,
            semantic_memories: 0,
          },
        },
      );
    }
    if (/\/api\/admin\/v1\/sync\/[^/]+\/now$/.test(url) && method === "POST") {
      const projectId = url.split("/").at(-2) ?? "";
      const configuredResponse = syncActionResponses[`${projectId}:now`];
      if (configuredResponse) return jsonResponse(configuredResponse);
      const target = syncProjects.find((project) => project.project_id === projectId);
      if (!target) return jsonResponse({ message: "Project not found" }, 404);
      target.dirty_count = Number(target.dirty_count ?? 0) + 1;
      const logs = syncLogsByProjectId[projectId] ?? [];
      logs.unshift({
        id: logs.length + 1,
        project_id: projectId,
        path: ".",
        event_type: "manual_sync",
        reason: "manual_sync_now",
        status: "queued",
        enqueued_at: "2026-04-30T00:20:00Z",
        updated_at: "2026-04-30T00:20:00Z",
        processed_at: null,
      });
      syncLogsByProjectId[projectId] = logs;
      return jsonResponse({
        project_id: projectId,
        status: "queued",
        dirty_count: target.dirty_count,
        action: "sync",
        memory_mode: "simple",
        job_id: "job-sync-now",
        workflow: "project-memory-sync",
      });
    }
    if (/\/api\/admin\/v1\/sync\/[^/]+\/reconcile$/.test(url) && method === "POST") {
      const projectId = url.split("/").at(-2) ?? "";
      const configuredResponse = syncActionResponses[`${projectId}:reconcile`];
      if (configuredResponse) return jsonResponse(configuredResponse);
      const target = syncProjects.find((project) => project.project_id === projectId);
      if (!target) return jsonResponse({ message: "Project not found" }, 404);
      target.requires_reconciliation = true;
      const logs = syncLogsByProjectId[projectId] ?? [];
      logs.unshift({
        id: logs.length + 1,
        project_id: projectId,
        path: ".",
        event_type: "reconcile",
        reason: "reconciliation_required:manual_reconcile",
        status: "queued",
        enqueued_at: "2026-04-30T00:21:00Z",
        updated_at: "2026-04-30T00:21:00Z",
        processed_at: null,
      });
      syncLogsByProjectId[projectId] = logs;
      return jsonResponse({
        project_id: projectId,
        status: "queued",
        dirty_count: target.dirty_count,
        action: "reconcile",
        memory_mode: "simple",
        job_id: "job-reconcile",
        workflow: "project-memory-sync",
      });
    }
    if (/\/api\/admin\/v1\/sync\/[^/]+\/rebuild$/.test(url) && method === "POST") {
      const projectId = url.split("/").at(-2) ?? "";
      const configuredResponse = syncActionResponses[`${projectId}:rebuild`];
      if (configuredResponse) return jsonResponse(configuredResponse);
      const target = syncProjects.find((project) => project.project_id === projectId);
      if (!target) return jsonResponse({ message: "Project not found" }, 404);
      target.requires_reconciliation = true;
      const logs = syncLogsByProjectId[projectId] ?? [];
      logs.unshift({
        id: logs.length + 1,
        project_id: projectId,
        path: ".",
        event_type: "rebuild",
        reason: "reconciliation_required:manual_rebuild",
        status: "queued",
        enqueued_at: "2026-04-30T00:22:00Z",
        updated_at: "2026-04-30T00:22:00Z",
        processed_at: null,
      });
      syncLogsByProjectId[projectId] = logs;
      return jsonResponse({
        project_id: projectId,
        status: "queued",
        dirty_count: target.dirty_count,
        action: "rebuild",
        memory_mode: "simple",
        job_id: "job-rebuild",
        workflow: "project-memory-sync",
      });
    }
    if (url.endsWith("/api/events/v1/sync/state") && method === "GET") {
      return jsonResponse({ version: 1, items: syncProjects });
    }
    if (url.endsWith("/api/admin/v1/mcp-clients") && method === "GET") {
      return jsonResponse({ mcp_clients: mcpClients });
    }
    if (url.endsWith("/api/admin/v1/mcp-clients") && method === "POST") {
      const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      const created = {
        id: `t${mcpClients.length + 1}`,
        label: String(payload.label ?? ""),
        project_ids: Array.isArray(payload.project_ids) ? payload.project_ids : [],
        capabilities: payload.capabilities ?? {},
        created_at: "2026-04-30T00:00:00Z",
        last_used_at: null,
        revoked_at: null,
      };
      mcpClients.push(created);
      return jsonResponse({
        ...created,
        token: "mcp-secret-token-create",
        config_snippet: '{"headers":{"Authorization":"Bearer mcp-secret-token-create"}}',
        ...createMcpClientResponse,
      }, 201);
    }
    if (url.includes("/api/admin/v1/mcp-clients/") && method === "PATCH") {
      const tokenId = url.split("/").pop() ?? "";
      const existing = mcpClients.find((client) => client.id === tokenId);
      if (!existing) {
        return jsonResponse({ message: "Token not found" }, 404);
      }
      const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      existing.project_ids = Array.isArray(payload.project_ids) ? payload.project_ids : [];
      return jsonResponse(existing);
    }
    if (url.includes("/api/admin/v1/mcp-clients/") && url.endsWith("/registration") && method === "DELETE") {
      const parts = url.split("/");
      const tokenId = parts[parts.length - 2] ?? "";
      const index = mcpClients.findIndex((client) => client.id === tokenId);
      if (index < 0) {
        return jsonResponse({ message: "Token not found" }, 404);
      }
      mcpClients.splice(index, 1);
      return jsonResponse({ deleted: true });
    }
    if (url.includes("/api/admin/v1/mcp-clients/") && method === "DELETE") {
      const tokenId = url.split("/").pop() ?? "";
      const existing = mcpClients.find((client) => client.id === tokenId);
      if (!existing) {
        return jsonResponse({ message: "Token not found" }, 404);
      }
      existing.revoked_at = "2026-04-30T00:10:00Z";
      return jsonResponse({ revoked: true });
    }
    if (url.includes("/api/admin/v1/mcp-clients/") && url.endsWith("/regenerate") && method === "POST") {
      const parts = url.split("/");
      const tokenId = parts[parts.length - 2] ?? "";
      const existing = mcpClients.find((client) => client.id === tokenId);
      if (!existing) {
        return jsonResponse({ message: "Token not found" }, 404);
      }
      return jsonResponse({
        ...existing,
        token: "mcp-secret-token-regenerated",
        config_snippet: '{"headers":{"Authorization":"Bearer mcp-secret-token-regenerated"}}',
        ...regenerateMcpClientResponse,
      });
    }
    if (url.endsWith("/api/admin/v1/workflows") && method === "GET") {
      return jsonResponse({ workflows });
    }
    if (url.endsWith("/api/admin/v1/workflows/sources") && method === "GET") {
      return jsonResponse({ sources: workflowSources });
    }
    if (url.endsWith("/api/admin/v1/workflows/sources") && method === "POST") {
      const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      const created = {
        source_id: `src-${workflowSources.length + 1}`,
        project_id: String(payload.project_id ?? ""),
        source_path: String(payload.source_path ?? ""),
        checksum: payload.checksum ?? null,
        discovered_at: "2026-04-30T00:20:00Z",
        last_loaded_at: null,
        status: "pending",
        error_message: null,
      };
      workflowSources.push(created);
      return jsonResponse(created, 201);
    }
    if (/\/api\/admin\/v1\/workflows\/sources\/[^/]+$/.test(url) && method === "DELETE") {
      const sourceId = url.split("/").pop() ?? "";
      const index = workflowSources.findIndex((source) => source.source_id === sourceId);
      if (index < 0) {
        return jsonResponse({ message: "Workflow source not found" }, 404);
      }
      workflowSources.splice(index, 1);
      return jsonResponse({ deleted: true });
    }
    if (url.endsWith("/api/admin/v1/workflows/schema") && method === "GET") {
      return jsonResponse(workflowSchema);
    }
    if (/\/api\/admin\/v1\/workflows\/[^/]+$/.test(url) && method === "GET") {
      const workflowName = decodeURIComponent(url.split("/").pop() ?? "");
      const found = workflows.find((workflow) => workflow.name === workflowName);
      if (!found) {
        return jsonResponse({ message: `Workflow not found: ${workflowName}` }, 404);
      }
      return jsonResponse({
        ...found,
        yaml_path: `${String(found.source_path ?? "/workspace/workflows")}`,
        raw_yaml: `name: ${workflowName}\ndescription: ${String(found.description ?? "")}\nblocks:\n  - id: lint\n    type: Shell\n`,
        load_logs: [`Loaded from ${String(found.source_path ?? "/workspace/workflows")}`],
        blocks: [{ id: "lint", type: "Shell" }],
      });
    }
    if (url.includes("/api/admin/v1/runs") && method === "GET") {
      if (/\/api\/admin\/v1\/runs\/[^/]+$/.test(url)) {
        const runId = decodeURIComponent(url.split("/").pop() ?? "");
        const detail = runDetailById[runId];
        if (!detail) {
          return jsonResponse({ message: `Run not found: ${runId}` }, 404);
        }
        return jsonResponse(detail);
      }

      const parsed = new URL(url, "http://localhost");
      const status = parsed.searchParams.get("status");
      const mode = parsed.searchParams.get("mode");
      const workflow = parsed.searchParams.get("workflow");
      const projectId = parsed.searchParams.get("project_id");
      const limit = Number(parsed.searchParams.get("limit") ?? "50");
      const offset = Number(parsed.searchParams.get("offset") ?? "0");
      const filtered = runs.filter((run) => {
        if (status && run.status !== status) return false;
        if (mode && run.execution_mode !== mode) return false;
        if (workflow && run.workflow_name !== workflow) return false;
        if (projectId && run.project_id !== projectId) return false;
        return true;
      });
      const paged = filtered.slice(offset, offset + limit);
      return jsonResponse({ runs: paged, total: filtered.length, limit, offset });
    }
    if (/\/api\/admin\/v1\/runs\/[^/]+\/cancel$/.test(url) && method === "POST") {
      if (cancelRunStatus !== 200) {
        return jsonResponse({ message: cancelRunErrorMessage }, cancelRunStatus);
      }
      const runId = url.split("/").at(-2) ?? "";
      return jsonResponse({ run_id: runId, outcome: "cancelled" });
    }
    if (/\/api\/admin\/v1\/runs\/[^/]+\/resume$/.test(url) && method === "POST") {
      if (resumeRunStatus !== 200) {
        return jsonResponse({ message: resumeRunErrorMessage }, resumeRunStatus);
      }
      const runId = url.split("/").at(-2) ?? "";
      const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      return jsonResponse({ run_id: runId, outcome: "resumed", response: payload.response ?? "" });
    }
    if (url.endsWith("/api/admin/v1/workflows/reload") && method === "POST") {
      const headers = new Headers(init?.headers);
      if (headers.get("X-CSRF-Token") !== "csrf-test-token") {
        return jsonResponse({ message: "missing csrf" }, 403);
      }
      if (workflowReloadStatus !== 200) {
        return jsonResponse({ message: workflowReloadErrorMessage }, workflowReloadStatus);
      }
      return jsonResponse({ status: "ok", source_count: 1, total: 1, workflow_names: ["a"] });
    }

    return jsonResponse({ message: `unhandled ${method} ${url}` }, 404);
  });

  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

describe("App", () => {
  it("renders required navigation links as real anchors", () => {
    renderAtPath("/");

    for (const href of NAV_LINKS) {
      const link = document.querySelector(`nav a[href="${href}"]`);
      expect(link).toBeTruthy();
    }

    const docsLink = screen.getByRole("link", { name: /docs/i });
    expect(docsLink.getAttribute("href")).toBe("/docs");
  });

  it("redirects unauthenticated root visits to /login", async () => {
    renderAtPath("/");

    await waitFor(() => {
      expect(window.location.pathname).toBe("/login");
      expect(screen.getByRole("heading", { level: 1, name: "Login" })).toBeTruthy();
    });
  });

  it("redirects admin API 401 responses to login", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse(
          { error: { code: "UNAUTHORIZED", message: "Authentication required", request_id: "test" } },
          401,
        ),
      ),
    );

    renderAtPath("/projects");

    await waitFor(() => {
      expect(window.location.pathname).toBe("/login");
      expect(screen.getByRole("heading", { level: 1, name: "Login" })).toBeTruthy();
      expect(screen.getByText("Your admin session expired. Sign in to continue.")).toBeTruthy();
    });
  });

  for (const [path, title] of Object.entries(TITLES_BY_PATH)) {
    it(`renders page title '${title}' for route '${path}'`, () => {
      renderAtPath(path);
      expect(screen.getByRole("heading", { level: 1, name: title })).toBeTruthy();
    });
  }

  it("renders a not-found title and a recovery link for unknown routes", () => {
    renderAtPath("/does-not-exist");

    expect(screen.getByRole("heading", { level: 1, name: "Page not found" })).toBeTruthy();

    const projectsLink = screen.getByRole("link", { name: /go to database/i });
    expect(projectsLink.getAttribute("href")).toBe("/database");
  });

  it("submits login form and navigates to setup", async () => {
    installApiMock();
    renderAtPath("/login");

    const passwordInput = screen.getByLabelText(/password/i);
    fireEvent.change(passwordInput, { target: { value: "admin-pass" } });
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(window.location.pathname).toBe("/setup");
      expect(screen.getByRole("heading", { level: 1, name: "Setup" })).toBeTruthy();
    });
  });

  it("renders workflows and sources, and supports add/delete/reload operations", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/workflows");

    const sourceTable = await screen.findByRole("table", { name: /workflow sources/i });
    expect(within(sourceTable).getByRole("row", { name: /expand workflow source \/workspace\/workflows/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /^new$/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /load schema/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /reload workflows/i })).toBeTruthy();

    fireEvent.click(within(sourceTable).getByRole("button", { name: /expand workflow source \/workspace\/workflows/i }));
    const loadedWorkflows = await screen.findByRole("table", { name: /workflows loaded from \/workspace\/workflows/i });
    expect(within(loadedWorkflows).getByRole("row", { name: /open workflow python-ci-pipeline details/i })).toBeTruthy();
    expect(within(loadedWorkflows).getByText("1.2.0")).toBeTruthy();
    expect(within(loadedWorkflows).getByText("python, ci")).toBeTruthy();

    fireEvent.click(within(loadedWorkflows).getByRole("row", { name: /open workflow python-ci-pipeline details/i }));
    const detailDialog = await screen.findByRole("dialog", { name: /workflow python-ci-pipeline/i });
    expect(within(detailDialog).getByRole("region", { name: /workflow yaml/i })).toBeTruthy();
    expect(detailDialog.querySelector(".workflow-yaml-viewer")).toBeTruthy();
    expect(detailDialog.querySelector(".yaml-token--key")).toBeTruthy();
    expect(within(detailDialog).queryByText(/technical json/i)).toBeNull();
    fireEvent.click(within(detailDialog).getByRole("button", { name: /close dialog/i }));

    fireEvent.click(screen.getByRole("button", { name: /^new$/i }));
    const sourceDialog = await screen.findByRole("dialog", { name: /new workflow source/i });
    fireEvent.change(within(sourceDialog).getByLabelText(/^source path$/i, { selector: "input" }), {
      target: { value: "/workspace/more-workflows" },
    });
    fireEvent.click(within(sourceDialog).getByRole("button", { name: /add workflow source/i }));

    await waitFor(() => {
      expect(screen.getByText(/workflow source added\./i)).toBeTruthy();
      expect(screen.getByRole("row", { name: /expand workflow source \/workspace\/more-workflows/i })).toBeTruthy();
      expect(screen.queryByRole("dialog", { name: /new workflow source/i })).toBeNull();
    });

    const sourceRowForDelete = within(sourceTable).getByRole("row", {
      name: /expand workflow source \/workspace\/workflows/i,
    });
    fireEvent.click(within(sourceRowForDelete).getByRole("button", { name: /^delete$/i }));
    await waitFor(() => {
      expect(screen.getByText(/source src-1 deleted/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /load schema/i }));
    await waitFor(() => {
      expect(screen.getByText(/workflow schema loaded/i)).toBeTruthy();
      expect(screen.getByText(/workflowschema/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /reload workflows/i }));

    await waitFor(() => {
      const reloadCall = fetchMock.mock.calls.find((call) => {
        const url = String(call[0]);
        const init = call[1] as RequestInit | undefined;
        return url.includes("/api/admin/v1/workflows/reload") && init?.method === "POST";
      });
      expect(reloadCall).toBeTruthy();

      const reloadHeaders = new Headers((reloadCall?.[1] as RequestInit | undefined)?.headers);
      expect(reloadHeaders.get("X-CSRF-Token")).toBe("csrf-test-token");
    });

    await waitFor(() => {
      expect(screen.getByText(/workflow registry reloaded: 1 workflows from 1 source/i)).toBeTruthy();
    });
  });

  it("renders workflow sources as expandable tables and opens workflow detail modals", async () => {
    installApiMock({
      workflows: [
        {
          name: "python-ci-pipeline",
          description: "Runs lint and test checks",
          version: "1.2.0",
          tags: ["python", "ci"],
          source_path: "/workspace/workflows/python-ci.yaml",
        },
        {
          name: "node-ci-pipeline",
          description: "Runs node checks",
          version: "2.0.0",
          tags: ["node"],
          source_path: "/workspace/node-workflows",
        },
      ],
      workflowSources: [
        {
          source_id: "src-1",
          project_id: "p1",
          source_path: "/workspace/workflows",
          checksum: null,
          discovered_at: "2026-04-30T00:00:00Z",
          last_loaded_at: "2026-04-30T00:10:00Z",
          status: "loaded",
          error_message: null,
        },
        {
          source_id: "src-2",
          project_id: "p1",
          source_path: "/workspace/node-workflows",
          checksum: null,
          discovered_at: "2026-04-30T00:00:00Z",
          last_loaded_at: null,
          status: "failed",
          error_message: "Invalid workflow definition at node.yaml",
        },
      ],
    });

    renderAtPath("/workflows");

    const sourceTable = await screen.findByRole("table", { name: /workflow sources/i });
    expect(within(sourceTable).getByRole("columnheader", { name: /^source$/i })).toBeTruthy();
    expect(within(sourceTable).getByRole("columnheader", { name: /^registry state$/i })).toBeTruthy();
    expect(within(sourceTable).getByRole("columnheader", { name: /^actions$/i })).toBeTruthy();
    expect(within(sourceTable).queryByRole("columnheader", { name: /^project$/i })).toBeNull();
    expect(within(sourceTable).queryByRole("columnheader", { name: /^workflows$/i })).toBeNull();

    const sourceRow = within(sourceTable).getByRole("row", {
      name: /expand workflow source \/workspace\/workflows status loaded workflows 1/i,
    });
    const expandButton = within(sourceRow).getByRole("button", { name: /expand workflow source \/workspace\/workflows/i });
    expect(expandButton.getAttribute("aria-expanded")).toBe("false");

    fireEvent.click(expandButton);
    expect(expandButton.getAttribute("aria-expanded")).toBe("true");
    expect(screen.queryByText(/^Source src-1$/i)).toBeNull();
    expect(screen.queryByText(/^Project p1$/i)).toBeNull();
    expect(screen.queryByText(/Discovered 2026-04-30/i)).toBeNull();
    expect(screen.queryByRole("button", { name: /validate src-1/i })).toBeNull();
    expect(within(sourceRow).getByRole("button", { name: /^delete$/i })).toBeTruthy();

    const workflowsTable = await screen.findByRole("table", { name: /workflows loaded from \/workspace\/workflows/i });
    const workflowRow = within(workflowsTable).getByRole("row", {
      name: /open workflow python-ci-pipeline details 1.2.0 python, ci/i,
    });
    expect(workflowRow.getAttribute("aria-haspopup")).toBe("dialog");

    fireEvent.click(workflowRow);

    const dialog = await screen.findByRole("dialog", { name: /workflow python-ci-pipeline/i });
    expect(within(dialog).getByRole("heading", { name: /workflow python-ci-pipeline/i })).toBeTruthy();
    expect(within(dialog).queryByText(/^Status$/i)).toBeNull();
    expect(within(dialog).queryByText(/^Loaded$/i)).toBeNull();
    expect(within(dialog).queryByText(/^Tags$/i)).toBeNull();
    expect(within(dialog).queryByText(/technical json/i)).toBeNull();
    const yamlRegion = within(dialog).getByRole("region", { name: /workflow yaml/i });
    expect(within(yamlRegion).getByText("YAML")).toBeTruthy();
    expect(yamlRegion.querySelector(".workflow-yaml-viewer")).toBeTruthy();
    expect(yamlRegion.textContent).toContain("name: python-ci-pipeline");
    const logRegion = within(dialog).getByRole("region", { name: /workflow load logs/i });
    expect(within(logRegion).getByText("Load logs")).toBeTruthy();
    expect(within(logRegion).getByText(/loaded from \/workspace\/workflows\/python-ci.yaml/i)).toBeTruthy();
  });

  it("adds workflow sources without project selection", async () => {
    const fetchMock = installApiMock({
      projects: [
        { id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp/main", fs_allowlist: [] },
        { id: "p2", name: "Docs Project", slug: "docs", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp/docs", fs_allowlist: [] },
      ],
    });
    renderAtPath("/workflows");

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^new$/i })).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /^new$/i }));
    const sourceDialog = await screen.findByRole("dialog", { name: /new workflow source/i });
    expect(within(sourceDialog).queryByLabelText(/^project$/i)).toBeNull();
    fireEvent.change(within(sourceDialog).getByLabelText(/^source path$/i, { selector: "input" }), {
      target: { value: "/workspace/docs-workflows" },
    });
    fireEvent.click(within(sourceDialog).getByRole("button", { name: /add workflow source/i }));

    await waitFor(() => {
      const createCallIndex = fetchMock.mock.calls.findIndex((call) => {
        const url = String(call[0]);
        const init = call[1] as RequestInit | undefined;
        return url.includes("/api/admin/v1/workflows/sources") && init?.method === "POST";
      });
      expect(createCallIndex).toBeGreaterThanOrEqual(0);
      const createCall = fetchMock.mock.calls[createCallIndex];
      const body = JSON.parse(String((createCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as Record<string, unknown>;
      expect(body).toEqual({ source_path: "/workspace/docs-workflows", checksum: null });
      const reloadCallIndex = fetchMock.mock.calls.findIndex((call, index) => {
        const url = String(call[0]);
        const init = call[1] as RequestInit | undefined;
        return index > createCallIndex && url.includes("/api/admin/v1/workflows/reload") && init?.method === "POST";
      });
      expect(reloadCallIndex).toBeGreaterThan(createCallIndex);
    });
  });

  it("shows actionable workflows empty state when no sources are configured", async () => {
    installApiMock({ projectCount: 0, workflows: [], workflowSources: [] });
    renderAtPath("/workflows");

    await waitFor(() => {
      expect(screen.getByText(/no workflow sources configured/i)).toBeTruthy();
    });

    expect(screen.getByText(/use new to add a workflow source path/i)).toBeTruthy();
    expect(screen.queryByRole("link", { name: /create a project first/i })).toBeNull();
  });

  it("shows readable errors for reload failures", async () => {
    installApiMock({
      workflowReloadStatus: 422,
      workflowReloadErrorMessage: "workflow reload failed because duplicate workflow names were detected",
    });
    renderAtPath("/workflows");

    fireEvent.click(screen.getByRole("button", { name: /reload workflows/i }));
    await waitFor(() => {
      expect(
        screen.getByText(/unable to reload workflows: workflow reload failed because duplicate workflow names were detected/i),
      ).toBeTruthy();
    });
  });

  it("renders system source with a system badge and no delete button", async () => {
    installApiMock({
      workflowSources: [
        {
          source_id: "sys-src-1",
          project_id: "system-project",
          source_path: "/builtin/templates/memory",
          checksum: null,
          discovered_at: "2026-01-01T00:00:00Z",
          last_loaded_at: "2026-01-01T00:00:00Z",
          status: "loaded",
          error_message: null,
          is_system: true,
        },
        {
          source_id: "src-1",
          project_id: "p1",
          source_path: "/workspace/workflows",
          checksum: null,
          discovered_at: "2026-04-30T00:00:00Z",
          last_loaded_at: "2026-04-30T00:10:00Z",
          status: "loaded",
          error_message: null,
          is_system: false,
        },
      ],
    });
    renderAtPath("/workflows");

    const sourceTable = await screen.findByRole("table", { name: /workflow sources/i });

    // System source row must show a system/read-only indicator
    const systemRow = within(sourceTable).getByRole("row", {
      name: /expand workflow source \/builtin\/templates\/memory/i,
    });
    expect(within(systemRow).getByText(/system/i)).toBeTruthy();

    // System source row must NOT have an active delete button
    expect(within(systemRow).queryByRole("button", { name: /^delete$/i })).toBeNull();

    // User source row still has delete button
    const userRow = within(sourceTable).getByRole("row", {
      name: /expand workflow source \/workspace\/workflows/i,
    });
    expect(within(userRow).getByRole("button", { name: /^delete$/i })).toBeTruthy();
  });

  it("groups system1-scan workflow under the system source", async () => {
    installApiMock({
      workflows: [
        {
          name: "system1-scan",
          description: "Built-in system scan workflow",
          version: "1.0.0",
          tags: ["system"],
          source_path: "/builtin/templates/memory/system1-scan.yaml",
        },
      ],
      workflowSources: [
        {
          source_id: "sys-src-1",
          project_id: "system-project",
          source_path: "/builtin/templates/memory",
          checksum: null,
          discovered_at: "2026-01-01T00:00:00Z",
          last_loaded_at: "2026-01-01T00:00:00Z",
          status: "loaded",
          error_message: null,
          is_system: true,
        },
      ],
    });
    renderAtPath("/workflows");

    const sourceTable = await screen.findByRole("table", { name: /workflow sources/i });
    const systemRow = within(sourceTable).getByRole("row", {
      name: /expand workflow source \/builtin\/templates\/memory/i,
    });
    fireEvent.click(within(systemRow).getByRole("button", { name: /expand workflow source/i }));

    const workflowsTable = await screen.findByRole("table", {
      name: /workflows loaded from \/builtin\/templates\/memory/i,
    });
    expect(within(workflowsTable).getByRole("row", { name: /open workflow system1-scan details/i })).toBeTruthy();
  });

  it("renders setup checklist with status cards and next-action links", async () => {
    installApiMock();
    renderAtPath("/setup");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /setup checklist/i })).toBeTruthy();
    });

    expect(screen.getByText(/admin session active/i)).toBeTruthy();
    expect(screen.getByText(/public status/i)).toBeTruthy();
    expect(screen.getByText(/^ok$/i)).toBeTruthy();
    expect(screen.getByText(/database configured/i)).toBeTruthy();
    expect(screen.getAllByText(/^no$/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/llm config loaded/i)).toBeTruthy();
    expect(screen.getByText(/projects registered/i)).toBeTruthy();
    expect(screen.getByText(/^1$/i)).toBeTruthy();
    expect(screen.getByText(/mcp clients issued/i)).toBeTruthy();
    expect(screen.getByText(/^0$/i)).toBeTruthy();

    expect(screen.getByRole("link", { name: /configure database/i }).getAttribute("href")).toBe(
      "/database",
    );
    expect(screen.getByRole("link", { name: /review llm configuration/i }).getAttribute("href")).toBe(
      "/llm",
    );
    expect(screen.getByRole("link", { name: /create chat llm profile/i }).getAttribute("href")).toBe(
      "/llm",
    );
    expect(screen.getByRole("link", { name: /create embedding profile/i }).getAttribute("href")).toBe(
      "/llm",
    );
    expect(screen.queryByRole("link", { name: /add first project/i })).toBeNull();
    expect(screen.getByRole("link", { name: /create mcp client token/i }).getAttribute("href")).toBe(
      "/mcp-clients",
    );
  });

  it("does not mark LLM configured when only config version exists", async () => {
    installApiMock({ llmConfig: { version: "1.0", providers: {}, profiles: {}, default_profile: null } });
    renderAtPath("/setup");

    await waitFor(() => {
      expect(screen.getByText(/llm config loaded/i)).toBeTruthy();
    });

    expect(screen.getAllByText(/^no$/i).length).toBeGreaterThan(1);
  });

  it("does not mark LLM configured when providers exist without profiles", async () => {
    installApiMock({
      llmConfig: {
        version: "1.0",
        providers: { openai: { type: "openai", model: "gpt-4.1-mini" } },
        profiles: {},
        default_profile: null,
      },
    });
    renderAtPath("/setup");

    await waitFor(() => {
      expect(screen.getByText(/llm config loaded/i)).toBeTruthy();
    });

    expect(screen.getAllByText(/^no$/i).length).toBeGreaterThan(1);
  });

  it("shows loading, empty, and error states for secrets", async () => {
    const listDeferred = deferred<Response>();
    installApiMock({ secretsListPromise: listDeferred.promise });
    renderAtPath("/secrets");

    expect(await screen.findByRole("status", { name: /loading secrets/i })).toBeTruthy();

    listDeferred.resolve(jsonResponse({ secrets: [] }));
    expect(await screen.findByText(/no secrets configured/i)).toBeTruthy();
    expect(screen.queryByText(/follow-up slice/i)).toBeNull();

    cleanup();
    installApiMock({ secrets: [], secretsListStatus: 500, secretsListErrorMessage: "vault unavailable" });
    renderAtPath("/secrets");

    expect((await screen.findByRole("alert")).textContent).toMatch(/vault unavailable/i);
  });

  it("loads secrets metadata with key-id fallback and accessible row activation", async () => {
    installApiMock({
      secrets: [
        {
          name: "OPENAI_API_KEY",
          key_id: "openai-prod",
          created_at: "2026-04-30T00:00:00Z",
          updated_at: "2026-04-30T01:00:00Z",
        },
        {
          name: "STRIPE_WEBHOOK_SECRET",
          key_id: null,
          created_at: "2026-04-29T00:00:00Z",
          updated_at: "2026-04-29T00:00:00Z",
        },
      ],
    });
    renderAtPath("/secrets");

    expect(await screen.findByRole("heading", { name: /secret configuration/i })).toBeTruthy();
    expect(screen.queryByText(/secrets admin controls are available/i)).toBeNull();
    expect(screen.getByText(/secrets: 2/i)).toBeTruthy();

    const table = screen.getByRole("table", { name: /configured secrets/i });
    expect(within(table).getByRole("columnheader", { name: /^name$/i })).toBeTruthy();
    expect(within(table).getByRole("columnheader", { name: /key id/i })).toBeTruthy();
    expect(within(table).getByRole("columnheader", { name: /created/i })).toBeTruthy();
    expect(within(table).getByRole("columnheader", { name: /updated/i })).toBeTruthy();
    expect(within(table).getByRole("row", {
      name: /open secret openai_api_key configuration/i,
      description: /opens the secret configuration dialog/i,
    })).toBeTruthy();
    expect(screen.getByRole("button", { name: /^new$/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /^reload$/i })).toBeTruthy();
    expect(within(table).getByText("openai-prod")).toBeTruthy();
    expect(within(table).getByText("Server default")).toBeTruthy();
    expect(screen.queryByRole("region", { name: /manage secret/i })).toBeNull();
    expect(screen.queryByLabelText(/^secret value$/i)).toBeNull();

    const fallbackRow = within(table).getByRole("row", { name: /open secret stripe_webhook_secret configuration/i });
    fallbackRow.focus();
    expect(document.activeElement).toBe(fallbackRow);
    fireEvent.keyDown(fallbackRow, { key: "Enter" });

    const dialog = await screen.findByRole("dialog", { name: /secret stripe_webhook_secret/i });
    expect((within(dialog).getByLabelText(/^secret name$/i) as HTMLInputElement).value).toBe("STRIPE_WEBHOOK_SECRET");
    const valueInput = within(dialog).getByLabelText(/^secret value$/i) as HTMLInputElement;
    expect(valueInput.type).toBe("password");
    expect(valueInput.value).toBe("");
    expect(within(dialog).getByRole("button", { name: /save secret/i })).toBeTruthy();
    expect(screen.queryByDisplayValue("sk-test-secret")).toBeNull();
  });

  it("upserts secrets through the API and clears the secret value after save", async () => {
    const fetchMock = installApiMock({ secrets: [] });
    renderAtPath("/secrets");

    await screen.findByRole("heading", { name: /secret configuration/i });
    fireEvent.click(screen.getByRole("button", { name: /^new$/i }));
    const dialog = await screen.findByRole("dialog", { name: /new secret/i });
    fireEvent.change(within(dialog).getByLabelText(/^secret name$/i), { target: { value: "STRIPE_API_KEY" } });
    fireEvent.change(within(dialog).getByLabelText(/^secret value$/i), { target: { value: "sk-test-secret" } });
    fireEvent.change(within(dialog).getByLabelText(/^key id$/i), { target: { value: "stripe-test" } });
    fireEvent.click(within(dialog).getByRole("button", { name: /register secret/i }));

    expect((await screen.findByRole("status", { name: /secret save status/i })).textContent).toMatch(
      /secret stripe_api_key saved/i,
    );
    await waitFor(() => expect(screen.queryByRole("dialog", { name: /new secret/i })).toBeNull());
    expect(screen.queryByDisplayValue("sk-test-secret")).toBeNull();

    const saveCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/secrets") && init?.method === "POST";
    });
    expect(saveCall).toBeTruthy();
    const headers = new Headers((saveCall?.[1] as RequestInit | undefined)?.headers);
    expect(headers.get("X-CSRF-Token")).toBe("csrf-test-token");
    const payload = JSON.parse(String((saveCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as Record<string, unknown>;
    expect(payload).toEqual({ name: "STRIPE_API_KEY", value: "sk-test-secret", key_id: "stripe-test" });
  });

  it("blocks invalid secret names before posting to the API", async () => {
    const fetchMock = installApiMock({ secrets: [] });
    renderAtPath("/secrets");

    await screen.findByRole("heading", { name: /secret configuration/i });
    fireEvent.click(screen.getByRole("button", { name: /^new$/i }));
    const dialog = await screen.findByRole("dialog", { name: /new secret/i });

    for (const invalidName of ["1INVALID", "BAD-NAME"]) {
      fireEvent.change(within(dialog).getByLabelText(/^secret name$/i), { target: { value: invalidName } });
      fireEvent.change(within(dialog).getByLabelText(/^secret value$/i), { target: { value: "sk-test-secret" } });
      fireEvent.click(within(dialog).getByRole("button", { name: /register secret/i }));

      expect((await within(dialog).findByRole("alert")).textContent).toMatch(
        /secret name must start with a letter or underscore and use only letters, numbers, or underscores/i,
      );
      await waitFor(() => {
        const saveCalls = fetchMock.mock.calls.filter((call) => {
          const url = String(call[0]);
          const init = call[1] as RequestInit | undefined;
          return url.includes("/api/admin/v1/secrets") && init?.method === "POST";
        });
        expect(saveCalls).toHaveLength(0);
      });
    }
  });

  it("requires typed confirmation before deleting a selected secret", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/secrets");

    await screen.findByRole("heading", { name: /secret configuration/i });
    const table = screen.getByRole("table", { name: /configured secrets/i });
    fireEvent.click(within(table).getByRole("row", { name: /open secret openai_api_key configuration/i }));

    const dialog = await screen.findByRole("dialog", { name: /secret openai_api_key/i });
    const deleteButton = within(dialog).getByRole("button", { name: /confirm delete openai_api_key/i });
    expect((deleteButton as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(within(dialog).getByLabelText(/confirm secret name/i), { target: { value: "OPENAI_API_KEY" } });
    expect((deleteButton as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(deleteButton);

    expect((await screen.findByRole("status", { name: /secret delete status/i })).textContent).toMatch(
      /secret openai_api_key deleted/i,
    );
    await waitFor(() => expect(screen.queryByRole("dialog", { name: /secret openai_api_key/i })).toBeNull());
    expect(within(table).queryByRole("row", { name: /open secret openai_api_key configuration/i })).toBeNull();

    const deleteCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/secrets/OPENAI_API_KEY") && init?.method === "DELETE";
    });
    expect(deleteCall).toBeTruthy();
  });

  it("loads SQLite-backed LLM config with provider and profile summaries", async () => {
    installApiMock({
      llmConfig: {
        version: "1.0",
        providers: {
          openai: {
            type: "openai",
            api_url: "https://api.openai.com/v1",
            api_key_secret: "OPENAI_API_KEY",
            model: "gpt-4.1-mini",
            timeout: 30,
            max_retries: 2,
            retry_delay: 1,
            extra_headers: { "X-Team": "platform" },
          },
        },
        profiles: {
          default: {
            provider: "openai",
            model: "gpt-4.1-mini",
            temperature: 0.2,
            max_tokens: 4096,
            description: "Primary runtime profile",
          },
        },
        default_profile: "default",
      },
    });

    renderAtPath("/llm");

    expect(await screen.findByRole("heading", { name: /llm configuration/i })).toBeTruthy();
    expect(screen.getByText(/source of truth: sqlite-backed/i)).toBeTruthy();
    expect(screen.getByText(/providers: 1/i)).toBeTruthy();
    expect(screen.getByText(/profiles: 1/i)).toBeTruthy();
    expect(screen.getByText(/default profile: default/i)).toBeTruthy();
    expect(screen.queryByText(/persist configuration/i)).toBeNull();
    expect(screen.getByRole("button", { name: /save llm configuration/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /discard local changes/i })).toBeTruthy();

    const providerTable = screen.getByRole("table", { name: /llm providers/i });
    expect(within(providerTable).getByRole("columnheader", { name: /provider id/i })).toBeTruthy();
    expect(within(providerTable).getByRole("columnheader", { name: /^type$/i })).toBeTruthy();
    expect(within(providerTable).getByRole("columnheader", { name: /^model$/i })).toBeTruthy();
    expect(within(providerTable).queryByRole("columnheader", { name: /operations/i })).toBeNull();
    const providerRow = within(providerTable).getByRole("row", {
      name: /edit provider openai/i,
      description: /opens the provider edit dialog/i,
    });
    expect(within(providerRow).getAllByText("openai").length).toBeGreaterThanOrEqual(2);
    expect(within(providerRow).getByText("gpt-4.1-mini")).toBeTruthy();
    expect(providerRow.getAttribute("aria-haspopup")).toBe("dialog");
    expect(providerRow.getAttribute("aria-keyshortcuts")).toBe("Enter Space");
    expect(within(providerTable).queryByRole("button", { name: /edit provider openai/i })).toBeNull();
    expect(within(providerTable).queryByRole("button", { name: /delete provider openai/i })).toBeNull();
    expect(screen.queryByLabelText(/llm provider details/i)).toBeNull();
    expect(screen.queryByText(/api key secret: openai_api_key/i)).toBeNull();
    const providersCard = screen.getByRole("heading", { name: /^providers$/i }).closest("article");
    expect(providersCard).toBeTruthy();
    const addProviderButton = within(providersCard as HTMLElement).getByRole("button", { name: /add provider/i });
    const reloadButton = within(providersCard as HTMLElement).getByRole("button", { name: /reload/i });
    expect(providerTable.compareDocumentPosition(addProviderButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(providerTable.compareDocumentPosition(reloadButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    const profileTable = screen.getByRole("table", { name: /llm profiles/i });
    expect(within(profileTable).getByRole("columnheader", { name: /profile id/i })).toBeTruthy();
    expect(within(profileTable).getByRole("columnheader", { name: /^provider$/i })).toBeTruthy();
    expect(within(profileTable).getByRole("columnheader", { name: /^model$/i })).toBeTruthy();
    expect(within(profileTable).getByRole("columnheader", { name: /temperature/i })).toBeTruthy();
    expect(within(profileTable).getByRole("columnheader", { name: /max tokens/i })).toBeTruthy();
    expect(within(profileTable).queryByRole("columnheader", { name: /operations/i })).toBeNull();
    const profileRow = within(profileTable).getByRole("row", {
      name: /edit profile default/i,
      description: /opens the profile edit dialog/i,
    });
    expect(within(profileRow).getByText("default")).toBeTruthy();
    expect(within(profileRow).getByText("openai")).toBeTruthy();
    expect(within(profileRow).getByText("gpt-4.1-mini")).toBeTruthy();
    expect(within(profileRow).getByText("0.2")).toBeTruthy();
    expect(within(profileRow).getByText("4096")).toBeTruthy();
    expect(profileRow.getAttribute("aria-haspopup")).toBe("dialog");
    expect(profileRow.getAttribute("aria-keyshortcuts")).toBe("Enter Space");
    expect(within(profileTable).queryByRole("button", { name: /edit profile default/i })).toBeNull();
    expect(within(profileTable).queryByRole("button", { name: /delete profile default/i })).toBeNull();
    expect(screen.queryByLabelText(/llm profile descriptions/i)).toBeNull();
    expect(screen.queryByText(/primary runtime profile/i)).toBeNull();
    const profilesCard = screen.getByRole("heading", { name: /^chat profiles$/i }).closest("article");
    expect(profilesCard).toBeTruthy();
    const addProfileButton = within(profilesCard as HTMLElement).getByRole("button", { name: /add profile/i });
    expect(profileTable.compareDocumentPosition(addProfileButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("renders embedding profiles in their own section", async () => {
    installApiMock({
      llmConfig: {
        version: "1.0",
        providers: {
          openai: {
            type: "openai",
            api_url: "https://api.openai.com/v1",
            api_key_secret: "OPENAI_API_KEY",
            model: "gpt-5-mini",
            timeout: 30,
            max_retries: 2,
            retry_delay: 1,
            extra_headers: {},
          },
        },
        profiles: {
          chat: {
            provider: "openai",
            model: "gpt-5-mini",
            temperature: 0.2,
            max_tokens: 4096,
            description: "Chat profile",
          },
          embedding: {
            provider: "openai",
            model: "text-embedding-3-small",
            temperature: null,
            max_tokens: null,
            description: "Required by semantic search and System 2 evidence retrieval",
          },
        },
        default_profile: "chat",
      },
    });

    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });
    expect(screen.getByRole("heading", { name: /^chat profiles$/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /^embedding profiles$/i })).toBeTruthy();
    expect(screen.getByText(/required by semantic search and system 2 evidence retrieval/i)).toBeTruthy();

    const chatTable = screen.getByRole("table", { name: /llm profiles/i });
    expect(within(chatTable).getByRole("row", { name: /edit profile chat/i })).toBeTruthy();
    expect(within(chatTable).queryByRole("row", { name: /edit profile embedding/i })).toBeNull();

    const embeddingTable = screen.getByRole("table", { name: /embedding profiles/i });
    expect(within(embeddingTable).getByRole("row", { name: /edit embedding profile embedding/i })).toBeTruthy();
    const embeddingCard = screen.getByRole("heading", { name: /^embedding profiles$/i }).closest("article");
    expect(embeddingCard).toBeTruthy();
    expect(within(embeddingCard as HTMLElement).getByRole("button", { name: /^add profile$/i })).toBeTruthy();
    expect(within(embeddingCard as HTMLElement).queryByRole("button", { name: /add embedding profile/i })).toBeNull();
  });

  it("uses embedding-specific fields when adding or editing embedding profiles", async () => {
    installApiMock({
      llmConfig: {
        version: "1.0",
        providers: {
          openai: {
            type: "openai",
            api_url: "https://api.openai.com/v1",
            api_key_secret: "OPENAI_API_KEY",
            model: "gpt-5-mini",
          },
        },
        profiles: {
          embedding: {
            provider: "openai",
            model: "text-embedding-3-small",
            temperature: null,
            max_tokens: 1536,
            description: "Embedding profile",
          },
        },
        default_profile: null,
      },
    });

    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });
    const embeddingTable = screen.getByRole("table", { name: /embedding profiles/i });
    fireEvent.click(within(embeddingTable).getByRole("row", { name: /edit embedding profile embedding/i }));

    const editDialog = await screen.findByRole("dialog", { name: /edit embedding profile embedding/i });
    expect(editDialog.querySelector("#llm-profile-dialog-title")?.classList.contains("visually-hidden")).toBe(true);
    expect((within(editDialog).getByLabelText(/^profile id$/i) as HTMLInputElement).value).toBe("embedding");
    expect((within(editDialog).getByLabelText(/^embedding provider$/i) as HTMLSelectElement).value).toBe("openai");
    expect((within(editDialog).getByLabelText(/^embedding model$/i) as HTMLInputElement).value).toBe(
      "text-embedding-3-small",
    );
    expect((within(editDialog).getByLabelText(/^dimensions$/i) as HTMLInputElement).value).toBe("1536");
    expect(within(editDialog).queryByLabelText(/temperature/i)).toBeNull();
    expect(within(editDialog).queryByLabelText(/max tokens/i)).toBeNull();
    fireEvent.click(within(editDialog).getByRole("button", { name: /cancel/i }));

    const embeddingCard = screen.getByRole("heading", { name: /^embedding profiles$/i }).closest("article");
    expect(embeddingCard).toBeTruthy();
    fireEvent.click(within(embeddingCard as HTMLElement).getByRole("button", { name: /^add profile$/i }));

    const addDialog = await screen.findByRole("dialog", { name: /add embedding profile/i });
    expect((within(addDialog).getByLabelText(/^profile id$/i) as HTMLInputElement).value).toBe("embedding");
    expect(within(addDialog).getByLabelText(/^embedding provider$/i)).toBeTruthy();
    expect(within(addDialog).getByLabelText(/^embedding model$/i)).toBeTruthy();
    expect(within(addDialog).getByLabelText(/^dimensions$/i)).toBeTruthy();
    expect(within(addDialog).queryByLabelText(/temperature/i)).toBeNull();
    expect(within(addDialog).queryByLabelText(/max tokens/i)).toBeNull();
  });

  it("paginates LLM provider and profile tables independently at ten rows per page", async () => {
    installApiMock({ llmConfig: buildLargeLlmConfig(12) });
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });

    const providerTable = screen.getByRole("table", { name: /llm providers/i });
    expect(within(providerTable).getByRole("row", { name: /edit provider provider-01/i })).toBeTruthy();
    expect(within(providerTable).getByRole("row", { name: /edit provider provider-10/i })).toBeTruthy();
    expect(within(providerTable).queryByRole("row", { name: /edit provider provider-11/i })).toBeNull();

    const providerPagination = screen.getByRole("navigation", { name: /llm providers pagination/i });
    expect(within(providerPagination).getByText("Showing 1-10 of 12")).toBeTruthy();
    fireEvent.click(within(providerPagination).getByRole("button", { name: /next page/i }));

    expect(within(providerTable).queryByRole("row", { name: /edit provider provider-01/i })).toBeNull();
    expect(within(providerTable).getByRole("row", { name: /edit provider provider-11/i })).toBeTruthy();
    expect(within(providerPagination).getByText("Showing 11-12 of 12")).toBeTruthy();

    const profileTable = screen.getByRole("table", { name: /llm profiles/i });
    expect(within(profileTable).getByRole("row", { name: /edit profile profile-01/i })).toBeTruthy();
    expect(within(profileTable).queryByRole("row", { name: /edit profile profile-11/i })).toBeNull();

    const profilePagination = screen.getByRole("navigation", { name: /llm profiles pagination/i });
    fireEvent.click(within(profilePagination).getByRole("button", { name: /next page/i }));

    expect(within(profileTable).queryByRole("row", { name: /edit profile profile-01/i })).toBeNull();
    expect(within(profileTable).getByRole("row", { name: /edit profile profile-11/i })).toBeTruthy();
    expect(within(profilePagination).getByText("Showing 11-12 of 12")).toBeTruthy();
  });

  it("saves full normalized LLM config after provider and profile edits", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });

    fireEvent.click(screen.getByRole("button", { name: /add provider/i }));
    const providerDialog = await screen.findByRole("dialog", { name: /provider/i });
    fireEvent.change(within(providerDialog).getByLabelText(/^provider id$/i), { target: { value: "openai" } });
    fireEvent.change(within(providerDialog).getByLabelText(/^provider type$/i), { target: { value: "openai" } });
    fireEvent.change(within(providerDialog).getByLabelText(/^provider model$/i), { target: { value: "gpt-4.1-mini" } });
    fireEvent.change(within(providerDialog).getByLabelText(/api key secret/i), { target: { value: "OPENAI_API_KEY" } });
    fireEvent.change(within(providerDialog).getByLabelText(/api url/i), { target: { value: "https://api.openai.com/v1" } });
    fireEvent.click(within(providerDialog).getByRole("button", { name: /^save$/i }));

    const profilesCard = screen.getByRole("heading", { name: /^chat profiles$/i }).closest("article");
    expect(profilesCard).toBeTruthy();
    fireEvent.click(within(profilesCard as HTMLElement).getByRole("button", { name: /^add profile$/i }));
    const profileDialog = await screen.findByRole("dialog", { name: /profile/i });
    fireEvent.change(within(profileDialog).getByLabelText(/^profile id$/i), { target: { value: "default" } });
    fireEvent.change(within(profileDialog).getByLabelText(/^profile provider$/i), { target: { value: "openai" } });
    fireEvent.change(within(profileDialog).getByLabelText(/^profile model$/i), { target: { value: "gpt-4.1-mini" } });
    fireEvent.change(within(profileDialog).getByLabelText(/temperature/i), { target: { value: "0.2" } });
    fireEvent.change(within(profileDialog).getByLabelText(/max tokens/i), { target: { value: "4096" } });
    fireEvent.change(within(profileDialog).getByLabelText(/description/i), { target: { value: "Primary runtime profile" } });
    fireEvent.click(within(profileDialog).getByRole("button", { name: /^save$/i }));

    const profileTable = await screen.findByRole("table", { name: /llm profiles/i });
    expect(within(profileTable).getByRole("row", { name: /edit profile default/i })).toBeTruthy();

    fireEvent.change(screen.getByLabelText(/default profile/i), { target: { value: "default" } });
    fireEvent.click(screen.getByRole("button", { name: /save llm configuration/i }));

    await screen.findByText(/llm configuration saved/i);

    const saveCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/llm/config") && init?.method === "PUT";
    });
    expect(saveCall).toBeTruthy();

    const headers = new Headers((saveCall?.[1] as RequestInit | undefined)?.headers);
    expect(headers.get("X-CSRF-Token")).toBe("csrf-test-token");
    const payload = JSON.parse(String((saveCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as Record<string, unknown>;
    expect(payload).toEqual({
      version: "1.0",
      providers: {
        openai: {
          type: "openai",
          api_url: "https://api.openai.com/v1",
          api_key_secret: "OPENAI_API_KEY",
          model: "gpt-4.1-mini",
          extra_headers: {},
          deployment_name: null,
          api_version: null,
        },
      },
      profiles: {
        default: {
          provider: "openai",
          model: "gpt-4.1-mini",
          temperature: 0.2,
          max_tokens: 4096,
          description: "Primary runtime profile",
        },
      },
      default_profile: "default",
    });
  });

  it("opens the provider modal when selecting an existing LLM provider row", async () => {
    installApiMock({
      llmConfig: {
        version: "1.0",
        providers: {
          openai: {
            type: "openai",
            api_url: "https://api.openai.com/v1",
            api_key_secret: "OPENAI_API_KEY",
            model: "gpt-4.1-mini",
          },
        },
        profiles: {},
        default_profile: null,
      },
    });
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });
    const providerTable = screen.getByRole("table", { name: /llm providers/i });
    fireEvent.click(within(providerTable).getByRole("row", { name: /edit provider openai/i }));

    const providerDialog = await screen.findByRole("dialog", { name: /edit provider openai/i });
    expect(within(providerDialog).queryByRole("status")).toBeNull();
    expect(within(providerDialog).getByRole("button", { name: /^save$/i })).toBeTruthy();
    expect(within(providerDialog).queryByRole("button", { name: /clear provider/i })).toBeNull();
    expect((within(providerDialog).getByLabelText(/^provider id$/i) as HTMLInputElement).value).toBe("openai");
    expect((within(providerDialog).getByLabelText(/^provider type$/i) as HTMLInputElement).value).toBe("openai");
    expect((within(providerDialog).getByLabelText(/^provider model$/i) as HTMLInputElement).value).toBe("gpt-4.1-mini");
    const extraHeadersField = within(providerDialog).getByLabelText(/extra headers json/i) as HTMLTextAreaElement;
    fireEvent.change(extraHeadersField, { target: { value: '{"X-Test":"ok"}' } });
    fireEvent.blur(extraHeadersField);
    expect(extraHeadersField.value).toBe('{\n  "X-Test": "ok"\n}');
  });

  it("opens the provider modal when pressing Enter on an existing LLM provider row", async () => {
    installApiMock({
      llmConfig: {
        version: "1.0",
        providers: {
          openai: {
            type: "openai",
            api_url: "https://api.openai.com/v1",
            api_key_secret: "OPENAI_API_KEY",
            model: "gpt-4.1-mini",
          },
        },
        profiles: {},
        default_profile: null,
      },
    });
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });
    const providerTable = screen.getByRole("table", { name: /llm providers/i });
    const providerRow = within(providerTable).getByRole("row", { name: /edit provider openai/i });
    providerRow.focus();
    expect(document.activeElement).toBe(providerRow);
    fireEvent.keyDown(providerRow, { key: "Enter" });

    const providerDialog = await screen.findByRole("dialog", { name: /edit provider openai/i });
    expect((within(providerDialog).getByLabelText(/^provider id$/i) as HTMLInputElement).value).toBe("openai");
    expect(document.activeElement).toBe(within(providerDialog).getByLabelText(/^provider id$/i));
  });

  it("opens the profile modal when selecting an existing LLM profile row", async () => {
    installApiMock({
      llmConfig: {
        version: "1.0",
        providers: { openai: { type: "openai", model: "gpt-4.1-mini" } },
        profiles: {
          default: {
            provider: "openai",
            model: "gpt-4.1-mini",
            temperature: 0.2,
            max_tokens: 4096,
            description: "Primary runtime profile",
          },
        },
        default_profile: "default",
      },
    });
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });
    const profileTable = screen.getByRole("table", { name: /llm profiles/i });
    fireEvent.click(within(profileTable).getByRole("row", { name: /edit profile default/i }));

    const profileDialog = await screen.findByRole("dialog", { name: /edit profile default/i });
    expect(within(profileDialog).queryByRole("status")).toBeNull();
    expect(within(profileDialog).getByRole("button", { name: /^save$/i })).toBeTruthy();
    expect(within(profileDialog).queryByRole("button", { name: /clear profile/i })).toBeNull();
    expect((within(profileDialog).getByLabelText(/^profile id$/i) as HTMLInputElement).value).toBe("default");
    expect((within(profileDialog).getByLabelText(/^profile provider$/i) as HTMLInputElement).value).toBe("openai");
    expect((within(profileDialog).getByLabelText(/^profile model$/i) as HTMLInputElement).value).toBe("gpt-4.1-mini");
    expect((within(profileDialog).getByLabelText(/temperature/i) as HTMLInputElement).value).toBe("0.2");
    expect((within(profileDialog).getByLabelText(/max tokens/i) as HTMLInputElement).value).toBe("4096");
  });

  it("opens the profile modal when pressing Space on an existing LLM profile row", async () => {
    installApiMock({
      llmConfig: {
        version: "1.0",
        providers: { openai: { type: "openai", model: "gpt-4.1-mini" } },
        profiles: { default: { provider: "openai", model: "gpt-4.1-mini", temperature: 0.2, max_tokens: 4096 } },
        default_profile: "default",
      },
    });
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });
    const profileTable = screen.getByRole("table", { name: /llm profiles/i });
    const profileRow = within(profileTable).getByRole("row", { name: /edit profile default/i });
    profileRow.focus();
    expect(document.activeElement).toBe(profileRow);
    fireEvent.keyDown(profileRow, { key: " " });

    const profileDialog = await screen.findByRole("dialog", { name: /edit profile default/i });
    expect((within(profileDialog).getByLabelText(/^profile id$/i) as HTMLInputElement).value).toBe("default");
    expect(document.activeElement).toBe(within(profileDialog).getByLabelText(/^profile id$/i));
  });

  it("keeps keyboard focus inside LLM modals and restores focus to the opener", async () => {
    installApiMock();
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });
    const opener = screen.getByRole("button", { name: /add provider/i });
    opener.focus();
    fireEvent.click(opener);

    const providerDialog = await screen.findByRole("dialog", { name: /provider/i });
    const firstField = within(providerDialog).getByLabelText(/^provider id$/i);
    await waitFor(() => expect(document.activeElement).toBe(firstField));

    const closeButton = within(providerDialog).getByRole("button", { name: /close/i });
    const cancelButton = within(providerDialog).getByRole("button", { name: /cancel/i });
    closeButton.focus();
    fireEvent.keyDown(closeButton, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(cancelButton);

    fireEvent.keyDown(cancelButton, { key: "Tab" });
    expect(document.activeElement).toBe(closeButton);

    fireEvent.keyDown(firstField, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: /provider/i })).toBeNull());
    expect(document.activeElement).toBe(opener);
  });

  it("locks background scrolling while an LLM modal is open and restores it on close", async () => {
    installApiMock();
    document.body.style.overflow = "auto";
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });
    fireEvent.click(screen.getByRole("button", { name: /add provider/i }));

    const providerDialog = await screen.findByRole("dialog", { name: /provider/i });
    expect(document.body.style.overflow).toBe("hidden");

    fireEvent.click(within(providerDialog).getByRole("button", { name: /cancel/i }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: /provider/i })).toBeNull());
    expect(document.body.style.overflow).toBe("auto");
  });

  it("deletes an LLM provider from the edit modal without exposing table actions", async () => {
    installApiMock({
      llmConfig: {
        version: "1.0",
        providers: { openai: { type: "openai", model: "gpt-4.1-mini" } },
        profiles: {},
        default_profile: null,
      },
    });
    renderAtPath("/llm");

    await screen.findByText(/default profile: none/i);
    const providerTable = screen.getByRole("table", { name: /llm providers/i });
    expect(within(providerTable).queryByRole("button", { name: /delete provider openai/i })).toBeNull();
    fireEvent.click(within(providerTable).getByRole("row", { name: /edit provider openai/i }));
    const providerDialog = await screen.findByRole("dialog", { name: /edit provider openai/i });
    const deleteButton = within(providerDialog).getByRole("button", { name: /^delete$/i });
    expect(deleteButton.className).toContain("danger-button");
    fireEvent.click(deleteButton);

    expect(await screen.findByText(/provider openai staged for deletion/i)).toBeTruthy();
    await waitFor(() => expect(screen.queryByRole("dialog", { name: /edit provider openai/i })).toBeNull());
  });

  it("deletes an LLM profile from the edit modal and clears the default profile", async () => {
    const fetchMock = installApiMock({
      llmConfig: {
        version: "1.0",
        providers: { openai: { type: "openai", model: "gpt-4.1-mini" } },
        profiles: { default: { provider: "openai", model: "gpt-4.1-mini", temperature: 0.2, max_tokens: 4096 } },
        default_profile: "default",
      },
    });
    renderAtPath("/llm");

    await screen.findByText(/default profile: default/i);
    const profileTable = screen.getByRole("table", { name: /llm profiles/i });
    expect(within(profileTable).queryByRole("button", { name: /delete profile default/i })).toBeNull();
    fireEvent.click(within(profileTable).getByRole("row", { name: /edit profile default/i }));
    const profileDialog = await screen.findByRole("dialog", { name: /edit profile default/i });
    const deleteButton = within(profileDialog).getByRole("button", { name: /^delete$/i });
    expect(deleteButton.className).toContain("danger-button");
    fireEvent.click(deleteButton);

    expect(await screen.findByText(/profile default staged for deletion/i)).toBeTruthy();
    await waitFor(() => expect(screen.queryByRole("dialog", { name: /edit profile default/i })).toBeNull());
    expect(screen.getByText(/default profile: none/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /save llm configuration/i }));
    await screen.findByText(/llm configuration saved/i);
    const saveCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/llm/config") && init?.method === "PUT";
    });
    const payload = JSON.parse(String((saveCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as Record<string, unknown>;
    expect(payload).toMatchObject({
      profiles: {},
      default_profile: null,
    });
  });

  it("rejects blank LLM profile model before saving configuration", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });

    fireEvent.click(screen.getByRole("button", { name: /add provider/i }));
    const providerDialog = await screen.findByRole("dialog", { name: /provider/i });
    expect(within(providerDialog).queryByRole("button", { name: /^delete$/i })).toBeNull();
    fireEvent.change(within(providerDialog).getByLabelText(/^provider id$/i), { target: { value: "openai" } });
    fireEvent.change(within(providerDialog).getByLabelText(/^provider type$/i), { target: { value: "openai" } });
    fireEvent.change(within(providerDialog).getByLabelText(/^provider model$/i), { target: { value: "gpt-4.1-mini" } });
    fireEvent.click(within(providerDialog).getByRole("button", { name: /^save$/i }));

    const profilesCard = screen.getByRole("heading", { name: /^chat profiles$/i }).closest("article");
    expect(profilesCard).toBeTruthy();
    fireEvent.click(within(profilesCard as HTMLElement).getByRole("button", { name: /^add profile$/i }));
    const profileDialog = await screen.findByRole("dialog", { name: /profile/i });
    expect(within(profileDialog).queryByRole("button", { name: /^delete$/i })).toBeNull();
    fireEvent.change(within(profileDialog).getByLabelText(/^profile id$/i), { target: { value: "default" } });
    fireEvent.change(within(profileDialog).getByLabelText(/^profile provider$/i), { target: { value: "openai" } });
    fireEvent.change(within(profileDialog).getByLabelText(/^profile model$/i), { target: { value: "   " } });
    fireEvent.click(within(profileDialog).getByRole("button", { name: /^save$/i }));

    expect(await within(profileDialog).findByText(/profile model is required/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /save llm configuration/i }));
    const saveCalls = fetchMock.mock.calls.filter((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/llm/config") && init?.method === "PUT";
    });
    expect(saveCalls).toHaveLength(0);
  });

  it("blocks deleting an LLM provider while profiles reference it", async () => {
    const fetchMock = installApiMock({
      llmConfig: {
        version: "1.0",
        providers: { openai: { type: "openai", model: "gpt-4.1-mini" } },
        profiles: { default: { provider: "openai", model: "gpt-4.1-mini" } },
        default_profile: "default",
      },
    });
    renderAtPath("/llm");

    await screen.findByText(/default profile: default/i);
    const providerTable = screen.getByRole("table", { name: /llm providers/i });
    fireEvent.click(within(providerTable).getByRole("row", { name: /edit provider openai/i }));
    const providerDialog = await screen.findByRole("dialog", { name: /edit provider openai/i });
    fireEvent.click(within(providerDialog).getByRole("button", { name: /^delete$/i }));

    await screen.findByText(/remove or reassign profiles first: default/i);
    expect(
      fetchMock.mock.calls.some((call) => {
        const url = String(call[0]);
        const init = call[1] as RequestInit | undefined;
        return url.includes("/api/admin/v1/llm/config") && init?.method === "PUT";
      }),
    ).toBe(false);
  });

  it("supports LLM YAML export, preview, and import migration flow", async () => {
    installApiMock();
    renderAtPath("/llm");

    await screen.findByRole("heading", { name: /llm configuration/i });

    const createObjectUrl = vi.fn(() => "blob:llm-config");
    const revokeObjectUrl = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createObjectUrl });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeObjectUrl });
    const linkClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);

    fireEvent.click(screen.getByRole("button", { name: /export yaml/i }));
    await waitFor(() => expect(createObjectUrl).toHaveBeenCalledWith(expect.any(Blob)));
    await waitFor(() => expect(linkClick).toHaveBeenCalled());
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:llm-config");

    const downloadedLink = linkClick.mock.contexts.find(
      (context): context is HTMLAnchorElement => context instanceof HTMLAnchorElement,
    );
    expect(downloadedLink?.download).toBe("llm-config.yaml");
    expect(downloadedLink?.href).toBe("blob:llm-config");

    fireEvent.click(screen.getByRole("button", { name: /^import yaml$/i }));
    const yamlDialog = await screen.findByRole("dialog", { name: /import yaml/i });
    fireEvent.change(within(yamlDialog).getByLabelText(/yaml import/i), {
      target: { value: "version: '1.0'\nproviders:\n  anthropic:\n    type: anthropic\n" },
    });
    fireEvent.click(within(yamlDialog).getByRole("button", { name: /preview import/i }));
    expect(await screen.findByText(/preview loaded: 1 providers, 1 profiles/i)).toBeTruthy();

    const uploadedYaml = "version: '1.0'\nproviders:\n  uploaded:\n    type: openai\n";
    const upload = new File([uploadedYaml], "llm-config.yaml", { type: "application/x-yaml" });
    fireEvent.change(within(yamlDialog).getByLabelText(/upload yaml file/i), {
      target: { files: [upload] },
    });
    await waitFor(() =>
      expect((within(yamlDialog).getByLabelText(/yaml import/i) as HTMLTextAreaElement).value).toBe(uploadedYaml),
    );

    fireEvent.click(within(yamlDialog).getByRole("button", { name: /^import yaml$/i }));
    expect(await screen.findByText(/llm yaml imported into sqlite-backed config/i)).toBeTruthy();
    const importCall = vi.mocked(fetch).mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.endsWith("/api/admin/v1/llm/import") && init?.method === "POST";
    });
    expect(importCall).toBeTruthy();
    expect(JSON.parse(String((importCall?.[1] as RequestInit | undefined)?.body ?? "{}"))).toEqual({ raw_yaml: uploadedYaml });
    expect(screen.getAllByText(/anthropic/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/default profile: review/i)).toBeTruthy();
  });

  it("routes zero-project setup CTA to the projects page", async () => {
    installApiMock({ projectCount: 0 });
    renderAtPath("/setup");

    const addProjectLink = await screen.findByRole("link", { name: /add first project/i });
    expect(addProjectLink.getAttribute("href")).toBe("/projects");
  });

  it("renders structured database profile fields and saves structured payload", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/database");

    await screen.findByRole("heading", { name: /database settings/i });
    expect(screen.getByLabelText(/^host$/i)).toBeTruthy();
    expect(screen.getByLabelText(/^port$/i)).toBeTruthy();
    expect(screen.getByLabelText(/^database$/i)).toBeTruthy();
    expect(screen.getByLabelText(/^username$/i)).toBeTruthy();
    expect(screen.getByLabelText(/^password$/i)).toBeTruthy();
    expect(screen.getByLabelText(/ssl mode/i)).toBeTruthy();
    expect(screen.getByLabelText(/extra parameters/i)).toBeTruthy();
    expect(screen.getByLabelText(/container name/i)).toBeTruthy();
    expect(screen.getByLabelText(/container image/i)).toBeTruthy();
    expect(screen.getByLabelText(/host port/i)).toBeTruthy();
    expect(screen.getByLabelText(/volume name/i)).toBeTruthy();
    expect(screen.getByLabelText(/advanced dsn import/i)).toBeTruthy();
    expect(screen.getByRole("region", { name: /profile status/i })).toBeTruthy();
    expect(screen.getAllByText(/connection profile/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/runtime container/i).length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: /runtime commands/i })).toBeTruthy();
    expect(screen.getByText(/workflowsctl \/ database-profile \/ local/i)).toBeTruthy();
    expect(screen.getAllByText(/connection profile/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/container command/i)).toBeTruthy();
    expect(screen.getByText(/connection test/i)).toBeTruthy();
    expect(screen.getByText(/persist settings/i)).toBeTruthy();
    expect(screen.queryByRole("checkbox", { name: /enable postgresql/i })).toBeNull();
    expect(screen.getAllByText(/configured: no/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/password: pending/i)).toBeTruthy();
    expect(screen.getByText(/legacy re-entry: clean/i)).toBeTruthy();

    fireEvent.change(screen.getByLabelText(/^host$/i), { target: { value: "127.0.0.1" } });
    fireEvent.change(screen.getByLabelText(/^port$/i), { target: { value: "5432" } });
    fireEvent.change(screen.getByLabelText(/^database$/i), { target: { value: "workflows" } });
    fireEvent.change(screen.getByLabelText(/^username$/i), { target: { value: "wf_user" } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "strong-pass" } });
    fireEvent.click(screen.getByRole("button", { name: /save settings/i }));

    await screen.findByText("Database settings saved.");

    const saveCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/database/settings") && init?.method === "PUT";
    });
    expect(saveCall).toBeTruthy();
    const payload = JSON.parse(String((saveCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as Record<string, unknown>;
    expect(Object.hasOwn(payload, "enabled")).toBe(false);
    expect(payload.host).toBe("127.0.0.1");
    expect(payload.port).toBe(5432);
    expect(payload.database).toBe("workflows");
    expect(payload.username).toBe("wf_user");
    expect(payload.password).toBe("strong-pass");
    expect(payload.password_clear).toBe(false);
    expect(payload.ssl_mode).toBe("prefer");
    expect(payload.container_host_port).toBe(5432);
    expect(payload.dsn_import).toBeNull();
    expect(Object.hasOwn(payload, "dsn")).toBe(false);
  });

  it("blocks save when required structured fields or password are missing", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/database");

    await screen.findByRole("heading", { name: /database settings/i });
    fireEvent.change(screen.getByLabelText(/^host$/i), { target: { value: "127.0.0.1" } });
    fireEvent.change(screen.getByLabelText(/^port$/i), { target: { value: "5432" } });
    fireEvent.change(screen.getByLabelText(/^database$/i), { target: { value: "workflows" } });
    fireEvent.change(screen.getByLabelText(/^username$/i), { target: { value: "wf_user" } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: /save settings/i }));

    await screen.findByText(/password is required unless you keep a configured password/i);
    expect((screen.getByLabelText(/^host$/i) as HTMLInputElement).value).toBe("127.0.0.1");

    const saveCalls = fetchMock.mock.calls.filter((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/database/settings") && init?.method === "PUT";
    });
    expect(saveCalls).toHaveLength(0);
  });

  it("blocks save when container image is empty", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/database");

    await screen.findByRole("heading", { name: /database settings/i });
    fireEvent.change(screen.getByLabelText(/^host$/i), { target: { value: "127.0.0.1" } });
    fireEvent.change(screen.getByLabelText(/^port$/i), { target: { value: "5432" } });
    fireEvent.change(screen.getByLabelText(/^database$/i), { target: { value: "workflows" } });
    fireEvent.change(screen.getByLabelText(/^username$/i), { target: { value: "wf_user" } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "strong-pass" } });
    fireEvent.change(screen.getByLabelText(/container image/i), { target: { value: "   " } });
    fireEvent.click(screen.getByRole("button", { name: /save settings/i }));

    await screen.findByText(/container image is required/i);

    const saveCalls = fetchMock.mock.calls.filter((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/database/settings") && init?.method === "PUT";
    });
    expect(saveCalls).toHaveLength(0);
  });

  it("imports a postgres dsn into fields and keeps raw dsn out of persisted state", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/database");

    await screen.findByRole("heading", { name: /database settings/i });
    fireEvent.change(screen.getByLabelText(/advanced dsn import/i), {
      target: { value: "postgresql://wf_user:typed-pass@db.internal:5433/workflows?sslmode=require&application_name=cli" },
    });
    fireEvent.click(screen.getByRole("button", { name: /import dsn/i }));

    expect((screen.getByLabelText(/^host$/i) as HTMLInputElement).value).toBe("db.internal");
    expect((screen.getByLabelText(/^port$/i) as HTMLInputElement).value).toBe("5433");
    expect((screen.getByLabelText(/^database$/i) as HTMLInputElement).value).toBe("workflows");
    expect((screen.getByLabelText(/^username$/i) as HTMLInputElement).value).toBe("wf_user");
    expect((screen.getByLabelText(/^password$/i) as HTMLInputElement).value).toBe("typed-pass");
    expect((screen.getByLabelText(/ssl mode/i) as HTMLSelectElement).value).toBe("require");
    expect((screen.getByLabelText(/extra parameters/i) as HTMLInputElement).value).toBe("application_name=cli");

    fireEvent.click(screen.getByRole("button", { name: /save settings/i }));
    await screen.findByText("Database settings saved.");

    const saveCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/database/settings") && init?.method === "PUT";
    });
    const payload = JSON.parse(String((saveCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as Record<string, unknown>;
    expect(payload.dsn_import).toBeNull();
    expect(Object.hasOwn(payload, "dsn")).toBe(false);
  });

  it("quotes metacharacter values in command preview and switches to configured-password after reload", async () => {
    installApiMock();
    renderAtPath("/database");

    await screen.findByRole("heading", { name: /database setup guidance/i });
    expect(screen.getAllByText(/<password>/i).length).toBeGreaterThan(0);

    fireEvent.change(screen.getByLabelText(/^username$/i), { target: { value: "wf user;echo" } });
    fireEvent.change(screen.getByLabelText(/^database$/i), { target: { value: "wf$db 'name'" } });
    fireEvent.change(screen.getByLabelText(/container name/i), { target: { value: "wf name;$(id)" } });
    fireEvent.change(screen.getByLabelText(/volume name/i), { target: { value: "wf data && rm -rf /" } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "p@ss ' \" ; $(whoami)" } });

    expect(screen.getAllByText(/POSTGRES_USER='wf user;echo'/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/POSTGRES_DB='wf\$db '"'"'name'"'"''/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/POSTGRES_PASSWORD='p@ss '"'"' " ; \$\(whoami\)'/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/'wf name;\$\(id\)'/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/'wf data && rm -rf \/':\/var\/lib\/postgresql\/data/i).length).toBeGreaterThan(0);

    cleanup();
    installApiMock({
      initialDatabaseSettings: {
        configured: true,
        host: "db.internal",
        port: 5433,
        database: "wf$db 'name'",
        username: "wf user;echo",
        password_configured: true,
        container_name: "wf name;$(id)",
        container_image: "pgvector/pgvector:pg17",
        container_host_port: 5432,
        volume_name: "wf data && rm -rf /",
      },
    });
    renderAtPath("/database");
    await screen.findByRole("heading", { name: /database setup guidance/i });
    await waitFor(() => {
      expect(screen.getAllByText(/POSTGRES_PASSWORD='<configured-password>'/i).length).toBeGreaterThan(0);
    });
    expect(screen.queryByText(/p@ss ' " ; \$\(whoami\)/i)).toBeNull();
  });

  it("shows connection-test status and inline errors with accessible status regions", async () => {
    installApiMock();
    renderAtPath("/database");

    await screen.findByRole("heading", { name: /database settings/i });
    fireEvent.click(screen.getByRole("button", { name: /save settings/i }));

    await screen.findAllByRole("alert");
    fireEvent.click(screen.getByRole("button", { name: /test connection/i }));

    await waitFor(() => {
      expect(screen.getByText(/connection status: degraded/i)).toBeTruthy();
      expect(screen.getByText(/verify host, port, credentials, and network reachability/i)).toBeTruthy();
      expect(screen.getAllByRole("status").length).toBeGreaterThan(0);
    });
  });

  it("shows copy controls and copies docker command with accessible success status", async () => {
    const writeTextMock = vi.fn(async () => undefined);
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: { writeText: writeTextMock },
    });

    installApiMock();
    renderAtPath("/database");
    await screen.findByRole("heading", { name: /database settings/i });

    const dockerPre = screen.getByText(/docker run --name workflows-postgres/i);
    const copyDockerButton = screen.getByRole("button", { name: /copy docker command/i });
    const copyPodmanButton = screen.getByRole("button", { name: /copy podman command/i });
    expect(copyDockerButton).toBeTruthy();
    expect(copyPodmanButton).toBeTruthy();
    expect(copyDockerButton.textContent?.trim()).toBe("");
    expect(copyPodmanButton.textContent?.trim()).toBe("");

    fireEvent.click(copyDockerButton);

    await waitFor(() => {
      expect(writeTextMock).toHaveBeenCalledTimes(1);
      expect(writeTextMock).toHaveBeenCalledWith(dockerPre.textContent ?? "");
    });
    expect(screen.getByRole("status", { name: /clipboard status/i })).toBeTruthy();
    expect(screen.getByText(/docker command copied to clipboard/i)).toBeTruthy();
  });

  it("copies the just-saved password without rendering it in the command preview", async () => {
    const writeTextMock = vi.fn(async (_text: string) => undefined);
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: { writeText: writeTextMock },
    });

    installApiMock();
    renderAtPath("/database");
    await screen.findByRole("heading", { name: /database settings/i });

    fireEvent.change(screen.getByLabelText(/^host$/i), { target: { value: "127.0.0.1" } });
    fireEvent.change(screen.getByLabelText(/^database$/i), { target: { value: "workflows" } });
    fireEvent.change(screen.getByLabelText(/^username$/i), { target: { value: "workflows" } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "saved-pass-123" } });
    fireEvent.click(screen.getByRole("button", { name: /save settings/i }));

    await screen.findByText("Database settings saved.");
    expect((screen.getByLabelText(/^password$/i) as HTMLInputElement).value).toBe("");
    expect(screen.queryByText(/saved-pass-123/i)).toBeNull();
    expect(screen.getAllByText(/<configured-password>/i).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: /copy docker command/i }));

    await waitFor(() => {
      expect(writeTextMock).toHaveBeenCalledTimes(1);
    });
    const copiedCommand = String(writeTextMock.mock.calls[0]?.[0] ?? "");
    expect(copiedCommand).toContain("POSTGRES_PASSWORD=saved-pass-123");
    expect(copiedCommand).not.toContain("<configured-password>");
  });

  it("falls back to textarea copy when async clipboard is unavailable", async () => {
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: undefined,
    });
    const execCommandMock = vi.fn(() => true);
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: execCommandMock,
    });

    renderAtPath("/database");
    await screen.findByRole("heading", { name: /database settings/i });

    fireEvent.click(screen.getByRole("button", { name: /copy docker command/i }));

    await waitFor(() => {
      expect(execCommandMock).toHaveBeenCalledWith("copy");
    });
    expect(screen.getByText(/docker command copied to clipboard/i)).toBeTruthy();
  });

  it("shows accessible error when clipboard copy fails", async () => {
    const writeTextMock = vi.fn(async () => {
      throw new Error("clipboard blocked");
    });
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: { writeText: writeTextMock },
    });
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: vi.fn(() => false),
    });

    renderAtPath("/database");
    await screen.findByRole("heading", { name: /database settings/i });

    fireEvent.click(screen.getByRole("button", { name: /copy podman command/i }));

    await waitFor(() => {
      expect(writeTextMock).toHaveBeenCalledTimes(1);
    });
    const clipboardAlert = await screen.findByText(/unable to copy command to clipboard/i);
    expect(clipboardAlert.getAttribute("role")).toBe("alert");
  });

  it("renders watcher table rows, opens a detail modal, and runs watcher commands", async () => {
    installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
    });
    renderAtPath("/watchers");

    const watcherTable = await screen.findByRole("table", { name: /registered watchers/i });
    const watcherRow = within(watcherTable).getByRole("row", {
      name: /open watcher main project status needs reconcile dirty files 2/i,
    });
    expect(watcherRow.className).toContain("watchers-table-row--warning");

    fireEvent.click(watcherRow);
    const dialog = await screen.findByRole("dialog", { name: /watcher main project/i });
    expect(within(dialog).getByText("Project ID")).toBeTruthy();
    expect(within(dialog).getByText("p1")).toBeTruthy();
    expect(within(dialog).getByText("Enabled")).toBeTruthy();
    expect(within(dialog).getByText("Dirty files")).toBeTruthy();

    fireEvent.click(within(dialog).getByRole("button", { name: /pause watcher p1/i }));

    await waitFor(() => {
      expect(screen.getByText(/watcher p1 paused/i)).toBeTruthy();
      expect(within(dialog).getByText("Paused")).toBeTruthy();
    });
  });

  it("renders watcher table status tones and long project ids without inline action buttons", async () => {
    const longProjectId = "0eeb6dcb-64d5-4383-b81a-127d09346240";
    installApiMock({
      projects: [
        {
          id: longProjectId,
          name: "Main Project",
          slug: "main",
          palace: "x",
          default_wing: "w",
          default_room: "r",
          fs_root: "/tmp",
          fs_allowlist: [],
        },
        {
          id: "secondary-project",
          name: "Secondary Project",
          slug: "secondary",
          palace: "x",
          default_wing: "w",
          default_room: "r",
          fs_root: "/tmp",
          fs_allowlist: [],
        },
      ],
      watchers: [
        {
          project_id: longProjectId,
          state: "enabled",
          dirty_count: 12,
          requires_reconciliation: true,
          last_event_at: "2026-04-30T00:04:00Z",
          updated_at: "2026-04-30T00:05:00Z",
        },
        {
          project_id: "secondary-project",
          state: "paused",
          dirty_count: 0,
          requires_reconciliation: false,
          last_event_at: null,
          updated_at: "2026-04-30T00:06:00Z",
        },
      ],
    });

    renderAtPath("/watchers");

    const watcherTable = await screen.findByRole("table", { name: /registered watchers/i });
    const mainRow = within(watcherTable).getByRole("row", {
      name: new RegExp(`open watcher main project status needs reconcile dirty files 12`, "i"),
    });
    expect(within(mainRow).getByText(longProjectId)).toBeTruthy();
    expect(mainRow.className).toContain("watchers-table-row--warning");

    const secondaryRow = within(watcherTable).getByRole("row", {
      name: /open watcher secondary project status paused dirty files 0/i,
    });
    expect(secondaryRow.className).toContain("watchers-table-row--info");
    expect(within(watcherTable).queryByRole("button", { name: /pause watcher/i })).toBeNull();
  });

  it("shows actionable empty state on watchers and sync pages", async () => {
    installApiMock({ projectCount: 0, watchers: [], syncProjects: [] });

    renderAtPath("/watchers");
    await waitFor(() => {
      expect(screen.getByText(/no watchers available/i)).toBeTruthy();
    });
    expect(screen.getByRole("link", { name: /create a project/i }).getAttribute("href")).toBe("/projects");

    cleanup();
    installApiMock({ projectCount: 0, watchers: [], syncProjects: [] });
    renderAtPath("/sync");
    await waitFor(() => {
      expect(screen.getByText(/no sync queue entries/i)).toBeTruthy();
    });
    expect(screen.getByRole("link", { name: /create a project/i }).getAttribute("href")).toBe("/projects");
  });

  it("renders sync dashboard rows and runs sync now action", async () => {
    const fetchMock = installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
    });
    renderAtPath("/sync");

    const syncTable = await screen.findByRole("table", { name: /registered sync projects/i });
    const syncRow = within(syncTable).getByRole("row", {
      name: /open sync main project status needs reconcile dirty files 2/i,
    });
    expect(syncRow.className).toContain("sync-table-row--warning");

    fireEvent.click(syncRow);
    const dialog = await screen.findByRole("dialog", { name: /sync main project/i });
    expect(within(dialog).getByText("Project ID")).toBeTruthy();
    expect(within(dialog).getByText("p1")).toBeTruthy();
    expect(within(dialog).getByText("Dirty files")).toBeTruthy();

    fireEvent.click(within(dialog).getByRole("button", { name: /sync dirty files p1/i }));

    await waitFor(() => {
      expect(screen.getByText(/dirty file sync queued for p1: job-sync-now/i)).toBeTruthy();
    });

    await waitFor(() => {
      const syncStateCalls = fetchMock.mock.calls.filter((call) => String(call[0]).includes("/api/events/v1/sync/state"));
      expect(syncStateCalls.length).toBeGreaterThan(1);
    });
  });

  it("shows system extraction details in the sync detail dialog", async () => {
    installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [], system2_enabled: true }],
      syncDetails: {
        p1: {
          project_id: "p1",
          system1_enabled: true,
          system1_state: "completed",
          system2_enabled: true,
          system2_state: "completed",
          embedding_profile_required: true,
          embedding_profile: "embedding",
          memory_backend_ready: true,
          counts: {
            source_items: 7,
            structural_evidence: 12,
            verification_cycles: 1,
            wings: 2,
            rooms: 3,
            compartments: 5,
            semantic_claims: 4,
            semantic_memories: 2,
          },
        },
      },
    });
    renderAtPath("/sync");

    const syncTable = await screen.findByRole("table", { name: /registered sync projects/i });
    fireEvent.click(within(syncTable).getByRole("row", { name: /open sync main project status needs reconcile/i }));

    const dialog = await screen.findByRole("dialog", { name: /sync main project/i });
    expect(await within(dialog).findByText(/system 1 structural extraction/i)).toBeTruthy();
    expect(within(dialog).getByText(/system 2 semantic extraction/i)).toBeTruthy();
    expect(within(dialog).getByText(/embedding profile: embedding/i)).toBeTruthy();
    const counts = within(dialog).getByLabelText(/collected memory palace counts/i);
    expect(within(counts).getByText("Wings")).toBeTruthy();
    expect(within(counts).getAllByText("2").length).toBeGreaterThan(0);
    expect(within(counts).getByText("Rooms")).toBeTruthy();
    expect(within(counts).getByText("3")).toBeTruthy();
    expect(within(counts).getByText("Compartments")).toBeTruthy();
    expect(within(counts).getByText("5")).toBeTruthy();
    expect(within(counts).getByText("Memories")).toBeTruthy();
  });

  it("shows sync action failure details returned by the API", async () => {
    installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
      syncActionResponses: {
        "p1:now": {
          project_id: "p1",
          status: "failed",
          dirty_count: 2,
          error: {
            code: "project_system1_sync_failed",
            message: "MEMORY_BACKEND_UNAVAILABLE: no memory PostgreSQL backend is configured.",
          },
        },
      },
    });
    renderAtPath("/sync");

    const syncTable = await screen.findByRole("table", { name: /registered sync projects/i });
    const syncRow = within(syncTable).getByRole("row", {
      name: /open sync main project status needs reconcile dirty files 2/i,
    });
    fireEvent.click(syncRow);

    const dialog = await screen.findByRole("dialog", { name: /sync main project/i });
    fireEvent.click(within(dialog).getByRole("button", { name: /sync dirty files p1/i }));

    await waitFor(() => {
      expect(screen.getByText(/memory_backend_unavailable: no memory postgresql backend is configured/i)).toBeTruthy();
    });
    expect(screen.queryByText(/dirty file sync queued for p1/i)).toBeNull();
  });

  it("renders recent sync activity in the sync detail dialog", async () => {
    const fetchMock = installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
      syncLogs: {
        p1: [
          {
            id: 2,
            project_id: "p1",
            path: "src/app.py",
            event_type: "modified",
            reason: "file_event",
            status: "queued",
            enqueued_at: "2026-04-30T00:12:00Z",
            updated_at: "2026-04-30T00:12:00Z",
            processed_at: null,
          },
          {
            id: 1,
            project_id: "p1",
            path: ".",
            event_type: "reconcile",
            reason: "reconciliation_required:manual_reconcile",
            status: "processed",
            enqueued_at: "2026-04-30T00:05:00Z",
            updated_at: "2026-04-30T00:06:00Z",
            processed_at: "2026-04-30T00:06:00Z",
          },
        ],
      },
    });
    renderAtPath("/sync");

    const syncTable = await screen.findByRole("table", { name: /registered sync projects/i });
    const syncRow = within(syncTable).getByRole("row", {
      name: /open sync main project status needs reconcile dirty files 2/i,
    });
    fireEvent.click(syncRow);

    const dialog = await screen.findByRole("dialog", { name: /sync main project/i });
    const activity = await within(dialog).findByRole("region", { name: /recent sync activity/i });

    expect(within(activity).getByText("src/app.py")).toBeTruthy();
    expect(within(activity).getByText(/file event/i)).toBeTruthy();
    expect(within(activity).getByText(/queued/i)).toBeTruthy();
    expect(within(activity).getByText(/processed/i)).toBeTruthy();
    await waitFor(() => {
      const logCalls = fetchMock.mock.calls.filter((call) => String(call[0]).includes("/api/admin/v1/sync/p1/logs"));
      expect(logCalls.length).toBeGreaterThan(0);
    });
  });

  it("opens sync rows for registered projects and runs targeted commands", async () => {
    const fetchMock = installApiMock({
      projects: [
        { id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp/main", fs_allowlist: [] },
        { id: "p2", name: "Docs Project", slug: "docs", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp/docs", fs_allowlist: [] },
      ],
      syncProjects: [
        { project_id: "p1", dirty_count: 2, requires_reconciliation: true },
        { project_id: "p2", dirty_count: 0, requires_reconciliation: false },
      ],
    });
    renderAtPath("/sync");

    const syncTable = await screen.findByRole("table", { name: /registered sync projects/i });
    const docsRow = within(syncTable).getByRole("row", {
      name: /open sync docs project status clean dirty files 0/i,
    });
    expect(docsRow.className).toContain("sync-table-row--success");

    fireEvent.click(docsRow);
    const dialog = await screen.findByRole("dialog", { name: /sync docs project/i });
    fireEvent.click(within(dialog).getByRole("button", { name: /sync dirty files p2/i }));

    await waitFor(() => {
      const syncCall = fetchMock.mock.calls.find((call) => {
        const url = String(call[0]);
        const init = call[1] as RequestInit | undefined;
        return url.includes("/api/admin/v1/sync/p2/now") && init?.method === "POST";
      });
      expect(syncCall).toBeTruthy();
    });
  });

  it("renders clean registered project in sync dashboard when sync state is empty", async () => {
    const fetchMock = installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
      syncProjects: [],
    });

    renderAtPath("/sync");

    const syncTable = await screen.findByRole("table", { name: /registered sync projects/i });
    const syncRow = within(syncTable).getByRole("row", {
      name: /open sync main project status clean dirty files 0/i,
    });
    expect(within(syncRow).getByText(/idle/i)).toBeTruthy();
    expect(syncRow.className).toContain("sync-table-row--success");

    fireEvent.click(syncRow);
    const dialog = await screen.findByRole("dialog", { name: /sync main project/i });
    const syncDetails = dialog.querySelector('dl[aria-label="Sync details"]');
    expect(syncDetails).toBeTruthy();
    expect(within(syncDetails as HTMLElement).getByText(/^0$/i)).toBeTruthy();
    expect(within(syncDetails as HTMLElement).getByText(/idle/i)).toBeTruthy();
    expect(within(syncDetails as HTMLElement).getByText(/clear/i)).toBeTruthy();
    expect(within(syncDetails as HTMLElement).getByText(/available as manual action/i)).toBeTruthy();
    expect(within(dialog).getByRole("button", { name: /sync dirty files p1/i })).toBeTruthy();
    expect(within(dialog).getByRole("button", { name: /reconcile project p1/i })).toBeTruthy();
    expect(within(dialog).getByRole("button", { name: /rebuild project p1/i })).toBeTruthy();

    expect(screen.queryByText(/no sync queue entries/i)).toBeNull();

    const syncNowCalls = fetchMock.mock.calls.filter((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/sync/p1/now") && init?.method === "POST";
    });
    expect(syncNowCalls).toHaveLength(0);
  });

  it("subscribes to named watcher and sync SSE events", async () => {
    installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
    });
    vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);

    renderAtPath("/watchers");
    await waitFor(() => {
      expect(screen.getByText(/main project/i)).toBeTruthy();
    });

    const watcherInstance = FakeEventSource.instances.find((item) => item.url.includes("/api/events/v1/watchers"));
    expect(watcherInstance).toBeTruthy();
    expect(watcherInstance?.listeners.has("watcher.status")).toBe(true);

    cleanup();
    FakeEventSource.instances = [];

    installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
    });
    vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);

    renderAtPath("/sync");
    await waitFor(() => {
      expect(screen.getByRole("table", { name: /registered sync projects/i })).toBeTruthy();
    });

    const syncInstance = FakeEventSource.instances.find((item) => item.url.includes("/api/events/v1/sync"));
    expect(syncInstance).toBeTruthy();
    expect(syncInstance?.listeners.has("sync.status")).toBe(true);
  });

  it("shows sync fallback warning after EventSource error and keeps manual refresh usable", async () => {
    const fetchMock = installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
    });

    class ErroringEventSource extends FakeEventSource {
      constructor(url: string) {
        super(url);
        queueMicrotask(() => this.emitError());
      }
    }

    vi.stubGlobal("EventSource", ErroringEventSource as unknown as typeof EventSource);

    renderAtPath("/sync");

    await waitFor(() => {
      expect(screen.getByText(/live sync updates are unavailable\. auto-refresh fallback is active\./i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /^refresh$/i }));

    await waitFor(() => {
      const syncStateCalls = fetchMock.mock.calls.filter((call) => String(call[0]).includes("/api/events/v1/sync/state"));
      expect(syncStateCalls.length).toBeGreaterThan(1);
    });
  });

  it("renders explicit sync, reconcile, and rebuild state indicators with honest telemetry copy", async () => {
    installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
      syncProjects: [{ project_id: "p1", dirty_count: 2, requires_reconciliation: true }],
    });

    renderAtPath("/sync");

    const syncTable = await screen.findByRole("table", { name: /registered sync projects/i });
    const syncRow = within(syncTable).getByRole("row", {
      name: /open sync main project status needs reconcile dirty files 2/i,
    });
    fireEvent.click(syncRow);
    const dialog = await screen.findByRole("dialog", { name: /sync main project/i });
    const syncDetails = dialog.querySelector('dl[aria-label="Sync details"]');
    expect(syncDetails).toBeTruthy();

    expect(within(syncDetails as HTMLElement).getByText(/queued/i)).toBeTruthy();
    expect(within(syncDetails as HTMLElement).getByText(/required/i)).toBeTruthy();
    expect(within(syncDetails as HTMLElement).getByText(/available as manual action/i)).toBeTruthy();
    expect(within(syncDetails as HTMLElement).getByText(/not reported by the current sync endpoint/i)).toBeTruthy();
  });

  it("renders registered projects as selectable table rows that open configuration modals", async () => {
    installApiMock({
      projects: [
        {
          id: "p1",
          name: "Workflow Service",
          slug: "workflow-service",
          palace: "wf-palace",
          default_wing: "platform",
          default_room: "runtime",
          fs_root: "/workspace/workflows",
          fs_allowlist: ["/workspace/workflows", "/workspace/shared"],
        },
      ],
    });

    renderAtPath("/projects");

    const projectTable = await screen.findByRole("table", { name: /registered projects/i });
    expect(within(projectTable).getByRole("columnheader", { name: /^project$/i })).toBeTruthy();
    expect(within(projectTable).getByRole("columnheader", { name: /^slug$/i })).toBeTruthy();
    expect(within(projectTable).getByRole("columnheader", { name: /^palace$/i })).toBeTruthy();
    expect(within(projectTable).queryByRole("columnheader", { name: /operations/i })).toBeNull();

    const projectRow = within(projectTable).getByRole("row", {
      name: /open project workflow service configuration/i,
      description: /opens the project configuration dialog/i,
    });
    expect(projectRow.getAttribute("aria-haspopup")).toBe("dialog");
    expect(projectRow.getAttribute("aria-keyshortcuts")).toBe("Enter Space");
    expect(within(projectTable).queryByRole("button", { name: /delete project p1/i })).toBeNull();

    fireEvent.click(projectRow);

    expect(await screen.findByRole("heading", { name: /project workflow service/i })).toBeTruthy();
    expect((screen.getByLabelText(/^name$/i) as HTMLInputElement).value).toBe("Workflow Service");
    expect((screen.getByLabelText(/^slug$/i) as HTMLInputElement).value).toBe("workflow-service");
    expect((screen.getByLabelText(/^palace$/i) as HTMLInputElement).value).toBe("wf-palace");
    expect((screen.getByLabelText(/^fs root$/i, { selector: "input" }) as HTMLInputElement).value).toBe(
      "/workspace/workflows",
    );
    expect((screen.getByLabelText(/^allowlist paths$/i, { selector: "textarea" }) as HTMLTextAreaElement).value).toBe(
      "/workspace/workflows\n/workspace/shared",
    );
  });

  it("opens project configuration from keyboard and saves edits through the project API", async () => {
    const fetchMock = installApiMock({
      projects: [
        {
          id: "p1",
          name: "Workflow Service",
          slug: "workflow-service",
          palace: "wf-palace",
          default_wing: "platform",
          default_room: "runtime",
          fs_root: "/workspace/workflows",
          fs_allowlist: ["/workspace/workflows"],
        },
      ],
    });

    renderAtPath("/projects");

    const projectTable = await screen.findByRole("table", { name: /registered projects/i });
    const projectRow = within(projectTable).getByRole("row", { name: /open project workflow service configuration/i });
    fireEvent.keyDown(projectRow, { key: "Enter" });

    expect(await screen.findByRole("heading", { name: /project workflow service/i })).toBeTruthy();
    fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: "Workflow Platform" } });
    fireEvent.change(screen.getByLabelText(/default room/i), { target: { value: "orchestration" } });
    fireEvent.change(screen.getByLabelText(/^allowlist paths$/i, { selector: "textarea" }), {
      target: { value: "/workspace/workflows\n/workspace/shared" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save project/i }));

    await screen.findByText(/project p1 saved/i);

    const updateCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/projects/p1") && init?.method === "PATCH";
    });
    expect(updateCall).toBeTruthy();
    expect(JSON.parse(String((updateCall?.[1] as RequestInit | undefined)?.body ?? "{}"))).toEqual({
      name: "Workflow Platform",
      slug: "workflow-service",
      palace: "wf-palace",
      default_wing: "platform",
      default_room: "orchestration",
      fs_root: "/workspace/workflows",
      fs_allowlist: ["/workspace/workflows", "/workspace/shared"],
      system2_enabled: false,
    });
  });

  it("paginates registered projects at ten rows per page", async () => {
    installApiMock({ projectCount: 12 });
    renderAtPath("/projects");

    const projectTable = await screen.findByRole("table", { name: /registered projects/i });
    expect(within(projectTable).getByRole("row", { name: /open project p1 configuration/i })).toBeTruthy();
    expect(within(projectTable).getByRole("row", { name: /open project p10 configuration/i })).toBeTruthy();
    expect(within(projectTable).queryByRole("row", { name: /open project p11 configuration/i })).toBeNull();

    const pagination = screen.getByRole("navigation", { name: /registered projects pagination/i });
    expect(within(pagination).getByText("Showing 1-10 of 12")).toBeTruthy();
    fireEvent.click(within(pagination).getByRole("button", { name: /next page/i }));

    expect(within(projectTable).queryByRole("row", { name: /open project p1 configuration/i })).toBeNull();
    expect(within(projectTable).getByRole("row", { name: /open project p11 configuration/i })).toBeTruthy();
    expect(within(pagination).getByText("Showing 11-12 of 12")).toBeTruthy();

    fireEvent.click(within(pagination).getByRole("button", { name: /previous page/i }));
    expect(within(projectTable).getByRole("row", { name: /open project p1 configuration/i })).toBeTruthy();
  });

  it("pre-fills new project onboarding identifiers from the project name", async () => {
    installApiMock();
    renderAtPath("/projects");
    await openNewProjectModal();

    expect((screen.getByLabelText(/default wing/i) as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText(/default room/i) as HTMLInputElement).value).toBe("");

    fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: "Forge Runtime" } });
    expect((screen.getByLabelText(/^slug$/i) as HTMLInputElement).value).toBe("forge-runtime");
    expect((screen.getByLabelText(/^palace$/i) as HTMLInputElement).value).toBe("forge-runtime");

    const summary = screen.getByRole("complementary", { name: /project onboarding summary/i });
    expect(within(summary).getByText("Forge Runtime")).toBeTruthy();
    expect(within(summary).getAllByText("forge-runtime")).toHaveLength(2);

    fireEvent.change(screen.getByLabelText(/^slug$/i), { target: { value: "forge-custom" } });
    expect((screen.getByLabelText(/^palace$/i) as HTMLInputElement).value).toBe("forge-custom");

    fireEvent.change(screen.getByLabelText(/^palace$/i), { target: { value: "manual-palace" } });
    fireEvent.change(screen.getByLabelText(/^slug$/i), { target: { value: "forge-final" } });
    expect((screen.getByLabelText(/^palace$/i) as HTMLInputElement).value).toBe("manual-palace");
  });

  it("shows system extraction settings during project onboarding", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/projects");
    await openNewProjectModal();

    expect(screen.getAllByText(/system 1 structural extraction/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/always on/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/does not use an llm or embedding profile/i)).toBeTruthy();

    const system2Toggle = screen.getByRole("checkbox", { name: /enable system 2 semantic extraction/i }) as HTMLInputElement;
    expect(system2Toggle.checked).toBe(false);
    expect(screen.getByText(/uses the chat llm profile and the embedding profile/i)).toBeTruthy();

    fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: "Semantic Registry" } });
    fireEvent.change(screen.getByLabelText(/^fs root$/i, { selector: "input" }), { target: { value: "/workspace/semantic" } });
    fireEvent.click(system2Toggle);
    fireEvent.click(screen.getByRole("button", { name: /register project/i }));

    await screen.findByText(/project registered successfully/i);

    const createCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/projects") && init?.method === "POST";
    });
    const createBody = JSON.parse(String((createCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as {
      system2_enabled?: boolean;
    };
    expect(createBody.system2_enabled).toBe(true);
  });

  it("creates a palace-level project without default wing or room", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/projects");
    await openNewProjectModal();

    const defaultWingInput = screen.getByLabelText(/default wing/i) as HTMLInputElement;
    const defaultRoomInput = screen.getByLabelText(/default room/i) as HTMLInputElement;
    expect(defaultWingInput.value).toBe("");
    expect(defaultRoomInput.value).toBe("");
    expect(defaultWingInput.required).toBe(false);
    expect(defaultRoomInput.required).toBe(false);

    fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: "Palace Registry" } });
    fireEvent.change(screen.getByLabelText(/^fs root$/i, { selector: "input" }), { target: { value: "/workspace/palace" } });
    fireEvent.click(screen.getByRole("button", { name: /register project/i }));

    await waitFor(() => {
      expect(screen.getByText(/project registered successfully/i)).toBeTruthy();
    });

    const createCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/projects") && init?.method === "POST";
    });
    expect(createCall).toBeTruthy();
    expect(JSON.parse(String((createCall?.[1] as RequestInit | undefined)?.body ?? "{}"))).toEqual({
      name: "Palace Registry",
      slug: "palace-registry",
      palace: "palace-registry",
      default_wing: null,
      default_room: null,
      fs_root: "/workspace/palace",
      fs_allowlist: [],
      system2_enabled: false,
    });
  });

  it("creates a project with normalized allowlist entries", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/projects");
    await openNewProjectModal();

    fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: "Workflow Service" } });
    fireEvent.change(screen.getByLabelText(/^slug$/i), { target: { value: "workflow-service" } });
    fireEvent.change(screen.getByLabelText(/^palace$/i), { target: { value: "wf-palace" } });
    fireEvent.change(screen.getByLabelText(/default wing/i), { target: { value: "platform" } });
    fireEvent.change(screen.getByLabelText(/default room/i), { target: { value: "runtime" } });
    fireEvent.change(screen.getByLabelText(/^fs root$/i, { selector: "input" }), { target: { value: "/workspace/workflows" } });
    fireEvent.change(screen.getByLabelText(/^allowlist paths$/i, { selector: "textarea" }), {
      target: { value: "/workspace/workflows, /workspace/shared\n/workspace/workflows" },
    });
    fireEvent.click(screen.getByRole("button", { name: /register project/i }));

    await waitFor(() => {
      expect(screen.getByText(/project registered successfully/i)).toBeTruthy();
    });

    const createCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/projects") && init?.method === "POST";
    });
    expect(createCall).toBeTruthy();

    const createBody = JSON.parse(String((createCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as {
      fs_allowlist?: string[];
    };
    expect(createBody.fs_allowlist).toEqual(["/workspace/workflows", "/workspace/shared"]);
  });

  it("defaults project fs root to the user home shortcut", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/projects");
    await openNewProjectModal();

    const fsRootInput = await screen.findByLabelText(/^fs root$/i, { selector: "input" });
    expect((fsRootInput as HTMLInputElement).value).toBe("~");

    fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: "Workflow Service" } });
    fireEvent.change(screen.getByLabelText(/^slug$/i), { target: { value: "workflow-service" } });
    fireEvent.change(screen.getByLabelText(/^palace$/i), { target: { value: "wf-palace" } });
    fireEvent.change(screen.getByLabelText(/default wing/i), { target: { value: "platform" } });
    fireEvent.change(screen.getByLabelText(/default room/i), { target: { value: "runtime" } });

    fireEvent.click(screen.getByRole("button", { name: /register project/i }));

    await waitFor(() => {
      expect(screen.getByText(/project registered successfully/i)).toBeTruthy();
    });

    const createCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/projects") && init?.method === "POST";
    });

    expect(createCall).toBeTruthy();
    const createBody = JSON.parse(String((createCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as {
      fs_root?: string;
    };
    expect(createBody.fs_root).toBe("~");
    expect((fsRootInput as HTMLInputElement).value).toBe("~");
  });

  it("preserves create-project payload shape when using browser-selected fs_root", async () => {
    const fetchMock = installApiMock({
      filesystemListingByPath: {
        "~": {
          root: "/srv/workflows",
          path: "/srv/workflows",
          parent: null,
          can_go_up: false,
          entries: [
            { name: "alpha", path: "/srv/workflows/alpha", type: "directory", selectable: true },
            { name: "shared", path: "/srv/workflows/shared", type: "directory", selectable: true },
          ],
        },
        "/srv/workflows/alpha": {
          root: "/srv/workflows",
          path: "/srv/workflows/alpha",
          parent: "/srv/workflows",
          can_go_up: true,
          entries: [],
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: "Workflow Service" } });
    fireEvent.change(screen.getByLabelText(/^slug$/i), { target: { value: "workflow-service" } });
    fireEvent.change(screen.getByLabelText(/^palace$/i), { target: { value: "wf-palace" } });
    fireEvent.change(screen.getByLabelText(/default wing/i), { target: { value: "platform" } });
    fireEvent.change(screen.getByLabelText(/default room/i), { target: { value: "runtime" } });

    fireEvent.click(screen.getByRole("button", { name: /browse fs root/i }));
    fireEvent.click(await screen.findByRole("option", { name: /^alpha\/$/i }));
    fireEvent.click(screen.getByRole("button", { name: /use selection/i }));

    fireEvent.change(screen.getByLabelText(/^allowlist paths$/i, { selector: "textarea" }), {
      target: { value: "/srv/workflows/shared\n/srv/workflows/manual" },
    });

    fireEvent.click(screen.getByRole("button", { name: /register project/i }));

    await waitFor(() => {
      expect(screen.getByText(/project registered successfully/i)).toBeTruthy();
    });

    const projectCreateRequest = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/projects") && init?.method === "POST";
    });

    expect(projectCreateRequest).toBeTruthy();
    expect(JSON.parse(String(projectCreateRequest?.[1] && (projectCreateRequest[1] as RequestInit).body))).toEqual({
      name: "Workflow Service",
      slug: "workflow-service",
      palace: "wf-palace",
      default_wing: "platform",
      default_room: "runtime",
      fs_root: "/srv/workflows/alpha",
      fs_allowlist: ["/srv/workflows/shared", "/srv/workflows/manual"],
      system2_enabled: false,
    });
  });

  it("replaces fs root from the server folder browser", async () => {
    const fetchMock = installApiMock({
      filesystemListingByPath: {
        "~": {
          root: "/srv/workflows",
          path: "/srv/workflows",
          parent: null,
          can_go_up: false,
          entries: [{ name: "alpha", path: "/srv/workflows/alpha", type: "directory", selectable: true }],
        },
        "/srv/workflows/alpha": {
          root: "/srv/workflows",
          path: "/srv/workflows/alpha",
          parent: "/srv/workflows",
          can_go_up: true,
          entries: [],
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    const browseButton = await screen.findByRole("button", { name: /browse fs root/i });
    fireEvent.click(browseButton);

    expect(await screen.findByRole("heading", { name: /browse server folders/i })).toBeTruthy();

    fireEvent.click(await screen.findByRole("option", { name: /^alpha\/$/i }));

    fireEvent.click(screen.getByRole("button", { name: /use selection/i }));

    const fsRootInput = screen.getByLabelText(/^fs root$/i);
    await waitFor(() => {
      expect((fsRootInput as HTMLInputElement).value).toBe("/srv/workflows/alpha");
      expect(screen.queryByRole("heading", { name: /browse server folders/i })).toBeNull();
      expect(document.activeElement).toBe(fsRootInput);
      expect(screen.getByText("FS root set to /srv/workflows/alpha.")).toBeTruthy();
    });

    expect(
      fetchMock.mock.calls.some((call) => {
        const requestUrl = new URL(String(call[0]), "https://admin.example.test");
        return requestUrl.pathname === "/api/admin/v1/filesystem/entries";
      }),
    ).toBe(true);
  });

  it("renders icon-only browse buttons with accessible names", async () => {
    installApiMock();
    renderAtPath("/projects");
    await openNewProjectModal();

    expect(await screen.findByRole("button", { name: "Browse FS root" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Browse allowlist paths" })).toBeTruthy();
    expect(screen.queryByText("Browse FS root")).toBeNull();
    expect(screen.queryByText("Browse allowlist paths")).toBeNull();
  });

  it("uses backend default start path when fs root is blank", async () => {
    const fetchMock = installApiMock({
      filesystemListingByPath: {
        "": {
          root: "/srv/workflows",
          path: "/srv/workflows",
          parent: null,
          can_go_up: false,
          entries: [{ name: "alpha", path: "/srv/workflows/alpha", type: "directory", selectable: true }],
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    fireEvent.change(await screen.findByLabelText(/^fs root$/i), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: /browse fs root/i }));

    await screen.findByRole("heading", { name: /browse server folders/i });
    fireEvent.click(await screen.findByRole("option", { name: /^alpha\/$/i }));
    fireEvent.click(screen.getByRole("button", { name: /use selection/i }));

    expect(
      fetchMock.mock.calls.some((call) => {
        const requestUrl = new URL(String(call[0]), "https://admin.example.test");
        return requestUrl.pathname === "/api/admin/v1/filesystem/entries" && !requestUrl.searchParams.has("path");
      }),
    ).toBe(true);
  });

  it("shows fallback warning when prefilled fs root start path is invalid and preserves manual value", async () => {
    installApiMock({
      filesystemListingByPath: {
        "": {
          root: "/srv/workflows",
          path: "/srv/workflows",
          parent: null,
          can_go_up: false,
          entries: [{ name: "shared", path: "/srv/workflows/shared", type: "directory", selectable: true }],
        },
      },
      filesystemErrorByPath: {
        "/srv/workflows/missing": {
          status: 404,
          body: { message: "path not found" },
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    const fsRootInput = await screen.findByLabelText(/^fs root$/i);
    fireEvent.change(fsRootInput, { target: { value: "/srv/workflows/missing" } });

    fireEvent.click(screen.getByRole("button", { name: /browse fs root/i }));

    expect(
      await screen.findByText("Could not open /srv/workflows/missing; showing the default browsing root."),
    ).toBeTruthy();
    expect((fsRootInput as HTMLInputElement).value).toBe("/srv/workflows/missing");
  });

  it("uses prefilled fs root as picker start path", async () => {
    const fetchMock = installApiMock({
      filesystemListingByPath: {
        "/srv/workflows/prefilled": {
          root: "/srv/workflows",
          path: "/srv/workflows/prefilled",
          parent: "/srv/workflows",
          can_go_up: true,
          entries: [],
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    fireEvent.change(await screen.findByLabelText(/^fs root$/i), { target: { value: "/srv/workflows/prefilled" } });
    fireEvent.click(screen.getByRole("button", { name: /browse fs root/i }));

    await screen.findByText("/srv/workflows/prefilled");

    expect(
      fetchMock.mock.calls.some((call) => {
        const requestUrl = new URL(String(call[0]), "https://admin.example.test");
        return (
          requestUrl.pathname === "/api/admin/v1/filesystem/entries" &&
          requestUrl.searchParams.get("path") === "/srv/workflows/prefilled"
        );
      }),
    ).toBe(true);
  });

  it("appends selected folder to allowlist and preserves manual entries", async () => {
    installApiMock({
      filesystemListingByPath: {
        "": {
          root: "/srv/workflows",
          path: "/srv/workflows",
          parent: null,
          can_go_up: false,
          entries: [{ name: "shared", path: "/srv/workflows/shared", type: "directory", selectable: true }],
        },
        "/srv/workflows/shared": {
          root: "/srv/workflows",
          path: "/srv/workflows/shared",
          parent: "/srv/workflows",
          can_go_up: true,
          entries: [],
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    const allowlistInput = await screen.findByLabelText(/^allowlist paths$/i, { selector: "textarea" });
    fireEvent.change(allowlistInput, { target: { value: "/srv/workflows/manual" } });

    fireEvent.click(screen.getByRole("button", { name: /browse allowlist paths/i }));
    expect(await screen.findByRole("heading", { name: /browse server folders/i })).toBeTruthy();

    fireEvent.click(screen.getByRole("option", { name: /^shared\/$/i }));

    fireEvent.click(screen.getByRole("button", { name: /use selection/i }));

    await waitFor(() => {
      expect((allowlistInput as HTMLTextAreaElement).value).toBe("/srv/workflows/manual\n/srv/workflows/shared");
      expect(screen.getByText("Added /srv/workflows/shared to allowlist paths.")).toBeTruthy();
      expect(screen.queryByRole("heading", { name: /browse server folders/i })).toBeNull();
      expect(document.activeElement).toBe(allowlistInput);
    });
  });

  it("does not append duplicate allowlist path after trimming", async () => {
    installApiMock({
      filesystemListingByPath: {
        "": {
          root: "/srv/workflows",
          path: "/srv/workflows",
          parent: null,
          can_go_up: false,
          entries: [{ name: "shared", path: "/srv/workflows/shared", type: "directory", selectable: true }],
        },
        "/srv/workflows/shared": {
          root: "/srv/workflows",
          path: "/srv/workflows/shared",
          parent: "/srv/workflows",
          can_go_up: true,
          entries: [],
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    const allowlistInput = await screen.findByLabelText(/^allowlist paths$/i, { selector: "textarea" });
    fireEvent.change(allowlistInput, { target: { value: "  /srv/workflows/shared " } });

    fireEvent.click(screen.getByRole("button", { name: /browse allowlist paths/i }));
    expect(await screen.findByRole("heading", { name: /browse server folders/i })).toBeTruthy();

    fireEvent.click(screen.getByRole("option", { name: /^shared\/$/i }));

    fireEvent.click(screen.getByRole("button", { name: /use selection/i }));

    await waitFor(() => {
      expect((allowlistInput as HTMLTextAreaElement).value).toBe("  /srv/workflows/shared ");
      expect(screen.getByText("/srv/workflows/shared is already in allowlist paths.")).toBeTruthy();
      expect(document.activeElement).toBe(allowlistInput);
    });
  });

  it("keeps use-selection disabled and shows empty-folder copy", async () => {
    installApiMock({
      filesystemListingByPath: {
        "": {
          root: "/srv/workflows",
          path: "/srv/workflows",
          parent: null,
          can_go_up: false,
          entries: [],
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    fireEvent.click(await screen.findByRole("button", { name: /browse fs root/i }));
    expect(await screen.findByText("No entries in this directory.")).toBeTruthy();

    const useFolderButton = screen.getByRole("button", { name: /use selection/i });
    expect(useFolderButton).toBeTruthy();
    expect((useFolderButton as HTMLButtonElement).disabled).toBe(true);
  });

  it("loads root listing when filesystem root is system root", async () => {
    installApiMock({
      filesystemListingByPath: {
        "": {
          root: "/",
          path: "/",
          parent: null,
          can_go_up: false,
          entries: [{ name: "tmp", path: "/tmp", type: "directory", selectable: true }],
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    fireEvent.click(await screen.findByRole("button", { name: /browse fs root/i }));

    expect(await screen.findByRole("heading", { name: /browse server folders/i })).toBeTruthy();
  });

  it("keeps cancel available after initial listing error and restores fs-root focus", async () => {
    installApiMock({
      filesystemErrorByPath: {
        "~": {
          status: 409,
          body: {
            detail: {
              code: "filesystem_browsing_root_missing",
              message: "Server folder browsing requires WORKFLOWS_SCAN_ROOT to be configured.",
            },
          },
        },
        "": {
          status: 409,
          body: {
            detail: {
              code: "filesystem_browsing_root_missing",
              message: "Server folder browsing requires WORKFLOWS_SCAN_ROOT to be configured.",
            },
          },
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    const fsRootInput = await screen.findByLabelText(/^fs root$/i);
    fireEvent.click(screen.getByRole("button", { name: /browse fs root/i }));

    expect(await screen.findByText("Unable to load directory listing.")).toBeTruthy();

    const cancelButton = screen.getByRole("button", { name: /^cancel$/i });
    fireEvent.click(cancelButton);

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: /browse server folders/i })).toBeNull();
      expect(document.activeElement).toBe(fsRootInput);
    });
  });

  it("returns focus to allowlist textarea when browser is canceled", async () => {
    installApiMock({
      filesystemListingByPath: {
        "": {
          root: "/srv/workflows",
          path: "/srv/workflows",
          parent: null,
          can_go_up: false,
          entries: [{ name: "shared", path: "/srv/workflows/shared", type: "directory", selectable: true }],
        },
      },
    });

    renderAtPath("/projects");
    await openNewProjectModal();

    const allowlistInput = await screen.findByLabelText(/^allowlist paths$/i, { selector: "textarea" });
    fireEvent.click(screen.getByRole("button", { name: /browse allowlist paths/i }));
    expect(await screen.findByRole("heading", { name: /browse server folders/i })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: /browse server folders/i })).toBeNull();
      expect(document.activeElement).toBe(allowlistInput);
    });
  });

  it("requires explicit confirmation before deleting a project", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/projects");

    const projectTable = await screen.findByRole("table", { name: /registered projects/i });
    const projectRow = within(projectTable).getByRole("row", { name: /open project p1 configuration/i });
    fireEvent.click(projectRow);

    expect(screen.getByText(/type delete to confirm removing p1/i)).toBeTruthy();
    expect(
      fetchMock.mock.calls.some((call) => {
        const url = String(call[0]);
        const init = call[1] as RequestInit | undefined;
        return url.includes("/api/admin/v1/projects/p1") && init?.method === "DELETE";
      }),
    ).toBe(false);

    fireEvent.change(screen.getByLabelText(/confirm deletion/i), { target: { value: "DELETE" } });
    fireEvent.click(screen.getByRole("button", { name: /confirm delete p1/i }));

    await waitFor(() => {
      expect(screen.getByText(/project p1 deleted/i)).toBeTruthy();
    });

    expect(
      fetchMock.mock.calls.some((call) => {
        const url = String(call[0]);
        const init = call[1] as RequestInit | undefined;
        return url.includes("/api/admin/v1/projects/p1") && init?.method === "DELETE";
      }),
    ).toBe(true);
  });

  it("renders watcher/default hints from project API payload when present", async () => {
    installApiMock({
      projects: [
        {
          id: "p1",
          name: "Workflow Service",
          slug: "workflow-service",
          palace: "wf-palace",
          default_wing: "platform",
          default_room: "runtime",
          fs_root: "/workspace/workflows",
          fs_allowlist: ["/workspace/workflows"],
          watcher_hint: "Watcher state defaults to enabled.",
          default_state_hint: "Default state routes to platform/runtime.",
          created_at: "",
          updated_at: "",
        },
      ],
    });
    renderAtPath("/projects");

    const projectTable = await screen.findByRole("table", { name: /registered projects/i });
    const projectRow = within(projectTable).getByRole("row", { name: /open project workflow service configuration/i });
    fireEvent.click(projectRow);

    expect(await screen.findByText(/watcher state defaults to enabled\./i)).toBeTruthy();
    expect(screen.getByText(/default state routes to platform\/runtime\./i)).toBeTruthy();
    expect(screen.queryByText(/watcher state hints are unavailable from the current projects api response\./i)).toBeNull();
  });

  it("renders honest fallback copy when watcher/default hints are absent", async () => {
    installApiMock({
      projects: [
        {
          id: "p1",
          name: "Workflow Service",
          slug: "workflow-service",
          palace: "wf-palace",
          default_wing: "platform",
          default_room: "runtime",
          fs_root: "/workspace/workflows",
          fs_allowlist: ["/workspace/workflows"],
          created_at: "",
          updated_at: "",
        },
      ],
    });
    renderAtPath("/projects");

    const projectTable = await screen.findByRole("table", { name: /registered projects/i });
    const projectRow = within(projectTable).getByRole("row", { name: /open project workflow service configuration/i });
    fireEvent.click(projectRow);

    expect(await screen.findByText(/watcher state hints are unavailable from the current projects api response\./i)).toBeTruthy();
    expect(screen.getByText(/default state hints are unavailable from the current projects api response\./i)).toBeTruthy();
  });

  it("renders MCP clients in a selectable table and opens creation from New", async () => {
    installApiMock({
      mcpClients: [
        {
          id: "t1",
          label: "ci-agent",
          project_ids: ["p1"],
          created_at: "2026-04-30T00:00:00Z",
          last_used_at: null,
          revoked_at: null,
        },
      ],
    });
    renderAtPath("/mcp-clients");

    const table = await screen.findByRole("table", { name: /registered mcp clients/i });
    expect(screen.queryByRole("list", { name: /mcp clients list/i })).toBeNull();
    expect(screen.queryByLabelText(/^client label$/i)).toBeNull();

    const row = within(table).getByRole("row", { name: /open mcp client ci-agent details/i });
    fireEvent.keyDown(row, { key: "Enter" });

    expect(await screen.findByRole("dialog", { name: /mcp client ci-agent/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /regenerate/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /revoke/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /delete/i })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /close dialog/i }));
    fireEvent.click(screen.getByRole("button", { name: /^new$/i }));

    expect(await screen.findByRole("dialog", { name: /new mcp client/i })).toBeTruthy();
    expect(screen.getByLabelText(/^client label$/i)).toBeTruthy();
    const createButton = screen.getByRole("button", { name: /create mcp client/i });
    expect(createButton).toBeTruthy();
    expect(createButton.hasAttribute("disabled")).toBe(false);
  });

  it("shows one-time mcp token after creation and hides it after reload", async () => {
    installApiMock();
    renderAtPath("/mcp-clients");

    await openNewMcpClientModal();

    fireEvent.change(screen.getByLabelText(/^client label$/i), { target: { value: "ci-agent" } });
    fireEvent.change(screen.getByLabelText(/project access/i, { selector: "select" }), { target: { value: "p1" } });
    fireEvent.click(screen.getByRole("button", { name: /create mcp client/i }));

    await waitFor(() => {
      expect(screen.getByText(/copy once: token and snippet/i)).toBeTruthy();
      expect(screen.getAllByText(/mcp-secret-token-create/i).length).toBeGreaterThan(0);
    });

    fireEvent.click(screen.getByRole("button", { name: /reload mcp clients/i }));

    await waitFor(() => {
      expect(screen.queryByText(/mcp-secret-token-create/i)).toBeNull();
    });
  });

  it("creates MCP clients from registered project selections", async () => {
    const fetchMock = installApiMock({
      projects: [
        { id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp/main", fs_allowlist: [] },
        { id: "p2", name: "Docs Project", slug: "docs", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp/docs", fs_allowlist: [] },
      ],
    });
    renderAtPath("/mcp-clients");

    await openNewMcpClientModal();

    fireEvent.change(screen.getByLabelText(/^client label$/i), { target: { value: "docs-agent" } });
    fireEvent.change(screen.getByLabelText(/project access/i, { selector: "select" }), { target: { value: "p2" } });
    fireEvent.click(screen.getByRole("button", { name: /create mcp client/i }));

    await waitFor(() => {
      const createCall = fetchMock.mock.calls.find((call) => {
        const url = String(call[0]);
        const init = call[1] as RequestInit | undefined;
        return url.includes("/api/admin/v1/mcp-clients") && init?.method === "POST";
      });
      expect(createCall).toBeTruthy();
      const body = JSON.parse(String((createCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as Record<string, unknown>;
      expect(body.project_ids).toEqual(["p2"]);
    });
  });

  it("creates MCP clients without project bindings by default", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/mcp-clients");

    await openNewMcpClientModal();

    fireEvent.change(screen.getByLabelText(/^client label$/i), { target: { value: "client-selects-project" } });
    fireEvent.click(screen.getByRole("button", { name: /create mcp client/i }));

    await waitFor(() => {
      const createCall = fetchMock.mock.calls.find((call) => {
        const url = String(call[0]);
        const init = call[1] as RequestInit | undefined;
        return url.includes("/api/admin/v1/mcp-clients") && init?.method === "POST";
      });
      expect(createCall).toBeTruthy();
      const body = JSON.parse(String((createCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as Record<string, unknown>;
      expect(body.project_ids).toEqual([]);
    });
  });

  it("keeps form label in token card when create response omits label", async () => {
    installApiMock({ createMcpClientResponse: { label: undefined } });
    renderAtPath("/mcp-clients");

    await openNewMcpClientModal();

    fireEvent.change(screen.getByLabelText(/^client label$/i), { target: { value: "ci-agent" } });
    fireEvent.change(screen.getByLabelText(/project access/i, { selector: "select" }), { target: { value: "p1" } });
    fireEvent.click(screen.getByRole("button", { name: /create mcp client/i }));

    await waitFor(() => {
      expect(screen.getByText(/copy once: token and snippet/i)).toBeTruthy();
      expect(screen.getAllByText(/mcp-secret-token-create/i).length).toBeGreaterThan(0);
      expect(screen.getByText("Client label: ci-agent")).toBeTruthy();
    });
  });

  it("shows readable error and no success card when create response is missing token", async () => {
    installApiMock({ createMcpClientResponse: { token: "" } });
    renderAtPath("/mcp-clients");

    await openNewMcpClientModal();

    fireEvent.change(screen.getByLabelText(/^client label$/i), { target: { value: "ci-agent" } });
    fireEvent.change(screen.getByLabelText(/project access/i, { selector: "select" }), { target: { value: "p1" } });
    fireEvent.click(screen.getByRole("button", { name: /create mcp client/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/token issuance response was incomplete; no secret was returned\./i),
      ).toBeTruthy();
    });

    expect(screen.queryByText(/mcp client created\. save the token now/i)).toBeNull();
    expect(screen.queryByText(/copy once: token and snippet/i)).toBeNull();
  });

  it("shows readable error and no success card when regenerate response is missing token", async () => {
    installApiMock({
      regenerateMcpClientResponse: { token: "" },
      mcpClients: [
        {
          id: "t1",
          label: "ci-agent",
          project_ids: ["p1"],
          created_at: "2026-04-30T00:00:00Z",
          last_used_at: null,
          revoked_at: null,
        },
      ],
    });
    renderAtPath("/mcp-clients");

    await openMcpClientDetailModal();

    fireEvent.click(screen.getByRole("button", { name: /regenerate/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/token issuance response was incomplete; no secret was returned\./i),
      ).toBeTruthy();
    });

    expect(screen.queryByText(/mcp token regenerated\. save the new token now/i)).toBeNull();
    expect(screen.queryByText(/copy once: token and snippet/i)).toBeNull();
  });

  it("shows regenerated one-time MCP token inside the open detail dialog", async () => {
    installApiMock({
      mcpClients: [
        {
          id: "t1",
          label: "ci-agent",
          project_ids: ["p1"],
          created_at: "2026-04-30T00:00:00Z",
          last_used_at: null,
          revoked_at: null,
        },
      ],
    });
    renderAtPath("/mcp-clients");

    await openMcpClientDetailModal();
    const dialog = screen.getByRole("dialog", { name: /mcp client ci-agent/i });

    fireEvent.click(within(dialog).getByRole("button", { name: /regenerate/i }));

    await waitFor(() => {
      expect(within(dialog).getByText(/copy once: token and snippet/i)).toBeTruthy();
      expect(within(dialog).getAllByText(/mcp-secret-token-regenerated/i).length).toBeGreaterThan(0);
    });
    const tokenCard = within(dialog).getByText(/copy once: token and snippet/i).closest(".token-card");
    expect(tokenCard).toBeTruthy();
    expect(document.activeElement).toBe(tokenCard);
  });

  it("falls back to token id when regenerate response label is empty", async () => {
    installApiMock({
      regenerateMcpClientResponse: { label: "   " },
      mcpClients: [
        {
          id: "t1",
          label: "ci-agent",
          project_ids: ["p1"],
          created_at: "2026-04-30T00:00:00Z",
          last_used_at: null,
          revoked_at: null,
        },
      ],
    });
    renderAtPath("/mcp-clients");

    await openMcpClientDetailModal();

    fireEvent.click(screen.getByRole("button", { name: /regenerate/i }));

    await waitFor(() => {
      expect(screen.getByText(/mcp token regenerated\. save the new token now/i)).toBeTruthy();
      expect(screen.getByText("Client label: t1")).toBeTruthy();
    });
  });

  it("updates MCP client project access from the detail dialog", async () => {
    const fetchMock = installApiMock({
      projects: [
        { id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp/main", fs_allowlist: [] },
        { id: "p2", name: "Docs Project", slug: "docs", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp/docs", fs_allowlist: [] },
      ],
      mcpClients: [
        {
          id: "t1",
          label: "ci-agent",
          project_ids: ["p1"],
          created_at: "2026-04-30T00:00:00Z",
          last_used_at: null,
          revoked_at: null,
        },
      ],
    });
    renderAtPath("/mcp-clients");

    await openMcpClientDetailModal();
    const dialog = screen.getByRole("dialog", { name: /mcp client ci-agent/i });
    fireEvent.change(within(dialog).getByLabelText(/allowed projects/i, { selector: "select" }), {
      target: { value: "p2" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: /save project access/i }));

    await waitFor(() => {
      expect(screen.getByText(/mcp client t1 project access updated/i)).toBeTruthy();
    });
    const patchCall = fetchMock.mock.calls.find((call) => {
      const init = call[1] as RequestInit | undefined;
      return String(call[0]).endsWith("/api/admin/v1/mcp-clients/t1") && init?.method === "PATCH";
    });
    expect(patchCall).toBeTruthy();
    const body = JSON.parse(String((patchCall?.[1] as RequestInit | undefined)?.body ?? "{}")) as Record<string, unknown>;
    expect(body.project_ids).toEqual(["p2"]);
  });

  it("deletes an MCP client from the detail dialog and removes it from the table", async () => {
    const fetchMock = installApiMock({
      mcpClients: [
        {
          id: "t1",
          label: "ci-agent",
          project_ids: ["p1"],
          created_at: "2026-04-30T00:00:00Z",
          last_used_at: null,
          revoked_at: "2026-04-30T00:10:00Z",
        },
      ],
    });
    renderAtPath("/mcp-clients");

    await openMcpClientDetailModal();
    fireEvent.click(screen.getByRole("button", { name: /delete/i }));

    await waitFor(() => {
      expect(screen.getByText(/mcp client t1 deleted/i)).toBeTruthy();
    });
    expect(screen.queryByRole("dialog", { name: /mcp client ci-agent/i })).toBeNull();
    expect(screen.queryByRole("row", { name: /open mcp client ci-agent details/i })).toBeNull();

    const deleteCall = fetchMock.mock.calls.find((call) => {
      const init = call[1] as RequestInit | undefined;
      return String(call[0]).endsWith("/api/admin/v1/mcp-clients/t1/registration") && init?.method === "DELETE";
    });
    expect(deleteCall).toBeTruthy();
  });

  it("renders the execution recorder, filters runs, and handles detail/cancel/resume actions", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/runs");

    const runsTable = await screen.findByRole("table", { name: /workflow execution runs/i });
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /execution recorder/i })).toBeTruthy();
      expect(within(runsTable).getByText(/python-ci-pipeline/i)).toBeTruthy();
      expect(within(runsTable).getByText(/run-1/i)).toBeTruthy();
      expect(within(runsTable).getByText(/deploy-gate/i)).toBeTruthy();
      expect(screen.getByText(/total match/i)).toBeTruthy();
      expect(screen.queryByRole("button", { name: /inspect run/i })).toBeNull();
      expect(screen.getByRole("button", { name: /^reload$/i })).toBeTruthy();
    });

    fireEvent.click(within(runsTable).getByRole("row", { name: /open run python-ci-pipeline detail/i }));
    const detailDialog = await screen.findByRole("dialog", { name: /run python-ci-pipeline/i });
    await waitFor(() => {
      expect(within(detailDialog).getAllByText(/run detail/i).length).toBeGreaterThan(0);
      expect(within(detailDialog).getAllByText(/required_param is required/i).length).toBeGreaterThan(0);
      expect(within(detailDialog).getAllByText(/validate_inputs/i).length).toBeGreaterThan(0);
      expect(within(detailDialog).getByText(/technical json/i)).toBeTruthy();
    });
    fireEvent.click(within(detailDialog).getByRole("button", { name: /close dialog/i }));

    fireEvent.change(screen.getByLabelText(/status filter/i), { target: { value: "completed" } });
    fireEvent.change(screen.getByLabelText(/^mode$/i), { target: { value: "async" } });
    fireEvent.change(screen.getByLabelText(/^workflow$/i), { target: { value: "node-ci-pipeline" } });
    fireEvent.change(screen.getByLabelText(/^limit$/i), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText(/^offset$/i), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: /^apply$/i }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find((entry) =>
        String(entry[0]).includes("/api/admin/v1/runs?status=completed&mode=async&workflow=node-ci-pipeline&limit=1&offset=0"),
      );
      expect(call).toBeTruthy();
    });

    await waitFor(() => {
      expect(screen.getByText(/run-3/i)).toBeTruthy();
    });

    fireEvent.change(screen.getByLabelText(/status filter/i), { target: { value: "paused" } });
    fireEvent.change(screen.getByLabelText(/^workflow$/i), { target: { value: "deploy-gate" } });
    fireEvent.change(screen.getByLabelText(/project id/i), { target: { value: "p1" } });
    fireEvent.click(screen.getByRole("button", { name: /^apply$/i }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find((entry) =>
        String(entry[0]).includes("/api/admin/v1/runs?status=paused&mode=async&workflow=deploy-gate&project_id=p1&limit=1&offset=0"),
      );
      expect(call).toBeTruthy();
    });

    const filteredRunsTable = screen.getByRole("table", { name: /workflow execution runs/i });
    fireEvent.click(within(filteredRunsTable).getByRole("row", { name: /open run deploy-gate detail/i }));
    const pausedDialog = await screen.findByRole("dialog", { name: /run deploy-gate/i });

    fireEvent.click(within(pausedDialog).getByRole("button", { name: /cancel run/i }));
    await waitFor(() => {
      expect(screen.getByText(/run run-2 cancellation requested/i)).toBeTruthy();
    });

    fireEvent.click(within(pausedDialog).getByRole("button", { name: /resume run/i }));
    const resumeDialog = await screen.findByRole("dialog", { name: /resume run-2/i });
    fireEvent.change(within(resumeDialog).getByLabelText(/response/i), { target: { value: "approved" } });
    fireEvent.click(within(resumeDialog).getByRole("button", { name: /submit resume/i }));

    await waitFor(() => {
      expect(screen.getByText(/run run-2 resume submitted/i)).toBeTruthy();
    });

    const resumeCall = fetchMock.mock.calls.find((entry) => {
      const url = String(entry[0]);
      const init = entry[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/runs/run-2/resume") && init?.method === "POST";
    });
    expect(resumeCall).toBeTruthy();
    expect(JSON.parse(String((resumeCall?.[1] as RequestInit | undefined)?.body ?? "{}"))).toEqual({ response: "approved" });
  });

  it("syntax-highlights JSON diagnostics and run inputs in the detail dialog", async () => {
    installApiMock({
      runDetailById: {
        "run-1": {
          run_id: "run-1",
          job_id: "job-1",
          workflow_name: "book-designer",
          status: "completed",
          execution_mode: "sync",
          created_at: "2026-04-30T00:00:00Z",
          started_at: "2026-04-30T00:01:00Z",
          finished_at: "2026-04-30T00:01:01Z",
          updated_at: "2026-04-30T00:01:01Z",
          duration_ms: 67,
          cancellable: false,
          project_id: "p1",
          token_id: "t1",
          result_summary: null,
          error_summary: null,
          inputs: {
            book_name: "The Adventures of Blinky the Bunny",
            topic: "Friendship and sharing",
            characters: 8,
          },
          outputs: {
            _error: "Output evaluation failed",
          },
          error: JSON.stringify({
            outputs: {
              _error: "Output evaluation failed: Failed to evaluate expression",
            },
            _note: "Check debug log for partial execution details",
          }),
          metadata: { workflow_name: "book-designer" },
          blocks: [],
          technical_json: {
            status: "completed",
            inputs: {
              book_name: "The Adventures of Blinky the Bunny",
              topic: "Friendship and sharing",
              characters: 8,
            },
          },
        },
      },
    });
    renderAtPath("/runs");

    const runsTable = await screen.findByRole("table", { name: /workflow execution runs/i });
    fireEvent.click(within(runsTable).getByRole("row", { name: /open run python-ci-pipeline detail/i }));
    const detailDialog = await screen.findByRole("dialog", { name: /run book-designer/i });

    const diagnosticSection = within(detailDialog).getByText("Diagnostic").closest("section");
    const inputsSection = within(detailDialog).getByText("Inputs").closest("section");
    expect(diagnosticSection?.querySelector(".runs-json-code")).toBeTruthy();
    expect(inputsSection?.querySelector(".runs-json-code")).toBeTruthy();
    expect(diagnosticSection?.querySelector(".json-token--key")?.textContent).toBe("\"outputs\"");
    expect(
      Array.from(inputsSection?.querySelectorAll(".json-token--key") ?? []).some((token) => token.textContent === "\"book_name\""),
    ).toBe(true);
    expect(
      Array.from(inputsSection?.querySelectorAll(".json-token--string") ?? []).some((token) =>
        token.textContent?.includes("The Adventures of Blinky the Bunny"),
      ),
    ).toBe(true);
    expect(
      Array.from(inputsSection?.querySelectorAll(".json-token--number") ?? []).some((token) => token.textContent === "8"),
    ).toBe(true);
  });

  it("shows clear runs empty copy", async () => {
    installApiMock({ runs: [] });
    renderAtPath("/runs");

    await waitFor(() => {
      expect(screen.getByText(/no runs match this view/i)).toBeTruthy();
      expect(screen.getByText(/records them in sqlite/i)).toBeTruthy();
      expect(screen.getByRole("button", { name: /^reload$/i })).toBeTruthy();
    });
  });

  it("shows readable runs errors for cancel and resume failures", async () => {
    installApiMock({ cancelRunStatus: 409, cancelRunErrorMessage: "Run is not cancellable", resumeRunStatus: 409, resumeRunErrorMessage: "Run is not resumable" });
    renderAtPath("/runs");

    const runsTable = await screen.findByRole("table", { name: /workflow execution runs/i });
    fireEvent.click(within(runsTable).getByRole("row", { name: /open run deploy-gate detail/i }));
    const detailDialog = await screen.findByRole("dialog", { name: /run deploy-gate/i });
    fireEvent.click(within(detailDialog).getByRole("button", { name: /cancel run/i }));
    await waitFor(() => {
      expect(screen.getByText(/run is not cancellable/i)).toBeTruthy();
    });

    fireEvent.click(within(detailDialog).getByRole("button", { name: /resume run/i }));
    const dialog = await screen.findByRole("dialog", { name: /resume run-2/i });
    fireEvent.change(within(dialog).getByLabelText(/response/i), { target: { value: "retry" } });
    fireEvent.click(within(dialog).getByRole("button", { name: /submit resume/i }));
    await waitFor(() => {
      expect(screen.getByText(/run is not resumable/i)).toBeTruthy();
    });
  });

  it("does not submit resume when the resume modal is cancelled", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/runs");

    const runsTable = await screen.findByRole("table", { name: /workflow execution runs/i });
    fireEvent.click(within(runsTable).getByRole("row", { name: /open run deploy-gate detail/i }));
    const detailDialog = await screen.findByRole("dialog", { name: /run deploy-gate/i });
    fireEvent.click(within(detailDialog).getByRole("button", { name: /resume run/i }));

    const dialog = await screen.findByRole("dialog", { name: /resume run-2/i });
    fireEvent.click(within(dialog).getByRole("button", { name: /^cancel$/i }));

    const resumeCall = fetchMock.mock.calls.find((entry) => {
      const url = String(entry[0]);
      const init = entry[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/runs/run-2/resume") && init?.method === "POST";
    });

    expect(resumeCall).toBeUndefined();
    expect(screen.queryByText(/run run-2 resume submitted/i)).toBeNull();
  });
});
