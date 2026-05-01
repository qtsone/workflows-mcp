import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

afterEach(() => {
  cleanup();
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
    workflows: customWorkflows,
    workflowSources: customWorkflowSources,
    workflowSchema: customWorkflowSchema,
    workflowReloadStatus = 200,
    workflowReloadErrorMessage = "reload failed due to duplicate names",
    workflowValidateStatus = 200,
    workflowValidateErrorMessage = "validation failed due to invalid YAML",
    runs: customRuns,
    runDetailById: customRunDetailById,
    cancelRunStatus = 200,
    cancelRunErrorMessage = "cancel failed",
    resumeRunStatus = 200,
    resumeRunErrorMessage = "resume failed",
    filesystemListingByPath,
    filesystemErrorByPath,
    initialDatabaseSettings,
  }: {
    projectCount?: number;
    projects?: Array<Record<string, unknown>>;
    createMcpClientResponse?: Record<string, unknown>;
    regenerateMcpClientResponse?: Record<string, unknown>;
    mcpClients?: Array<Record<string, unknown>>;
    watchers?: Array<Record<string, unknown>>;
    syncProjects?: Array<Record<string, unknown>>;
    workflows?: Array<Record<string, unknown>>;
    workflowSources?: Array<Record<string, unknown>>;
    workflowSchema?: Record<string, unknown>;
    workflowReloadStatus?: number;
    workflowReloadErrorMessage?: string;
    workflowValidateStatus?: number;
    workflowValidateErrorMessage?: string;
    runs?: Array<Record<string, unknown>>;
    runDetailById?: Record<string, Record<string, unknown>>;
    cancelRunStatus?: number;
    cancelRunErrorMessage?: string;
    resumeRunStatus?: number;
    resumeRunErrorMessage?: string;
    filesystemListingByPath?: Record<string, Record<string, unknown>>;
    filesystemErrorByPath?: Record<string, { status: number; body: Record<string, unknown> }>;
    initialDatabaseSettings?: Record<string, unknown>;
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
        status: "paused",
        created_at: "2026-04-30T00:00:00Z",
        started_at: "2026-04-30T00:01:00Z",
        finished_at: null,
        updated_at: "2026-04-30T00:02:00Z",
        cancellable: true,
        project_id: "p1",
        token_id: "t1",
      },
      {
        run_id: "run-2",
        job_id: "job-2",
        workflow_name: "node-ci-pipeline",
        status: "completed",
        created_at: "2026-04-30T01:00:00Z",
        started_at: "2026-04-30T01:01:00Z",
        finished_at: "2026-04-30T01:04:00Z",
        updated_at: "2026-04-30T01:04:00Z",
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
        status: "paused",
        created_at: "2026-04-30T00:00:00Z",
        started_at: "2026-04-30T00:01:00Z",
        finished_at: null,
        updated_at: "2026-04-30T00:02:00Z",
        cancellable: true,
        project_id: "p1",
        token_id: "t1",
        result_summary: "Awaiting user input",
        error_summary: null,
        metadata: { prompt: "Approve deployment?", gate: "prod" },
      },
    };
  const pathEntriesListingByPath: Record<string, Record<string, unknown>> = filesystemListingByPath ?? {};
  const pathEntriesErrorByPath = filesystemErrorByPath ?? {};
  let databaseSettings: Record<string, unknown> = {
    enabled: false,
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
        configured: payload.enabled === true,
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
      return jsonResponse({ version: "1.0", providers: {}, profiles: {} });
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
        created_at: "2026-04-30T00:00:00Z",
        updated_at: "2026-04-30T00:00:00Z",
      };
      projects.push(created);
      return jsonResponse(created, 201);
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
    if (/\/api\/admin\/v1\/sync\/[^/]+\/now$/.test(url) && method === "POST") {
      const projectId = url.split("/").at(-2) ?? "";
      const target = syncProjects.find((project) => project.project_id === projectId);
      if (!target) return jsonResponse({ message: "Project not found" }, 404);
      target.dirty_count = Number(target.dirty_count ?? 0) + 1;
      return jsonResponse({ project_id: projectId, status: "queued", dirty_count: target.dirty_count });
    }
    if (/\/api\/admin\/v1\/sync\/[^/]+\/reconcile$/.test(url) && method === "POST") {
      const projectId = url.split("/").at(-2) ?? "";
      const target = syncProjects.find((project) => project.project_id === projectId);
      if (!target) return jsonResponse({ message: "Project not found" }, 404);
      target.requires_reconciliation = true;
      return jsonResponse({ project_id: projectId, dirty_count: target.dirty_count, requires_reconciliation: true });
    }
    if (/\/api\/admin\/v1\/sync\/[^/]+\/rebuild$/.test(url) && method === "POST") {
      const projectId = url.split("/").at(-2) ?? "";
      const target = syncProjects.find((project) => project.project_id === projectId);
      if (!target) return jsonResponse({ message: "Project not found" }, 404);
      target.requires_reconciliation = true;
      return jsonResponse({ project_id: projectId, dirty_count: target.dirty_count, requires_reconciliation: true });
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
    if (url.includes("/api/admin/v1/mcp-clients/") && method === "DELETE") {
      const tokenId = url.split("/").pop() ?? "";
      const index = mcpClients.findIndex((client) => client.id === tokenId);
      if (index < 0) {
        return jsonResponse({ message: "Token not found" }, 404);
      }
      mcpClients.splice(index, 1);
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
    if (/\/api\/admin\/v1\/workflows\/sources\/[^/]+\/validate$/.test(url) && method === "POST") {
      if (workflowValidateStatus !== 200) {
        return jsonResponse({ message: workflowValidateErrorMessage }, workflowValidateStatus);
      }
      return jsonResponse({ valid: true, workflow_names: ["python-ci-pipeline"], total: 1 });
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
      const limit = Number(parsed.searchParams.get("limit") ?? "50");
      const offset = Number(parsed.searchParams.get("offset") ?? "0");
      const filtered = status ? runs.filter((run) => run.status === status) : runs;
      const paged = filtered.slice(offset, offset + limit);
      return jsonResponse({ runs: paged });
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

  it("renders workflows and sources, and supports add/validate/delete/reload operations", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/workflows");

    await waitFor(() => {
      expect(screen.getByText(/python-ci-pipeline/i)).toBeTruthy();
      expect(screen.getByText(/runs lint and test checks/i)).toBeTruthy();
      expect(screen.getByText(/version: 1.2.0/i)).toBeTruthy();
      expect(screen.getByText(/tags: python, ci/i)).toBeTruthy();
      expect(screen.getByText(/project: p1/i)).toBeTruthy();
      expect(screen.getByText("Source path: /workspace/workflows/python-ci.yaml")).toBeTruthy();
      expect(screen.getByText("Path: /workspace/workflows")).toBeTruthy();
      expect(screen.getByRole("button", { name: /load schema/i })).toBeTruthy();
      expect(screen.getByRole("button", { name: /reload workflows/i })).toBeTruthy();
    });

    fireEvent.change(screen.getByLabelText(/project id/i), { target: { value: "p1" } });
    fireEvent.change(screen.getByLabelText(/source path/i), { target: { value: "/workspace/more-workflows" } });
    fireEvent.click(screen.getByRole("button", { name: /add workflow source/i }));

    await waitFor(() => {
      expect(screen.getByText(/workflow source added for project p1/i)).toBeTruthy();
      expect(screen.getByText(/path: \/workspace\/more-workflows/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /validate src-1/i }));
    await waitFor(() => {
      expect(screen.getByText(/validation passed for src-1/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /delete src-1/i }));
    await waitFor(() => {
      expect(screen.getByText(/source src-1 deleted/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /load schema/i }));
    await waitFor(() => {
      expect(screen.getByText(/workflow schema loaded/i)).toBeTruthy();
      expect(screen.getByText(/workflowschema/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /view details/i }));
    await waitFor(() => {
      expect(screen.getByText(/workflow details for python-ci-pipeline/i)).toBeTruthy();
      expect(screen.getByText(/"id": "lint"/i)).toBeTruthy();
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

  it("shows actionable workflows empty state when projects and sources are missing", async () => {
    installApiMock({ projectCount: 0, workflows: [], workflowSources: [] });
    renderAtPath("/workflows");

    await waitFor(() => {
      expect(screen.getByText(/no workflow sources configured/i)).toBeTruthy();
    });

    expect(screen.getByRole("link", { name: /create a project first/i }).getAttribute("href")).toBe("/projects");
    expect(screen.getByText(/once a project exists, add a workflow source path below/i)).toBeTruthy();
  });

  it("shows readable errors for validate and reload failures", async () => {
    installApiMock({
      workflowReloadStatus: 422,
      workflowReloadErrorMessage: "workflow reload failed because duplicate workflow names were detected",
      workflowValidateStatus: 422,
      workflowValidateErrorMessage: "source validation failed: invalid definition",
    });
    renderAtPath("/workflows");

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /validate src-1/i })).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /validate src-1/i }));
    await waitFor(() => {
      expect(screen.getByText(/unable to validate source src-1: source validation failed: invalid definition/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /reload workflows/i }));
    await waitFor(() => {
      expect(
        screen.getByText(/unable to reload workflows: workflow reload failed because duplicate workflow names were detected/i),
      ).toBeTruthy();
    });
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
    expect(screen.getByText(/^no$/i)).toBeTruthy();
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
    expect(screen.queryByRole("link", { name: /add first project/i })).toBeNull();
    expect(screen.getByRole("link", { name: /create mcp client token/i }).getAttribute("href")).toBe(
      "/mcp-clients",
    );
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
    expect(screen.getByText(/workflowsctl \/ database-profile \/ local/i)).toBeTruthy();
    expect(screen.getByText(/connection profile/i)).toBeTruthy();
    expect(screen.getByText(/container command/i)).toBeTruthy();
    expect(screen.getByText(/connection test/i)).toBeTruthy();
    expect(screen.getByText(/persist settings/i)).toBeTruthy();
    expect(screen.getByText(/enabled: off/i)).toBeTruthy();
    expect(screen.getAllByText(/configured: no/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/password: pending/i)).toBeTruthy();
    expect(screen.getByText(/legacy re-entry: clean/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("checkbox", { name: /enable postgresql metadata backend/i }));
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
    expect(payload.enabled).toBe(true);
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

  it("blocks enabled save when required structured fields or password are missing", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/database");

    await screen.findByRole("heading", { name: /database settings/i });
    fireEvent.click(screen.getByRole("checkbox", { name: /enable postgresql metadata backend/i }));
    fireEvent.change(screen.getByLabelText(/^host$/i), { target: { value: "127.0.0.1" } });
    fireEvent.change(screen.getByLabelText(/^port$/i), { target: { value: "5432" } });
    fireEvent.change(screen.getByLabelText(/^database$/i), { target: { value: "workflows" } });
    fireEvent.change(screen.getByLabelText(/^username$/i), { target: { value: "wf_user" } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: /save settings/i }));

    await screen.findByText(/password is required unless you clear it/i);
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
    fireEvent.click(screen.getByRole("checkbox", { name: /enable postgresql metadata backend/i }));
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

    fireEvent.click(screen.getByRole("checkbox", { name: /enable postgresql metadata backend/i }));
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
        enabled: true,
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
    fireEvent.click(screen.getByRole("checkbox", { name: /enable postgresql metadata backend/i }));
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

    renderAtPath("/database");
    await screen.findByRole("heading", { name: /database settings/i });

    const dockerPre = screen.getByText(/docker run --name workflows-postgres/i);
    expect(screen.getByRole("button", { name: /copy docker command/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /copy podman command/i })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /copy docker command/i }));

    await waitFor(() => {
      expect(writeTextMock).toHaveBeenCalledTimes(1);
      expect(writeTextMock).toHaveBeenCalledWith(dockerPre.textContent ?? "");
    });
    expect(screen.getByRole("status", { name: /clipboard status/i })).toBeTruthy();
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

    renderAtPath("/database");
    await screen.findByRole("heading", { name: /database settings/i });

    fireEvent.click(screen.getByRole("button", { name: /copy podman command/i }));

    await waitFor(() => {
      expect(writeTextMock).toHaveBeenCalledTimes(1);
    });
    expect(screen.getByRole("alert")).toBeTruthy();
    expect(screen.getByText(/unable to copy command to clipboard/i)).toBeTruthy();
  });

  it("renders watcher dashboard rows and allows pause action", async () => {
    installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
    });
    renderAtPath("/watchers");

    await waitFor(() => {
      expect(screen.getByText(/main project/i)).toBeTruthy();
      expect(screen.getByText(/state: enabled/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /pause watcher p1/i }));

    await waitFor(() => {
      expect(screen.getByText(/watcher p1 paused/i)).toBeTruthy();
      expect(screen.getByText(/state: paused/i)).toBeTruthy();
    });
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

    await waitFor(() => {
      expect(screen.getByText(/main project/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /sync now p1/i }));

    await waitFor(() => {
      expect(screen.getByText(/sync requested for p1/i)).toBeTruthy();
    });

    await waitFor(() => {
      const syncStateCalls = fetchMock.mock.calls.filter((call) => String(call[0]).includes("/api/events/v1/sync/state"));
      expect(syncStateCalls.length).toBeGreaterThan(1);
    });
  });

  it("renders clean registered project in sync dashboard when sync state is empty", async () => {
    const fetchMock = installApiMock({
      projects: [{ id: "p1", name: "Main Project", slug: "main", palace: "x", default_wing: "w", default_room: "r", fs_root: "/tmp", fs_allowlist: [] }],
      syncProjects: [],
    });

    renderAtPath("/sync");

    await waitFor(() => {
      expect(screen.getByText(/main project/i)).toBeTruthy();
      expect(screen.getByText(/dirty count: 0/i)).toBeTruthy();
      expect(screen.getByText(/sync state: idle/i)).toBeTruthy();
      expect(screen.getByText(/reconcile state: clear/i)).toBeTruthy();
      expect(screen.getByText(/rebuild state: available as manual action/i)).toBeTruthy();
      expect(screen.getByRole("button", { name: /sync now p1/i })).toBeTruthy();
      expect(screen.getByRole("button", { name: /reconcile p1/i })).toBeTruthy();
      expect(screen.getByRole("button", { name: /rebuild p1/i })).toBeTruthy();
    });

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
      expect(screen.getByText(/main project/i)).toBeTruthy();
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

    await waitFor(() => {
      expect(screen.getByText(/sync state: queued/i)).toBeTruthy();
      expect(screen.getByText(/reconcile state: required/i)).toBeTruthy();
      expect(screen.getByText(/rebuild state: available as manual action/i)).toBeTruthy();
      expect(screen.getByText(/lifecycle telemetry: not reported by the current sync endpoint/i)).toBeTruthy();
    });
  });

  it("creates a project with normalized allowlist entries", async () => {
    const fetchMock = installApiMock();
    renderAtPath("/projects");

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /register project/i })).toBeTruthy();
    });

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

    await screen.findByRole("button", { name: /register project/i });

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

    const deleteButton = await screen.findByRole("button", { name: /delete project p1/i });
    fireEvent.click(deleteButton);

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

    expect(await screen.findByText(/watcher state hints are unavailable from the current projects api response\./i)).toBeTruthy();
    expect(screen.getByText(/default state hints are unavailable from the current projects api response\./i)).toBeTruthy();
  });

  it("shows one-time mcp token after creation and hides it after reload", async () => {
    installApiMock();
    renderAtPath("/mcp-clients");

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /create mcp client/i })).toBeTruthy();
    });

    fireEvent.change(screen.getByLabelText(/^client label$/i), { target: { value: "ci-agent" } });
    fireEvent.change(screen.getByLabelText(/project ids/i), { target: { value: "p1" } });
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

  it("keeps form label in token card when create response omits label", async () => {
    installApiMock({ createMcpClientResponse: { label: undefined } });
    renderAtPath("/mcp-clients");

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /create mcp client/i })).toBeTruthy();
    });

    fireEvent.change(screen.getByLabelText(/^client label$/i), { target: { value: "ci-agent" } });
    fireEvent.change(screen.getByLabelText(/project ids/i), { target: { value: "p1" } });
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

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /create mcp client/i })).toBeTruthy();
    });

    fireEvent.change(screen.getByLabelText(/^client label$/i), { target: { value: "ci-agent" } });
    fireEvent.change(screen.getByLabelText(/project ids/i), { target: { value: "p1" } });
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

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /regenerate/i })).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /regenerate/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/token issuance response was incomplete; no secret was returned\./i),
      ).toBeTruthy();
    });

    expect(screen.queryByText(/mcp token regenerated\. save the new token now/i)).toBeNull();
    expect(screen.queryByText(/copy once: token and snippet/i)).toBeNull();
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

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /regenerate/i })).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /regenerate/i }));

    await waitFor(() => {
      expect(screen.getByText(/mcp token regenerated\. save the new token now/i)).toBeTruthy();
      expect(screen.getByText("Client label: t1")).toBeTruthy();
    });
  });

  it("renders runs list, supports status/limit/offset query, and handles detail/cancel/resume actions", async () => {
    const fetchMock = installApiMock();
    const promptSpy = vi.spyOn(window, "prompt").mockReturnValue("approved");
    renderAtPath("/runs");

    await waitFor(() => {
      expect(screen.getByText(/python-ci-pipeline/i)).toBeTruthy();
      expect(screen.getByText(/run id: run-1/i)).toBeTruthy();
      expect(screen.getByText(/job id: job-1/i)).toBeTruthy();
      expect(screen.getByText(/project id: p1/i)).toBeTruthy();
      expect(screen.getByText(/token id: t1/i)).toBeTruthy();
      expect(screen.getByText(/cancellable: yes/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /view run run-1/i }));
    await waitFor(() => {
      expect(screen.getByText(/run detail: run-1/i)).toBeTruthy();
      expect(screen.getByText(/result summary: awaiting user input/i)).toBeTruthy();
      expect(screen.getByText(/error summary: none/i)).toBeTruthy();
      expect(screen.getByText(/metadata keys: prompt, gate/i)).toBeTruthy();
      expect(screen.getByText(/technical json/i)).toBeTruthy();
    });

    fireEvent.change(screen.getByLabelText(/status filter/i), { target: { value: "completed" } });
    fireEvent.change(screen.getByLabelText(/^limit$/i), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText(/^offset$/i), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: /apply run filters/i }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find((entry) => String(entry[0]).includes("/api/admin/v1/runs?status=completed&limit=1&offset=0"));
      expect(call).toBeTruthy();
    });

    await waitFor(() => {
      expect(screen.getByText(/run id: run-2/i)).toBeTruthy();
      expect(screen.queryByRole("button", { name: /cancel run run-2/i })).toBeNull();
    });

    fireEvent.change(screen.getByLabelText(/status filter/i), { target: { value: "paused" } });
    fireEvent.click(screen.getByRole("button", { name: /apply run filters/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /cancel run run-1/i })).toBeTruthy();
      expect(screen.getByRole("button", { name: /resume run run-1/i })).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /cancel run run-1/i }));
    await waitFor(() => {
      expect(screen.getByText(/run run-1 cancellation requested/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /resume run run-1/i }));
    await waitFor(() => {
      expect(promptSpy).toHaveBeenCalled();
      expect(screen.getByText(/run run-1 resume submitted/i)).toBeTruthy();
    });

    const resumeCall = fetchMock.mock.calls.find((entry) => {
      const url = String(entry[0]);
      const init = entry[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/runs/run-1/resume") && init?.method === "POST";
    });
    expect(resumeCall).toBeTruthy();
    expect(JSON.parse(String((resumeCall?.[1] as RequestInit | undefined)?.body ?? "{}"))).toEqual({ response: "approved" });
  });

  it("shows clear runs empty and filter limitation copy", async () => {
    installApiMock({ runs: [] });
    renderAtPath("/runs");

    await waitFor(() => {
      expect(screen.getByText(/project and workflow filters are not available yet/i)).toBeTruthy();
      expect(screen.getByText(/no runs found for the current filter/i)).toBeTruthy();
      expect(screen.getByRole("button", { name: /reload runs/i })).toBeTruthy();
    });
  });

  it("shows readable runs errors for load, cancel, and resume failures", async () => {
    installApiMock({ cancelRunStatus: 409, cancelRunErrorMessage: "Run is not cancellable", resumeRunStatus: 409, resumeRunErrorMessage: "Run is not resumable" });
    vi.spyOn(window, "prompt").mockReturnValue("retry");
    renderAtPath("/runs");

    const cancelButton = await screen.findByRole("button", { name: /cancel run run-1/i });
    fireEvent.click(cancelButton);
    await waitFor(() => {
      expect(screen.getByText(/run is not cancellable/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /resume run run-1/i }));
    await waitFor(() => {
      expect(screen.getByText(/run is not resumable/i)).toBeTruthy();
    });
  });

  it("does not submit resume when prompt is cancelled", async () => {
    const fetchMock = installApiMock();
    const promptSpy = vi.spyOn(window, "prompt").mockReturnValue(null);
    renderAtPath("/runs");

    const resumeButton = await screen.findByRole("button", { name: /resume run run-1/i });
    fireEvent.click(resumeButton);

    await waitFor(() => {
      expect(promptSpy).toHaveBeenCalled();
    });

    const resumeCall = fetchMock.mock.calls.find((entry) => {
      const url = String(entry[0]);
      const init = entry[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/runs/run-1/resume") && init?.method === "POST";
    });

    expect(resumeCall).toBeUndefined();
    expect(screen.queryByText(/run run-1 resume submitted/i)).toBeNull();
  });
});
