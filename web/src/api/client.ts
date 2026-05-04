import createClient from "openapi-fetch";

import type { paths } from "./generated/schema";

type HttpMethod = "GET" | "HEAD" | "OPTIONS" | "POST" | "PUT" | "PATCH" | "DELETE";

type CsrfTokenProvider = () => string | null | undefined;

export interface CreateApiFetchOptions {
  baseUrl?: string;
  csrfTokenProvider?: CsrfTokenProvider;
  onUnauthorized?: (error: ApiHttpError) => void;
  fetchImpl?: typeof fetch;
}

export interface ApiErrorPayload {
  code?: string;
  message?: string;
  [key: string]: unknown;
}

export type DatabaseSslMode = "disable" | "prefer" | "require" | "verify-ca" | "verify-full";

export interface SaveDatabaseSettingsPayload {
  enabled: boolean;
  host: string;
  port: number;
  database: string;
  username: string;
  password?: string | null;
  password_clear: boolean;
  ssl_mode: DatabaseSslMode;
  extra_params: string;
  container_name: string;
  container_image: string;
  container_host_port: number;
  volume_name: string;
  dsn_import: string | null;
}

export class ApiHttpError extends Error {
  readonly name = "ApiHttpError";
  readonly status: number;
  readonly statusText: string;
  readonly code?: string;
  readonly body: unknown;

  constructor(params: {
    status: number;
    statusText: string;
    message: string;
    code?: string;
    body: unknown;
  }) {
    super(params.message);
    this.status = params.status;
    this.statusText = params.statusText;
    this.code = params.code;
    this.body = params.body;
  }
}

export class ApiUrlPolicyError extends Error {
  readonly name = "ApiUrlPolicyError";
  readonly inputUrl: string;

  constructor(message: string, inputUrl: string) {
    super(message);
    this.inputUrl = inputUrl;
  }
}

const SAFE_METHODS: ReadonlySet<HttpMethod> = new Set(["GET", "HEAD", "OPTIONS"]);

function isAbsoluteUrl(url: string): boolean {
  return /^[a-z][a-z\d+\-.]*:\/\//i.test(url);
}

function isProtocolRelativeUrl(url: string): boolean {
  return url.startsWith("//");
}

function isRelativeUrl(url: string): boolean {
  return !isAbsoluteUrl(url) && !isProtocolRelativeUrl(url);
}

function getLocationOrigin(): string | undefined {
  if (typeof globalThis.location?.origin === "string" && globalThis.location.origin.length > 0) {
    return globalThis.location.origin;
  }
  return undefined;
}

function resolveAndValidateUrl(baseUrl: string | undefined, input: RequestInfo | URL): string {
  const inputString = typeof input === "string" ? input : input.toString();

  if (isProtocolRelativeUrl(inputString)) {
    throw new ApiUrlPolicyError("Protocol-relative URLs are not allowed", inputString);
  }

  if (!baseUrl) {
    if (isRelativeUrl(inputString)) {
      return inputString;
    }

    const locationOrigin = getLocationOrigin();
    if (!locationOrigin) {
      throw new ApiUrlPolicyError(
        "Absolute URLs are not allowed without a configured baseUrl",
        inputString,
      );
    }

    const resolved = new URL(inputString);
    if (resolved.origin !== locationOrigin) {
      throw new ApiUrlPolicyError("Cross-origin URLs are not allowed", inputString);
    }
    return resolved.toString();
  }

  const baseOrigin = new URL(baseUrl).origin;
  const resolved = new URL(inputString, baseUrl);
  if (resolved.origin !== baseOrigin) {
    throw new ApiUrlPolicyError("Cross-origin URLs are not allowed", inputString);
  }
  return resolved.toString();
}

function methodFrom(init: RequestInit | undefined): HttpMethod {
  return (init?.method?.toUpperCase() ?? "GET") as HttpMethod;
}

function normalizeErrorPayload(body: unknown): ApiErrorPayload | undefined {
  if (!body || typeof body !== "object") {
    return undefined;
  }
  const obj = body as Record<string, unknown>;
  if (typeof obj.detail === "string") {
    return { message: obj.detail };
  }
  if (Array.isArray(obj.detail)) {
    return { message: "Request validation failed." };
  }
  if (obj.detail && typeof obj.detail === "object") {
    const detail = obj.detail as Record<string, unknown>;
    if (typeof detail.message === "string" || typeof detail.code === "string") {
      return detail as ApiErrorPayload;
    }
  }
  return body as ApiErrorPayload;
}

async function parseErrorBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  try {
    const text = await response.text();
    return text.length > 0 ? text : null;
  } catch {
    return null;
  }
}

export function createApiFetch(options: CreateApiFetchOptions = {}): typeof fetch {
  const fetchImpl = options.fetchImpl ?? fetch;

  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const method = methodFrom(init);
    const headers = new Headers(init?.headers);

    if (!SAFE_METHODS.has(method)) {
      const token = options.csrfTokenProvider?.();
      if (token) {
        headers.set("X-CSRF-Token", token);
      }
    }

    const resolvedUrl = resolveAndValidateUrl(options.baseUrl, input);

    const response = await fetchImpl(resolvedUrl, {
      ...init,
      method,
      credentials: "include",
      headers,
    });

    if (response.ok) {
      return response;
    }

    const body = await parseErrorBody(response);
    const payload = normalizeErrorPayload(body);
    const message = payload?.message ?? (response.statusText || `HTTP ${response.status}`);
    const error = new ApiHttpError({
      status: response.status,
      statusText: response.statusText,
      code: payload?.code,
      message,
      body,
    });

    if (response.status === 401) {
      options.onUnauthorized?.(error);
    }

    throw error;
  };
}

export function createApiClient(options: CreateApiFetchOptions = {}) {
  let csrfToken: string | null = null;
  let csrfRefreshInFlight: Promise<string | null> | null = null;
  const fetcher = createApiFetch({
    ...options,
    csrfTokenProvider: () => csrfToken ?? options.csrfTokenProvider?.() ?? null,
  });
  const openApiClient = createClient<paths>({
    baseUrl: options.baseUrl,
    fetch: fetcher,
  });

  let sessionActive = false;

  type LoginResponse = { ok?: boolean } & Record<string, unknown>;
  type SessionResponse = Record<string, unknown>;
  type CsrfResponse = { csrf_token?: string; token?: string } & Record<string, unknown>;
  type SystemStatusResponse = { status: string; warning?: string } & Record<string, unknown>;
  type SetupStatusResponse = { docker: string; image: string; podman: string; notes: string[] };
  type DatabaseSettingsResponse = Record<string, unknown>;
  type DatabaseConnectionTestResponse = Record<string, unknown>;
  type ProjectPayload = Record<string, unknown>;
  type ProjectsListResponse = { projects: Array<Record<string, unknown>> };
  type PathEntryResponse = {
    name: string;
    path: string;
    type: "directory" | "file";
    selectable: boolean;
  };
  type PathEntriesListingResponse = {
    root: string;
    path: string;
    parent: string | null;
    can_go_up: boolean;
    entries: PathEntryResponse[];
  };
  type WatcherResponse = Record<string, unknown>;
  type WatchersListResponse = { watchers: Array<Record<string, unknown>> };
  type SyncResponse = Record<string, unknown>;
  type SyncListResponse = { projects: Array<Record<string, unknown>> };
  type SyncLogsResponse = { project_id: string; entries: Array<Record<string, unknown>> };
  type MCPClientPayload = Record<string, unknown>;
  type MCPClientsListResponse = { mcp_clients: Array<Record<string, unknown>> };
  type WorkflowSourcePayload = Record<string, unknown>;
  type WorkflowsListResponse = { workflows: Array<Record<string, unknown>> };
  type WorkflowDetailResponse = Record<string, unknown>;
  type WorkflowSchemaResponse = Record<string, unknown>;
  type WorkflowSourcesListResponse = { sources: Array<Record<string, unknown>> };
  type WorkflowValidateResponse = Record<string, unknown>;
  type RunPayload = Record<string, unknown>;
  type RunsListResponse = {
    runs: Array<Record<string, unknown>>;
    total?: number;
    limit?: number;
    offset?: number;
  };
  type LlmConfigResponse = Record<string, unknown>;
  type LlmRawYamlResponse = { raw_yaml: string } & Record<string, unknown>;
  type SecretPayload = Record<string, unknown>;
  type SecretsListResponse = { secrets: Array<Record<string, unknown>> };
  type WorkflowReloadResponse = {
    status: string;
    source_count: number;
    total: number;
    workflow_names: string[];
  };

  const json = async <T>(response: Response): Promise<T> => {
    return (await response.json()) as T;
  };

  const readCsrfToken = (payload: CsrfResponse): string | null => {
    const candidate = payload.csrf_token ?? payload.token;
    return typeof candidate === "string" && candidate.length > 0 ? candidate : null;
  };

  const ensureCsrf = async (): Promise<string | null> => {
    if (csrfToken) {
      return csrfToken;
    }
    if (csrfRefreshInFlight) {
      return csrfRefreshInFlight;
    }

    csrfRefreshInFlight = (async () => {
      const payload = await json<CsrfResponse>(await fetcher("/api/admin/v1/auth/csrf"));
      csrfToken = readCsrfToken(payload);
      return csrfToken;
    })();

    try {
      return await csrfRefreshInFlight;
    } finally {
      csrfRefreshInFlight = null;
    }
  };

  const runsListPath = (query?: {
    status?: string;
    mode?: string;
    workflow?: string;
    projectId?: string;
    limit?: number;
    offset?: number;
  }): string => {
    if (!query) {
      return "/api/admin/v1/runs";
    }

    const params = new URLSearchParams();
    if (query.status) params.set("status", query.status);
    if (query.mode) params.set("mode", query.mode);
    if (query.workflow) params.set("workflow", query.workflow);
    if (query.projectId) params.set("project_id", query.projectId);
    if (typeof query.limit === "number") params.set("limit", String(query.limit));
    if (typeof query.offset === "number") params.set("offset", String(query.offset));
    const suffix = params.toString();
    return `/api/admin/v1/runs${suffix ? `?${suffix}` : ""}`;
  };

  const pathEntriesListPath = (options?: {
    path?: string;
    selectionType?: "folder" | "file";
    extensions?: string[];
  }): string => {
    if (!options) {
      return "/api/admin/v1/filesystem/entries";
    }

    const params = new URLSearchParams();
    if (options.path) {
      params.set("path", options.path);
    }
    if (options.selectionType) {
      params.set("selection_type", options.selectionType);
    }
    if (options.extensions && options.extensions.length > 0) {
      const normalized = options.extensions.map((value) => value.trim()).filter((value) => value.length > 0);
      if (normalized.length > 0) {
        params.set("extensions", normalized.join(","));
      }
    }

    const suffix = params.toString();
    return `/api/admin/v1/filesystem/entries${suffix ? `?${suffix}` : ""}`;
  };

  const api = {
    ...openApiClient,
    getStoredCsrfToken(): string | null {
      return csrfToken;
    },
    isSessionActive(): boolean {
      return sessionActive;
    },
    async login(password: string): Promise<LoginResponse> {
      const loginPayload = await json<LoginResponse>(
        await fetcher("/api/admin/v1/auth/login", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ password }),
        }),
      );
      sessionActive = true;
      await api.getSession();
      await api.getCsrf();
      return loginPayload;
    },
    async getSession(): Promise<SessionResponse> {
      const payload = await json<SessionResponse>(await fetcher("/api/admin/v1/auth/session"));
      sessionActive = true;
      return payload;
    },
    async getCsrf(): Promise<CsrfResponse> {
      const payload = await json<CsrfResponse>(await fetcher("/api/admin/v1/auth/csrf"));
      csrfToken = readCsrfToken(payload);
      return payload;
    },
    async logout(): Promise<{ authenticated?: boolean } & Record<string, unknown>> {
      await ensureCsrf();
      const payload = await json<{ authenticated?: boolean } & Record<string, unknown>>(
        await fetcher("/api/admin/v1/auth/logout", { method: "POST" }),
      );
      sessionActive = false;
      csrfToken = null;
      return payload;
    },
    async getSystemStatus(): Promise<SystemStatusResponse> {
      return json<SystemStatusResponse>(await fetcher("/api/public/v1/system/status"));
    },
    async getSetupStatus(): Promise<SetupStatusResponse> {
      return json<SetupStatusResponse>(await fetcher("/api/admin/v1/database/setup"));
    },
    async getDatabaseSettings(): Promise<DatabaseSettingsResponse> {
      return json<DatabaseSettingsResponse>(await fetcher("/api/admin/v1/database/settings"));
    },
    async saveDatabaseSettings(payload: SaveDatabaseSettingsPayload): Promise<DatabaseSettingsResponse> {
      await ensureCsrf();
      return json<DatabaseSettingsResponse>(
        await fetcher("/api/admin/v1/database/settings", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    },
    async testDatabaseConnection(): Promise<DatabaseConnectionTestResponse> {
      await ensureCsrf();
      return json<DatabaseConnectionTestResponse>(
        await fetcher("/api/admin/v1/database/connection-test", { method: "POST" }),
      );
    },
    async listProjects(): Promise<ProjectsListResponse> {
      return json<ProjectsListResponse>(await fetcher("/api/admin/v1/projects"));
    },
    async listServerPathEntries(options?: {
      path?: string;
      selectionType?: "folder" | "file";
      extensions?: string[];
    }): Promise<PathEntriesListingResponse> {
      return json<PathEntriesListingResponse>(await fetcher(pathEntriesListPath(options)));
    },
    async createProject(payload: ProjectPayload): Promise<ProjectPayload> {
      await ensureCsrf();
      return json<ProjectPayload>(
        await fetcher("/api/admin/v1/projects", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    },
    async getProject(projectId: string): Promise<ProjectPayload> {
      return json<ProjectPayload>(await fetcher(`/api/admin/v1/projects/${encodeURIComponent(projectId)}`));
    },
    async updateProject(projectId: string, payload: ProjectPayload): Promise<ProjectPayload> {
      await ensureCsrf();
      return json<ProjectPayload>(
        await fetcher(`/api/admin/v1/projects/${encodeURIComponent(projectId)}`, {
          method: "PATCH",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    },
    async deleteProject(projectId: string): Promise<{ deleted: boolean } & Record<string, unknown>> {
      await ensureCsrf();
      return json<{ deleted: boolean } & Record<string, unknown>>(
        await fetcher(`/api/admin/v1/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" }),
      );
    },
    async listWatchers(): Promise<WatchersListResponse> {
      return json<WatchersListResponse>(await fetcher("/api/admin/v1/watchers"));
    },
    async getWatcher(projectId: string): Promise<WatcherResponse> {
      return json<WatcherResponse>(await fetcher(`/api/admin/v1/watchers/${encodeURIComponent(projectId)}`));
    },
    async pauseWatcher(projectId: string): Promise<WatcherResponse> {
      await ensureCsrf();
      return json<WatcherResponse>(
        await fetcher(`/api/admin/v1/watchers/${encodeURIComponent(projectId)}/pause`, { method: "POST" }),
      );
    },
    async resumeWatcher(projectId: string): Promise<WatcherResponse> {
      await ensureCsrf();
      return json<WatcherResponse>(
        await fetcher(`/api/admin/v1/watchers/${encodeURIComponent(projectId)}/resume`, { method: "POST" }),
      );
    },
    async disableWatcher(projectId: string): Promise<WatcherResponse> {
      await ensureCsrf();
      return json<WatcherResponse>(
        await fetcher(`/api/admin/v1/watchers/${encodeURIComponent(projectId)}/disable`, { method: "POST" }),
      );
    },
    async listSyncQueue(): Promise<SyncListResponse> {
      return json<SyncListResponse>(await fetcher("/api/admin/v1/sync"));
    },
    async listSyncLogs(projectId: string, limit = 25): Promise<SyncLogsResponse> {
      const params = new URLSearchParams({ limit: String(limit) });
      return json<SyncLogsResponse>(
        await fetcher(`/api/admin/v1/sync/${encodeURIComponent(projectId)}/logs?${params.toString()}`),
      );
    },
    async syncNow(projectId: string): Promise<SyncResponse> {
      await ensureCsrf();
      return json<SyncResponse>(
        await fetcher(`/api/admin/v1/sync/${encodeURIComponent(projectId)}/now`, { method: "POST" }),
      );
    },
    async reconcileSync(projectId: string): Promise<SyncResponse> {
      await ensureCsrf();
      return json<SyncResponse>(
        await fetcher(`/api/admin/v1/sync/${encodeURIComponent(projectId)}/reconcile`, {
          method: "POST",
        }),
      );
    },
    async rebuildSync(projectId: string): Promise<SyncResponse> {
      await ensureCsrf();
      return json<SyncResponse>(
        await fetcher(`/api/admin/v1/sync/${encodeURIComponent(projectId)}/rebuild`, {
          method: "POST",
        }),
      );
    },
    async listMcpClients(): Promise<MCPClientsListResponse> {
      return json<MCPClientsListResponse>(await fetcher("/api/admin/v1/mcp-clients"));
    },
    async createMcpClient(payload: MCPClientPayload): Promise<MCPClientPayload> {
      await ensureCsrf();
      return json<MCPClientPayload>(
        await fetcher("/api/admin/v1/mcp-clients", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    },
    async updateMcpClient(tokenId: string, payload: MCPClientPayload): Promise<MCPClientPayload> {
      await ensureCsrf();
      return json<MCPClientPayload>(
        await fetcher(`/api/admin/v1/mcp-clients/${encodeURIComponent(tokenId)}`, {
          method: "PATCH",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    },
    async revokeMcpClient(tokenId: string): Promise<{ revoked: boolean } & Record<string, unknown>> {
      await ensureCsrf();
      return json<{ revoked: boolean } & Record<string, unknown>>(
        await fetcher(`/api/admin/v1/mcp-clients/${encodeURIComponent(tokenId)}`, { method: "DELETE" }),
      );
    },
    async deleteMcpClient(tokenId: string): Promise<{ deleted: boolean } & Record<string, unknown>> {
      await ensureCsrf();
      return json<{ deleted: boolean } & Record<string, unknown>>(
        await fetcher(`/api/admin/v1/mcp-clients/${encodeURIComponent(tokenId)}/registration`, { method: "DELETE" }),
      );
    },
    async regenerateMcpClient(tokenId: string): Promise<MCPClientPayload> {
      await ensureCsrf();
      return json<MCPClientPayload>(
        await fetcher(`/api/admin/v1/mcp-clients/${encodeURIComponent(tokenId)}/regenerate`, {
          method: "POST",
        }),
      );
    },
    async listWorkflows(): Promise<WorkflowsListResponse> {
      return json<WorkflowsListResponse>(await fetcher("/api/admin/v1/workflows"));
    },
    async getWorkflowDetail(workflowName: string): Promise<WorkflowDetailResponse> {
      return json<WorkflowDetailResponse>(
        await fetcher(`/api/admin/v1/workflows/${encodeURIComponent(workflowName)}`),
      );
    },
    async getWorkflowSchema(): Promise<WorkflowSchemaResponse> {
      return json<WorkflowSchemaResponse>(await fetcher("/api/admin/v1/workflows/schema"));
    },
    async listWorkflowSources(): Promise<WorkflowSourcesListResponse> {
      return json<WorkflowSourcesListResponse>(await fetcher("/api/admin/v1/workflows/sources"));
    },
    async createWorkflowSource(payload: WorkflowSourcePayload): Promise<WorkflowSourcePayload> {
      await ensureCsrf();
      return json<WorkflowSourcePayload>(
        await fetcher("/api/admin/v1/workflows/sources", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    },
    async deleteWorkflowSource(sourceId: string): Promise<{ deleted: boolean } & Record<string, unknown>> {
      await ensureCsrf();
      return json<{ deleted: boolean } & Record<string, unknown>>(
        await fetcher(`/api/admin/v1/workflows/sources/${encodeURIComponent(sourceId)}`, {
          method: "DELETE",
        }),
      );
    },
    async validateWorkflowSource(sourceId: string): Promise<WorkflowValidateResponse> {
      await ensureCsrf();
      return json<WorkflowValidateResponse>(
        await fetcher(`/api/admin/v1/workflows/sources/${encodeURIComponent(sourceId)}/validate`, {
          method: "POST",
        }),
      );
    },
    async listRuns(query?: {
      status?: string;
      mode?: string;
      workflow?: string;
      projectId?: string;
      limit?: number;
      offset?: number;
    }): Promise<RunsListResponse> {
      return json<RunsListResponse>(await fetcher(runsListPath(query)));
    },
    async getRunDetail(runId: string): Promise<RunPayload> {
      return json<RunPayload>(await fetcher(`/api/admin/v1/runs/${encodeURIComponent(runId)}`));
    },
    async cancelRun(runId: string): Promise<RunPayload> {
      await ensureCsrf();
      return json<RunPayload>(
        await fetcher(`/api/admin/v1/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" }),
      );
    },
    async resumeRun(runId: string, response = ""): Promise<RunPayload> {
      await ensureCsrf();
      return json<RunPayload>(
        await fetcher(`/api/admin/v1/runs/${encodeURIComponent(runId)}/resume`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ response }),
        }),
      );
    },
    async getLlmConfig(): Promise<LlmConfigResponse> {
      return json<LlmConfigResponse>(await fetcher("/api/admin/v1/llm/config"));
    },
    async updateLlmConfig(payload: LlmConfigResponse): Promise<LlmConfigResponse> {
      await ensureCsrf();
      return json<LlmConfigResponse>(
        await fetcher("/api/admin/v1/llm/config", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    },
    async previewLlmConfig(raw_yaml: string): Promise<LlmConfigResponse> {
      await ensureCsrf();
      return json<LlmConfigResponse>(
        await fetcher("/api/admin/v1/llm/preview", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ raw_yaml }),
        }),
      );
    },
    async importLlmConfig(raw_yaml: string): Promise<LlmConfigResponse> {
      await ensureCsrf();
      return json<LlmConfigResponse>(
        await fetcher("/api/admin/v1/llm/import", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ raw_yaml }),
        }),
      );
    },
    async exportLlmConfig(): Promise<LlmRawYamlResponse> {
      return json<LlmRawYamlResponse>(await fetcher("/api/admin/v1/llm/export"));
    },
    async listSecrets(): Promise<SecretsListResponse> {
      return json<SecretsListResponse>(await fetcher("/api/admin/v1/secrets"));
    },
    async upsertSecret(payload: SecretPayload): Promise<SecretPayload> {
      await ensureCsrf();
      return json<SecretPayload>(
        await fetcher("/api/admin/v1/secrets", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    },
    async deleteSecret(name: string): Promise<{ deleted: boolean } & Record<string, unknown>> {
      await ensureCsrf();
      return json<{ deleted: boolean } & Record<string, unknown>>(
        await fetcher(`/api/admin/v1/secrets/${encodeURIComponent(name)}`, { method: "DELETE" }),
      );
    },
    async reloadWorkflows(): Promise<WorkflowReloadResponse> {
      await ensureCsrf();
      return json<WorkflowReloadResponse>(
        await fetcher("/api/admin/v1/workflows/reload", {
          method: "POST",
        }),
      );
    },
  };

  return api;
}
