import "./App.css";
import { ChangeEvent, FormEvent, KeyboardEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

import { ApiHttpError, createApiClient, type DatabaseSslMode } from "../api/client";
import {
  createEventStream,
  createPollingFallback,
  fetchSyncState,
  fetchWatcherState,
  SyncStateItem,
  WatcherStateItem,
} from "../api/events";
import { ServerPathPicker, type PathListing } from "./ServerPathPicker";
import { ActionButton, CopyIcon, FolderIcon, PageHeader, Panel, ProjectMultiSelect, ProjectSelect, StatusBadge } from "./ui";

type RouteDefinition = {
  path: string;
  label: string;
  title: string;
  description: string;
};

const ROUTES: readonly RouteDefinition[] = [
  {
    path: "/login",
    label: "Login",
    title: "Login",
    description: "Sign in to access workflow administration.",
  },
  {
    path: "/setup",
    label: "Setup",
    title: "Setup",
    description: "Configure the initial workflows admin settings.",
  },
  {
    path: "/projects",
    label: "Projects",
    title: "Projects",
    description: "Review registered projects and onboarding readiness before watcher setup.",
  },
  {
    path: "/database",
    label: "Database",
    title: "Database",
    description: "Inspect database connectivity, schema readiness, and setup posture.",
  },
  {
    path: "/llm",
    label: "LLM",
    title: "LLM",
    description: "Review model provider status and language-model service connectivity.",
  },
  {
    path: "/secrets",
    label: "Secrets",
    title: "Secrets",
    description: "Audit configured secret keys and operational availability indicators.",
  },
  {
    path: "/watchers",
    label: "Watchers",
    title: "Watchers",
    description: "Manage watcher state and monitor reconciliation indicators.",
  },
  {
    path: "/sync",
    label: "Sync",
    title: "Sync",
    description: "Monitor project sync queues and reconciliation requirements.",
  },
  {
    path: "/mcp-clients",
    label: "MCP Clients",
    title: "MCP Clients",
    description: "Review and manage MCP client tokens and access state.",
  },
  {
    path: "/workflows",
    label: "Workflows",
    title: "Workflows",
    description: "Review workflow templates and orchestration definitions.",
  },
  {
    path: "/runs",
    label: "Runs",
    title: "Runs",
    description: "Track workflow run status and execution history.",
  },
] as const;

const PRIMARY_NAV_PATHS = new Set([
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
]);

const DEFAULT_PAGE = {
  title: "Workflows Admin",
  description: "Choose a section to begin managing the workflows platform.",
} as const;

const DEFAULT_PROJECT_FS_ROOT = "~";

const NOT_FOUND_PAGE = {
  title: "Page not found",
  description: "The requested route is not part of the admin shell.",
} as const;

type SetupDashboardData = {
  sessionActive: boolean;
  systemStatus: string;
  databaseConfigured: boolean;
  llmConfigured: boolean;
  projectCount: number;
  mcpClientCount: number;
};

type SetupDashboardState = {
  loading: boolean;
  error: string | null;
  data: SetupDashboardData | null;
};

type DatabaseSettingsModel = {
  enabled: boolean;
  configured: boolean;
  updatedAt: string;
  host: string;
  port: number;
  database: string;
  username: string;
  passwordConfigured: boolean;
  sslMode: DatabaseSslMode;
  extraParams: string;
  containerName: string;
  containerImage: string;
  containerHostPort: number;
  volumeName: string;
};

type DatabaseProfileForm = {
  enabled: boolean;
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  passwordConfigured: boolean;
  passwordClear: boolean;
  sslMode: DatabaseSslMode;
  extraParams: string;
  containerName: string;
  containerImage: string;
  containerHostPort: string;
  volumeName: string;
  dsnImport: string;
};

type ProviderConfig = {
  type: string;
  api_url: string | null;
  api_key_secret: string | null;
  model: string | null;
  timeout: number | null;
  max_retries: number | null;
  retry_delay: number | null;
  extra_headers: Record<string, string>;
  deployment_name: string | null;
  api_version: string | null;
};

type ProfileConfig = {
  provider: string;
  model: string;
  temperature: number | null;
  max_tokens: number | null;
  description: string | null;
};

type LlmConfigModel = {
  version: "1.0";
  providers: Record<string, ProviderConfig>;
  profiles: Record<string, ProfileConfig>;
  default_profile: string | null;
};

type ProviderConfigPayload = Omit<ProviderConfig, "timeout" | "max_retries" | "retry_delay"> & {
  timeout?: number;
  max_retries?: number;
  retry_delay?: number;
};

type ProfileConfigPayload = ProfileConfig;

type LlmConfigPayload = {
  version: "1.0";
  providers: Record<string, ProviderConfigPayload>;
  profiles: Record<string, ProfileConfigPayload>;
  default_profile: string | null;
};

type LlmProviderForm = {
  id: string;
  type: string;
  apiUrl: string;
  apiKeySecret: string;
  model: string;
  timeout: string;
  maxRetries: string;
  retryDelay: string;
  extraHeaders: string;
  deploymentName: string;
  apiVersion: string;
};

type LlmProfileForm = {
  id: string;
  provider: string;
  model: string;
  temperature: string;
  maxTokens: string;
  description: string;
};

type ModalShellProps = {
  titleId: string;
  title: string;
  eyebrow: string;
  children: ReactNode;
  onClose: () => void;
};

type ConnectionTestModel = {
  ok: boolean;
  status: string;
  configured: boolean;
  blockers: string[];
  actionable: string[];
};

type ProjectModel = {
  id: string;
  name: string;
  slug: string;
  palace: string;
  defaultWing: string;
  defaultRoom: string;
  fsRoot: string;
  fsAllowlist: string[];
  watcherHint: string | null;
  defaultStateHint: string | null;
};

type MpcClientModel = {
  id: string;
  label: string;
  projectIds: string[];
  createdAt: string;
  lastUsedAt: string | null;
  revokedAt: string | null;
};

type WatcherDashboardRow = {
  projectId: string;
  projectName: string | null;
  state: string;
  dirtyCount: number;
  requiresReconciliation: boolean;
  lastEventAt: string | null;
  updatedAt: string;
};

type SyncDashboardRow = {
  projectId: string;
  projectName: string | null;
  dirtyCount: number;
  requiresReconciliation: boolean;
  syncState: string;
  reconcileState: string;
  rebuildState: string;
  lifecycleTelemetry: string;
};

function buildSyncDashboardRows(items: SyncStateItem[], projects: ProjectModel[]): SyncDashboardRow[] {
  const projectNameById = new Map(projects.map((project) => [project.id, project.name]));
  const syncItemByProjectId = new Map(items.map((item) => [item.project_id, item]));

  return projects.map((project) => {
    const item = syncItemByProjectId.get(project.id);
    const dirtyCount = item?.dirty_count ?? 0;
    const requiresReconciliation = item?.requires_reconciliation ?? false;

    return {
      projectId: project.id,
      projectName: projectNameById.get(project.id) ?? null,
      dirtyCount,
      requiresReconciliation,
      syncState: typeof item?.sync_state === "string" ? item.sync_state : dirtyCount > 0 ? "queued" : "idle",
      reconcileState:
        typeof item?.reconcile_state === "string"
          ? item.reconcile_state
          : requiresReconciliation
            ? "required"
            : "clear",
      rebuildState: typeof item?.rebuild_state === "string" ? item.rebuild_state : "available as manual action",
      lifecycleTelemetry:
        typeof item?.updated_at === "string" && item.updated_at.trim().length > 0
          ? `updated_at=${item.updated_at}`
          : "not reported by the current sync endpoint",
    };
  });
}

type WorkflowSummaryModel = {
  name: string;
  description: string;
  version: string;
  tags: string[];
  sourcePath: string | null;
};

type WorkflowSourceModel = {
  sourceId: string;
  projectId: string;
  sourcePath: string;
  status: string | null;
  discoveredAt: string;
  lastLoadedAt: string | null;
  errorMessage: string | null;
};

type RunRowModel = {
  runId: string;
  jobId: string;
  workflowName: string;
  status: string;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  updatedAt: string;
  cancellable: boolean;
  projectId: string | null;
  tokenId: string | null;
};

type RunDetailModel = RunRowModel & {
  resultSummary: string | null;
  errorSummary: string | null;
  metadata: Record<string, unknown>;
  technicalJson: string;
};

type OneTimeMcpSecret = {
  token: string;
  configSnippet: string | null;
  label: string;
};

type FolderBrowserTarget = "fsRoot" | "allowlist";

const MCP_SECRET_MISSING_ERROR = "Token issuance response was incomplete; no secret was returned.";
const VALID_NAME_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$/;

const DEFAULT_DATABASE_FORM: DatabaseProfileForm = {
  enabled: false,
  host: "",
  port: "5432",
  database: "",
  username: "",
  password: "",
  passwordConfigured: false,
  passwordClear: false,
  sslMode: "prefer",
  extraParams: "",
  containerName: "workflows-postgres",
  containerImage: "pgvector/pgvector:pg17",
  containerHostPort: "5432",
  volumeName: "workflows-postgres-data",
  dsnImport: "",
};

const EMPTY_LLM_CONFIG: LlmConfigModel = {
  version: "1.0",
  providers: {},
  profiles: {},
  default_profile: null,
};

const DEFAULT_LLM_PROVIDER_FORM: LlmProviderForm = {
  id: "",
  type: "",
  apiUrl: "",
  apiKeySecret: "",
  model: "",
  timeout: "",
  maxRetries: "",
  retryDelay: "",
  extraHeaders: "{}",
  deploymentName: "",
  apiVersion: "",
};

const DEFAULT_LLM_PROFILE_FORM: LlmProfileForm = {
  id: "",
  provider: "",
  model: "",
  temperature: "",
  maxTokens: "",
  description: "",
};

function ModalShell({ titleId, title, eyebrow, children, onClose }: ModalShellProps) {
  const dialogRef = useRef<HTMLElement | null>(null);
  const openerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    openerRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const dialog = dialogRef.current;
    const initialFocusable = dialog ? getInitialModalFocusElement(dialog) : null;
    initialFocusable?.focus();

    return () => {
      openerRef.current?.focus();
    };
  }, []);

  const onDialogKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
      return;
    }

    if (event.key !== "Tab") return;

    const dialog = dialogRef.current;
    if (!dialog) return;
    const focusableElements = getModalFocusableElements(dialog);
    if (focusableElements.length === 0) {
      event.preventDefault();
      return;
    }

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];
    if (event.shiftKey && document.activeElement === firstElement) {
      event.preventDefault();
      lastElement.focus();
      return;
    }
    if (!event.shiftKey && document.activeElement === lastElement) {
      event.preventDefault();
      firstElement.focus();
    }
  };

  return (
    <div className="llm-modal__backdrop">
      <section
        ref={dialogRef}
        className="llm-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onKeyDown={onDialogKeyDown}
      >
        <header className="llm-modal__header">
          <div>
            <p className="llm-modal__eyebrow">{eyebrow}</p>
            <h2 id={titleId}>{title}</h2>
          </div>
          <button type="button" className="llm-modal__close" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="llm-modal__body">{children}</div>
      </section>
    </div>
  );
}

function getModalFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  ).filter((element) => !element.hasAttribute("disabled") && element.getAttribute("aria-hidden") !== "true");
}

function getInitialModalFocusElement(container: HTMLElement): HTMLElement | null {
  const body = container.querySelector<HTMLElement>(".llm-modal__body");
  return (body ? getModalFocusableElements(body)[0] : null) ?? getModalFocusableElements(container)[0] ?? null;
}

function toNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function toUserError(error: unknown, fallback: string): string {
  if (error instanceof ApiHttpError) {
    const bodyObject = toObject(error.body);
    const detail = toObject(bodyObject.detail);
    if (typeof detail.message === "string" && detail.message.trim().length > 0) {
      return detail.message;
    }
    if (typeof bodyObject.message === "string" && bodyObject.message.trim().length > 0) {
      return bodyObject.message;
    }
    return error.message;
  }
  if (error instanceof Error && error.message.trim().length > 0) return error.message;
  return fallback;
}

function toObject(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function toNullableString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

function toNullableNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function toStringMap(value: unknown): Record<string, string> {
  const obj = toObject(value);
  return Object.fromEntries(
    Object.entries(obj)
      .filter((entry): entry is [string, string] => typeof entry[1] === "string")
      .map(([key, val]) => [key, val]),
  );
}

function toLlmProviderConfig(payload: unknown): ProviderConfig {
  const obj = toObject(payload);
  return {
    type: typeof obj.type === "string" ? obj.type : "",
    api_url: toNullableString(obj.api_url),
    api_key_secret: toNullableString(obj.api_key_secret),
    model: toNullableString(obj.model),
    timeout: toNullableNumber(obj.timeout),
    max_retries: toNullableNumber(obj.max_retries),
    retry_delay: toNullableNumber(obj.retry_delay),
    extra_headers: toStringMap(obj.extra_headers),
    deployment_name: toNullableString(obj.deployment_name),
    api_version: toNullableString(obj.api_version),
  };
}

function toLlmProfileConfig(payload: unknown): ProfileConfig {
  const obj = toObject(payload);
  return {
    provider: typeof obj.provider === "string" ? obj.provider : "",
    model: toNullableString(obj.model) ?? "",
    temperature: toNullableNumber(obj.temperature),
    max_tokens: toNullableNumber(obj.max_tokens),
    description: toNullableString(obj.description),
  };
}

function toLlmConfigModel(payload: unknown): LlmConfigModel {
  const obj = toObject(payload);
  const providersObj = toObject(obj.providers);
  const profilesObj = toObject(obj.profiles);
  const providers = Object.fromEntries(
    Object.entries(providersObj).map(([key, value]) => [key, toLlmProviderConfig(value)]),
  );
  const profiles = Object.fromEntries(
    Object.entries(profilesObj).map(([key, value]) => [key, toLlmProfileConfig(value)]),
  );
  const defaultProfile = toNullableString(obj.default_profile);
  return {
    version: "1.0",
    providers,
    profiles,
    default_profile: defaultProfile && profiles[defaultProfile] ? defaultProfile : null,
  };
}

function toLlmProviderForm(id: string, provider: ProviderConfig): LlmProviderForm {
  return {
    id,
    type: provider.type,
    apiUrl: provider.api_url ?? "",
    apiKeySecret: provider.api_key_secret ?? "",
    model: provider.model ?? "",
    timeout: provider.timeout === null ? "" : String(provider.timeout),
    maxRetries: provider.max_retries === null ? "" : String(provider.max_retries),
    retryDelay: provider.retry_delay === null ? "" : String(provider.retry_delay),
    extraHeaders: JSON.stringify(provider.extra_headers, null, 2),
    deploymentName: provider.deployment_name ?? "",
    apiVersion: provider.api_version ?? "",
  };
}

function toLlmProfileForm(id: string, profile: ProfileConfig): LlmProfileForm {
  return {
    id,
    provider: profile.provider,
    model: profile.model,
    temperature: profile.temperature === null ? "" : String(profile.temperature),
    maxTokens: profile.max_tokens === null ? "" : String(profile.max_tokens),
    description: profile.description ?? "",
  };
}

function parseExtraHeaders(value: string): Record<string, string> {
  const trimmed = value.trim();
  if (trimmed.length === 0) return {};
  const parsed = JSON.parse(trimmed) as unknown;
  const obj = toObject(parsed);
  const invalid = Object.entries(obj).find(([, val]) => typeof val !== "string");
  if (invalid) throw new Error("Extra headers must be a JSON object with string values.");
  return toStringMap(obj);
}

function numberField(value: string, label: string): number | null {
  const trimmed = value.trim();
  if (trimmed.length === 0) return null;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) throw new Error(`${label} must be a number.`);
  return parsed;
}

function readUploadText(file: File): Promise<string> {
  if (typeof file.text === "function") return file.text();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
    reader.onerror = () => reject(reader.error ?? new Error("Unable to read YAML file."));
    reader.readAsText(file);
  });
}

function toDatabaseSettingsModel(payload: unknown): DatabaseSettingsModel {
  const obj = toObject(payload);
  const sslModeRaw = typeof obj.ssl_mode === "string" ? obj.ssl_mode : "prefer";
  const sslMode: DatabaseSslMode =
    sslModeRaw === "disable" ||
    sslModeRaw === "prefer" ||
    sslModeRaw === "require" ||
    sslModeRaw === "verify-ca" ||
    sslModeRaw === "verify-full"
      ? sslModeRaw
      : "prefer";
  return {
    enabled: obj.enabled === true,
    configured: obj.configured === true,
    updatedAt: typeof obj.updated_at === "string" ? obj.updated_at : "Unavailable",
    host: typeof obj.host === "string" ? obj.host : "",
    port: typeof obj.port === "number" ? obj.port : 5432,
    database: typeof obj.database === "string" ? obj.database : "",
    username: typeof obj.username === "string" ? obj.username : "",
    passwordConfigured: obj.password_configured === true,
    sslMode,
    extraParams: typeof obj.extra_params === "string" ? obj.extra_params : "",
    containerName: typeof obj.container_name === "string" ? obj.container_name : DEFAULT_DATABASE_FORM.containerName,
    containerImage: typeof obj.container_image === "string" ? obj.container_image : DEFAULT_DATABASE_FORM.containerImage,
    containerHostPort: typeof obj.container_host_port === "number" ? obj.container_host_port : 5432,
    volumeName: typeof obj.volume_name === "string" ? obj.volume_name : DEFAULT_DATABASE_FORM.volumeName,
  };
}

function toDatabaseForm(settings: DatabaseSettingsModel): DatabaseProfileForm {
  return {
    enabled: settings.enabled,
    host: settings.host,
    port: String(settings.port),
    database: settings.database,
    username: settings.username,
    password: "",
    passwordConfigured: settings.passwordConfigured,
    passwordClear: false,
    sslMode: settings.sslMode,
    extraParams: settings.extraParams,
    containerName: settings.containerName,
    containerImage: settings.containerImage,
    containerHostPort: String(settings.containerHostPort),
    volumeName: settings.volumeName,
    dsnImport: "",
  };
}

function parsePostgresDsnForForm(input: string): Partial<DatabaseProfileForm> {
  const url = new URL(input.trim());
  if (!["postgres:", "postgresql:"].includes(url.protocol)) {
    throw new Error("Only postgresql:// or postgres:// DSN values are supported.");
  }
  const params = new URLSearchParams(url.search);
  const sslmode = params.get("sslmode");
  if (sslmode) params.delete("sslmode");
  const sslMode: DatabaseSslMode =
    sslmode === "disable" || sslmode === "prefer" || sslmode === "require" || sslmode === "verify-ca" || sslmode === "verify-full"
      ? sslmode
      : "prefer";

  return {
    host: decodeURIComponent(url.hostname),
    port: url.port || "5432",
    database: decodeURIComponent(url.pathname.replace(/^\//, "")),
    username: decodeURIComponent(url.username),
    password: decodeURIComponent(url.password),
    sslMode,
    extraParams: params.toString(),
  };
}

function quoteShellPreview(value: string): string {
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(value)) return value;
  return `'${value.replace(/'/g, `'"'"'`)}'`;
}

function toConnectionTestModel(payload: unknown): ConnectionTestModel {
  const obj = toObject(payload);
  const blockers = Array.isArray(obj.blockers) ? obj.blockers.filter((item): item is string => typeof item === "string") : [];
  const actionable = Array.isArray(obj.actionable)
    ? obj.actionable.filter((item): item is string => typeof item === "string")
    : [];
  return {
    ok: obj.ok === true,
    status: typeof obj.status === "string" ? obj.status : "unknown",
    configured: obj.configured === true,
    blockers,
    actionable,
  };
}

function toProjectModel(payload: unknown): ProjectModel {
  const obj = toObject(payload);
  const fsAllowlist = Array.isArray(obj.fs_allowlist)
    ? obj.fs_allowlist.filter((item): item is string => typeof item === "string")
    : [];
  return {
    id: typeof obj.id === "string" ? obj.id : "",
    name: typeof obj.name === "string" ? obj.name : "",
    slug: typeof obj.slug === "string" ? obj.slug : "",
    palace: typeof obj.palace === "string" ? obj.palace : "",
    defaultWing: typeof obj.default_wing === "string" ? obj.default_wing : "",
    defaultRoom: typeof obj.default_room === "string" ? obj.default_room : "",
    fsRoot: typeof obj.fs_root === "string" ? obj.fs_root : "",
    fsAllowlist,
    watcherHint: typeof obj.watcher_hint === "string" ? obj.watcher_hint : null,
    defaultStateHint: typeof obj.default_state_hint === "string" ? obj.default_state_hint : null,
  };
}

function toMcpClientModel(payload: unknown): MpcClientModel {
  const obj = toObject(payload);
  return {
    id: typeof obj.id === "string" ? obj.id : "",
    label: typeof obj.label === "string" ? obj.label : "",
    projectIds: Array.isArray(obj.project_ids)
      ? obj.project_ids.filter((item): item is string => typeof item === "string")
      : [],
    createdAt: typeof obj.created_at === "string" ? obj.created_at : "",
    lastUsedAt: typeof obj.last_used_at === "string" ? obj.last_used_at : null,
    revokedAt: typeof obj.revoked_at === "string" ? obj.revoked_at : null,
  };
}

function normalizePathList(input: string): string[] {
  const unique = new Set<string>();
  for (const part of input.split(/[\n,]/)) {
    const trimmed = part.trim();
    if (trimmed.length > 0) unique.add(trimmed);
  }
  return Array.from(unique);
}

function toWorkflowSummaryModel(payload: unknown): WorkflowSummaryModel {
  const obj = toObject(payload);
  const tags = Array.isArray(obj.tags) ? obj.tags.filter((item): item is string => typeof item === "string") : [];
  return {
    name: typeof obj.name === "string" ? obj.name : "",
    description: typeof obj.description === "string" ? obj.description : "",
    version: typeof obj.version === "string" ? obj.version : "",
    tags,
    sourcePath: typeof obj.source_path === "string" ? obj.source_path : null,
  };
}

function toWorkflowSourceModel(payload: unknown): WorkflowSourceModel {
  const obj = toObject(payload);
  return {
    sourceId: typeof obj.source_id === "string" ? obj.source_id : "",
    projectId: typeof obj.project_id === "string" ? obj.project_id : "",
    sourcePath: typeof obj.source_path === "string" ? obj.source_path : "",
    status: typeof obj.status === "string" ? obj.status : null,
    discoveredAt: typeof obj.discovered_at === "string" ? obj.discovered_at : "",
    lastLoadedAt: typeof obj.last_loaded_at === "string" ? obj.last_loaded_at : null,
    errorMessage: typeof obj.error_message === "string" ? obj.error_message : null,
  };
}

function toRunRowModel(payload: unknown): RunRowModel {
  const obj = toObject(payload);
  return {
    runId: typeof obj.run_id === "string" ? obj.run_id : "",
    jobId: typeof obj.job_id === "string" ? obj.job_id : "",
    workflowName: typeof obj.workflow_name === "string" ? obj.workflow_name : "",
    status: typeof obj.status === "string" ? obj.status : "unknown",
    createdAt: typeof obj.created_at === "string" ? obj.created_at : "Unavailable",
    startedAt: typeof obj.started_at === "string" ? obj.started_at : null,
    finishedAt: typeof obj.finished_at === "string" ? obj.finished_at : null,
    updatedAt: typeof obj.updated_at === "string" ? obj.updated_at : "Unavailable",
    cancellable: obj.cancellable === true,
    projectId: typeof obj.project_id === "string" ? obj.project_id : null,
    tokenId: typeof obj.token_id === "string" ? obj.token_id : null,
  };
}

function toRunDetailModel(payload: unknown): RunDetailModel {
  const obj = toObject(payload);
  const row = toRunRowModel(payload);
  const metadata = toObject(obj.metadata);
  return {
    ...row,
    resultSummary: typeof obj.result_summary === "string" ? obj.result_summary : null,
    errorSummary: typeof obj.error_summary === "string" ? obj.error_summary : null,
    metadata,
    technicalJson: JSON.stringify(obj, null, 2),
  };
}

type NavigateOptions = { replace?: boolean };

function navigateBrowser(path: string, { replace = false }: NavigateOptions = {}): void {
  if (window.location.pathname === path) return;
  if (replace) {
    window.history.replaceState({}, "", path);
  } else {
    window.history.pushState({}, "", path);
  }
  window.dispatchEvent(new PopStateEvent("popstate"));
}

async function copyTextToClipboard(text: string): Promise<void> {
  try {
    if (typeof navigator.clipboard?.writeText === "function") {
      await navigator.clipboard.writeText(text);
      return;
    }
  } catch {
    // Fall through to the selection-based copy path for embedded browsers.
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  textarea.style.opacity = "0";

  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();

  try {
    if (typeof document.execCommand !== "function" || !document.execCommand("copy")) {
      throw new Error("Clipboard fallback failed");
    }
  } finally {
    textarea.remove();
  }
}

export function App(): JSX.Element {
  const [currentPath, setCurrentPath] = useState(() => window.location.pathname);
  const route = ROUTES.find((entry) => entry.path === currentPath);
  const [password, setPassword] = useState("");
  const [loginState, setLoginState] = useState<"idle" | "pending" | "ok" | "error">("idle");
  const [loginMessage, setLoginMessage] = useState<string>("");
  const [contentState, setContentState] = useState<string>("");
  const [setupDashboard, setSetupDashboard] = useState<SetupDashboardState>({
    loading: false,
    error: null,
    data: null,
  });
  const [dbGuidance, setDbGuidance] = useState<{ image: string; docker: string; podman: string; notes: string[] } | null>(null);
  const [dbSettings, setDbSettings] = useState<DatabaseSettingsModel | null>(null);
  const [dbForm, setDbForm] = useState<DatabaseProfileForm>(DEFAULT_DATABASE_FORM);
  const [dbLoadError, setDbLoadError] = useState<string>("");
  const [dbSaveMessage, setDbSaveMessage] = useState<string>("");
  const [dbSavePending, setDbSavePending] = useState(false);
  const [dbFieldErrors, setDbFieldErrors] = useState<Partial<Record<keyof DatabaseProfileForm, string>>>({});
  const [dbTestPending, setDbTestPending] = useState(false);
  const [dbConnectionResult, setDbConnectionResult] = useState<ConnectionTestModel | null>(null);
  const [dbConnectionMessage, setDbConnectionMessage] = useState("");
  const [dbCopyStatus, setDbCopyStatus] = useState("");
  const [dbCopyError, setDbCopyError] = useState("");
  const [dbSavedPasswordForCopy, setDbSavedPasswordForCopy] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectModel[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectMessage, setProjectMessage] = useState("");
  const [projectError, setProjectError] = useState("");
  const [projectDeleteTarget, setProjectDeleteTarget] = useState<ProjectModel | null>(null);
  const [projectDeleteConfirm, setProjectDeleteConfirm] = useState("");
  const [projectDeletePending, setProjectDeletePending] = useState(false);
  const [projectCreatePending, setProjectCreatePending] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectSlug, setProjectSlug] = useState("");
  const [projectPalace, setProjectPalace] = useState("");
  const [projectDefaultWing, setProjectDefaultWing] = useState("");
  const [projectDefaultRoom, setProjectDefaultRoom] = useState("");
  const [projectFsRoot, setProjectFsRoot] = useState(DEFAULT_PROJECT_FS_ROOT);
  const [projectAllowlistInput, setProjectAllowlistInput] = useState("");
  const projectFsRootRef = useRef<HTMLInputElement | null>(null);
  const projectAllowlistRef = useRef<HTMLTextAreaElement | null>(null);
  const [pathPickerOpen, setPathPickerOpen] = useState(false);
  const [pathPickerTarget, setPathPickerTarget] = useState<FolderBrowserTarget>("fsRoot");

  const [mcpClients, setMcpClients] = useState<MpcClientModel[]>([]);
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpError, setMcpError] = useState("");
  const [mcpMessage, setMcpMessage] = useState("");
  const [mcpCreatePending, setMcpCreatePending] = useState(false);
  const [mcpMutationPendingId, setMcpMutationPendingId] = useState<string | null>(null);
  const [mcpLabel, setMcpLabel] = useState("");
  const [mcpSelectedProjectIds, setMcpSelectedProjectIds] = useState<string[]>([]);
  const [oneTimeMcpSecret, setOneTimeMcpSecret] = useState<OneTimeMcpSecret | null>(null);
  const [watchersRows, setWatchersRows] = useState<WatcherDashboardRow[]>([]);
  const [watchersLoading, setWatchersLoading] = useState(false);
  const [watchersError, setWatchersError] = useState("");
  const [watchersMessage, setWatchersMessage] = useState("");
  const [watchersPendingAction, setWatchersPendingAction] = useState<string | null>(null);
  const [syncRows, setSyncRows] = useState<SyncDashboardRow[]>([]);
  const [syncLoading, setSyncLoading] = useState(false);
  const [syncError, setSyncError] = useState("");
  const [syncMessage, setSyncMessage] = useState("");
  const [syncPendingAction, setSyncPendingAction] = useState<string | null>(null);
  const [selectedSyncProjectId, setSelectedSyncProjectId] = useState("");
  const [workflowsLoading, setWorkflowsLoading] = useState(false);
  const [workflowsError, setWorkflowsError] = useState("");
  const [workflowsMessage, setWorkflowsMessage] = useState("");
  const [workflowsList, setWorkflowsList] = useState<WorkflowSummaryModel[]>([]);
  const [workflowSources, setWorkflowSources] = useState<WorkflowSourceModel[]>([]);
  const [workflowSourceProjectId, setWorkflowSourceProjectId] = useState("");
  const [workflowSourcePath, setWorkflowSourcePath] = useState("");
  const [workflowSourceChecksum, setWorkflowSourceChecksum] = useState("");
  const [workflowSourcePending, setWorkflowSourcePending] = useState(false);
  const [workflowActionPendingId, setWorkflowActionPendingId] = useState<string | null>(null);
  const [workflowProjectsCount, setWorkflowProjectsCount] = useState(0);
  const [selectedWorkflowName, setSelectedWorkflowName] = useState<string | null>(null);
  const [workflowDetailText, setWorkflowDetailText] = useState<string>("");
  const [workflowSchemaText, setWorkflowSchemaText] = useState<string>("");
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState("");
  const [runsMessage, setRunsMessage] = useState("");
  const [runsRows, setRunsRows] = useState<RunRowModel[]>([]);
  const [runDetail, setRunDetail] = useState<RunDetailModel | null>(null);
  const [runFilterStatus, setRunFilterStatus] = useState("");
  const [runFilterLimit, setRunFilterLimit] = useState("50");
  const [runFilterOffset, setRunFilterOffset] = useState("0");
  const [runActionPendingId, setRunActionPendingId] = useState<string | null>(null);
  const [llmConfig, setLlmConfig] = useState<LlmConfigModel>(EMPTY_LLM_CONFIG);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmSaving, setLlmSaving] = useState(false);
  const [llmError, setLlmError] = useState("");
  const [llmMessage, setLlmMessage] = useState("");
  const [llmProviderForm, setLlmProviderForm] = useState<LlmProviderForm>(DEFAULT_LLM_PROVIDER_FORM);
  const [llmProviderModalOpen, setLlmProviderModalOpen] = useState(false);
  const [llmProfileForm, setLlmProfileForm] = useState<LlmProfileForm>(DEFAULT_LLM_PROFILE_FORM);
  const [llmYamlImport, setLlmYamlImport] = useState("");
  const [llmYamlModalOpen, setLlmYamlModalOpen] = useState(false);
  const [llmPreview, setLlmPreview] = useState<LlmConfigModel | null>(null);

  const api = useMemo(
    () =>
      createApiClient({
        onUnauthorized: () => {
          setContentState("");
          setPathPickerOpen(false);
          setLoginState("idle");
          setLoginMessage("Your admin session expired. Sign in to continue.");
          navigateBrowser("/login", { replace: true });
        },
      }),
    [],
  );

  const loadProjects = async (): Promise<ProjectModel[]> => {
    setProjectsLoading(true);
    setProjectError("");
    try {
      const result = await api.listProjects();
      const projectModels = result.projects.map(toProjectModel);
      const validProjectIds = new Set(projectModels.map((project) => project.id));
      setProjects(projectModels);
      setMcpSelectedProjectIds((current) => current.filter((projectId) => validProjectIds.has(projectId)));
      if (selectedSyncProjectId && !validProjectIds.has(selectedSyncProjectId)) {
        setSelectedSyncProjectId("");
      }
      if (workflowSourceProjectId && !validProjectIds.has(workflowSourceProjectId)) {
        setWorkflowSourceProjectId("");
      }
      return projectModels;
    } catch (error) {
      setProjectError(toUserError(error, "Unable to load projects. Confirm your admin session and retry."));
      return [];
    } finally {
      setProjectsLoading(false);
    }
  };

  const loadMcpClients = async (): Promise<void> => {
    setMcpLoading(true);
    setMcpError("");
    setOneTimeMcpSecret(null);
    try {
      const result = await api.listMcpClients();
      setMcpClients(result.mcp_clients.map(toMcpClientModel));
    } catch (error) {
      setMcpError(toUserError(error, "Unable to load MCP clients. Confirm your admin session and retry."));
    } finally {
      setMcpLoading(false);
    }
  };

  const loadWorkflowsPageData = async (): Promise<void> => {
    setWorkflowsLoading(true);
    setWorkflowsError("");
    try {
      await api.getSession();
      await api.getCsrf();
      const [projectsPayload, workflowsPayload, sourcesPayload] = await Promise.all([
        api.listProjects(),
        api.listWorkflows(),
        api.listWorkflowSources(),
      ]);
      const projectModels = projectsPayload.projects.map(toProjectModel);
      setProjects(projectModels);
      setWorkflowProjectsCount(projectModels.length);
      setWorkflowsList(workflowsPayload.workflows.map(toWorkflowSummaryModel));
      setWorkflowSources(sourcesPayload.sources.map(toWorkflowSourceModel));
    } catch (error) {
      setWorkflowsError(toUserError(error, "Unable to load workflows dashboard. Confirm your admin session and retry."));
    } finally {
      setWorkflowsLoading(false);
    }
  };

  const loadRuns = async (): Promise<void> => {
    setRunsLoading(true);
    setRunsError("");
    try {
      const limit = Number.parseInt(runFilterLimit, 10);
      const offset = Number.parseInt(runFilterOffset, 10);
      const payload = await api.listRuns({
        status: runFilterStatus.trim() || undefined,
        limit: Number.isFinite(limit) && limit > 0 ? limit : 50,
        offset: Number.isFinite(offset) && offset >= 0 ? offset : 0,
      });
      setRunsRows(payload.runs.map(toRunRowModel));
    } catch (error) {
      setRunsError(toUserError(error, "Unable to load runs. Confirm your admin session and retry."));
    } finally {
      setRunsLoading(false);
    }
  };

  const loadLlmConfig = async (): Promise<void> => {
    setLlmLoading(true);
    setLlmError("");
    try {
      const payload = await api.getLlmConfig();
      setLlmConfig(toLlmConfigModel(payload));
    } catch (error) {
      setLlmError(toUserError(error, "Unable to load LLM configuration."));
    } finally {
      setLlmLoading(false);
    }
  };

  const onApplyRunsFilters = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    await loadRuns();
  };

  const onViewRunDetail = async (runId: string): Promise<void> => {
    setRunActionPendingId(`detail:${runId}`);
    setRunsError("");
    setRunsMessage("");
    try {
      const payload = await api.getRunDetail(runId);
      setRunDetail(toRunDetailModel(payload));
    } catch (error) {
      setRunsError(toUserError(error, `Unable to load run detail for ${runId}.`));
    } finally {
      setRunActionPendingId(null);
    }
  };

  const onCancelRun = async (runId: string): Promise<void> => {
    setRunActionPendingId(`cancel:${runId}`);
    setRunsError("");
    setRunsMessage("");
    try {
      await api.cancelRun(runId);
      setRunsMessage(`Run ${runId} cancellation requested.`);
      await loadRuns();
    } catch (error) {
      setRunsError(toUserError(error, `Unable to cancel run ${runId}.`));
    } finally {
      setRunActionPendingId(null);
    }
  };

  const onResumeRun = async (runId: string): Promise<void> => {
    setRunActionPendingId(`resume:${runId}`);
    setRunsError("");
    setRunsMessage("");
    try {
      const promptResult = window.prompt("Resume response text (optional)", "");
      if (promptResult === null) {
        setRunsMessage("Resume cancelled.");
        return;
      }
      const responseText = promptResult;
      await api.resumeRun(runId, responseText);
      setRunsMessage(`Run ${runId} resume submitted.`);
      await loadRuns();
      await onViewRunDetail(runId);
    } catch (error) {
      setRunsError(toUserError(error, `Unable to resume run ${runId}.`));
    } finally {
      setRunActionPendingId(null);
    }
  };

  const navigate = navigateBrowser;

  useEffect(() => {
    const onPopState = (): void => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener("popstate", onPopState);
    return () => {
      window.removeEventListener("popstate", onPopState);
    };
  }, []);

  useEffect(() => {
    if (currentPath === "/") {
      navigate("/login", { replace: true });
    }
  }, [currentPath]);

  useEffect(() => {
    let cancelled = false;
    let stopRealtime = (): void => undefined;

    const toProjectNameById = (items: ProjectModel[]): Map<string, string> => {
      return new Map(items.map((project) => [project.id, project.name]));
    };

    const toWatcherRows = (items: WatcherStateItem[], projectNameById: Map<string, string>): WatcherDashboardRow[] => {
      return items.map((item) => ({
        projectId: item.project_id,
        projectName: projectNameById.get(item.project_id) ?? null,
        state: item.state,
        dirtyCount: item.dirty_count,
        requiresReconciliation: item.requires_reconciliation,
        lastEventAt: item.last_event_at,
        updatedAt: item.updated_at,
      }));
    };

    const run = async (): Promise<void> => {
      try {
        if (currentPath === "/setup") {
          if (!cancelled) {
            setSetupDashboard({ loading: true, error: null, data: null });
          }
          const [session, system, dbSettingsResponse, llmConfig, projects, mcpClients] = await Promise.all([
            api.getSession(),
            api.getSystemStatus(),
            api.getDatabaseSettings(),
            api.getLlmConfig(),
            api.listProjects(),
            api.listMcpClients(),
          ]);
          const settings = toDatabaseSettingsModel(dbSettingsResponse);
          const llmObj = toObject(llmConfig);
          const providers = toObject(llmObj.providers);
          const profiles = toObject(llmObj.profiles);
          const llmConfigured = Object.keys(providers).length > 0 && Object.keys(profiles).length > 0;
          if (!cancelled) {
            setSetupDashboard({
              loading: false,
              error: null,
              data: {
                sessionActive: Object.keys(toObject(session)).length > 0,
                systemStatus: typeof toObject(system).status === "string" ? String(toObject(system).status) : "unknown",
                databaseConfigured: settings.configured,
                llmConfigured,
                projectCount: Array.isArray(toObject(projects).projects) ? (toObject(projects).projects as unknown[]).length : 0,
                mcpClientCount: Array.isArray(toObject(mcpClients).mcp_clients)
                  ? (toObject(mcpClients).mcp_clients as unknown[]).length
                  : 0,
              },
            });
          }
          return;
        }
        if (currentPath === "/database") {
          if (!cancelled) {
            setDbLoadError("");
            setDbSaveMessage("");
            setDbConnectionMessage("");
            setDbConnectionResult(null);
            setDbSavedPasswordForCopy(null);
          }
          await api.getCsrf();
          const [guidancePayload, settingsPayload] = await Promise.all([
            api.getSetupStatus(),
            api.getDatabaseSettings(),
          ]);
          const guidanceObj = toObject(guidancePayload);
          const settings = toDatabaseSettingsModel(settingsPayload);
          if (!cancelled) {
            setDbGuidance({
              image: typeof guidanceObj.image === "string" ? guidanceObj.image : "Unavailable",
              docker: typeof guidanceObj.docker === "string" ? guidanceObj.docker : "Unavailable",
              podman: typeof guidanceObj.podman === "string" ? guidanceObj.podman : "Unavailable",
              notes: Array.isArray(guidanceObj.notes)
                ? guidanceObj.notes.filter((note): note is string => typeof note === "string")
                : [],
            });
            setDbSettings(settings);
            setDbForm(toDatabaseForm(settings));
            setDbFieldErrors({});
          }
          return;
        }
        if (currentPath === "/projects") {
          if (!cancelled) {
            setContentState("");
            setProjectMessage("");
            setProjectError("");
            await loadProjects();
          }
          return;
        }
        if (currentPath === "/llm") {
          if (!cancelled) {
            setContentState("");
            setLlmMessage("");
            setLlmError("");
            setLlmPreview(null);
            setLlmProviderModalOpen(false);
            setLlmYamlModalOpen(false);
            await loadLlmConfig();
          }
          return;
        }
        if (currentPath === "/secrets") {
          if (!cancelled) setContentState("Secrets admin controls are available in a follow-up slice.");
          return;
        }
        if (currentPath === "/watchers") {
          if (!cancelled) {
            setContentState("");
            setWatchersLoading(true);
            setWatchersError("");
            setWatchersMessage("");
          }
          const projectsPayload = await api.listProjects();
          const projectModels = projectsPayload.projects.map(toProjectModel);
          if (!cancelled) setProjects(projectModels);
          const projectNameById = toProjectNameById(projectModels);
          const statePayload = await fetchWatcherState();
          if (!cancelled) {
            setWatchersRows(toWatcherRows(statePayload.items, projectNameById));
            setWatchersLoading(false);
          }

          const poller = createPollingFallback({
            intervalMs: 15000,
            fetcher: async () => fetchWatcherState(),
            onData: (payload) => {
              if (!cancelled) setWatchersRows(toWatcherRows(payload.items, projectNameById));
            },
            onError: (error) => {
              if (!cancelled) setWatchersError(toUserError(error, "Unable to refresh watcher status."));
            },
          });

          let stream: { stop: () => void } = { stop: () => undefined };
          try {
            stream = createEventStream({
              url: "/api/events/v1/watchers",
              eventName: "watcher.status",
              onMessage: (payload: { items?: WatcherStateItem[] }) => {
                const items = Array.isArray(payload.items) ? payload.items : [];
                if (!cancelled) setWatchersRows(toWatcherRows(items, projectNameById));
              },
              onError: () => {
                poller.start();
                if (!cancelled) {
                  setWatchersMessage("Live watcher updates are unavailable. Auto-refresh fallback is active.");
                }
              },
            });
          } catch {
            poller.start();
            if (!cancelled) setWatchersMessage("Live watcher updates are unavailable. Auto-refresh fallback is active.");
          }

          stopRealtime = (): void => {
            stream.stop();
            poller.stop();
          };
          return;
        }
        if (currentPath === "/sync") {
          if (!cancelled) {
            setContentState("");
            setSyncLoading(true);
            setSyncError("");
            setSyncMessage("");
          }
          const projectsPayload = await api.listProjects();
          const projectModels = projectsPayload.projects.map(toProjectModel);
          if (!cancelled) {
            setProjects(projectModels);
            if (selectedSyncProjectId && !projectModels.some((project) => project.id === selectedSyncProjectId)) {
              setSelectedSyncProjectId("");
            }
          }
          const statePayload = await fetchSyncState();
          if (!cancelled) {
            setSyncRows(buildSyncDashboardRows(statePayload.items, projectModels));
            setSyncLoading(false);
          }

          const poller = createPollingFallback({
            intervalMs: 15000,
            fetcher: async () => fetchSyncState(),
            onData: (payload) => {
              if (!cancelled) setSyncRows(buildSyncDashboardRows(payload.items, projectModels));
            },
            onError: (error) => {
              if (!cancelled) setSyncError(toUserError(error, "Unable to refresh sync queue status."));
            },
          });

          let stream: { stop: () => void } = { stop: () => undefined };
          try {
            stream = createEventStream({
              url: "/api/events/v1/sync",
              eventName: "sync.status",
              onMessage: (payload: { items?: SyncStateItem[] }) => {
                const items = Array.isArray(payload.items) ? payload.items : [];
                if (!cancelled) setSyncRows(buildSyncDashboardRows(items, projectModels));
              },
              onError: () => {
                poller.start();
                if (!cancelled) {
                  setSyncMessage("Live sync updates are unavailable. Auto-refresh fallback is active.");
                }
              },
            });
          } catch {
            poller.start();
            if (!cancelled) setSyncMessage("Live sync updates are unavailable. Auto-refresh fallback is active.");
          }

          stopRealtime = (): void => {
            stream.stop();
            poller.stop();
          };
          return;
        }
        if (currentPath === "/mcp-clients") {
          if (!cancelled) {
            setContentState("");
            setMcpMessage("");
            setMcpError("");
            await Promise.all([loadProjects(), loadMcpClients()]);
          }
          return;
        }
        if (currentPath === "/workflows") {
          if (!cancelled) {
            setContentState("");
            setWorkflowsMessage("");
            setWorkflowsError("");
            setSelectedWorkflowName(null);
            setWorkflowDetailText("");
            setWorkflowSchemaText("");
            await loadWorkflowsPageData();
          }
          return;
        }
        if (currentPath === "/runs") {
          if (!cancelled) {
            setContentState("");
            setRunsMessage("");
            setRunsError("");
            setRunDetail(null);
            await loadRuns();
          }
        }
      } catch (error) {
        if (!cancelled) {
          const message = toUserError(error, "Failed to load route data");
          if (currentPath === "/setup") {
            setSetupDashboard({ loading: false, error: message, data: null });
          } else if (currentPath === "/database") {
            setDbLoadError(message);
          } else if (currentPath === "/watchers") {
            setWatchersLoading(false);
            setWatchersError(message);
          } else if (currentPath === "/sync") {
            setSyncLoading(false);
            setSyncError(message);
          } else {
            setContentState(message);
          }
        }
      }
    };

    void run();
    return () => {
      cancelled = true;
      stopRealtime();
    };
  }, [api, currentPath]);

  const refreshWatchersState = async (): Promise<void> => {
    setWatchersError("");
    try {
      const projectNameById = new Map((await api.listProjects()).projects.map(toProjectModel).map((p) => [p.id, p.name]));
      const payload = await fetchWatcherState();
      setWatchersRows(
        payload.items.map((item) => ({
          projectId: item.project_id,
          projectName: projectNameById.get(item.project_id) ?? null,
          state: item.state,
          dirtyCount: item.dirty_count,
          requiresReconciliation: item.requires_reconciliation,
          lastEventAt: item.last_event_at,
          updatedAt: item.updated_at,
        })),
      );
    } catch (error) {
      setWatchersError(toUserError(error, "Unable to refresh watcher status."));
    }
  };

  const refreshSyncState = async (): Promise<void> => {
    setSyncError("");
    try {
      const projects = (await api.listProjects()).projects.map(toProjectModel);
      const payload = await fetchSyncState();
      setProjects(projects);
      if (selectedSyncProjectId && !projects.some((project) => project.id === selectedSyncProjectId)) {
        setSelectedSyncProjectId("");
      }
      setSyncRows(buildSyncDashboardRows(payload.items, projects));
    } catch (error) {
      setSyncError(toUserError(error, "Unable to refresh sync queue status."));
    }
  };

  const onWatcherAction = async (projectId: string, action: "pause" | "resume" | "disable"): Promise<void> => {
    setWatchersPendingAction(`${action}:${projectId}`);
    setWatchersError("");
    setWatchersMessage("");
    try {
      if (action === "pause") await api.pauseWatcher(projectId);
      if (action === "resume") await api.resumeWatcher(projectId);
      if (action === "disable") await api.disableWatcher(projectId);
      setWatchersMessage(`Watcher ${projectId} ${action}d.`);
      await refreshWatchersState();
    } catch (error) {
      setWatchersError(toUserError(error, `Unable to ${action} watcher ${projectId}.`));
    } finally {
      setWatchersPendingAction(null);
    }
  };

  const onSyncAction = async (projectId: string, action: "now" | "reconcile" | "rebuild"): Promise<void> => {
    setSyncPendingAction(`${action}:${projectId}`);
    setSyncError("");
    setSyncMessage("");
    try {
      if (action === "now") {
        await api.syncNow(projectId);
        setSyncMessage(`Sync requested for ${projectId}.`);
      }
      if (action === "reconcile") {
        await api.reconcileSync(projectId);
        setSyncMessage(`Reconcile requested for ${projectId}.`);
      }
      if (action === "rebuild") {
        await api.rebuildSync(projectId);
        setSyncMessage(`Rebuild requested for ${projectId}.`);
      }
      await refreshSyncState();
    } catch (error) {
      setSyncError(toUserError(error, `Unable to ${action} sync state for ${projectId}.`));
    } finally {
      setSyncPendingAction(null);
    }
  };

  const onLoginSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setLoginState("pending");
    setLoginMessage("");
    try {
      await api.login(password);
      setLoginState("ok");
      setLoginMessage("Signed in");
      setPassword("");
      navigate("/setup", { replace: true });
    } catch (error) {
      const message = error instanceof ApiHttpError ? error.message : "Login failed";
      setLoginState("error");
      setLoginMessage(message);
    }
  };

  const onReloadWorkflows = async (): Promise<void> => {
    setWorkflowsError("");
    setWorkflowsMessage("");
    try {
      const payload = await api.reloadWorkflows();
      setWorkflowsMessage(`Workflow registry reloaded: ${payload.total} workflows from ${payload.source_count} source(s).`);
      await loadWorkflowsPageData();
    } catch (error) {
      setWorkflowsError(`Unable to reload workflows: ${toUserError(error, "reload failed")}`);
    }
  };

  const onCreateWorkflowSource = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setWorkflowSourcePending(true);
    setWorkflowsError("");
    setWorkflowsMessage("");
    try {
      await api.createWorkflowSource({
        project_id: workflowSourceProjectId.trim(),
        source_path: workflowSourcePath.trim(),
        checksum: workflowSourceChecksum.trim() || null,
      });
      setWorkflowsMessage(`Workflow source added for project ${workflowSourceProjectId.trim()}.`);
      setWorkflowSourcePath("");
      setWorkflowSourceChecksum("");
      await loadWorkflowsPageData();
    } catch (error) {
      setWorkflowsError(toUserError(error, "Unable to add workflow source."));
    } finally {
      setWorkflowSourcePending(false);
    }
  };

  const onDeleteWorkflowSource = async (sourceId: string): Promise<void> => {
    setWorkflowActionPendingId(`delete:${sourceId}`);
    setWorkflowsError("");
    setWorkflowsMessage("");
    try {
      await api.deleteWorkflowSource(sourceId);
      setWorkflowsMessage(`Source ${sourceId} deleted.`);
      await loadWorkflowsPageData();
    } catch (error) {
      setWorkflowsError(toUserError(error, `Unable to delete source ${sourceId}.`));
    } finally {
      setWorkflowActionPendingId(null);
    }
  };

  const onValidateWorkflowSource = async (sourceId: string): Promise<void> => {
    setWorkflowActionPendingId(`validate:${sourceId}`);
    setWorkflowsError("");
    setWorkflowsMessage("");
    try {
      const result = await api.validateWorkflowSource(sourceId);
      setWorkflowsMessage(`Validation passed for ${sourceId}: ${result.total} workflow(s) discovered.`);
    } catch (error) {
      setWorkflowsError(`Unable to validate source ${sourceId}: ${toUserError(error, "validation failed")}`);
    } finally {
      setWorkflowActionPendingId(null);
    }
  };

  const onLoadWorkflowSchema = async (): Promise<void> => {
    setWorkflowsError("");
    setWorkflowsMessage("");
    try {
      const schema = await api.getWorkflowSchema();
      setWorkflowSchemaText(JSON.stringify(schema, null, 2));
      setWorkflowsMessage("Workflow schema loaded.");
    } catch (error) {
      setWorkflowsError(toUserError(error, "Unable to load workflow schema."));
    }
  };

  const onViewWorkflowDetails = async (workflowName: string): Promise<void> => {
    setSelectedWorkflowName(workflowName);
    setWorkflowsError("");
    setWorkflowsMessage("");
    try {
      const details = await api.getWorkflowDetail(workflowName);
      setWorkflowDetailText(JSON.stringify(details, null, 2));
    } catch (error) {
      setWorkflowsError(toUserError(error, `Unable to load workflow details for ${workflowName}.`));
      setWorkflowDetailText("");
    }
  };

  const onImportDatabaseDsn = (): void => {
    try {
      const parsed = parsePostgresDsnForForm(dbForm.dsnImport);
      setDbForm((current) => ({
        ...current,
        ...parsed,
        dsnImport: "",
        passwordConfigured: current.passwordConfigured || (parsed.password?.length ?? 0) > 0,
        passwordClear: false,
      }));
      setDbFieldErrors((current) => ({ ...current, dsnImport: undefined }));
      setDbSaveMessage("DSN imported into structured profile fields.");
    } catch (error) {
      setDbFieldErrors((current) => ({ ...current, dsnImport: toUserError(error, "Invalid PostgreSQL DSN.") }));
    }
  };

  const onSaveDatabaseSettings = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const errors: Partial<Record<keyof DatabaseProfileForm, string>> = {};
    const port = Number.parseInt(dbForm.port, 10);
    const hostPort = Number.parseInt(dbForm.containerHostPort, 10);
    const savedPasswordForCopy = dbForm.passwordClear || dbForm.password.length === 0 ? null : dbForm.password;

    if (dbForm.enabled) {
      if (dbForm.host.trim().length === 0) errors.host = "Host is required.";
      if (dbForm.database.trim().length === 0) errors.database = "Database is required.";
      if (dbForm.username.trim().length === 0) errors.username = "Username is required.";
      if (!Number.isInteger(port) || port < 1 || port > 65535) errors.port = "Port must be an integer between 1 and 65535.";
      if (!Number.isInteger(hostPort) || hostPort < 1 || hostPort > 65535) {
        errors.containerHostPort = "Host port must be an integer between 1 and 65535.";
      }
      if (!VALID_NAME_PATTERN.test(dbForm.containerName)) {
        errors.containerName = "Container name must match ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$.";
      }
      if (!VALID_NAME_PATTERN.test(dbForm.volumeName)) {
        errors.volumeName = "Volume name must match ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$.";
      }
      if (dbForm.containerImage.trim().length === 0) {
        errors.containerImage = "Container image is required.";
      }
      if (!dbForm.passwordClear && dbForm.password.trim().length === 0 && !dbForm.passwordConfigured) {
        errors.password = "Password is required unless you clear it or keep a configured password.";
      }
    }

    setDbFieldErrors(errors);
    if (Object.keys(errors).length > 0) {
      setDbSaveMessage("Resolve the highlighted database settings errors.");
      return;
    }

    setDbSavePending(true);
    setDbSaveMessage("");
    try {
      const payload = await api.saveDatabaseSettings({
        enabled: dbForm.enabled,
        host: dbForm.host.trim(),
        port,
        database: dbForm.database.trim(),
        username: dbForm.username.trim(),
        password: dbForm.password.length > 0 ? dbForm.password : null,
        password_clear: dbForm.passwordClear,
        ssl_mode: dbForm.sslMode,
        extra_params: dbForm.extraParams.trim(),
        container_name: dbForm.containerName.trim(),
        container_image: dbForm.containerImage.trim(),
        container_host_port: hostPort,
        volume_name: dbForm.volumeName.trim(),
        dsn_import: null,
      });
      const settings = toDatabaseSettingsModel(payload);
      setDbSettings(settings);
      setDbForm((current) => ({ ...toDatabaseForm(settings), password: "", dsnImport: current.dsnImport }));
      setDbSavedPasswordForCopy(savedPasswordForCopy);
      setDbSaveMessage("Database settings saved.");
    } catch (error) {
      setDbSaveMessage(toUserError(error, "Unable to save database settings."));
    } finally {
      setDbSavePending(false);
    }
  };

  const onTestDatabaseConnection = async (): Promise<void> => {
    setDbTestPending(true);
    setDbConnectionMessage("");
    try {
      const result = toConnectionTestModel(await api.testDatabaseConnection());
      setDbConnectionResult(result);
      setDbConnectionMessage(
        result.ok ? "Connection test passed." : `Connection check requires attention (${result.status}).`,
      );
    } catch (error) {
      setDbConnectionMessage(toUserError(error, "Unable to run database connection test."));
    } finally {
      setDbTestPending(false);
    }
  };

  const onCopyDatabaseCommand = async (label: "Docker" | "Podman", command: string): Promise<void> => {
    setDbCopyStatus("");
    setDbCopyError("");
    try {
      await copyTextToClipboard(command);
      setDbCopyStatus(`${label} command copied to clipboard.`);
    } catch {
      setDbCopyError("Unable to copy command to clipboard.");
    }
  };

  const onCreateProject = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setProjectCreatePending(true);
    setProjectError("");
    setProjectMessage("");
    try {
      await api.createProject({
        name: projectName.trim(),
        slug: projectSlug.trim(),
        palace: projectPalace.trim(),
        default_wing: projectDefaultWing.trim(),
        default_room: projectDefaultRoom.trim(),
        fs_root: projectFsRoot.trim(),
        fs_allowlist: normalizePathList(projectAllowlistInput),
      });
      setProjectMessage("Project registered successfully.");
      setProjectName("");
      setProjectSlug("");
      setProjectPalace("");
      setProjectDefaultWing("");
      setProjectDefaultRoom("");
      setProjectFsRoot(DEFAULT_PROJECT_FS_ROOT);
      setProjectAllowlistInput("");
      await loadProjects();
    } catch (error) {
      setProjectError(toUserError(error, "Unable to register project. Review values and retry."));
    } finally {
      setProjectCreatePending(false);
    }
  };

  const focusPathPickerTarget = (): void => {
    if (pathPickerTarget === "fsRoot") {
      projectFsRootRef.current?.focus();
      return;
    }
    projectAllowlistRef.current?.focus();
  };

  const openPathPicker = (target: FolderBrowserTarget): void => {
    setPathPickerTarget(target);
    setPathPickerOpen(true);
  };

  const closePathPicker = (): void => {
    setPathPickerOpen(false);
    queueMicrotask(() => {
      focusPathPickerTarget();
    });
  };

  const listServerPathEntries = async (options?: {
    path?: string;
    selectionType?: "folder" | "file";
    extensions?: string[];
  }): Promise<PathListing> => {
    const payload = await api.listServerPathEntries(options);
    return {
      root: payload.root,
      path: payload.path,
      parent: payload.parent,
      canGoUp: payload.can_go_up,
      entries: payload.entries,
    };
  };

  const onSelectPath = (selectedPath: string): void => {
    if (pathPickerTarget === "fsRoot") {
      setProjectFsRoot(selectedPath);
      setProjectMessage(`FS root set to ${selectedPath}.`);
    } else {
      setProjectAllowlistInput((current) => {
        const selected = selectedPath.trim();
        if (selected.length === 0) {
          return current;
        }

        const entries = current
          .split(/[,\n]/)
          .map((item) => item.trim())
          .filter((item) => item.length > 0);
        const hasDuplicate = entries.some((item) => item === selected);

        if (hasDuplicate) {
          setProjectMessage(`${selected} is already in allowlist paths.`);
          return current;
        }

        const next = current.trim().length === 0 ? selected : `${current.trimEnd()}\n${selected}`;
        setProjectMessage(`Added ${selected} to allowlist paths.`);
        return next;
      });
    }
    closePathPicker();
  };

  const onConfirmDeleteProject = async (): Promise<void> => {
    if (!projectDeleteTarget) return;
    setProjectDeletePending(true);
    setProjectError("");
    setProjectMessage("");
    try {
      await api.deleteProject(projectDeleteTarget.id);
      setProjectMessage(`Project ${projectDeleteTarget.id} deleted.`);
      setProjectDeleteTarget(null);
      setProjectDeleteConfirm("");
      await loadProjects();
    } catch (error) {
      setProjectError(toUserError(error, "Unable to delete project. Retry if this project is still required."));
    } finally {
      setProjectDeletePending(false);
    }
  };

  const onCreateMcpClient = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setMcpCreatePending(true);
    setMcpError("");
    setMcpMessage("");
    try {
      const result = await api.createMcpClient({
        label: mcpLabel.trim(),
        project_ids: mcpSelectedProjectIds,
      });
      const token = toNonEmptyString(result.token);
      if (!token) {
        setOneTimeMcpSecret(null);
        await loadMcpClients();
        setMcpError(MCP_SECRET_MISSING_ERROR);
        return;
      }
      const secretLabel = toNonEmptyString(result.label) ?? mcpLabel.trim();
      setOneTimeMcpSecret({
        token,
        configSnippet: toNonEmptyString(result.config_snippet),
        label: secretLabel,
      });
      setMcpMessage("MCP client created. Save the token now—it will not be shown again.");
      setMcpLabel("");
      setMcpSelectedProjectIds([]);
      await loadMcpClients();
      setOneTimeMcpSecret({
        token,
        configSnippet: toNonEmptyString(result.config_snippet),
        label: secretLabel,
      });
    } catch (error) {
      setMcpError(toUserError(error, "Unable to create MCP client. Confirm label and project access."));
    } finally {
      setMcpCreatePending(false);
    }
  };

  const onRevokeMcpClient = async (tokenId: string): Promise<void> => {
    setMcpMutationPendingId(tokenId);
    setMcpError("");
    setMcpMessage("");
    try {
      await api.revokeMcpClient(tokenId);
      setMcpMessage(`MCP client ${tokenId} revoked.`);
      await loadMcpClients();
    } catch (error) {
      setMcpError(toUserError(error, "Unable to revoke MCP client token."));
    } finally {
      setMcpMutationPendingId(null);
    }
  };

  const onRegenerateMcpClient = async (tokenId: string): Promise<void> => {
    setMcpMutationPendingId(tokenId);
    setMcpError("");
    setMcpMessage("");
    try {
      const result = await api.regenerateMcpClient(tokenId);
      const token = toNonEmptyString(result.token);
      if (!token) {
        setOneTimeMcpSecret(null);
        await loadMcpClients();
        setMcpError(MCP_SECRET_MISSING_ERROR);
        return;
      }
      setOneTimeMcpSecret({
        token,
        configSnippet: toNonEmptyString(result.config_snippet),
        label: toNonEmptyString(result.label) ?? tokenId,
      });
      setMcpMessage("MCP token regenerated. Save the new token now—it will not be shown again.");
      await loadMcpClients();
      setOneTimeMcpSecret({
        token,
        configSnippet: toNonEmptyString(result.config_snippet),
        label: toNonEmptyString(result.label) ?? tokenId,
      });
    } catch (error) {
      setMcpError(toUserError(error, "Unable to regenerate MCP client token."));
    } finally {
      setMcpMutationPendingId(null);
    }
  };

  const normalizedLlmConfig = (config: LlmConfigModel): LlmConfigModel => {
    const profiles = Object.fromEntries(
      Object.entries(config.profiles).map(([id, profile]) => [
        id,
        {
          provider: profile.provider.trim(),
          model: profile.model.trim(),
          temperature: profile.temperature,
          max_tokens: profile.max_tokens,
          description: profile.description,
        },
      ]),
    );
    const defaultProfile = config.default_profile && profiles[config.default_profile] ? config.default_profile : null;
    return {
      version: "1.0",
      providers: Object.fromEntries(
        Object.entries(config.providers).map(([id, provider]) => [
          id,
          {
            type: provider.type.trim(),
            api_url: provider.api_url,
            api_key_secret: provider.api_key_secret,
            model: provider.model,
            timeout: provider.timeout,
            max_retries: provider.max_retries,
            retry_delay: provider.retry_delay,
            extra_headers: provider.extra_headers,
            deployment_name: provider.deployment_name,
            api_version: provider.api_version,
          },
        ]),
      ),
      profiles,
      default_profile: defaultProfile,
    };
  };

  const llmConfigPayload = (config: LlmConfigModel): LlmConfigPayload => {
    const normalized = normalizedLlmConfig(config);
    const providers = Object.fromEntries(
      Object.entries(normalized.providers).map(([id, provider]) => {
        const payload: ProviderConfigPayload = {
          type: provider.type,
          api_url: provider.api_url,
          api_key_secret: provider.api_key_secret,
          model: provider.model,
          extra_headers: provider.extra_headers,
          deployment_name: provider.deployment_name,
          api_version: provider.api_version,
        };
        if (provider.timeout !== null) payload.timeout = provider.timeout;
        if (provider.max_retries !== null) payload.max_retries = provider.max_retries;
        if (provider.retry_delay !== null) payload.retry_delay = provider.retry_delay;
        return [id, payload];
      }),
    );
    return { ...normalized, providers };
  };

  const validateLlmConfig = (config: LlmConfigModel): string | null => {
    for (const [profileId, profile] of Object.entries(config.profiles)) {
      if (!config.providers[profile.provider]) {
        return `Profile ${profileId} references missing provider ${profile.provider}.`;
      }
      if (profile.model.trim().length === 0) {
        return `Profile ${profileId} model is required.`;
      }
    }
    if (config.default_profile && !config.profiles[config.default_profile]) {
      return `Default profile ${config.default_profile} does not exist.`;
    }
    return null;
  };

  const openNewLlmProviderModal = (): void => {
    setLlmError("");
    setLlmMessage("");
    setLlmProviderForm(DEFAULT_LLM_PROVIDER_FORM);
    setLlmProviderModalOpen(true);
  };

  const closeLlmProviderModal = (): void => {
    setLlmProviderForm(DEFAULT_LLM_PROVIDER_FORM);
    setLlmProviderModalOpen(false);
  };

  const openLlmYamlModal = (): void => {
    setLlmError("");
    setLlmMessage("");
    setLlmPreview(null);
    setLlmYamlModalOpen(true);
  };

  const closeLlmYamlModal = (): void => {
    setLlmPreview(null);
    setLlmYamlModalOpen(false);
  };

  const onSaveLlmProvider = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    setLlmError("");
    setLlmMessage("");
    try {
      const id = llmProviderForm.id.trim();
      if (!VALID_NAME_PATTERN.test(id)) throw new Error("Provider ID must match ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$.");
      if (llmProviderForm.type.trim().length === 0) throw new Error("Provider type is required.");
      const provider: ProviderConfig = {
        type: llmProviderForm.type.trim(),
        api_url: toNullableString(llmProviderForm.apiUrl),
        api_key_secret: toNullableString(llmProviderForm.apiKeySecret),
        model: toNullableString(llmProviderForm.model),
        timeout: numberField(llmProviderForm.timeout, "Timeout"),
        max_retries: numberField(llmProviderForm.maxRetries, "Max retries"),
        retry_delay: numberField(llmProviderForm.retryDelay, "Retry delay"),
        extra_headers: parseExtraHeaders(llmProviderForm.extraHeaders),
        deployment_name: toNullableString(llmProviderForm.deploymentName),
        api_version: toNullableString(llmProviderForm.apiVersion),
      };
      setLlmConfig((current) => normalizedLlmConfig({
        ...current,
        providers: { ...current.providers, [id]: provider },
      }));
      setLlmProviderForm(DEFAULT_LLM_PROVIDER_FORM);
      setLlmProviderModalOpen(false);
      setLlmMessage(`Provider ${id} staged. Save LLM configuration to persist.`);
    } catch (error) {
      setLlmError(toUserError(error, "Unable to stage provider."));
    }
  };

  const onEditLlmProvider = (providerId: string): void => {
    const provider = llmConfig.providers[providerId];
    if (!provider) return;
    setLlmError("");
    setLlmProviderForm(toLlmProviderForm(providerId, provider));
    setLlmProviderModalOpen(true);
    setLlmMessage(`Editing provider ${providerId}.`);
  };

  const onDeleteLlmProvider = (providerId: string): void => {
    const referencingProfiles = Object.entries(llmConfig.profiles)
      .filter(([, profile]) => profile.provider === providerId)
      .map(([profileId]) => profileId);
    setLlmError("");
    setLlmMessage("");
    if (referencingProfiles.length > 0) {
      setLlmError(`Cannot delete provider ${providerId}; remove or reassign profiles first: ${referencingProfiles.join(", ")}.`);
      return;
    }
    setLlmConfig((current) => {
      const nextProviders = { ...current.providers };
      delete nextProviders[providerId];
      return normalizedLlmConfig({ ...current, providers: nextProviders });
    });
    setLlmMessage(`Provider ${providerId} staged for deletion. Save LLM configuration to persist.`);
  };

  const onSaveLlmProfile = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    setLlmError("");
    setLlmMessage("");
    try {
      const id = llmProfileForm.id.trim();
      if (!VALID_NAME_PATTERN.test(id)) throw new Error("Profile ID must match ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$.");
      if (!llmConfig.providers[llmProfileForm.provider]) throw new Error("Profile provider must reference an existing provider.");
      const model = llmProfileForm.model.trim();
      if (model.length === 0) throw new Error("Profile model is required.");
      const profile: ProfileConfig = {
        provider: llmProfileForm.provider,
        model,
        temperature: numberField(llmProfileForm.temperature, "Temperature"),
        max_tokens: numberField(llmProfileForm.maxTokens, "Max tokens"),
        description: toNullableString(llmProfileForm.description),
      };
      setLlmConfig((current) => normalizedLlmConfig({
        ...current,
        profiles: { ...current.profiles, [id]: profile },
      }));
      setLlmProfileForm(DEFAULT_LLM_PROFILE_FORM);
      setLlmMessage(`Profile ${id} staged. Save LLM configuration to persist.`);
    } catch (error) {
      setLlmError(toUserError(error, "Unable to stage profile."));
    }
  };

  const onEditLlmProfile = (profileId: string): void => {
    const profile = llmConfig.profiles[profileId];
    if (!profile) return;
    setLlmProfileForm(toLlmProfileForm(profileId, profile));
    setLlmMessage(`Editing profile ${profileId}.`);
  };

  const onDeleteLlmProfile = (profileId: string): void => {
    setLlmError("");
    setLlmMessage("");
    setLlmConfig((current) => {
      const nextProfiles = { ...current.profiles };
      delete nextProfiles[profileId];
      return normalizedLlmConfig({
        ...current,
        profiles: nextProfiles,
        default_profile: current.default_profile === profileId ? null : current.default_profile,
      });
    });
    setLlmMessage(`Profile ${profileId} staged for deletion. Save LLM configuration to persist.`);
  };

  const onSaveLlmConfig = async (): Promise<void> => {
    setLlmSaving(true);
    setLlmError("");
    setLlmMessage("");
    try {
      const normalized = normalizedLlmConfig(llmConfig);
      const validationError = validateLlmConfig(normalized);
      if (validationError) {
        setLlmError(validationError);
        return;
      }
      const payload = llmConfigPayload(normalized);
      const saved = await api.updateLlmConfig(payload);
      setLlmConfig(toLlmConfigModel(saved));
      setLlmMessage("LLM configuration saved.");
    } catch (error) {
      setLlmError(toUserError(error, "Unable to save LLM configuration."));
    } finally {
      setLlmSaving(false);
    }
  };

  const onExportLlmYaml = async (): Promise<void> => {
    setLlmError("");
    setLlmMessage("");
    try {
      const payload = await api.exportLlmConfig();
      const rawYaml = typeof payload.raw_yaml === "string" ? payload.raw_yaml : JSON.stringify(payload, null, 2);
      const blob = new Blob([rawYaml], { type: "application/x-yaml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "llm-config.yaml";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setLlmMessage("LLM YAML downloaded.");
    } catch (error) {
      setLlmError(toUserError(error, "Unable to export LLM YAML."));
    }
  };

  const onUploadLlmYaml = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.currentTarget.files?.[0];
    if (!file) return;
    setLlmError("");
    setLlmMessage("");
    try {
      const rawYaml = await readUploadText(file);
      setLlmYamlImport(rawYaml);
      setLlmPreview(null);
    } catch (error) {
      setLlmError(toUserError(error, "Unable to read YAML file."));
    }
  };

  const onPreviewLlmYaml = async (): Promise<void> => {
    setLlmError("");
    setLlmMessage("");
    try {
      const preview = toLlmConfigModel(await api.previewLlmConfig(llmYamlImport));
      setLlmPreview(preview);
      setLlmMessage(
        `Preview loaded: ${Object.keys(preview.providers).length} providers, ${Object.keys(preview.profiles).length} profiles.`,
      );
    } catch (error) {
      setLlmError(toUserError(error, "Unable to preview LLM YAML."));
    }
  };

  const onImportLlmYaml = async (): Promise<void> => {
    setLlmError("");
    setLlmMessage("");
    try {
      const imported = toLlmConfigModel(await api.importLlmConfig(llmYamlImport));
      setLlmConfig(imported);
      setLlmPreview(null);
      setLlmYamlModalOpen(false);
      setLlmMessage("LLM YAML imported into SQLite-backed config.");
    } catch (error) {
      setLlmError(toUserError(error, "Unable to import LLM YAML."));
    }
  };

  const page = route
    ? { title: route.title, description: route.description, unknown: false }
    : currentPath === "/"
      ? { ...DEFAULT_PAGE, unknown: false }
      : { ...NOT_FOUND_PAGE, unknown: true };

  const previewPassword =
    dbForm.password.length > 0
      ? dbForm.password
      : dbForm.passwordConfigured
        ? "<configured-password>"
        : "<password>";
  const previewHost = dbForm.host.trim().length > 0 ? dbForm.host.trim() : "<host>";
  const previewDatabase = dbForm.database.trim().length > 0 ? dbForm.database.trim() : "<database>";
  const previewUsername = dbForm.username.trim().length > 0 ? dbForm.username.trim() : "<username>";
  const previewPort = dbForm.port.trim().length > 0 ? dbForm.port.trim() : "5432";
  const previewContainerPort = dbForm.containerHostPort.trim().length > 0 ? dbForm.containerHostPort.trim() : "5432";
  const buildRuntimeCommand = (engine: "docker" | "podman", password: string): string =>
    `${engine} run --name ${quoteShellPreview(dbForm.containerName)} ` +
    `-e POSTGRES_DB=${quoteShellPreview(previewDatabase)} ` +
    `-e POSTGRES_USER=${quoteShellPreview(previewUsername)} ` +
    `-e POSTGRES_PASSWORD=${quoteShellPreview(password)} ` +
    `-p ${quoteShellPreview(previewContainerPort)}:5432 ` +
    `-v ${quoteShellPreview(dbForm.volumeName)}:/var/lib/postgresql/data ` +
    `${quoteShellPreview(dbForm.containerImage)}`;
  const dockerPreview = buildRuntimeCommand("docker", previewPassword);
  const podmanPreview = buildRuntimeCommand("podman", previewPassword);
  const copyPassword =
    dbForm.password.length > 0
      ? dbForm.password
      : !dbForm.passwordClear && dbSavedPasswordForCopy
        ? dbSavedPasswordForCopy
        : previewPassword;
  const dockerCopyCommand = buildRuntimeCommand("docker", copyPassword);
  const podmanCopyCommand = buildRuntimeCommand("podman", copyPassword);
  const dsnPreview =
    `postgresql://${quoteShellPreview(previewUsername)}:${quoteShellPreview(previewPassword)}` +
    `@${quoteShellPreview(previewHost)}:${quoteShellPreview(previewPort)}/${quoteShellPreview(previewDatabase)}`;
  const projectReferenceById = new Map(projects.map((project) => [project.id, `${project.name} (${project.id})`]));
  const formatProjectReference = (projectId: string): string => projectReferenceById.get(projectId) ?? projectId;
  const visibleSyncRows = selectedSyncProjectId
    ? syncRows.filter((row) => row.projectId === selectedSyncProjectId)
    : syncRows;
  const selectedSyncRow = selectedSyncProjectId
    ? syncRows.find((row) => row.projectId === selectedSyncProjectId) ?? null
    : null;
  const syncDirtyTotal = syncRows.reduce((total, row) => total + row.dirtyCount, 0);
  const syncReconcileCount = syncRows.filter((row) => row.requiresReconciliation).length;
  const syncQueuedCount = syncRows.filter((row) => row.dirtyCount > 0 || row.syncState !== "idle").length;
  const llmProviderEntries = Object.entries(llmConfig.providers);
  const llmProfileEntries = Object.entries(llmConfig.profiles);
  const llmDefaultProfileLabel = llmConfig.default_profile ?? "None";

  return (
    <div className="admin-shell">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>

      <header className="app-header" role="banner">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">wm</span>
          <div>
            <div className="brand">workflows-mcp</div>
            <p>Administration control plane</p>
          </div>
        </div>
        <div className="header-meta" aria-label="Admin shell status">
          <StatusBadge tone={currentPath === "/login" ? "warning" : "success"}>
            {currentPath === "/login" ? "Session required" : "Console ready"}
          </StatusBadge>
          <a href="/docs">Docs</a>
        </div>
      </header>

      <div className="layout">
        <nav aria-label="Primary navigation" className="app-nav">
          <div className="nav-header">
            <h2 className="nav-title">Control plane</h2>
            <p>Configure projects, registry state, access, and run history.</p>
          </div>
          <ul>
            {ROUTES.filter((entry) => PRIMARY_NAV_PATHS.has(entry.path)).map((entry, index) => (
              <li key={entry.path}>
                <a
                  href={entry.path}
                  aria-current={entry.path === currentPath ? "page" : undefined}
                  onClick={(event) => {
                    event.preventDefault();
                    navigate(entry.path);
                  }}
                >
                  <span className="nav-index" aria-hidden="true">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <span>{entry.label}</span>
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <main id="main-content" className="app-main">
          <PageHeader
            eyebrow={currentPath === "/login" ? "Access" : page.unknown ? "Routing" : "Admin workspace"}
            title={page.title}
            description={page.description}
            actions={
              currentPath !== "/login" && !page.unknown ? (
                <StatusBadge tone="info">{route?.label ?? "Overview"}</StatusBadge>
              ) : undefined
            }
          />
          {currentPath === "/login" ? (
            <form className="admin-form" onSubmit={(event) => void onLoginSubmit(event)}>
              <label htmlFor="admin-password">Password</label>
              <input
                id="admin-password"
                name="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
              <ActionButton type="submit" variant="primary" disabled={loginState === "pending"}>
                Sign in
              </ActionButton>
              {loginMessage ? <p role="status">{loginMessage}</p> : null}
            </form>
          ) : null}

          {currentPath !== "/login" && !page.unknown && contentState ? (
            <p role="status">{contentState}</p>
          ) : null}

          {currentPath === "/setup" ? (
            <section className="admin-section" aria-labelledby="setup-checklist-title">
              <h2 id="setup-checklist-title">Setup checklist</h2>
              {setupDashboard.loading ? <p role="status">Loading setup dashboard…</p> : null}
              {setupDashboard.error ? <p role="alert">{setupDashboard.error}</p> : null}
              {setupDashboard.data ? (
                <>
                  <ul className="checklist-grid" aria-label="Setup status checks">
                    <li>
                      <strong>Admin session active</strong>
                      <StatusBadge tone={setupDashboard.data.sessionActive ? "success" : "warning"}>
                        {setupDashboard.data.sessionActive ? "Yes" : "No"}
                      </StatusBadge>
                    </li>
                    <li>
                      <strong>Public status</strong>
                      <StatusBadge tone="info">{setupDashboard.data.systemStatus}</StatusBadge>
                    </li>
                    <li>
                      <strong>Database configured</strong>
                      <StatusBadge tone={setupDashboard.data.databaseConfigured ? "success" : "warning"}>
                        {setupDashboard.data.databaseConfigured ? "Yes" : "No"}
                      </StatusBadge>
                    </li>
                    <li>
                      <strong>LLM config loaded</strong>
                      <StatusBadge tone={setupDashboard.data.llmConfigured ? "success" : "warning"}>
                        {setupDashboard.data.llmConfigured ? "Yes" : "No"}
                      </StatusBadge>
                    </li>
                    <li>
                      <strong>Projects registered</strong>
                      <StatusBadge tone={setupDashboard.data.projectCount > 0 ? "success" : "warning"}>
                        {setupDashboard.data.projectCount}
                      </StatusBadge>
                    </li>
                    <li>
                      <strong>MCP clients issued</strong>
                      <StatusBadge tone={setupDashboard.data.mcpClientCount > 0 ? "success" : "warning"}>
                        {setupDashboard.data.mcpClientCount}
                      </StatusBadge>
                    </li>
                  </ul>
                  <Panel
                    title="Next actions"
                    description="Complete the essentials before running automation in shared environments."
                  >
                    <ul>
                      {!setupDashboard.data.databaseConfigured ? (
                        <li>
                          <a href="/database">Configure database</a>
                        </li>
                      ) : null}
                      <li>
                        <a href="/llm">Review LLM configuration</a>
                      </li>
                      {setupDashboard.data.projectCount === 0 ? (
                        <li>
                          <a href="/projects">Add first project</a>
                        </li>
                      ) : null}
                      {setupDashboard.data.mcpClientCount === 0 ? (
                        <li>
                          <a href="/mcp-clients">Create MCP client token</a>
                        </li>
                      ) : null}
                    </ul>
                  </Panel>
                </>
              ) : null}
            </section>
          ) : null}

          {currentPath === "/database" ? (
            <section className="admin-section database-workbench" aria-label="Database management">
              {dbLoadError ? <p role="alert">{dbLoadError}</p> : null}

              <section className="database-status-panel" aria-label="Profile status">
                <div className="database-status-panel__copy">
                  <p className="database-kicker">workflowsctl / database-profile / local</p>
                  <h2>PostgreSQL metadata profile</h2>
                  <p>Configure the durable store used by workflow metadata, sync queues, and run history.</p>
                </div>
                <dl className="database-status-grid" aria-label="Database profile status chips">
                  <div>
                    <dt>Backend</dt>
                    <dd>
                      <StatusBadge tone={dbForm.enabled ? "success" : "warning"}>
                        enabled: {dbForm.enabled ? "on" : "off"}
                      </StatusBadge>
                    </dd>
                  </div>
                  <div>
                    <dt>Profile</dt>
                    <dd>
                      <StatusBadge tone={dbSettings?.configured ? "success" : "warning"}>
                        configured: {dbSettings?.configured ? "yes" : "no"}
                      </StatusBadge>
                    </dd>
                  </div>
                  <div>
                    <dt>Secret</dt>
                    <dd>
                      <StatusBadge tone={dbForm.password.length > 0 || dbForm.passwordConfigured ? "success" : "warning"}>
                        password: {dbForm.password.length > 0 ? "typed" : dbForm.passwordConfigured ? "configured" : "pending"}
                      </StatusBadge>
                    </dd>
                  </div>
                  <div>
                    <dt>Import</dt>
                    <dd>
                      <StatusBadge tone={dbForm.dsnImport.trim().length > 0 ? "info" : "neutral"}>
                        legacy re-entry: {dbForm.dsnImport.trim().length > 0 ? "staged" : "clean"}
                      </StatusBadge>
                    </dd>
                  </div>
                </dl>
              </section>

              <div className="database-workbench__grid">
                <article className="admin-card database-profile-card" aria-labelledby="db-settings-title">
                  <div className="database-card-header">
                    <div>
                      <p className="database-kicker">Profile</p>
                      <h2 id="db-settings-title">Database settings</h2>
                      {dbSettings ? (
                        <p>
                          Configured: {dbSettings.configured ? "Yes" : "No"} · Last updated: {dbSettings.updatedAt}
                        </p>
                      ) : null}
                    </div>
                  </div>

                  <form className="admin-form database-profile-form" onSubmit={(event) => void onSaveDatabaseSettings(event)}>
                    <label className="database-toggle">
                      <input
                        type="checkbox"
                        checked={dbForm.enabled}
                        onChange={(event) => setDbForm((current) => ({ ...current, enabled: event.target.checked }))}
                      />
                      <span>
                        <strong>Enable PostgreSQL metadata backend</strong>
                        <small>Use this profile for shared metadata persistence instead of local-only storage.</small>
                      </span>
                    </label>

                    <fieldset className="database-fieldset">
                      <legend>Connection profile</legend>
                      <p>Endpoint, credentials, and TLS behavior for the PostgreSQL database.</p>
                      <div className="database-form-grid">
                        <div className="field-group">
                          <label htmlFor="database-host">Host</label>
                          <input id="database-host" value={dbForm.host} onChange={(event) => setDbForm((current) => ({ ...current, host: event.target.value }))} />
                          {dbFieldErrors.host ? <p role="alert">{dbFieldErrors.host}</p> : null}
                        </div>
                        <div className="field-group">
                          <label htmlFor="database-port">Port</label>
                          <input id="database-port" value={dbForm.port} onChange={(event) => setDbForm((current) => ({ ...current, port: event.target.value }))} />
                          {dbFieldErrors.port ? <p role="alert">{dbFieldErrors.port}</p> : null}
                        </div>
                        <div className="field-group">
                          <label htmlFor="database-name">Database</label>
                          <input id="database-name" value={dbForm.database} onChange={(event) => setDbForm((current) => ({ ...current, database: event.target.value }))} />
                          {dbFieldErrors.database ? <p role="alert">{dbFieldErrors.database}</p> : null}
                        </div>
                        <div className="field-group">
                          <label htmlFor="database-username">Username</label>
                          <input id="database-username" value={dbForm.username} onChange={(event) => setDbForm((current) => ({ ...current, username: event.target.value }))} />
                          {dbFieldErrors.username ? <p role="alert">{dbFieldErrors.username}</p> : null}
                        </div>
                        <div className="field-group">
                          <label htmlFor="database-password">Password</label>
                          <input
                            id="database-password"
                            type="password"
                            autoComplete="off"
                            value={dbForm.password}
                            onChange={(event) => setDbForm((current) => ({ ...current, password: event.target.value, passwordClear: false }))}
                          />
                          {dbFieldErrors.password ? <p role="alert">{dbFieldErrors.password}</p> : null}
                        </div>
                        <div className="field-group">
                          <label htmlFor="database-ssl-mode">SSL mode</label>
                          <select id="database-ssl-mode" value={dbForm.sslMode} onChange={(event) => setDbForm((current) => ({ ...current, sslMode: event.target.value as DatabaseSslMode }))}>
                            <option value="disable">disable</option>
                            <option value="prefer">prefer</option>
                            <option value="require">require</option>
                            <option value="verify-ca">verify-ca</option>
                            <option value="verify-full">verify-full</option>
                          </select>
                        </div>
                        <div className="field-group database-form-grid__wide">
                          <label htmlFor="database-extra-params">Extra parameters</label>
                          <input id="database-extra-params" value={dbForm.extraParams} onChange={(event) => setDbForm((current) => ({ ...current, extraParams: event.target.value }))} />
                        </div>
                      </div>
                      <label className="database-inline-check">
                        <input
                          type="checkbox"
                          checked={dbForm.passwordClear}
                          onChange={(event) => setDbForm((current) => ({ ...current, passwordClear: event.target.checked }))}
                        />
                        Clear configured password
                      </label>
                    </fieldset>

                    <fieldset className="database-fieldset">
                      <legend>Runtime container</legend>
                      <p>Values used to generate local Docker and Podman bootstrap commands.</p>
                      <div className="database-form-grid">
                        <div className="field-group">
                          <label htmlFor="database-container-name">Container name</label>
                          <input id="database-container-name" value={dbForm.containerName} onChange={(event) => setDbForm((current) => ({ ...current, containerName: event.target.value }))} />
                          {dbFieldErrors.containerName ? <p role="alert">{dbFieldErrors.containerName}</p> : null}
                        </div>
                        <div className="field-group">
                          <label htmlFor="database-container-image">Container image</label>
                          <input id="database-container-image" value={dbForm.containerImage} onChange={(event) => setDbForm((current) => ({ ...current, containerImage: event.target.value }))} />
                          {dbFieldErrors.containerImage ? <p role="alert">{dbFieldErrors.containerImage}</p> : null}
                        </div>
                        <div className="field-group">
                          <label htmlFor="database-host-port">Host port</label>
                          <input id="database-host-port" value={dbForm.containerHostPort} onChange={(event) => setDbForm((current) => ({ ...current, containerHostPort: event.target.value }))} />
                          {dbFieldErrors.containerHostPort ? <p role="alert">{dbFieldErrors.containerHostPort}</p> : null}
                        </div>
                        <div className="field-group">
                          <label htmlFor="database-volume-name">Volume name</label>
                          <input id="database-volume-name" value={dbForm.volumeName} onChange={(event) => setDbForm((current) => ({ ...current, volumeName: event.target.value }))} />
                          {dbFieldErrors.volumeName ? <p role="alert">{dbFieldErrors.volumeName}</p> : null}
                        </div>
                      </div>
                    </fieldset>

                    <div className="database-form-actions">
                      <button type="submit" disabled={dbSavePending}>
                        Save settings
                      </button>
                      {dbSaveMessage ? <p role="status">{dbSaveMessage}</p> : null}
                    </div>
                  </form>
                </article>

                <aside className="database-operations" aria-label="Database operator guidance">
                  <article className="admin-card database-steps-card" aria-labelledby="db-rail-steps-title">
                    <p className="database-kicker">Runbook</p>
                    <h2 id="db-rail-steps-title">Operator steps</h2>
                    <ol className="database-step-list">
                      <li>
                        <span>01</span>
                        <div>
                          <strong>Connection profile</strong>
                          <p>Fill endpoint and credential fields.</p>
                        </div>
                      </li>
                      <li>
                        <span>02</span>
                        <div>
                          <strong>Container command</strong>
                          <p>Review generated Docker or Podman command.</p>
                        </div>
                      </li>
                      <li>
                        <span>03</span>
                        <div>
                          <strong>Connection test</strong>
                          <p>Probe reachability before saving shared settings.</p>
                        </div>
                      </li>
                      <li>
                        <span>04</span>
                        <div>
                          <strong>Persist settings</strong>
                          <p>Save the structured profile without storing raw DSNs.</p>
                        </div>
                      </li>
                    </ol>
                  </article>

                  {dbGuidance ? (
                    <article className="admin-card database-guidance-card" aria-labelledby="db-setup-guidance-title">
                      <p className="database-kicker">Recommended runtime</p>
                      <h2 id="db-setup-guidance-title">Database setup guidance</h2>
                      <p>Recommended image: {dbGuidance.image}</p>
                      {dbGuidance.notes.length > 0 ? (
                        <ul>
                          {dbGuidance.notes.map((note) => (
                            <li key={note}>{note}</li>
                          ))}
                        </ul>
                      ) : null}
                    </article>
                  ) : null}

                  <article className="admin-card database-side-panel">
                    <p className="database-kicker">Legacy import</p>
                    <h2>Advanced DSN import</h2>
                    <label htmlFor="database-dsn-import">Advanced DSN import</label>
                    <textarea
                      id="database-dsn-import"
                      value={dbForm.dsnImport}
                      onChange={(event) => setDbForm((current) => ({ ...current, dsnImport: event.target.value }))}
                      placeholder="postgresql://user:password@host:5432/workflows?sslmode=require"
                    />
                    <button type="button" onClick={onImportDatabaseDsn}>Import DSN</button>
                    {dbFieldErrors.dsnImport ? <p role="alert">{dbFieldErrors.dsnImport}</p> : null}
                  </article>
                </aside>
              </div>

              <article className="admin-card database-command-pane" aria-labelledby="db-runtime-title">
                <div className="database-command-pane__header">
                  <div>
                    <p className="database-kicker">Verification and bootstrap</p>
                    <h2 id="db-runtime-title">Runtime commands</h2>
                    <p>Test the active profile and copy generated commands for local metadata storage.</p>
                  </div>
                  <button type="button" onClick={() => void onTestDatabaseConnection()} disabled={dbTestPending}>
                    Test connection
                  </button>
                </div>

                <div className="database-runtime-grid">
                  <section className="database-check-card" aria-labelledby="db-test-title">
                    <h3 id="db-test-title">Connection check</h3>
                    {dbConnectionMessage ? <p role="status">{dbConnectionMessage}</p> : null}
                    {dbConnectionResult ? (
                      <div role="status" aria-live="polite">
                        <p>Connection status: {dbConnectionResult.status}</p>
                        <p>Configured: {dbConnectionResult.configured ? "Yes" : "No"}</p>
                        {dbConnectionResult.blockers.length > 0 ? (
                          <ul>
                            {dbConnectionResult.blockers.map((blocker) => (
                              <li key={blocker}>{blocker}</li>
                            ))}
                          </ul>
                        ) : null}
                        {dbConnectionResult.actionable.length > 0 ? (
                          <ul>
                            {dbConnectionResult.actionable.map((step) => (
                              <li key={step}>{step}</li>
                            ))}
                          </ul>
                        ) : null}
                      </div>
                    ) : (
                      <p>Run a check after updating fields to confirm network and credential readiness.</p>
                    )}
                  </section>

                  <section className="database-command-card" aria-labelledby="db-command-title">
                    <h3 id="db-command-title">Command preview</h3>
                    <div className="database-command-block">
                      <p>Connection DSN preview</p>
                      <pre>{dsnPreview}</pre>
                    </div>
                    <div className="database-command-block">
                      <div className="database-command-block__header">
                        <p><strong>Docker</strong></p>
                        <button
                          type="button"
                          className="icon-button database-copy-button"
                          aria-label="Copy Docker command"
                          title="Copy Docker command"
                          onClick={() => void onCopyDatabaseCommand("Docker", dockerCopyCommand)}
                        >
                          <CopyIcon />
                        </button>
                      </div>
                      <pre>{dockerPreview}</pre>
                    </div>
                    <div className="database-command-block">
                      <div className="database-command-block__header">
                        <p><strong>Podman</strong></p>
                        <button
                          type="button"
                          className="icon-button database-copy-button"
                          aria-label="Copy Podman command"
                          title="Copy Podman command"
                          onClick={() => void onCopyDatabaseCommand("Podman", podmanCopyCommand)}
                        >
                          <CopyIcon />
                        </button>
                      </div>
                      <pre>{podmanPreview}</pre>
                    </div>
                    {dbCopyStatus ? (
                      <p role="status" aria-live="polite" aria-label="Clipboard status">
                        {dbCopyStatus}
                      </p>
                    ) : null}
                    {dbCopyError ? <p role="alert">{dbCopyError}</p> : null}
                  </section>
                </div>
              </article>
            </section>
          ) : null}

          {currentPath === "/llm" ? (
            <section className="admin-section llm-workbench" aria-label="LLM configuration management">
              <section className="llm-status-panel" aria-label="LLM source of truth">
                <div>
                  <p className="database-kicker">runtime configuration</p>
                  <h2>LLM configuration</h2>
                  <p>Source of truth: SQLite-backed.</p>
                </div>
                <dl className="llm-status-grid">
                  <div>
                    <dt>Providers</dt>
                    <dd>Providers: {llmProviderEntries.length}</dd>
                  </div>
                  <div>
                    <dt>Profiles</dt>
                    <dd>Profiles: {llmProfileEntries.length}</dd>
                  </div>
                  <div>
                    <dt>Default</dt>
                    <dd>Default profile: {llmDefaultProfileLabel}</dd>
                  </div>
                </dl>
              </section>

              <div className="llm-status-stack">
                {llmLoading ? <p role="status">Loading LLM configuration…</p> : null}
                {llmMessage ? <p role="status">{llmMessage}</p> : null}
                {llmError ? <p role="alert">{llmError}</p> : null}
              </div>

              <div className="llm-workbench__grid llm-workbench__grid--single">
                <article className="admin-card" aria-labelledby="llm-providers-title">
                  <div className="inline-actions">
                    <h2 id="llm-providers-title">Providers</h2>
                    <div className="inline-actions">
                      <button type="button" onClick={openNewLlmProviderModal}>
                        Add provider
                      </button>
                      <button type="button" className="secondary-button" onClick={() => void loadLlmConfig()}>
                        Reload
                      </button>
                    </div>
                  </div>
                  {llmProviderEntries.length === 0 ? <p>No providers configured.</p> : null}
                  <div className="llm-table-wrap">
                    <table className="llm-provider-table" aria-label="LLM providers">
                      <thead>
                        <tr>
                          <th scope="col">Provider ID</th>
                          <th scope="col">Type</th>
                          <th scope="col">Model</th>
                          <th scope="col">Endpoint</th>
                          <th scope="col">Secret</th>
                          <th scope="col">Operations</th>
                        </tr>
                      </thead>
                      <tbody>
                      {llmProviderEntries.map(([providerId, provider]) => (
                        <tr
                          key={providerId}
                          onClick={() => onEditLlmProvider(providerId)}
                        >
                          <th scope="row">{providerId}</th>
                          <td>{provider.type || "Not set"}</td>
                          <td>{provider.model ?? "Not set"}</td>
                          <td>{provider.api_url ?? "Default endpoint"}</td>
                          <td>{provider.api_key_secret ?? "Not set"}</td>
                          <td>
                            <div className="inline-actions">
                              <button
                                type="button"
                                className="secondary-button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onEditLlmProvider(providerId);
                                }}
                              >
                                Edit provider {providerId}
                              </button>
                              <button
                                type="button"
                                className="secondary-button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onDeleteLlmProvider(providerId);
                                }}
                              >
                                Delete provider {providerId}
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                      </tbody>
                    </table>
                  </div>
                  {llmProviderEntries.length > 0 ? (
                    <div className="llm-provider-details" aria-label="LLM provider details">
                      {llmProviderEntries.map(([providerId, provider]) => (
                        <p key={providerId}>
                          <strong>{providerId}</strong> API key secret: {provider.api_key_secret ?? "Not set"} · Headers:{" "}
                          {Object.keys(provider.extra_headers).length}
                          {provider.deployment_name ? ` · Deployment: ${provider.deployment_name}` : ""}
                          {provider.api_version ? ` · API version: ${provider.api_version}` : ""}
                        </p>
                      ))}
                    </div>
                  ) : null}
                </article>
              </div>

              <div className="llm-workbench__grid">
                <article className="admin-card" aria-labelledby="llm-profiles-title">
                  <h2 id="llm-profiles-title">Profiles</h2>
                  {llmProfileEntries.length === 0 ? <p>No profiles configured.</p> : null}
                  {llmProfileEntries.length > 0 ? (
                    <ul className="entity-list" aria-label="LLM profiles list">
                      {llmProfileEntries.map(([profileId, profile]) => (
                        <li key={profileId} className="entity-item">
                          <div>
                            <strong>{profileId}</strong>
                            <p>Provider: {profile.provider || "Not set"}</p>
                            <p>Model: {profile.model ?? "Provider default"}</p>
                            <p>Temperature: {profile.temperature ?? "default"}</p>
                            <p>Max tokens: {profile.max_tokens ?? "default"}</p>
                            {profile.description ? <p>{profile.description}</p> : null}
                          </div>
                          <div className="inline-actions">
                            <button type="button" className="secondary-button" onClick={() => onEditLlmProfile(profileId)}>
                              Edit profile {profileId}
                            </button>
                            <button type="button" onClick={() => onDeleteLlmProfile(profileId)}>
                              Delete profile {profileId}
                            </button>
                          </div>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </article>

                <article className="admin-card" aria-labelledby="llm-profile-form-title">
                  <h2 id="llm-profile-form-title">Profile editor</h2>
                  <form className="admin-form llm-form" onSubmit={onSaveLlmProfile}>
                    <div className="field-group">
                      <label htmlFor="llm-profile-id">Profile ID</label>
                      <input
                        id="llm-profile-id"
                        value={llmProfileForm.id}
                        onChange={(event) => setLlmProfileForm((current) => ({ ...current, id: event.target.value }))}
                        required
                      />
                    </div>
                    <div className="field-group">
                      <label htmlFor="llm-profile-provider">Profile provider</label>
                      <select
                        id="llm-profile-provider"
                        value={llmProfileForm.provider}
                        onChange={(event) => setLlmProfileForm((current) => ({ ...current, provider: event.target.value }))}
                        required
                      >
                        <option value="">Select provider</option>
                        {llmProviderEntries.map(([providerId]) => (
                          <option key={providerId} value={providerId}>
                            {providerId}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="field-group">
                      <label htmlFor="llm-profile-model">Profile model</label>
                      <input
                        id="llm-profile-model"
                        value={llmProfileForm.model}
                        onChange={(event) => setLlmProfileForm((current) => ({ ...current, model: event.target.value }))}
                        placeholder="gpt-4.1-mini"
                      />
                    </div>
                    <div className="llm-two-column">
                      <div className="field-group">
                        <label htmlFor="llm-profile-temperature">Temperature</label>
                        <input
                          id="llm-profile-temperature"
                          value={llmProfileForm.temperature}
                          onChange={(event) => setLlmProfileForm((current) => ({ ...current, temperature: event.target.value }))}
                          inputMode="decimal"
                        />
                      </div>
                      <div className="field-group">
                        <label htmlFor="llm-profile-max-tokens">Max tokens</label>
                        <input
                          id="llm-profile-max-tokens"
                          value={llmProfileForm.maxTokens}
                          onChange={(event) => setLlmProfileForm((current) => ({ ...current, maxTokens: event.target.value }))}
                          inputMode="numeric"
                        />
                      </div>
                    </div>
                    <div className="field-group">
                      <label htmlFor="llm-profile-description">Description</label>
                      <input
                        id="llm-profile-description"
                        value={llmProfileForm.description}
                        onChange={(event) => setLlmProfileForm((current) => ({ ...current, description: event.target.value }))}
                      />
                    </div>
                    <div className="inline-actions">
                      <button type="submit" disabled={llmProviderEntries.length === 0}>
                        Save profile
                      </button>
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => setLlmProfileForm(DEFAULT_LLM_PROFILE_FORM)}
                      >
                        Clear profile form
                      </button>
                    </div>
                  </form>
                </article>
              </div>

              <article className="admin-card llm-persist-panel" aria-labelledby="llm-save-title">
                <h2 id="llm-save-title">Configuration changes</h2>
                <div className="llm-two-column">
                  <div className="field-group">
                    <label htmlFor="llm-default-profile">Default profile</label>
                    <select
                      id="llm-default-profile"
                      value={llmConfig.default_profile ?? ""}
                      onChange={(event) =>
                        setLlmConfig((current) =>
                          normalizedLlmConfig({ ...current, default_profile: event.target.value || null }),
                        )
                      }
                    >
                      <option value="">No default profile</option>
                      {llmProfileEntries.map(([profileId]) => (
                        <option key={profileId} value={profileId}>
                          {profileId}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="llm-save-actions">
                    <button type="button" disabled={llmSaving} onClick={() => void onSaveLlmConfig()}>
                      Save LLM configuration
                    </button>
                    <button type="button" className="secondary-button" onClick={() => void loadLlmConfig()}>
                      Discard local changes
                    </button>
                  </div>
                </div>
              </article>

              <article className="admin-card llm-yaml-panel" aria-labelledby="llm-yaml-title">
                <div className="inline-actions">
                  <h2 id="llm-yaml-title">YAML migration</h2>
                  <button type="button" onClick={openLlmYamlModal}>
                    Import YAML
                  </button>
                  <button type="button" className="secondary-button" onClick={() => void onExportLlmYaml()}>
                    Export YAML
                  </button>
                </div>
                <p>Import legacy YAML or download the current SQLite-backed LLM configuration.</p>
              </article>

              {llmProviderModalOpen ? (
                <ModalShell
                  titleId="llm-provider-dialog-title"
                  title={llmProviderForm.id.trim() ? `Edit provider ${llmProviderForm.id.trim()}` : "Add provider"}
                  eyebrow="Provider"
                  onClose={closeLlmProviderModal}
                >
                  <form className="admin-form llm-form" onSubmit={onSaveLlmProvider}>
                    <div className="field-group">
                      <label htmlFor="llm-provider-id">Provider ID</label>
                      <input
                        id="llm-provider-id"
                        value={llmProviderForm.id}
                        onChange={(event) => setLlmProviderForm((current) => ({ ...current, id: event.target.value }))}
                        required
                      />
                    </div>
                    <div className="field-group">
                      <label htmlFor="llm-provider-type">Provider type</label>
                      <input
                        id="llm-provider-type"
                        value={llmProviderForm.type}
                        onChange={(event) => setLlmProviderForm((current) => ({ ...current, type: event.target.value }))}
                        placeholder="openai"
                        required
                      />
                    </div>
                    <div className="field-group">
                      <label htmlFor="llm-provider-model">Provider model</label>
                      <input
                        id="llm-provider-model"
                        value={llmProviderForm.model}
                        onChange={(event) => setLlmProviderForm((current) => ({ ...current, model: event.target.value }))}
                        placeholder="gpt-4.1-mini"
                      />
                    </div>
                    <div className="field-group">
                      <label htmlFor="llm-provider-api-key-secret">API key secret</label>
                      <input
                        id="llm-provider-api-key-secret"
                        value={llmProviderForm.apiKeySecret}
                        onChange={(event) => setLlmProviderForm((current) => ({ ...current, apiKeySecret: event.target.value }))}
                        placeholder="OPENAI_API_KEY"
                      />
                    </div>
                    <div className="field-group">
                      <label htmlFor="llm-provider-api-url">API URL</label>
                      <input
                        id="llm-provider-api-url"
                        value={llmProviderForm.apiUrl}
                        onChange={(event) => setLlmProviderForm((current) => ({ ...current, apiUrl: event.target.value }))}
                        placeholder="https://api.openai.com/v1"
                      />
                    </div>
                    <div className="llm-three-column">
                      <div className="field-group">
                        <label htmlFor="llm-provider-timeout">Timeout</label>
                        <input
                          id="llm-provider-timeout"
                          value={llmProviderForm.timeout}
                          onChange={(event) => setLlmProviderForm((current) => ({ ...current, timeout: event.target.value }))}
                          inputMode="decimal"
                        />
                      </div>
                      <div className="field-group">
                        <label htmlFor="llm-provider-max-retries">Max retries</label>
                        <input
                          id="llm-provider-max-retries"
                          value={llmProviderForm.maxRetries}
                          onChange={(event) => setLlmProviderForm((current) => ({ ...current, maxRetries: event.target.value }))}
                          inputMode="numeric"
                        />
                      </div>
                      <div className="field-group">
                        <label htmlFor="llm-provider-retry-delay">Retry delay</label>
                        <input
                          id="llm-provider-retry-delay"
                          value={llmProviderForm.retryDelay}
                          onChange={(event) => setLlmProviderForm((current) => ({ ...current, retryDelay: event.target.value }))}
                          inputMode="decimal"
                        />
                      </div>
                    </div>
                    <div className="field-group">
                      <label htmlFor="llm-provider-extra-headers">Extra headers JSON</label>
                      <textarea
                        id="llm-provider-extra-headers"
                        value={llmProviderForm.extraHeaders}
                        onChange={(event) => setLlmProviderForm((current) => ({ ...current, extraHeaders: event.target.value }))}
                      />
                    </div>
                    <div className="llm-two-column">
                      <div className="field-group">
                        <label htmlFor="llm-provider-deployment-name">Deployment name</label>
                        <input
                          id="llm-provider-deployment-name"
                          value={llmProviderForm.deploymentName}
                          onChange={(event) => setLlmProviderForm((current) => ({ ...current, deploymentName: event.target.value }))}
                        />
                      </div>
                      <div className="field-group">
                        <label htmlFor="llm-provider-api-version">API version</label>
                        <input
                          id="llm-provider-api-version"
                          value={llmProviderForm.apiVersion}
                          onChange={(event) => setLlmProviderForm((current) => ({ ...current, apiVersion: event.target.value }))}
                        />
                      </div>
                    </div>
                    <div className="inline-actions">
                      <button type="submit">Save provider</button>
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => setLlmProviderForm(DEFAULT_LLM_PROVIDER_FORM)}
                      >
                        Clear provider
                      </button>
                      <button type="button" className="secondary-button" onClick={closeLlmProviderModal}>
                        Cancel
                      </button>
                    </div>
                  </form>
                </ModalShell>
              ) : null}

              {llmYamlModalOpen ? (
                <ModalShell titleId="llm-yaml-dialog-title" title="Import YAML" eyebrow="YAML migration" onClose={closeLlmYamlModal}>
                  <div className="admin-form llm-form">
                    <div className="field-group">
                      <label htmlFor="llm-yaml-import">YAML import</label>
                      <textarea
                        id="llm-yaml-import"
                        value={llmYamlImport}
                        onChange={(event) => {
                          setLlmYamlImport(event.target.value);
                          setLlmPreview(null);
                        }}
                        placeholder="version: '1.0'"
                      />
                    </div>
                    <div className="field-group">
                      <label htmlFor="llm-yaml-upload">Upload YAML file</label>
                      <input id="llm-yaml-upload" type="file" accept=".yaml,.yml,application/x-yaml,text/yaml,text/plain" onChange={(event) => void onUploadLlmYaml(event)} />
                    </div>
                    {llmPreview ? (
                      <p className="llm-preview-summary">
                        Preview default profile: {llmPreview.default_profile ?? "None"} · Providers:{" "}
                        {Object.keys(llmPreview.providers).join(", ") || "None"}
                      </p>
                    ) : null}
                    <div className="inline-actions">
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={llmYamlImport.trim().length === 0}
                        onClick={() => void onPreviewLlmYaml()}
                      >
                        Preview import
                      </button>
                      <button
                        type="button"
                        disabled={llmYamlImport.trim().length === 0}
                        onClick={() => void onImportLlmYaml()}
                      >
                        Import YAML
                      </button>
                      <button type="button" className="secondary-button" onClick={closeLlmYamlModal}>
                        Cancel
                      </button>
                    </div>
                  </div>
                </ModalShell>
              ) : null}
            </section>
          ) : null}

          {currentPath === "/projects" ? (
            <section className="admin-section" aria-label="Projects management">
              <article className="admin-card">
                <h2>Registered projects</h2>
                <p>Topology inference and guided onboarding controls are not yet available from the backend API.</p>
                {projectsLoading ? <p role="status">Loading projects…</p> : null}
                {projectError ? <p role="alert">{projectError}</p> : null}
                {projectMessage ? <p role="status">{projectMessage}</p> : null}
                {!projectsLoading && projects.length === 0 ? (
                  <p>No projects yet. Register the first project below to continue setup.</p>
                ) : null}
                {projects.length > 0 ? (
                  <ul className="entity-list" aria-label="Projects list">
                    {projects.map((project) => (
                      <li key={project.id} className="entity-item">
                        <div>
                          <strong>{project.name}</strong>
                          <p>
                            Slug: {project.slug} · Palace: {project.palace} · Default wing/room: {project.defaultWing}/
                            {project.defaultRoom}
                          </p>
                          <p>FS root: {project.fsRoot}</p>
                          <p>
                            Allowlist: {project.fsAllowlist.length > 0 ? project.fsAllowlist.join(", ") : "No explicit allowlist"}
                          </p>
                          <p>
                            Watcher default hint: {project.watcherHint ?? "Watcher state hints are unavailable from the current projects API response."}
                          </p>
                          <p>
                            Default state hint: {project.defaultStateHint ?? "Default state hints are unavailable from the current projects API response."}
                          </p>
                        </div>
                        <button type="button" onClick={() => setProjectDeleteTarget(project)}>
                          Delete project {project.id}
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </article>

              {projectDeleteTarget ? (
                <article className="admin-card">
                  <h3>Delete project</h3>
                  <p>Type DELETE to confirm removing {projectDeleteTarget.id}.</p>
                  <label htmlFor="project-delete-confirm">Confirm deletion</label>
                  <input
                    id="project-delete-confirm"
                    type="text"
                    value={projectDeleteConfirm}
                    onChange={(event) => setProjectDeleteConfirm(event.target.value)}
                  />
                  <div className="actions-row inline-actions">
                    <button
                      type="button"
                      disabled={projectDeletePending || projectDeleteConfirm !== "DELETE"}
                      onClick={() => void onConfirmDeleteProject()}
                    >
                      Confirm delete {projectDeleteTarget.id}
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => {
                        setProjectDeleteTarget(null);
                        setProjectDeleteConfirm("");
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </article>
              ) : null}

              <article className="admin-card">
                <h2>Register project</h2>
                <form className="admin-form" onSubmit={(event) => void onCreateProject(event)}>
                  <label htmlFor="project-name">Name</label>
                  <input id="project-name" value={projectName} onChange={(event) => setProjectName(event.target.value)} required />
                  <label htmlFor="project-slug">Slug</label>
                  <input id="project-slug" value={projectSlug} onChange={(event) => setProjectSlug(event.target.value)} required />
                  <label htmlFor="project-palace">Palace</label>
                  <input id="project-palace" value={projectPalace} onChange={(event) => setProjectPalace(event.target.value)} required />
                  <label htmlFor="project-default-wing">Default wing</label>
                  <input
                    id="project-default-wing"
                    value={projectDefaultWing}
                    onChange={(event) => setProjectDefaultWing(event.target.value)}
                    required
                  />
                  <label htmlFor="project-default-room">Default room</label>
                  <input
                    id="project-default-room"
                    value={projectDefaultRoom}
                    onChange={(event) => setProjectDefaultRoom(event.target.value)}
                    required
                  />
                  <label htmlFor="project-fs-root">FS root</label>
                  <div className="inline-actions">
                    <input
                      id="project-fs-root"
                      ref={projectFsRootRef}
                      value={projectFsRoot}
                      onChange={(event) => setProjectFsRoot(event.target.value)}
                      required
                    />
                    <button
                      type="button"
                      className="secondary-button icon-button"
                      aria-label="Browse FS root"
                      title="Browse FS root"
                      onClick={() => openPathPicker("fsRoot")}
                    >
                      <FolderIcon />
                    </button>
                  </div>
                  <label htmlFor="project-allowlist">Allowlist paths</label>
                  <textarea
                    id="project-allowlist"
                    ref={projectAllowlistRef}
                    value={projectAllowlistInput}
                    onChange={(event) => setProjectAllowlistInput(event.target.value)}
                    placeholder="/workspace/workflows, /workspace/shared"
                  />
                  <button
                    type="button"
                    className="secondary-button icon-button"
                    aria-label="Browse allowlist paths"
                    title="Browse allowlist paths"
                    onClick={() => openPathPicker("allowlist")}
                  >
                    <FolderIcon />
                  </button>
                  {pathPickerOpen ? (
                    <ServerPathPicker
                      key={pathPickerTarget}
                      title="Browse server folders"
                      selectionMode="folder"
                      startPath={pathPickerTarget === "fsRoot" ? projectFsRoot : undefined}
                      listEntries={listServerPathEntries}
                      onSelect={onSelectPath}
                      onCancel={closePathPicker}
                    />
                  ) : null}
                  <button type="submit" disabled={projectCreatePending}>
                    Register project
                  </button>
                </form>
              </article>
            </section>
          ) : null}

          {currentPath === "/mcp-clients" ? (
            <section className="admin-section" aria-label="MCP clients management">
              <article className="admin-card">
                <h2>Issue MCP client token</h2>
                <form className="admin-form" onSubmit={(event) => void onCreateMcpClient(event)}>
                  <label htmlFor="mcp-label">Client label</label>
                  <input id="mcp-label" value={mcpLabel} onChange={(event) => setMcpLabel(event.target.value)} required />
                  <ProjectMultiSelect
                    id="mcp-project-ids"
                    label="Project access"
                    projects={projects}
                    selectedIds={mcpSelectedProjectIds}
                    onChange={setMcpSelectedProjectIds}
                    disabled={projectsLoading || mcpCreatePending}
                    helperText={
                      projects.length > 0
                        ? "Select every project this token may access."
                        : "Create a project before issuing MCP client tokens."
                    }
                  />
                  {projectError ? <p role="alert">{projectError}</p> : null}
                  <button type="submit" disabled={mcpCreatePending || mcpSelectedProjectIds.length === 0}>
                    Create MCP client
                  </button>
                </form>
                {mcpMessage ? <p role="status">{mcpMessage}</p> : null}
                {mcpError ? <p role="alert">{mcpError}</p> : null}
                {oneTimeMcpSecret ? (
                  <div className="token-card" role="status" aria-live="polite">
                    <h3>Copy once: token and snippet</h3>
                    <p>Client label: {oneTimeMcpSecret.label}</p>
                    <pre>{oneTimeMcpSecret.token}</pre>
                    {oneTimeMcpSecret.configSnippet ? (
                      <pre>{oneTimeMcpSecret.configSnippet}</pre>
                    ) : (
                      <p>No configuration snippet was returned. Use the token directly in your MCP client settings.</p>
                    )}
                  </div>
                ) : null}
              </article>

              <article className="admin-card">
                <div className="inline-actions">
                  <h2>Registered MCP clients</h2>
                  <button type="button" className="secondary-button" onClick={() => void loadMcpClients()}>
                    Reload MCP clients
                  </button>
                </div>
                {mcpLoading ? <p role="status">Loading MCP clients…</p> : null}
                {!mcpLoading && mcpClients.length === 0 ? (
                  <p>No MCP clients yet. Create one token to unblock integrations.</p>
                ) : null}
                {mcpClients.length > 0 ? (
                  <ul className="entity-list" aria-label="MCP clients list">
                    {mcpClients.map((client) => (
                      <li key={client.id} className="entity-item">
                        <div>
                          <strong>{client.label}</strong>
                          <p>ID: {client.id}</p>
                          <p>Projects: {client.projectIds.length > 0 ? client.projectIds.map(formatProjectReference).join(", ") : "None"}</p>
                          <p>Created: {client.createdAt || "Unknown"}</p>
                          <p>Last used: {client.lastUsedAt ?? "Never"}</p>
                          <p>Status: {client.revokedAt ? `Revoked at ${client.revokedAt}` : "Active"}</p>
                        </div>
                        <div className="inline-actions">
                          <button
                            type="button"
                            className="secondary-button"
                            disabled={mcpMutationPendingId === client.id}
                            onClick={() => void onRegenerateMcpClient(client.id)}
                          >
                            Regenerate
                          </button>
                          <button
                            type="button"
                            disabled={mcpMutationPendingId === client.id}
                            onClick={() => void onRevokeMcpClient(client.id)}
                          >
                            Revoke
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </article>
            </section>
          ) : null}

          {currentPath === "/watchers" ? (
            <section className="admin-section" aria-label="Watchers dashboard">
              <article className="admin-card">
                <div className="inline-actions">
                  <h2>Watcher status</h2>
                  <button type="button" className="secondary-button" onClick={() => void refreshWatchersState()}>
                    Refresh
                  </button>
                </div>
                {watchersLoading ? <p role="status">Loading watcher status…</p> : null}
                {watchersMessage ? <p role="status">{watchersMessage}</p> : null}
                {watchersError ? <p role="alert">{watchersError}</p> : null}
                {!watchersLoading && watchersRows.length === 0 ? (
                  <p>
                    No watchers available yet. <a href="/projects">Create a project</a> to initialize watcher management.
                  </p>
                ) : null}
                {watchersRows.length > 0 ? (
                  <ul className="entity-list" aria-label="Watchers list">
                    {watchersRows.map((row) => (
                      <li key={row.projectId} className="entity-item">
                        <div>
                          <strong>{row.projectName ?? row.projectId}</strong>
                          <p>Project ID: {row.projectId}</p>
                          <p>State: {row.state}</p>
                          <p>Dirty count: {row.dirtyCount}</p>
                          <p>Requires reconciliation: {row.requiresReconciliation ? "Yes" : "No"}</p>
                          <p>Last event: {row.lastEventAt ?? "Not yet recorded"}</p>
                          <p>Updated: {row.updatedAt || "Unavailable"}</p>
                        </div>
                        <div className="inline-actions">
                          <button type="button" className="secondary-button" disabled={watchersPendingAction !== null} onClick={() => void onWatcherAction(row.projectId, "pause")}>Pause watcher {row.projectId}</button>
                          <button type="button" className="secondary-button" disabled={watchersPendingAction !== null} onClick={() => void onWatcherAction(row.projectId, "resume")}>Resume watcher {row.projectId}</button>
                          <button type="button" disabled={watchersPendingAction !== null} onClick={() => void onWatcherAction(row.projectId, "disable")}>Disable watcher {row.projectId}</button>
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </article>
            </section>
          ) : null}

          {currentPath === "/sync" ? (
            <section className="admin-section sync-workbench" aria-label="Sync dashboard">
              <article className="sync-hero-panel">
                <div className="sync-hero-panel__copy">
                  <p className="database-kicker">Live sync queue</p>
                  <h2>Project sync operations</h2>
                  <p>
                    Review registered projects, spot queued files, and run targeted sync maintenance without copying project IDs.
                  </p>
                </div>
                <div className="sync-toolbar">
                  <ProjectSelect
                    id="sync-project-filter"
                    label="Project"
                    projects={projects}
                    value={selectedSyncProjectId}
                    onChange={setSelectedSyncProjectId}
                    allLabel="All projects"
                    disabled={syncLoading && projects.length === 0}
                    helperText={
                      selectedSyncRow
                        ? `Focused on ${selectedSyncRow.projectName ?? selectedSyncRow.projectId}.`
                        : "Choose a project to narrow the queue."
                    }
                  />
                  <button type="button" className="secondary-button" onClick={() => void refreshSyncState()}>
                    Refresh
                  </button>
                </div>
              </article>

              <dl className="sync-stats-grid" aria-label="Sync summary">
                <div>
                  <dt>Projects</dt>
                  <dd>{syncRows.length}</dd>
                </div>
                <div>
                  <dt>Dirty files</dt>
                  <dd>{syncDirtyTotal}</dd>
                </div>
                <div>
                  <dt>Queued</dt>
                  <dd>{syncQueuedCount}</dd>
                </div>
                <div>
                  <dt>Needs reconcile</dt>
                  <dd>{syncReconcileCount}</dd>
                </div>
              </dl>

              <div className="sync-status-stack">
                {syncLoading ? <p role="status">Loading sync queue status…</p> : null}
                {syncMessage ? <p role="status">{syncMessage}</p> : null}
                {syncError ? <p role="alert">{syncError}</p> : null}
              </div>

              {!syncLoading && syncRows.length === 0 ? (
                <article className="sync-empty-state">
                  <h2>Queue is empty</h2>
                  <p>
                    No sync queue entries yet. <a href="/projects">Create a project</a> to enqueue files for sync.
                  </p>
                </article>
              ) : null}

              {syncRows.length > 0 ? (
                <section className="sync-project-grid" aria-label="Sync projects list">
                  {visibleSyncRows.map((row) => {
                    const statusTone = row.requiresReconciliation ? "warning" : row.dirtyCount > 0 ? "info" : "success";
                    const statusLabel = row.requiresReconciliation ? "Needs reconcile" : row.dirtyCount > 0 ? "Queued" : "Clean";
                    return (
                      <article key={row.projectId} className="sync-project-card">
                        <div className="sync-card-header">
                          <div>
                            <p className="database-kicker">{row.projectId}</p>
                            <h3>{row.projectName ?? row.projectId}</h3>
                          </div>
                          <StatusBadge tone={statusTone}>{statusLabel}</StatusBadge>
                        </div>
                        <div className="sync-card-body">
                          <div className="sync-primary-metric">
                            <span>{row.dirtyCount}</span>
                            <p>Dirty count: {row.dirtyCount}</p>
                          </div>
                          <div className="sync-state-list">
                            <p>Project ID: {row.projectId}</p>
                            <p>Sync state: {row.syncState}</p>
                            <p>Reconcile state: {row.reconcileState}</p>
                            <p>Rebuild state: {row.rebuildState}</p>
                            <p>Requires reconciliation: {row.requiresReconciliation ? "Yes" : "No"}</p>
                            <p>Lifecycle telemetry: {row.lifecycleTelemetry}</p>
                          </div>
                        </div>
                        <div className="sync-card-actions">
                          <button
                            type="button"
                            className="secondary-button"
                            aria-label={`Sync now ${row.projectId}`}
                            disabled={syncPendingAction !== null}
                            onClick={() => void onSyncAction(row.projectId, "now")}
                          >
                            Sync now
                          </button>
                          <button
                            type="button"
                            className="secondary-button"
                            aria-label={`Reconcile ${row.projectId}`}
                            disabled={syncPendingAction !== null}
                            onClick={() => void onSyncAction(row.projectId, "reconcile")}
                          >
                            Reconcile
                          </button>
                          <button
                            type="button"
                            aria-label={`Rebuild ${row.projectId}`}
                            disabled={syncPendingAction !== null}
                            onClick={() => void onSyncAction(row.projectId, "rebuild")}
                          >
                            Rebuild
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </section>
              ) : null}
            </section>
          ) : null}

          {currentPath === "/workflows" ? (
            <section className="admin-section" aria-label="Workflows management">
              <article className="admin-card">
                <div className="inline-actions">
                  <h2>Workflow registry</h2>
                  <button type="button" className="secondary-button" onClick={() => void loadWorkflowsPageData()}>
                    Refresh
                  </button>
                  <button type="button" onClick={() => void onReloadWorkflows()}>
                    Reload workflows
                  </button>
                  <button type="button" className="secondary-button" onClick={() => void onLoadWorkflowSchema()}>
                    Load schema
                  </button>
                </div>
                {workflowsLoading ? <p role="status">Loading workflows dashboard…</p> : null}
                {workflowsMessage ? <p role="status">{workflowsMessage}</p> : null}
                {workflowsError ? <p role="alert">{workflowsError}</p> : null}
                {workflowsList.length === 0 ? (
                  <p>
                    No workflows discovered yet. Add at least one source and reload the registry.
                  </p>
                ) : (
                  <ul className="entity-list" aria-label="Workflows list">
                    {workflowsList.map((workflow) => (
                      <li key={workflow.name} className="entity-item">
                        <div>
                          <strong>{workflow.name}</strong>
                          <p>{workflow.description || "No description"}</p>
                          <p>Version: {workflow.version || "Not set"}</p>
                          <p>Tags: {workflow.tags.length > 0 ? workflow.tags.join(", ") : "None"}</p>
                          <p>Source path: {workflow.sourcePath ?? "Unknown"}</p>
                        </div>
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => void onViewWorkflowDetails(workflow.name)}
                        >
                          View details
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </article>

              <article className="admin-card">
                <h2>Workflow sources</h2>
                {workflowSources.length === 0 ? (
                  <p>
                    No workflow sources configured.
                    {workflowProjectsCount === 0 ? (
                      <>
                        {" "}
                        <a href="/projects">Create a project first</a>, then add a workflow source path.
                      </>
                    ) : (
                      " Add a workflow source path below to begin discovery."
                    )}
                  </p>
                ) : (
                  <ul className="entity-list" aria-label="Workflow sources list">
                    {workflowSources.map((source) => (
                      <li key={source.sourceId} className="entity-item">
                        <div>
                          <strong>{source.sourceId}</strong>
                          <p>Project: {source.projectId}</p>
                          <p>Path: {source.sourcePath}</p>
                          <p>Status: {source.status ?? "Unknown"}</p>
                          <p>Discovered: {source.discoveredAt || "Unavailable"}</p>
                          <p>Last loaded: {source.lastLoadedAt ?? "Never"}</p>
                          {source.errorMessage ? <p>Last error: {source.errorMessage}</p> : null}
                        </div>
                        <div className="inline-actions">
                          <button
                            type="button"
                            className="secondary-button"
                            disabled={workflowActionPendingId !== null}
                            onClick={() => void onValidateWorkflowSource(source.sourceId)}
                          >
                            Validate {source.sourceId}
                          </button>
                          <button
                            type="button"
                            disabled={workflowActionPendingId !== null}
                            onClick={() => void onDeleteWorkflowSource(source.sourceId)}
                          >
                            Delete {source.sourceId}
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}

                <form className="admin-form" onSubmit={(event) => void onCreateWorkflowSource(event)}>
                  <ProjectSelect
                    id="workflow-source-project-id"
                    label="Project"
                    projects={projects}
                    value={workflowSourceProjectId}
                    onChange={setWorkflowSourceProjectId}
                    helperText={
                      projects.length > 0
                        ? "Workflow sources are attached to one registered project."
                        : "Create a project before adding workflow sources."
                    }
                    required
                  />
                  <label htmlFor="workflow-source-path">Source path</label>
                  <input
                    id="workflow-source-path"
                    value={workflowSourcePath}
                    onChange={(event) => setWorkflowSourcePath(event.target.value)}
                    placeholder="/workspace/workflows"
                    required
                  />
                  <label htmlFor="workflow-source-checksum">Checksum (optional)</label>
                  <input
                    id="workflow-source-checksum"
                    value={workflowSourceChecksum}
                    onChange={(event) => setWorkflowSourceChecksum(event.target.value)}
                    placeholder="sha256:..."
                  />
                  {workflowProjectsCount === 0 ? (
                    <p>Once a project exists, add a workflow source path below.</p>
                  ) : null}
                  <button
                    type="submit"
                    disabled={workflowSourcePending || workflowProjectsCount === 0 || workflowSourceProjectId.trim().length === 0}
                  >
                    Add workflow source
                  </button>
                </form>
              </article>

              {selectedWorkflowName && workflowDetailText ? (
                <article className="admin-card">
                  <h2>Workflow details for {selectedWorkflowName}</h2>
                  <pre>{workflowDetailText}</pre>
                </article>
              ) : null}

              {workflowSchemaText ? (
                <article className="admin-card">
                  <h2>Workflow schema</h2>
                  <pre>{workflowSchemaText}</pre>
                </article>
              ) : null}
            </section>
          ) : null}

          {currentPath === "/runs" ? (
            <section className="admin-section" aria-label="Runs management">
              <article className="admin-card">
                <div className="inline-actions">
                  <h2>Run history</h2>
                  <button type="button" className="secondary-button" onClick={() => void loadRuns()}>
                    Reload runs
                  </button>
                </div>
                <p className="runs-filter-note">
                  Project and workflow filters are not available yet because the backend runs endpoint currently supports
                  only status, limit, and offset query parameters.
                </p>
                <form className="admin-form runs-filter-form" onSubmit={(event) => void onApplyRunsFilters(event)}>
                  <label htmlFor="runs-status-filter">Status filter</label>
                  <input
                    id="runs-status-filter"
                    value={runFilterStatus}
                    onChange={(event) => setRunFilterStatus(event.target.value)}
                    placeholder="paused"
                  />
                  <label htmlFor="runs-limit">Limit</label>
                  <input
                    id="runs-limit"
                    value={runFilterLimit}
                    onChange={(event) => setRunFilterLimit(event.target.value)}
                    inputMode="numeric"
                  />
                  <label htmlFor="runs-offset">Offset</label>
                  <input
                    id="runs-offset"
                    value={runFilterOffset}
                    onChange={(event) => setRunFilterOffset(event.target.value)}
                    inputMode="numeric"
                  />
                  <button type="submit" disabled={runsLoading || runActionPendingId !== null}>
                    Apply run filters
                  </button>
                </form>
                {runsLoading ? <p role="status">Loading runs…</p> : null}
                {runsMessage ? <p role="status">{runsMessage}</p> : null}
                {runsError ? <p role="alert">{runsError}</p> : null}
                {!runsLoading && runsRows.length === 0 ? <p>No runs found for the current filter. Adjust status or pagination and retry.</p> : null}
                {runsRows.length > 0 ? (
                  <ul className="entity-list" aria-label="Runs list">
                    {runsRows.map((run) => (
                      <li key={run.runId} className="entity-item">
                        <div>
                          <strong>{run.workflowName || "Unknown workflow"}</strong>
                          <p>Run ID: {run.runId || "Unavailable"}</p>
                          <p>Job ID: {run.jobId || "Unavailable"}</p>
                          <p>Status: {run.status}</p>
                          <p>Created: {run.createdAt}</p>
                          <p>Started: {run.startedAt ?? "Not started"}</p>
                          <p>Finished: {run.finishedAt ?? "Not finished"}</p>
                          <p>Updated: {run.updatedAt}</p>
                          <p>Project ID: {run.projectId ?? "Not provided"}</p>
                          <p>Token ID: {run.tokenId ?? "Not provided"}</p>
                          <p>Cancellable: {run.cancellable ? "Yes" : "No"}</p>
                        </div>
                        <div className="inline-actions">
                          <button
                            type="button"
                            className="secondary-button"
                            disabled={runActionPendingId !== null}
                            onClick={() => void onViewRunDetail(run.runId)}
                          >
                            View run {run.runId}
                          </button>
                          {run.cancellable ? (
                            <button type="button" disabled={runActionPendingId !== null} onClick={() => void onCancelRun(run.runId)}>
                              Cancel run {run.runId}
                            </button>
                          ) : null}
                          <button
                            type="button"
                            className="secondary-button"
                            disabled={runActionPendingId !== null}
                            onClick={() => void onResumeRun(run.runId)}
                          >
                            Resume run {run.runId}
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </article>

              {runDetail ? (
                <article className="admin-card">
                  <h2>Run detail: {runDetail.runId}</h2>
                  <p>Workflow: {runDetail.workflowName || "Unknown"}</p>
                  <p>Status: {runDetail.status}</p>
                  <p>Result summary: {runDetail.resultSummary ?? "None"}</p>
                  <p>Error summary: {runDetail.errorSummary ?? "None"}</p>
                  <p>
                    Metadata keys: {Object.keys(runDetail.metadata).length > 0 ? Object.keys(runDetail.metadata).join(", ") : "None"}
                  </p>
                  <h3>Technical JSON</h3>
                  <pre>{runDetail.technicalJson}</pre>
                </article>
              ) : null}
            </section>
          ) : null}

          {page.unknown ? (
            <p>
              <a href="/database">Go to Database</a>
            </p>
          ) : null}
        </main>
      </div>
    </div>
  );
}
