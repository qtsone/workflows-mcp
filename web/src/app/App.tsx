import "./App.css";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

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
import { ActionButton, FolderIcon, PageHeader, Panel, StatusBadge } from "./ui";

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
  const [mcpProjectIdsInput, setMcpProjectIdsInput] = useState("");
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

  const loadProjects = async (): Promise<void> => {
    setProjectsLoading(true);
    setProjectError("");
    try {
      const result = await api.listProjects();
      setProjects(result.projects.map(toProjectModel));
    } catch (error) {
      setProjectError(toUserError(error, "Unable to load projects. Confirm your admin session and retry."));
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
      setWorkflowProjectsCount(projectsPayload.projects.length);
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
          const llmConfigured =
            Object.keys(providers).length > 0 ||
            Object.keys(profiles).length > 0 ||
            Object.keys(llmObj).length > 0;
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
          if (!cancelled) setContentState("LLM admin controls are available in a follow-up slice.");
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
            await loadMcpClients();
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
      if (typeof navigator.clipboard?.writeText !== "function") {
        throw new Error("Clipboard API unavailable");
      }
      await navigator.clipboard.writeText(command);
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
        project_ids: normalizePathList(mcpProjectIdsInput),
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
      setMcpProjectIdsInput("");
      await loadMcpClients();
      setOneTimeMcpSecret({
        token,
        configSnippet: toNonEmptyString(result.config_snippet),
        label: secretLabel,
      });
    } catch (error) {
      setMcpError(toUserError(error, "Unable to create MCP client. Confirm label and project IDs."));
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
  const dockerPreview =
    `docker run --name ${quoteShellPreview(dbForm.containerName)} ` +
    `-e POSTGRES_DB=${quoteShellPreview(previewDatabase)} ` +
    `-e POSTGRES_USER=${quoteShellPreview(previewUsername)} ` +
    `-e POSTGRES_PASSWORD=${quoteShellPreview(previewPassword)} ` +
    `-p ${quoteShellPreview(previewContainerPort)}:5432 ` +
    `-v ${quoteShellPreview(dbForm.volumeName)}:/var/lib/postgresql/data ` +
    `${quoteShellPreview(dbForm.containerImage)}`;
  const podmanPreview = dockerPreview.replace(/^docker/, "podman");
  const dsnPreview =
    `postgresql://${quoteShellPreview(previewUsername)}:${quoteShellPreview(previewPassword)}` +
    `@${quoteShellPreview(previewHost)}:${quoteShellPreview(previewPort)}/${quoteShellPreview(previewDatabase)}`;

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
            <section className="admin-section database-console" aria-label="Database management">
              {dbLoadError ? <p role="alert">{dbLoadError}</p> : null}
              <div className="database-terminal-bar">
                <p className="database-terminal-title">workflowsctl / database-profile / local</p>
                <div className="database-terminal-chips" aria-label="Database profile status chips">
                  <span className="database-chip">enabled: {dbForm.enabled ? "on" : "off"}</span>
                  <span className="database-chip">configured: {dbSettings?.configured ? "yes" : "no"}</span>
                  <span className="database-chip">
                    password: {dbForm.password.length > 0 ? "typed" : dbForm.passwordConfigured ? "configured" : "pending"}
                  </span>
                  <span className="database-chip">legacy re-entry: {dbForm.dsnImport.trim().length > 0 ? "staged" : "clean"}</span>
                </div>
              </div>
              <div className="database-grid">
                <aside className="database-rail">
                  <article className="admin-card" aria-labelledby="db-rail-steps-title">
                    <h2 id="db-rail-steps-title">Operator steps</h2>
                    <ol>
                      <li>Connection profile</li>
                      <li>Container command</li>
                      <li>Connection test</li>
                      <li>Persist settings</li>
                    </ol>
                  </article>
                  {dbGuidance ? (
                    <article className="admin-card" aria-labelledby="db-setup-guidance-title">
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
                    <h3>Advanced DSN import</h3>
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

                <article className="admin-card" aria-labelledby="db-settings-title">
                <h2 id="db-settings-title">Database settings</h2>
                {dbSettings ? (
                  <p>
                    Configured: {dbSettings.configured ? "Yes" : "No"} · Last updated: {dbSettings.updatedAt}
                  </p>
                ) : null}
                <form className="admin-form" onSubmit={(event) => void onSaveDatabaseSettings(event)}>
                  <label>
                    <input
                      type="checkbox"
                      checked={dbForm.enabled}
                      onChange={(event) => setDbForm((current) => ({ ...current, enabled: event.target.checked }))}
                    />
                    Enable PostgreSQL metadata backend
                  </label>
                  <label htmlFor="database-host">Host</label>
                  <input id="database-host" value={dbForm.host} onChange={(event) => setDbForm((current) => ({ ...current, host: event.target.value }))} />
                  {dbFieldErrors.host ? <p role="alert">{dbFieldErrors.host}</p> : null}
                  <label htmlFor="database-port">Port</label>
                  <input id="database-port" value={dbForm.port} onChange={(event) => setDbForm((current) => ({ ...current, port: event.target.value }))} />
                  {dbFieldErrors.port ? <p role="alert">{dbFieldErrors.port}</p> : null}
                  <label htmlFor="database-name">Database</label>
                  <input id="database-name" value={dbForm.database} onChange={(event) => setDbForm((current) => ({ ...current, database: event.target.value }))} />
                  {dbFieldErrors.database ? <p role="alert">{dbFieldErrors.database}</p> : null}
                  <label htmlFor="database-username">Username</label>
                  <input id="database-username" value={dbForm.username} onChange={(event) => setDbForm((current) => ({ ...current, username: event.target.value }))} />
                  {dbFieldErrors.username ? <p role="alert">{dbFieldErrors.username}</p> : null}
                  <label htmlFor="database-password">Password</label>
                  <input
                    id="database-password"
                    type="password"
                    autoComplete="off"
                    value={dbForm.password}
                    onChange={(event) => setDbForm((current) => ({ ...current, password: event.target.value, passwordClear: false }))}
                  />
                  <label>
                    <input
                      type="checkbox"
                      checked={dbForm.passwordClear}
                      onChange={(event) => setDbForm((current) => ({ ...current, passwordClear: event.target.checked }))}
                    />
                    Clear configured password
                  </label>
                  {dbFieldErrors.password ? <p role="alert">{dbFieldErrors.password}</p> : null}
                  <label htmlFor="database-ssl-mode">SSL mode</label>
                  <select id="database-ssl-mode" value={dbForm.sslMode} onChange={(event) => setDbForm((current) => ({ ...current, sslMode: event.target.value as DatabaseSslMode }))}>
                    <option value="disable">disable</option>
                    <option value="prefer">prefer</option>
                    <option value="require">require</option>
                    <option value="verify-ca">verify-ca</option>
                    <option value="verify-full">verify-full</option>
                  </select>
                  <label htmlFor="database-extra-params">Extra parameters</label>
                  <input id="database-extra-params" value={dbForm.extraParams} onChange={(event) => setDbForm((current) => ({ ...current, extraParams: event.target.value }))} />
                  <label htmlFor="database-container-name">Container name</label>
                  <input id="database-container-name" value={dbForm.containerName} onChange={(event) => setDbForm((current) => ({ ...current, containerName: event.target.value }))} />
                  {dbFieldErrors.containerName ? <p role="alert">{dbFieldErrors.containerName}</p> : null}
                  <label htmlFor="database-container-image">Container image</label>
                  <input id="database-container-image" value={dbForm.containerImage} onChange={(event) => setDbForm((current) => ({ ...current, containerImage: event.target.value }))} />
                  {dbFieldErrors.containerImage ? <p role="alert">{dbFieldErrors.containerImage}</p> : null}
                  <label htmlFor="database-host-port">Host port</label>
                  <input id="database-host-port" value={dbForm.containerHostPort} onChange={(event) => setDbForm((current) => ({ ...current, containerHostPort: event.target.value }))} />
                  {dbFieldErrors.containerHostPort ? <p role="alert">{dbFieldErrors.containerHostPort}</p> : null}
                  <label htmlFor="database-volume-name">Volume name</label>
                  <input id="database-volume-name" value={dbForm.volumeName} onChange={(event) => setDbForm((current) => ({ ...current, volumeName: event.target.value }))} />
                  {dbFieldErrors.volumeName ? <p role="alert">{dbFieldErrors.volumeName}</p> : null}
                  <button type="submit" disabled={dbSavePending}>
                    Save settings
                  </button>
                </form>
                {dbSaveMessage ? <p role="status">{dbSaveMessage}</p> : null}
                </article>

                <article className="admin-card database-command-pane" aria-labelledby="db-test-title">
                <h2 id="db-test-title">Connection check</h2>
                <button type="button" onClick={() => void onTestDatabaseConnection()} disabled={dbTestPending}>
                  Test connection
                </button>
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
                ) : null}
                  <h3>Command preview</h3>
                  <p>Connection DSN preview</p>
                  <pre>{dsnPreview}</pre>
                  <p><strong>Docker</strong></p>
                  <pre>{dockerPreview}</pre>
                  <button type="button" onClick={() => void onCopyDatabaseCommand("Docker", dockerPreview)}>
                    Copy Docker command
                  </button>
                  <p><strong>Podman</strong></p>
                  <pre>{podmanPreview}</pre>
                  <button type="button" onClick={() => void onCopyDatabaseCommand("Podman", podmanPreview)}>
                    Copy Podman command
                  </button>
                  {dbCopyStatus ? (
                    <p role="status" aria-live="polite" aria-label="Clipboard status">
                      {dbCopyStatus}
                    </p>
                  ) : null}
                  {dbCopyError ? <p role="alert">{dbCopyError}</p> : null}
                </article>
              </div>
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
                  <label htmlFor="mcp-project-ids">Project IDs</label>
                  <input
                    id="mcp-project-ids"
                    value={mcpProjectIdsInput}
                    onChange={(event) => setMcpProjectIdsInput(event.target.value)}
                    placeholder="project-1,project-2"
                    required
                  />
                  <button type="submit" disabled={mcpCreatePending}>
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
                          <p>Projects: {client.projectIds.length > 0 ? client.projectIds.join(", ") : "None"}</p>
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
            <section className="admin-section" aria-label="Sync dashboard">
              <article className="admin-card">
                <div className="inline-actions">
                  <h2>Project sync queue</h2>
                  <button type="button" className="secondary-button" onClick={() => void refreshSyncState()}>
                    Refresh
                  </button>
                </div>
                {syncLoading ? <p role="status">Loading sync queue status…</p> : null}
                {syncMessage ? <p role="status">{syncMessage}</p> : null}
                {syncError ? <p role="alert">{syncError}</p> : null}
                {!syncLoading && syncRows.length === 0 ? (
                  <p>
                    No sync queue entries yet. <a href="/projects">Create a project</a> to enqueue files for sync.
                  </p>
                ) : null}
                {syncRows.length > 0 ? (
                  <ul className="entity-list" aria-label="Sync projects list">
                    {syncRows.map((row) => (
                      <li key={row.projectId} className="entity-item">
                        <div>
                          <strong>{row.projectName ?? row.projectId}</strong>
                          <p>Project ID: {row.projectId}</p>
                          <p>Dirty count: {row.dirtyCount}</p>
                          <p>Sync state: {row.syncState}</p>
                          <p>Reconcile state: {row.reconcileState}</p>
                          <p>Rebuild state: {row.rebuildState}</p>
                          <p>Requires reconciliation: {row.requiresReconciliation ? "Yes" : "No"}</p>
                          <p>Lifecycle telemetry: {row.lifecycleTelemetry}</p>
                        </div>
                        <div className="inline-actions">
                          <button type="button" className="secondary-button" disabled={syncPendingAction !== null} onClick={() => void onSyncAction(row.projectId, "now")}>Sync now {row.projectId}</button>
                          <button type="button" className="secondary-button" disabled={syncPendingAction !== null} onClick={() => void onSyncAction(row.projectId, "reconcile")}>Reconcile {row.projectId}</button>
                          <button type="button" disabled={syncPendingAction !== null} onClick={() => void onSyncAction(row.projectId, "rebuild")}>Rebuild {row.projectId}</button>
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </article>
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
                  <label htmlFor="workflow-source-project-id">Project ID</label>
                  <input
                    id="workflow-source-project-id"
                    value={workflowSourceProjectId}
                    onChange={(event) => setWorkflowSourceProjectId(event.target.value)}
                    placeholder="p1"
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
                  <button type="submit" disabled={workflowSourcePending || workflowProjectsCount === 0}>
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
