import "./App.css";
import { ChangeEvent, FormEvent, Fragment, KeyboardEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

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

type SecretModel = {
  name: string;
  keyId: string | null;
  createdAt: string;
  updatedAt: string;
};

type SecretForm = {
  name: string;
  value: string;
  keyId: string;
};

type ModalShellProps = {
  titleId: string;
  title: string;
  eyebrow: string;
  className?: string;
  children: ReactNode;
  onClose: () => void;
};

type LlmFeedbackMessagesProps = {
  error: string;
  message: string;
  className?: string;
};

type TablePaginationProps = {
  label: string;
  page: number;
  totalItems: number;
  onPageChange: (page: number) => void;
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

type StatusTone = "neutral" | "info" | "success" | "warning" | "danger";

function toWatcherRows(items: WatcherStateItem[], projectNameById: Map<string, string>): WatcherDashboardRow[] {
  return items.map((item) => ({
    projectId: item.project_id,
    projectName: projectNameById.get(item.project_id) ?? null,
    state: item.state,
    dirtyCount: item.dirty_count,
    requiresReconciliation: item.requires_reconciliation,
    lastEventAt: item.last_event_at,
    updatedAt: item.updated_at,
  }));
}

function watcherStatusTone(row: WatcherDashboardRow): StatusTone {
  if (row.requiresReconciliation) return "warning";
  if (row.state === "enabled") return "success";
  if (row.state === "paused") return "info";
  if (row.state === "disabled") return "danger";
  return "neutral";
}

function watcherStatusLabel(row: WatcherDashboardRow): string {
  return row.requiresReconciliation ? "Needs reconcile" : formatStatusValue(row.state);
}

type SyncDashboardRow = {
  projectId: string;
  projectName: string | null;
  dirtyCount: number;
  requiresReconciliation: boolean;
  syncState: string;
  reconcileState: string;
  rebuildState: string;
  updatedAt: string | null;
  lifecycleTelemetry: string;
};

type SyncLogEntryModel = {
  id: number;
  projectId: string;
  path: string;
  eventType: string;
  reason: string;
  status: string;
  enqueuedAt: string;
  updatedAt: string;
  processedAt: string | null;
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
      updatedAt: typeof item?.updated_at === "string" && item.updated_at.trim().length > 0 ? item.updated_at : null,
      lifecycleTelemetry:
        typeof item?.updated_at === "string" && item.updated_at.trim().length > 0
          ? `updated_at=${item.updated_at}`
          : "not reported by the current sync endpoint",
    };
  });
}

function syncStatusTone(row: SyncDashboardRow): StatusTone {
  const states = [row.syncState, row.reconcileState, row.rebuildState].join(" ").toLowerCase();
  if (states.includes("fail") || states.includes("error")) return "danger";
  if (row.requiresReconciliation) return "warning";
  if (row.dirtyCount > 0 || row.syncState !== "idle") return "info";
  return "success";
}

function syncStatusLabel(row: SyncDashboardRow): string {
  if (row.requiresReconciliation) return "Needs reconcile";
  if (row.dirtyCount > 0) return "Queued";
  if (row.syncState !== "idle") return formatStatusValue(row.syncState);
  return "Clean";
}

function formatStatusValue(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) return "Unknown";
  return trimmed
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function formatLogValue(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) return "Unknown";
  return trimmed
    .split(/[:\-_\s]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

type YamlTokenSegment = {
  text: string;
  className?: string;
};

function findYamlCommentStart(text: string): number {
  let quote: "\"" | "'" | null = null;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const previous = index > 0 ? text[index - 1] : "";
    if (char === "\"" && quote !== "'" && previous !== "\\") {
      quote = quote === "\"" ? null : "\"";
    } else if (char === "'" && quote !== "\"") {
      quote = quote === "'" ? null : "'";
    } else if (char === "#" && quote === null && (index === 0 || /\s/.test(previous))) {
      return index;
    }
  }
  return -1;
}

function yamlScalarSegments(text: string): YamlTokenSegment[] {
  const commentStart = findYamlCommentStart(text);
  const mainText = commentStart >= 0 ? text.slice(0, commentStart) : text;
  const commentText = commentStart >= 0 ? text.slice(commentStart) : "";
  const tokenPattern =
    /({{[^}]+}}|"(?:\\.|[^"\\])*"|'(?:''|[^'])*'|\b(?:true|false|null|yes|no|on|off)\b|-?\b\d+(?:\.\d+)?\b|\[[^\]]*\]|\{[^{}]*\})/gi;
  const segments: YamlTokenSegment[] = [];
  let cursor = 0;
  for (const match of mainText.matchAll(tokenPattern)) {
    const index = match.index ?? 0;
    if (index > cursor) segments.push({ text: mainText.slice(cursor, index) });
    const value = match[0];
    let className = "yaml-token yaml-token--scalar";
    if (value.startsWith("{{")) className = "yaml-token yaml-token--template";
    else if (value.startsWith("\"") || value.startsWith("'")) className = "yaml-token yaml-token--string";
    else if (/^-?\d/.test(value)) className = "yaml-token yaml-token--number";
    else if (value.startsWith("[") || value.startsWith("{")) className = "yaml-token yaml-token--collection";
    segments.push({ text: value, className });
    cursor = index + value.length;
  }
  if (cursor < mainText.length) segments.push({ text: mainText.slice(cursor) });
  if (commentText) segments.push({ text: commentText, className: "yaml-token yaml-token--comment" });
  return segments;
}

function renderYamlSegments(segments: YamlTokenSegment[], keyPrefix: string): ReactNode {
  return segments.map((segment, index) =>
    segment.className ? (
      <span key={`${keyPrefix}-${index}`} className={segment.className}>
        {segment.text}
      </span>
    ) : (
      <Fragment key={`${keyPrefix}-${index}`}>{segment.text}</Fragment>
    ),
  );
}

function renderYamlLineContent(line: string, lineIndex: number): ReactNode {
  const indentMatch = line.match(/^\s*/);
  const indent = indentMatch?.[0] ?? "";
  const body = line.slice(indent.length);
  const keyPrefix = `yaml-${lineIndex}`;
  if (body.length === 0) return <Fragment>{indent || " "}</Fragment>;
  if (body.startsWith("#")) {
    return (
      <>
        {indent}
        <span className="yaml-token yaml-token--comment">{body}</span>
      </>
    );
  }

  const sequenceMatch = body.match(/^-\s+(.*)$/);
  const prefix = sequenceMatch ? "- " : "";
  const content = sequenceMatch ? sequenceMatch[1] : body;
  const keyMatch = content.match(/^([^:#]+):(.*)$/);

  return (
    <>
      {indent}
      {prefix ? <span className="yaml-token yaml-token--dash">{prefix}</span> : null}
      {keyMatch ? (
        <>
          <span className="yaml-token yaml-token--key">{keyMatch[1]}</span>
          <span className="yaml-token yaml-token--punctuation">:</span>
          {renderYamlSegments(yamlScalarSegments(keyMatch[2]), keyPrefix)}
        </>
      ) : (
        renderYamlSegments(yamlScalarSegments(content), keyPrefix)
      )}
    </>
  );
}

function YamlCodeBlock({ yaml }: { yaml: string }): ReactNode {
  const lines = yaml.split(/\r?\n/);
  return (
    <div className="workflow-yaml-viewer">
      <div className="workflow-yaml-viewer__toolbar" aria-hidden="true">
        <span>syntax highlighted</span>
        <span>{lines.length} lines</span>
      </div>
      <pre className="workflow-yaml-viewer__code" tabIndex={0}>
        <code>
          {lines.map((line, index) => (
            <span className="yaml-line" key={`${index}-${line}`}>
              <span className="yaml-line-number" aria-hidden="true">
                {index + 1}
              </span>
              <span className="yaml-line-content">{renderYamlLineContent(line, index)}</span>
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}

function formatJsonCodeValue(value: unknown, parseString = false): string {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (parseString && trimmed.length > 0) {
      try {
        return JSON.stringify(JSON.parse(trimmed) as unknown, null, 2);
      } catch {
        return value;
      }
    }
    return value;
  }

  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

function jsonTokenClassName(token: string, source: string, index: number): string {
  if (token.startsWith("\"")) {
    return /^\s*:/.test(source.slice(index + token.length)) ? "json-token json-token--key" : "json-token json-token--string";
  }
  if (/^-?\d/.test(token)) return "json-token json-token--number";
  if (token === "true" || token === "false") return "json-token json-token--boolean";
  if (token === "null") return "json-token json-token--null";
  return "json-token json-token--punctuation";
}

function renderJsonSegments(source: string): ReactNode {
  const tokenPattern = /"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b|[{}\[\],:]/g;
  const segments: ReactNode[] = [];
  let cursor = 0;

  for (const match of source.matchAll(tokenPattern)) {
    const index = match.index ?? 0;
    if (index > cursor) segments.push(<Fragment key={`json-text-${cursor}`}>{source.slice(cursor, index)}</Fragment>);

    const token = match[0];
    segments.push(
      <span key={`json-token-${index}`} className={jsonTokenClassName(token, source, index)}>
        {token}
      </span>,
    );
    cursor = index + token.length;
  }

  if (cursor < source.length) segments.push(<Fragment key={`json-text-${cursor}`}>{source.slice(cursor)}</Fragment>);
  return segments;
}

function JsonCodeBlock({ value, parseString = false }: { value: unknown; parseString?: boolean }): ReactNode {
  const source = formatJsonCodeValue(value, parseString);
  return (
    <pre className="runs-json-code" tabIndex={0}>
      <code>{renderJsonSegments(source)}</code>
    </pre>
  );
}

type WorkflowSummaryModel = {
  name: string;
  description: string;
  version: string;
  tags: string[];
  sourcePath: string | null;
};

type WorkflowDetailModel = WorkflowSummaryModel & {
  yamlPath: string | null;
  rawYaml: string | null;
  loadLogs: string[];
};

type WorkflowSourceModel = {
  sourceId: string;
  projectId: string;
  sourcePath: string;
  status: string | null;
  discoveredAt: string;
  lastLoadedAt: string | null;
  errorMessage: string | null;
  isSystem: boolean;
};

type RunRowModel = {
  runId: string;
  jobId: string;
  workflowName: string;
  status: string;
  executionMode: string;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  updatedAt: string;
  durationMs: number | null;
  cancellable: boolean;
  projectId: string | null;
  tokenId: string | null;
};

type RunBlockModel = {
  blockId: string;
  blockType: string | null;
  status: string | null;
  outcome: string | null;
  durationMs: number | null;
  message: string | null;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  metadata: Record<string, unknown>;
};

type RunDetailModel = RunRowModel & {
  resultSummary: string | null;
  errorSummary: string | null;
  inputs: Record<string, unknown>;
  outputs: unknown;
  error: string | null;
  metadata: Record<string, unknown>;
  blocks: RunBlockModel[];
  technicalJson: string;
};

type OneTimeMcpSecret = {
  token: string;
  configSnippet: string | null;
  label: string;
  clientId?: string | null;
};

type FolderBrowserTarget = "fsRoot" | "allowlist";
type ProjectModalMode = "new" | "edit";

const MCP_SECRET_MISSING_ERROR = "Token issuance response was incomplete; no secret was returned.";
const VALID_NAME_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$/;
const SECRET_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

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

const DEFAULT_SECRET_FORM: SecretForm = {
  name: "",
  value: "",
  keyId: "",
};

const TABLE_PAGE_SIZE = 10;

function ModalShell({ titleId, title, eyebrow, className, children, onClose }: ModalShellProps) {
  const dialogRef = useRef<HTMLElement | null>(null);
  const openerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    openerRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const dialog = dialogRef.current;
    const initialFocusable = dialog ? getInitialModalFocusElement(dialog) : null;
    initialFocusable?.focus();

    return () => {
      document.body.style.overflow = previousBodyOverflow;
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
        className={["llm-modal", className].filter(Boolean).join(" ")}
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
          <button type="button" className="llm-modal__close" aria-label="Close dialog" title="Close" onClick={onClose}>
            <svg aria-hidden="true" viewBox="0 0 24 24" focusable="false">
              <path d="M6 6l12 12M18 6 6 18" />
            </svg>
          </button>
        </header>
        <div className="llm-modal__body">{children}</div>
      </section>
    </div>
  );
}

function LlmFeedbackMessages({ error, message, className }: LlmFeedbackMessagesProps) {
  return (
    <>
      {message ? <p className={className} role="status">{message}</p> : null}
      {error ? <p className={className} role="alert">{error}</p> : null}
    </>
  );
}

function OneTimeMcpTokenCard({ secret }: { secret: OneTimeMcpSecret }) {
  const cardRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    cardRef.current?.focus();
  }, [secret.token]);

  return (
    <div ref={cardRef} className="token-card" role="status" aria-live="polite" tabIndex={0}>
      <h3>Copy once: token and snippet</h3>
      <p>Client label: {secret.label}</p>
      <pre>{secret.token}</pre>
      {secret.configSnippet ? (
        <pre>{secret.configSnippet}</pre>
      ) : (
        <p>No configuration snippet was returned. Use the token directly in your MCP client settings.</p>
      )}
    </div>
  );
}

function pageCountFor(totalItems: number): number {
  return Math.max(1, Math.ceil(totalItems / TABLE_PAGE_SIZE));
}

function clampTablePage(page: number, totalItems: number): number {
  return Math.min(Math.max(1, page), pageCountFor(totalItems));
}

function paginateItems<T>(items: T[], page: number): T[] {
  const currentPage = clampTablePage(page, items.length);
  const startIndex = (currentPage - 1) * TABLE_PAGE_SIZE;
  return items.slice(startIndex, startIndex + TABLE_PAGE_SIZE);
}

function TablePagination({ label, page, totalItems, onPageChange }: TablePaginationProps) {
  if (totalItems <= TABLE_PAGE_SIZE) return null;

  const currentPage = clampTablePage(page, totalItems);
  const pageCount = pageCountFor(totalItems);
  const firstItem = (currentPage - 1) * TABLE_PAGE_SIZE + 1;
  const lastItem = Math.min(totalItems, currentPage * TABLE_PAGE_SIZE);

  return (
    <nav className="table-pagination" aria-label={`${label} pagination`}>
      <p>
        Showing {firstItem}-{lastItem} of {totalItems}
        <span>10/page</span>
      </p>
      <div className="table-pagination__controls">
        <button
          type="button"
          className="secondary-button"
          disabled={currentPage === 1}
          onClick={() => onPageChange(currentPage - 1)}
        >
          Previous page
        </button>
        <span>
          Page {currentPage} of {pageCount}
        </span>
        <button
          type="button"
          className="secondary-button"
          disabled={currentPage === pageCount}
          onClick={() => onPageChange(currentPage + 1)}
        >
          Next page
        </button>
      </div>
    </nav>
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

function syncActionFailureMessage(payload: unknown): string | null {
  const obj = toObject(payload);
  if (typeof obj.status !== "string" || obj.status.toLowerCase() !== "failed") return null;
  const error = toObject(obj.error);
  if (typeof error.message === "string" && error.message.trim().length > 0) {
    return error.message;
  }
  return "Project sync failed.";
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

function formatJsonFieldValue(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) return "{}";
  return JSON.stringify(JSON.parse(trimmed) as unknown, null, 2);
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

function toSyncLogEntryModel(payload: unknown): SyncLogEntryModel {
  const obj = toObject(payload);
  const rawId = typeof obj.id === "number" ? obj.id : Number(obj.id);
  return {
    id: Number.isFinite(rawId) ? rawId : 0,
    projectId: typeof obj.project_id === "string" ? obj.project_id : "",
    path: typeof obj.path === "string" ? obj.path : "",
    eventType: typeof obj.event_type === "string" ? obj.event_type : "",
    reason: typeof obj.reason === "string" ? obj.reason : "",
    status: typeof obj.status === "string" ? obj.status : "unknown",
    enqueuedAt: typeof obj.enqueued_at === "string" ? obj.enqueued_at : "",
    updatedAt: typeof obj.updated_at === "string" ? obj.updated_at : "",
    processedAt: typeof obj.processed_at === "string" ? obj.processed_at : null,
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

function toSecretModel(payload: unknown): SecretModel {
  const obj = toObject(payload);
  return {
    name: typeof obj.name === "string" ? obj.name : "",
    keyId: toNullableString(obj.key_id),
    createdAt: typeof obj.created_at === "string" ? obj.created_at : "Unavailable",
    updatedAt: typeof obj.updated_at === "string" ? obj.updated_at : "Unavailable",
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

function sameStringSet(left: string[], right: string[]): boolean {
  if (left.length !== right.length) return false;
  const rightValues = new Set(right);
  return left.every((value) => rightValues.has(value));
}

function slugifyProjectIdentifier(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 128);
}

function toStringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function toWorkflowSummaryModel(payload: unknown): WorkflowSummaryModel {
  const obj = toObject(payload);
  return {
    name: typeof obj.name === "string" ? obj.name : "",
    description: typeof obj.description === "string" ? obj.description : "",
    version: typeof obj.version === "string" ? obj.version : "",
    tags: toStringList(obj.tags),
    sourcePath: typeof obj.source_path === "string" ? obj.source_path : null,
  };
}

function toWorkflowDetailModel(payload: unknown): WorkflowDetailModel {
  const summary = toWorkflowSummaryModel(payload);
  const obj = toObject(payload);
  return {
    ...summary,
    yamlPath: toNullableString(obj.yaml_path),
    rawYaml: toNullableString(obj.raw_yaml),
    loadLogs: toStringList(obj.load_logs),
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
    isSystem: obj.is_system === true,
  };
}

function workflowBelongsToSource(workflow: WorkflowSummaryModel, source: WorkflowSourceModel): boolean {
  if (!workflow.sourcePath) return false;
  const workflowPath = workflow.sourcePath.replace(/\/+$/, "");
  const sourcePath = source.sourcePath.replace(/\/+$/, "");
  return workflowPath === sourcePath || workflowPath.startsWith(`${sourcePath}/`);
}

function workflowSourceStatus(source: WorkflowSourceModel, workflowCount: number): string {
  if (source.status && source.status.trim().length > 0) return source.status;
  return workflowCount > 0 ? "loaded" : "pending";
}

function statusToneForValue(status: string | null | undefined): StatusTone {
  const normalized = (status ?? "").trim().toLowerCase();
  if (["loaded", "completed", "success", "ok", "clean", "active", "enabled"].includes(normalized)) return "success";
  if (["failed", "failure", "error", "invalid", "blocked"].includes(normalized)) return "danger";
  if (["pending", "queued", "running", "loading", "validating"].includes(normalized)) return "warning";
  if (["paused", "unknown", ""].includes(normalized)) return "neutral";
  return "info";
}

function formatRunTimestamp(value: string | null): string {
  if (!value) return "Not recorded";
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(parsed));
}

function formatDurationMs(value: number | null): string {
  if (value === null) return "Pending";
  if (value < 1000) return `${value} ms`;
  const seconds = value / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

function runCanResume(run: RunRowModel): boolean {
  return run.status.toLowerCase() === "paused";
}

function toRunBlockModel(payload: unknown): RunBlockModel {
  const obj = toObject(payload);
  return {
    blockId: typeof obj.block_id === "string" ? obj.block_id : "",
    blockType: typeof obj.block_type === "string" ? obj.block_type : null,
    status: typeof obj.status === "string" ? obj.status : null,
    outcome: typeof obj.outcome === "string" ? obj.outcome : null,
    durationMs: toNullableNumber(obj.duration_ms),
    message: typeof obj.message === "string" ? obj.message : null,
    inputs: toObject(obj.inputs),
    outputs: toObject(obj.outputs),
    metadata: toObject(obj.metadata),
  };
}

function toRunRowModel(payload: unknown): RunRowModel {
  const obj = toObject(payload);
  return {
    runId: typeof obj.run_id === "string" ? obj.run_id : "",
    jobId: typeof obj.job_id === "string" ? obj.job_id : "",
    workflowName: typeof obj.workflow_name === "string" ? obj.workflow_name : "",
    status: typeof obj.status === "string" ? obj.status : "unknown",
    executionMode: typeof obj.execution_mode === "string" ? obj.execution_mode : "async",
    createdAt: typeof obj.created_at === "string" ? obj.created_at : "Unavailable",
    startedAt: typeof obj.started_at === "string" ? obj.started_at : null,
    finishedAt: typeof obj.finished_at === "string" ? obj.finished_at : null,
    updatedAt: typeof obj.updated_at === "string" ? obj.updated_at : "Unavailable",
    durationMs: toNullableNumber(obj.duration_ms),
    cancellable: obj.cancellable === true,
    projectId: typeof obj.project_id === "string" ? obj.project_id : null,
    tokenId: typeof obj.token_id === "string" ? obj.token_id : null,
  };
}

function toRunDetailModel(payload: unknown): RunDetailModel {
  const obj = toObject(payload);
  const row = toRunRowModel(payload);
  const metadata = toObject(obj.metadata);
  const blocks = Array.isArray(obj.blocks) ? obj.blocks.map(toRunBlockModel) : [];
  return {
    ...row,
    resultSummary: typeof obj.result_summary === "string" ? obj.result_summary : null,
    errorSummary: typeof obj.error_summary === "string" ? obj.error_summary : null,
    inputs: toObject(obj.inputs),
    outputs: obj.outputs,
    error: typeof obj.error === "string" ? obj.error : null,
    metadata,
    blocks,
    technicalJson: JSON.stringify(obj.technical_json ?? obj, null, 2),
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
  const [projectModalMode, setProjectModalMode] = useState<ProjectModalMode | null>(null);
  const [projectModalProject, setProjectModalProject] = useState<ProjectModel | null>(null);
  const [projectDeleteConfirm, setProjectDeleteConfirm] = useState("");
  const [projectDeletePending, setProjectDeletePending] = useState(false);
  const [projectSavePending, setProjectSavePending] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectSlug, setProjectSlug] = useState("");
  const [projectSlugTouched, setProjectSlugTouched] = useState(false);
  const [projectPalace, setProjectPalace] = useState("");
  const [projectPalaceTouched, setProjectPalaceTouched] = useState(false);
  const [projectDefaultWing, setProjectDefaultWing] = useState("");
  const [projectDefaultRoom, setProjectDefaultRoom] = useState("");
  const [projectFsRoot, setProjectFsRoot] = useState(DEFAULT_PROJECT_FS_ROOT);
  const [projectAllowlistInput, setProjectAllowlistInput] = useState("");
  const [projectsTablePage, setProjectsTablePage] = useState(1);
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
  const [mcpProjectEditIds, setMcpProjectEditIds] = useState<string[]>([]);
  const [oneTimeMcpSecret, setOneTimeMcpSecret] = useState<OneTimeMcpSecret | null>(null);
  const [mcpCreateModalOpen, setMcpCreateModalOpen] = useState(false);
  const [selectedMcpClientId, setSelectedMcpClientId] = useState<string | null>(null);
  const [mcpClientsTablePage, setMcpClientsTablePage] = useState(1);
  const [watchersRows, setWatchersRows] = useState<WatcherDashboardRow[]>([]);
  const [watchersLoading, setWatchersLoading] = useState(false);
  const [watchersError, setWatchersError] = useState("");
  const [watchersMessage, setWatchersMessage] = useState("");
  const [watchersPendingAction, setWatchersPendingAction] = useState<string | null>(null);
  const [watchersTablePage, setWatchersTablePage] = useState(1);
  const [selectedWatcherProjectId, setSelectedWatcherProjectId] = useState<string | null>(null);
  const [syncRows, setSyncRows] = useState<SyncDashboardRow[]>([]);
  const [syncLoading, setSyncLoading] = useState(false);
  const [syncError, setSyncError] = useState("");
  const [syncMessage, setSyncMessage] = useState("");
  const [syncPendingAction, setSyncPendingAction] = useState<string | null>(null);
  const [syncTablePage, setSyncTablePage] = useState(1);
  const [selectedSyncProjectId, setSelectedSyncProjectId] = useState<string | null>(null);
  const [syncLogs, setSyncLogs] = useState<SyncLogEntryModel[]>([]);
  const [syncLogsLoading, setSyncLogsLoading] = useState(false);
  const [syncLogsError, setSyncLogsError] = useState("");
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
  const [workflowSourcesTablePage, setWorkflowSourcesTablePage] = useState(1);
  const [workflowSourceModalOpen, setWorkflowSourceModalOpen] = useState(false);
  const [expandedWorkflowSourceIds, setExpandedWorkflowSourceIds] = useState<Set<string>>(new Set());
  const [workflowSourcePathPickerOpen, setWorkflowSourcePathPickerOpen] = useState(false);
  const [selectedWorkflowName, setSelectedWorkflowName] = useState<string | null>(null);
  const [selectedWorkflowDetail, setSelectedWorkflowDetail] = useState<WorkflowDetailModel | null>(null);
  const [workflowDetailLoading, setWorkflowDetailLoading] = useState(false);
  const [workflowSchemaText, setWorkflowSchemaText] = useState<string>("");
  const workflowSourcePathRef = useRef<HTMLInputElement | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState("");
  const [runsMessage, setRunsMessage] = useState("");
  const [runsRows, setRunsRows] = useState<RunRowModel[]>([]);
  const [runDetail, setRunDetail] = useState<RunDetailModel | null>(null);
  const [runFilterStatus, setRunFilterStatus] = useState("");
  const [runFilterMode, setRunFilterMode] = useState("");
  const [runFilterWorkflow, setRunFilterWorkflow] = useState("");
  const [runFilterProjectId, setRunFilterProjectId] = useState("");
  const [runFilterLimit, setRunFilterLimit] = useState("50");
  const [runFilterOffset, setRunFilterOffset] = useState("0");
  const [runsTotal, setRunsTotal] = useState(0);
  const [runActionPendingId, setRunActionPendingId] = useState<string | null>(null);
  const [runResumeTarget, setRunResumeTarget] = useState<RunRowModel | null>(null);
  const [runResumeResponse, setRunResumeResponse] = useState("");
  const [llmConfig, setLlmConfig] = useState<LlmConfigModel>(EMPTY_LLM_CONFIG);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmSaving, setLlmSaving] = useState(false);
  const [llmError, setLlmError] = useState("");
  const [llmMessage, setLlmMessage] = useState("");
  const [llmProviderForm, setLlmProviderForm] = useState<LlmProviderForm>(DEFAULT_LLM_PROVIDER_FORM);
  const [llmProviderModalOpen, setLlmProviderModalOpen] = useState(false);
  const [llmProviderEditId, setLlmProviderEditId] = useState<string | null>(null);
  const [llmProfileForm, setLlmProfileForm] = useState<LlmProfileForm>(DEFAULT_LLM_PROFILE_FORM);
  const [llmProfileModalOpen, setLlmProfileModalOpen] = useState(false);
  const [llmProfileEditId, setLlmProfileEditId] = useState<string | null>(null);
  const [llmYamlImport, setLlmYamlImport] = useState("");
  const [llmYamlModalOpen, setLlmYamlModalOpen] = useState(false);
  const [llmPreview, setLlmPreview] = useState<LlmConfigModel | null>(null);
  const [llmProvidersTablePage, setLlmProvidersTablePage] = useState(1);
  const [llmProfilesTablePage, setLlmProfilesTablePage] = useState(1);
  const [secrets, setSecrets] = useState<SecretModel[]>([]);
  const [secretsLoading, setSecretsLoading] = useState(false);
  const [secretsError, setSecretsError] = useState("");
  const [secretsMessage, setSecretsMessage] = useState("");
  const [secretForm, setSecretForm] = useState<SecretForm>(DEFAULT_SECRET_FORM);
  const [selectedSecretName, setSelectedSecretName] = useState<string | null>(null);
  const [secretModalOpen, setSecretModalOpen] = useState(false);
  const [secretSavePending, setSecretSavePending] = useState(false);
  const [secretDeletePending, setSecretDeletePending] = useState(false);
  const [secretDeleteConfirm, setSecretDeleteConfirm] = useState("");

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

  useEffect(() => {
    setProjectsTablePage((current) => clampTablePage(current, projects.length));
  }, [projects.length]);

  useEffect(() => {
    setWatchersTablePage((current) => clampTablePage(current, watchersRows.length));
  }, [watchersRows.length]);

  useEffect(() => {
    setSyncTablePage((current) => clampTablePage(current, syncRows.length));
  }, [syncRows.length]);

  useEffect(() => {
    setMcpClientsTablePage((current) => clampTablePage(current, mcpClients.length));
  }, [mcpClients.length]);

  useEffect(() => {
    setLlmProvidersTablePage((current) => clampTablePage(current, Object.keys(llmConfig.providers).length));
    setLlmProfilesTablePage((current) => clampTablePage(current, Object.keys(llmConfig.profiles).length));
  }, [llmConfig.providers, llmConfig.profiles]);

  const loadProjects = async (): Promise<ProjectModel[]> => {
    setProjectsLoading(true);
    setProjectError("");
    try {
      const result = await api.listProjects();
      const projectModels = result.projects.map(toProjectModel);
      const validProjectIds = new Set(projectModels.map((project) => project.id));
      setProjects(projectModels);
      setMcpSelectedProjectIds((current) => current.filter((projectId) => validProjectIds.has(projectId)));
      setMcpProjectEditIds((current) => current.filter((projectId) => validProjectIds.has(projectId)));
      if (selectedSyncProjectId && !validProjectIds.has(selectedSyncProjectId)) {
        setSelectedSyncProjectId(null);
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
      const clientModels = result.mcp_clients.map(toMcpClientModel);
      setMcpClients(clientModels);
      if (selectedMcpClientId && !clientModels.some((client) => client.id === selectedMcpClientId)) {
        setSelectedMcpClientId(null);
      }
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
        mode: runFilterMode.trim() || undefined,
        workflow: runFilterWorkflow.trim() || undefined,
        projectId: runFilterProjectId.trim() || undefined,
        limit: Number.isFinite(limit) && limit > 0 ? limit : 50,
        offset: Number.isFinite(offset) && offset >= 0 ? offset : 0,
      });
      setRunsRows(payload.runs.map(toRunRowModel));
      setRunsTotal(typeof payload.total === "number" ? payload.total : payload.runs.length);
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

  const loadSecrets = async (): Promise<void> => {
    setSecretsLoading(true);
    setSecretsError("");
    try {
      const payload = await api.listSecrets();
      setSecrets(payload.secrets.map(toSecretModel));
    } catch (error) {
      setSecretsError(toUserError(error, "Unable to load secrets."));
    } finally {
      setSecretsLoading(false);
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
    const target = runsRows.find((run) => run.runId === runId) ?? runDetail;
    setRunResumeTarget(target);
    setRunResumeResponse("");
  };

  const onSubmitRunResume = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const target = runResumeTarget;
    if (!target) return;
    const runId = target.runId;
    setRunActionPendingId(`resume:${runId}`);
    setRunsError("");
    setRunsMessage("");
    try {
      await api.resumeRun(runId, runResumeResponse);
      setRunsMessage(`Run ${runId} resume submitted.`);
      setRunResumeTarget(null);
      setRunResumeResponse("");
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
          if (!cancelled) {
            setContentState("");
            setSecretsMessage("");
            setSecretsError("");
            setSecretModalOpen(false);
            setSelectedSecretName(null);
            setSecretDeleteConfirm("");
            setSecretForm(DEFAULT_SECRET_FORM);
            await loadSecrets();
          }
          return;
        }
        if (currentPath === "/watchers") {
          if (!cancelled) {
            setContentState("");
            setWatchersLoading(true);
            setWatchersError("");
            setWatchersMessage("");
            setSelectedWatcherProjectId(null);
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
            setSyncLogs([]);
            setSyncLogsError("");
            setSyncLogsLoading(false);
            setSelectedSyncProjectId(null);
          }
          const projectsPayload = await api.listProjects();
          const projectModels = projectsPayload.projects.map(toProjectModel);
          if (!cancelled) {
            setProjects(projectModels);
            if (selectedSyncProjectId && !projectModels.some((project) => project.id === selectedSyncProjectId)) {
              setSelectedSyncProjectId(null);
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
            setSelectedWorkflowDetail(null);
            setWorkflowDetailLoading(false);
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
            setRunResumeTarget(null);
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
      setWatchersRows(toWatcherRows(payload.items, projectNameById));
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
        setSelectedSyncProjectId(null);
      }
      setSyncRows(buildSyncDashboardRows(payload.items, projects));
    } catch (error) {
      setSyncError(toUserError(error, "Unable to refresh sync queue status."));
    }
  };

  const loadSyncLogs = async (projectId: string): Promise<void> => {
    setSyncLogsLoading(true);
    setSyncLogsError("");
    try {
      const payload = await api.listSyncLogs(projectId);
      setSyncLogs(payload.entries.map(toSyncLogEntryModel));
    } catch (error) {
      setSyncLogs([]);
      setSyncLogsError(toUserError(error, `Unable to load sync activity for ${projectId}.`));
    } finally {
      setSyncLogsLoading(false);
    }
  };

  const refreshSyncDetails = async (projectId: string): Promise<void> => {
    await Promise.all([refreshSyncState(), loadSyncLogs(projectId)]);
  };

  const openWatcherStatusModal = (row: WatcherDashboardRow): void => {
    setSelectedWatcherProjectId(row.projectId);
    setWatchersError("");
    setWatchersMessage("");
  };

  const closeWatcherStatusModal = (): void => {
    setSelectedWatcherProjectId(null);
  };

  const openSyncStatusModal = (row: SyncDashboardRow): void => {
    setSelectedSyncProjectId(row.projectId);
    setSyncError("");
    setSyncMessage("");
    setSyncLogs([]);
    setSyncLogsError("");
    void loadSyncLogs(row.projectId);
  };

  const closeSyncStatusModal = (): void => {
    setSelectedSyncProjectId(null);
    setSyncLogs([]);
    setSyncLogsError("");
    setSyncLogsLoading(false);
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
        const result = await api.syncNow(projectId);
        const failureMessage = syncActionFailureMessage(result);
        if (failureMessage) throw new Error(failureMessage);
        setSyncMessage(`Sync requested for ${projectId}.`);
      }
      if (action === "reconcile") {
        const result = await api.reconcileSync(projectId);
        const failureMessage = syncActionFailureMessage(result);
        if (failureMessage) throw new Error(failureMessage);
        setSyncMessage(`Reconcile requested for ${projectId}.`);
      }
      if (action === "rebuild") {
        const result = await api.rebuildSync(projectId);
        const failureMessage = syncActionFailureMessage(result);
        if (failureMessage) throw new Error(failureMessage);
        setSyncMessage(`Rebuild requested for ${projectId}.`);
      }
      await refreshSyncDetails(projectId);
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
    const projectId = workflowSourceProjectId.trim();
    setWorkflowSourcePending(true);
    setWorkflowsError("");
    setWorkflowsMessage("");
    try {
      await api.createWorkflowSource({
        project_id: projectId,
        source_path: workflowSourcePath.trim(),
        checksum: workflowSourceChecksum.trim() || null,
      });
      let nextMessage = "";
      let nextError = "";
      try {
        const payload = await api.reloadWorkflows();
        nextMessage = `Workflow source added for project ${projectId}. Workflow registry reloaded: ${payload.total} workflows from ${payload.source_count} source(s).`;
      } catch (reloadError) {
        nextError = `Workflow source added, but reload failed: ${toUserError(reloadError, "reload failed")}`;
      }
      setWorkflowSourcePath("");
      setWorkflowSourceChecksum("");
      setWorkflowSourcePathPickerOpen(false);
      setWorkflowSourceModalOpen(false);
      await loadWorkflowsPageData();
      if (nextMessage) {
        setWorkflowsMessage(nextMessage);
      }
      if (nextError) {
        setWorkflowsError(nextError);
      }
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

  const toggleWorkflowSource = (sourceId: string): void => {
    setExpandedWorkflowSourceIds((current) => {
      const next = new Set(current);
      if (next.has(sourceId)) {
        next.delete(sourceId);
      } else {
        next.add(sourceId);
      }
      return next;
    });
  };

  const closeWorkflowDetailModal = (): void => {
    setSelectedWorkflowName(null);
    setSelectedWorkflowDetail(null);
    setWorkflowDetailLoading(false);
  };

  const onViewWorkflowDetails = async (workflowName: string): Promise<void> => {
    setSelectedWorkflowName(workflowName);
    setSelectedWorkflowDetail(null);
    setWorkflowDetailLoading(true);
    setWorkflowsError("");
    setWorkflowsMessage("");
    try {
      const details = await api.getWorkflowDetail(workflowName);
      setSelectedWorkflowDetail(toWorkflowDetailModel(details));
    } catch (error) {
      setWorkflowsError(toUserError(error, `Unable to load workflow details for ${workflowName}.`));
      setSelectedWorkflowDetail(null);
    } finally {
      setWorkflowDetailLoading(false);
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

  const resetProjectForm = (): void => {
    setProjectName("");
    setProjectSlug("");
    setProjectSlugTouched(false);
    setProjectPalace("");
    setProjectPalaceTouched(false);
    setProjectDefaultWing("");
    setProjectDefaultRoom("");
    setProjectFsRoot(DEFAULT_PROJECT_FS_ROOT);
    setProjectAllowlistInput("");
  };

  const setProjectFormFromProject = (project: ProjectModel): void => {
    setProjectName(project.name);
    setProjectSlug(project.slug);
    setProjectSlugTouched(true);
    setProjectPalace(project.palace);
    setProjectPalaceTouched(true);
    setProjectDefaultWing(project.defaultWing);
    setProjectDefaultRoom(project.defaultRoom);
    setProjectFsRoot(project.fsRoot || DEFAULT_PROJECT_FS_ROOT);
    setProjectAllowlistInput(project.fsAllowlist.join("\n"));
  };

  const onProjectNameChange = (value: string): void => {
    setProjectName(value);
    if (projectModalMode !== "new") return;

    const derivedIdentifier = slugifyProjectIdentifier(value);
    if (!projectSlugTouched) {
      setProjectSlug(derivedIdentifier);
    }
    if (!projectPalaceTouched) {
      setProjectPalace(derivedIdentifier);
    }
  };

  const onProjectSlugChange = (value: string): void => {
    setProjectSlug(value);
    setProjectSlugTouched(true);
    if (projectModalMode === "new" && !projectPalaceTouched) {
      setProjectPalace(value);
    }
  };

  const onProjectPalaceChange = (value: string): void => {
    setProjectPalace(value);
    setProjectPalaceTouched(true);
  };

  const openNewProjectModal = (): void => {
    setProjectError("");
    setProjectMessage("");
    setProjectDeleteConfirm("");
    setProjectModalProject(null);
    resetProjectForm();
    setPathPickerOpen(false);
    setProjectModalMode("new");
  };

  const openProjectConfigModal = (project: ProjectModel): void => {
    setProjectError("");
    setProjectMessage("");
    setProjectDeleteConfirm("");
    setProjectModalProject(project);
    setProjectFormFromProject(project);
    setPathPickerOpen(false);
    setProjectModalMode("edit");
  };

  const closeProjectModal = (): void => {
    setProjectModalMode(null);
    setProjectModalProject(null);
    setProjectDeleteConfirm("");
    setPathPickerOpen(false);
    resetProjectForm();
  };

  const projectFormPayload = (): Record<string, unknown> => ({
    name: projectName.trim(),
    slug: projectSlug.trim(),
    palace: projectPalace.trim(),
    default_wing: projectDefaultWing.trim() || null,
    default_room: projectDefaultRoom.trim() || null,
    fs_root: projectFsRoot.trim(),
    fs_allowlist: normalizePathList(projectAllowlistInput),
  });

  const onSaveProject = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setProjectSavePending(true);
    setProjectError("");
    setProjectMessage("");
    try {
      const payload = projectFormPayload();
      if (projectModalMode === "edit" && projectModalProject) {
        await api.updateProject(projectModalProject.id, payload);
        setProjectMessage(`Project ${projectModalProject.id} saved.`);
      } else {
        await api.createProject(payload);
        setProjectMessage("Project registered successfully.");
      }
      closeProjectModal();
      await loadProjects();
    } catch (error) {
      setProjectError(toUserError(error, "Unable to save project. Review values and retry."));
    } finally {
      setProjectSavePending(false);
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

  const closeWorkflowSourcePathPicker = (): void => {
    setWorkflowSourcePathPickerOpen(false);
    queueMicrotask(() => workflowSourcePathRef.current?.focus());
  };

  const openNewWorkflowSourceModal = (): void => {
    setWorkflowsError("");
    setWorkflowsMessage("");
    setWorkflowSourcePath("");
    setWorkflowSourceChecksum("");
    setWorkflowSourcePathPickerOpen(false);
    setWorkflowSourceModalOpen(true);
  };

  const closeWorkflowSourceModal = (): void => {
    setWorkflowSourceModalOpen(false);
    setWorkflowSourcePathPickerOpen(false);
    setWorkflowSourcePath("");
    setWorkflowSourceChecksum("");
    setWorkflowsError("");
  };

  const onSelectWorkflowSourcePath = (selectedPath: string): void => {
    setWorkflowSourcePath(selectedPath);
    setWorkflowsMessage(`Workflow source path set to ${selectedPath}.`);
    closeWorkflowSourcePathPicker();
  };

  const onConfirmDeleteProject = async (): Promise<void> => {
    if (!projectModalProject) return;
    const target = projectModalProject;
    setProjectDeletePending(true);
    setProjectError("");
    setProjectMessage("");
    try {
      await api.deleteProject(target.id);
      setProjectMessage(`Project ${target.id} deleted.`);
      closeProjectModal();
      await loadProjects();
    } catch (error) {
      setProjectError(toUserError(error, "Unable to delete project. Retry if this project is still required."));
    } finally {
      setProjectDeletePending(false);
    }
  };

  const openNewMcpClientModal = (): void => {
    setMcpError("");
    setMcpMessage("");
    setMcpCreateModalOpen(true);
  };

  const closeNewMcpClientModal = (): void => {
    setMcpCreateModalOpen(false);
  };

  const openMcpClientDetailModal = (client: MpcClientModel): void => {
    setSelectedMcpClientId(client.id);
    setMcpProjectEditIds(client.projectIds);
    setMcpError("");
    setMcpMessage("");
  };

  const closeMcpClientDetailModal = (): void => {
    setSelectedMcpClientId(null);
    setMcpProjectEditIds([]);
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
        clientId: toNonEmptyString(result.id),
      });
      setMcpMessage("MCP client created. Save the token now—it will not be shown again.");
      setMcpLabel("");
      setMcpSelectedProjectIds([]);
      await loadMcpClients();
      setOneTimeMcpSecret({
        token,
        configSnippet: toNonEmptyString(result.config_snippet),
        label: secretLabel,
        clientId: toNonEmptyString(result.id),
      });
      setMcpCreateModalOpen(false);
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

  const onDeleteMcpClient = async (tokenId: string): Promise<void> => {
    setMcpMutationPendingId(tokenId);
    setMcpError("");
    setMcpMessage("");
    try {
      await api.deleteMcpClient(tokenId);
      setSelectedMcpClientId(null);
      setOneTimeMcpSecret((current) => (current?.clientId === tokenId ? null : current));
      setMcpMessage(`MCP client ${tokenId} deleted.`);
      await loadMcpClients();
    } catch (error) {
      setMcpError(toUserError(error, "Unable to delete MCP client token."));
    } finally {
      setMcpMutationPendingId(null);
    }
  };

  const onUpdateMcpClientProjects = async (tokenId: string): Promise<void> => {
    setMcpMutationPendingId(tokenId);
    setMcpError("");
    setMcpMessage("");
    try {
      const updated = await api.updateMcpClient(tokenId, {
        project_ids: mcpProjectEditIds,
      });
      const updatedProjects = Array.isArray(updated.project_ids)
        ? updated.project_ids.filter((item): item is string => typeof item === "string")
        : mcpProjectEditIds;
      setMcpProjectEditIds(updatedProjects);
      setMcpMessage(`MCP client ${tokenId} project access updated.`);
      await loadMcpClients();
    } catch (error) {
      setMcpError(toUserError(error, "Unable to update MCP client project access."));
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
        clientId: tokenId,
      });
      setMcpMessage("MCP token regenerated. Save the new token now—it will not be shown again.");
      await loadMcpClients();
      setOneTimeMcpSecret({
        token,
        configSnippet: toNonEmptyString(result.config_snippet),
        label: toNonEmptyString(result.label) ?? tokenId,
        clientId: tokenId,
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
    setLlmProviderEditId(null);
    setLlmProviderForm(DEFAULT_LLM_PROVIDER_FORM);
    setLlmProviderModalOpen(true);
  };

  const closeLlmProviderModal = (): void => {
    setLlmProviderEditId(null);
    setLlmProviderForm(DEFAULT_LLM_PROVIDER_FORM);
    setLlmProviderModalOpen(false);
  };

  const openNewLlmProfileModal = (): void => {
    setLlmError("");
    setLlmMessage("");
    setLlmProfileEditId(null);
    setLlmProfileForm(DEFAULT_LLM_PROFILE_FORM);
    setLlmProfileModalOpen(true);
  };

  const closeLlmProfileModal = (): void => {
    setLlmProfileEditId(null);
    setLlmProfileForm(DEFAULT_LLM_PROFILE_FORM);
    setLlmProfileModalOpen(false);
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
      setLlmProviderEditId(null);
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
    setLlmMessage("");
    setLlmProviderEditId(providerId);
    setLlmProviderForm(toLlmProviderForm(providerId, provider));
    setLlmProviderModalOpen(true);
  };

  const onDeleteLlmProvider = (providerId: string): boolean => {
    const referencingProfiles = Object.entries(llmConfig.profiles)
      .filter(([, profile]) => profile.provider === providerId)
      .map(([profileId]) => profileId);
    setLlmError("");
    setLlmMessage("");
    if (referencingProfiles.length > 0) {
      setLlmError(`Cannot delete provider ${providerId}; remove or reassign profiles first: ${referencingProfiles.join(", ")}.`);
      return false;
    }
    setLlmConfig((current) => {
      const nextProviders = { ...current.providers };
      delete nextProviders[providerId];
      return normalizedLlmConfig({ ...current, providers: nextProviders });
    });
    setLlmMessage(`Provider ${providerId} staged for deletion. Save LLM configuration to persist.`);
    return true;
  };

  const onDeleteCurrentLlmProvider = (): void => {
    if (!llmProviderEditId) return;
    if (onDeleteLlmProvider(llmProviderEditId)) {
      closeLlmProviderModal();
    }
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
      setLlmProfileEditId(null);
      setLlmProfileModalOpen(false);
      setLlmMessage(`Profile ${id} staged. Save LLM configuration to persist.`);
    } catch (error) {
      setLlmError(toUserError(error, "Unable to stage profile."));
    }
  };

  const onEditLlmProfile = (profileId: string): void => {
    const profile = llmConfig.profiles[profileId];
    if (!profile) return;
    setLlmError("");
    setLlmMessage("");
    setLlmProfileEditId(profileId);
    setLlmProfileForm(toLlmProfileForm(profileId, profile));
    setLlmProfileModalOpen(true);
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

  const onDeleteCurrentLlmProfile = (): void => {
    if (!llmProfileEditId) return;
    onDeleteLlmProfile(llmProfileEditId);
    closeLlmProfileModal();
  };

  const onFormatLlmProviderExtraHeaders = (): void => {
    setLlmProviderForm((current) => {
      try {
        return { ...current, extraHeaders: formatJsonFieldValue(current.extraHeaders) };
      } catch {
        return current;
      }
    });
  };

  const openNewSecretModal = (): void => {
    setSelectedSecretName(null);
    setSecretForm(DEFAULT_SECRET_FORM);
    setSecretDeleteConfirm("");
    setSecretsMessage("");
    setSecretsError("");
    setSecretModalOpen(true);
  };

  const openSecretManagement = (secret: SecretModel): void => {
    setSelectedSecretName(secret.name);
    setSecretForm({ name: secret.name, value: "", keyId: secret.keyId ?? "" });
    setSecretDeleteConfirm("");
    setSecretsMessage("");
    setSecretsError("");
    setSecretModalOpen(true);
  };

  const closeSecretModal = (): void => {
    setSecretModalOpen(false);
    setSelectedSecretName(null);
    setSecretForm(DEFAULT_SECRET_FORM);
    setSecretDeleteConfirm("");
    setSecretsError("");
  };

  const onSaveSecret = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const name = secretForm.name.trim();
    const value = secretForm.value;
    const keyId = secretForm.keyId.trim();

    setSecretsError("");
    setSecretsMessage("");

    if (!SECRET_NAME_PATTERN.test(name)) {
      setSecretsError("Secret name must start with a letter or underscore and use only letters, numbers, or underscores.");
      return;
    }
    if (value.length === 0) {
      setSecretsError("Enter a new secret value before saving.");
      return;
    }

    setSecretSavePending(true);
    try {
      await api.upsertSecret({ name, value, key_id: keyId || null });
      await loadSecrets();
      closeSecretModal();
      setSecretsMessage(`Secret ${name} saved.`);
    } catch (error) {
      setSecretsError(toUserError(error, `Unable to save secret ${name}.`));
    } finally {
      setSecretSavePending(false);
    }
  };

  const onDeleteSelectedSecret = async (): Promise<void> => {
    if (!selectedSecretName || secretDeleteConfirm !== selectedSecretName) return;

    setSecretDeletePending(true);
    setSecretsError("");
    setSecretsMessage("");
    try {
      await api.deleteSecret(selectedSecretName);
      await loadSecrets();
      closeSecretModal();
      setSecretsMessage(`Secret ${selectedSecretName} deleted.`);
    } catch (error) {
      setSecretsError(toUserError(error, `Unable to delete secret ${selectedSecretName}.`));
    } finally {
      setSecretDeletePending(false);
    }
  };

  const onConfigurationTableRowKeyDown = (event: KeyboardEvent<HTMLTableRowElement>, onActivate: () => void): void => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    onActivate();
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
  const paginatedMcpClients = paginateItems(mcpClients, mcpClientsTablePage);
  const selectedMcpClient = selectedMcpClientId
    ? mcpClients.find((client) => client.id === selectedMcpClientId) ?? null
    : null;
  const mcpProjectEditChanged = selectedMcpClient
    ? !sameStringSet(selectedMcpClient.projectIds, mcpProjectEditIds)
    : false;
  const selectedMcpClientOneTimeSecret =
    selectedMcpClient && oneTimeMcpSecret
      ? oneTimeMcpSecret.clientId === selectedMcpClient.id ||
        oneTimeMcpSecret.label === selectedMcpClient.id ||
        oneTimeMcpSecret.label === selectedMcpClient.label
        ? oneTimeMcpSecret
        : null
      : null;
  const paginatedSyncRows = paginateItems(syncRows, syncTablePage);
  const selectedSyncRow = selectedSyncProjectId
    ? syncRows.find((row) => row.projectId === selectedSyncProjectId) ?? null
    : null;
  const selectedSyncLabel = selectedSyncRow?.projectName ?? selectedSyncRow?.projectId ?? "";
  const syncDirtyTotal = syncRows.reduce((total, row) => total + row.dirtyCount, 0);
  const syncReconcileCount = syncRows.filter((row) => row.requiresReconciliation).length;
  const syncQueuedCount = syncRows.filter((row) => row.dirtyCount > 0 || row.syncState !== "idle").length;
  const watchersDirtyTotal = watchersRows.reduce((total, row) => total + row.dirtyCount, 0);
  const watchersReconcileCount = watchersRows.filter((row) => row.requiresReconciliation).length;
  const watchersEnabledCount = watchersRows.filter((row) => row.state === "enabled").length;
  const paginatedWatchers = paginateItems(watchersRows, watchersTablePage);
  const selectedWatcher = selectedWatcherProjectId
    ? watchersRows.find((row) => row.projectId === selectedWatcherProjectId) ?? null
    : null;
  const selectedWatcherLabel = selectedWatcher?.projectName ?? selectedWatcher?.projectId ?? "";
  const llmProviderEntries = Object.entries(llmConfig.providers);
  const llmProfileEntries = Object.entries(llmConfig.profiles);
  const paginatedLlmProviderEntries = paginateItems(llmProviderEntries, llmProvidersTablePage);
  const paginatedLlmProfileEntries = paginateItems(llmProfileEntries, llmProfilesTablePage);
  const llmDefaultProfileLabel = llmConfig.default_profile ?? "None";
  const llmModalOpen = llmProviderModalOpen || llmProfileModalOpen || llmYamlModalOpen;
  const selectedSecret = selectedSecretName
    ? secrets.find((secret) => secret.name === selectedSecretName) ?? null
    : null;
  const secretModalTitle = selectedSecret ? `Secret ${selectedSecret.name}` : "New secret";
  const secretDeleteAllowed = Boolean(selectedSecret && secretDeleteConfirm === selectedSecret.name);
  const projectModalOpen = projectModalMode !== null;
  const paginatedProjects = paginateItems(projects, projectsTablePage);
  const projectAllowlistCount = projects.filter((project) => project.fsAllowlist.length > 0).length;
  const projectHintCount = projects.filter((project) => project.watcherHint || project.defaultStateHint).length;
  const projectModalTitle =
    projectModalMode === "edit" && projectModalProject
      ? `Project ${projectModalProject.name || projectModalProject.id}`
      : "New project";
  const projectSummaryName = projectName.trim() || "Untitled project";
  const projectSummarySlug = projectSlug.trim() || "Slug pending";
  const projectSummaryPalace = projectPalace.trim() || "Palace pending";
  const projectSummaryDefault = `${projectDefaultWing.trim() || "Not set"}/${projectDefaultRoom.trim() || "Not set"}`;
  const projectSummaryFsRoot = projectFsRoot.trim() || "FS root pending";
  const projectSummaryAllowlist = normalizePathList(projectAllowlistInput);
  const projectModalClassName = projectModalMode === "new" ? "project-modal project-modal--new" : "project-modal";
  const workflowsBySourceId = new Map(
    workflowSources.map((source) => [
      source.sourceId,
      workflowsList.filter((workflow) => workflowBelongsToSource(workflow, source)),
    ]),
  );
  const paginatedWorkflowSources = paginateItems(workflowSources, workflowSourcesTablePage);
  const loadedWorkflowSourceCount = workflowSources.filter((source) =>
    workflowSourceStatus(source, workflowsBySourceId.get(source.sourceId)?.length ?? 0).toLowerCase() === "loaded"
  ).length;
  const failedWorkflowSourceCount = workflowSources.filter((source) =>
    workflowSourceStatus(source, workflowsBySourceId.get(source.sourceId)?.length ?? 0).toLowerCase() === "failed"
  ).length;
  const selectedWorkflowLogs =
    selectedWorkflowDetail?.loadLogs && selectedWorkflowDetail.loadLogs.length > 0
      ? selectedWorkflowDetail.loadLogs
      : selectedWorkflowDetail
        ? ["No load warnings or errors were reported for this workflow."]
        : [];
  const workflowModalOpen = workflowSourceModalOpen || selectedWorkflowName !== null;
  const runsStatusCounts = runsRows.reduce<Record<string, number>>((counts, run) => {
    const key = run.status.toLowerCase() || "unknown";
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});
  const runsFailureCount = (runsStatusCounts.failed ?? 0) + (runsStatusCounts.failure ?? 0);
  const runsActiveCount = (runsStatusCounts.queued ?? 0) + (runsStatusCounts.running ?? 0) + (runsStatusCounts.paused ?? 0);
  const runsCompletedCount = runsStatusCounts.completed ?? 0;
  const selectedRunSummary = runDetail?.error ?? runDetail?.errorSummary ?? runDetail?.resultSummary ?? null;

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

          {currentPath === "/secrets" ? (
            <section className="admin-section llm-workbench secrets-workbench" aria-label="Secrets management">
              <section className="llm-status-panel" aria-label="Secrets registry summary">
                <div>
                  <p className="database-kicker">secret registry</p>
                  <h2>Secret configuration</h2>
                  <p>Rows open the secret configuration dialog. Values remain write-only.</p>
                </div>
                <dl className="llm-status-grid">
                  <div>
                    <dt>Configured</dt>
                    <dd>Secrets: {secrets.length}</dd>
                  </div>
                  <div>
                    <dt>Selected</dt>
                    <dd>Selected: {selectedSecret?.name ?? "None"}</dd>
                  </div>
                  <div>
                    <dt>Value visibility</dt>
                    <dd>Values: write-only</dd>
                  </div>
                </dl>
              </section>

              <div className="llm-status-stack">
                {secretsLoading ? (
                  <p role="status" aria-label="Loading secrets">
                    Loading secrets...
                  </p>
                ) : null}
                {secretsError ? <p role="alert">{secretsError}</p> : null}
                {!secretModalOpen && secretsMessage ? (
                  <p
                    role="status"
                    aria-label={secretsMessage.includes("deleted") ? "Secret delete status" : "Secret save status"}
                  >
                    {secretsMessage}
                  </p>
                ) : null}
              </div>

              <div className="llm-workbench__grid llm-workbench__grid--single">
                <article className="admin-card" aria-labelledby="secrets-table-title">
                  <div className="inline-actions projects-table-actions">
                    <h2 id="secrets-table-title">Configured secrets</h2>
                    <button type="button" onClick={openNewSecretModal}>
                      New
                    </button>
                    <button type="button" className="secondary-button" onClick={() => void loadSecrets()}>
                      Reload
                    </button>
                  </div>
                  <p id="secret-row-action-hint" className="visually-hidden">
                    Opens the secret configuration dialog. Press Enter or Space to activate.
                  </p>
                  {secrets.length === 0 && !secretsLoading ? (
                    <p>No secrets configured. Use New to register the first secret.</p>
                  ) : null}
                  <div className="llm-table-wrap">
                    <table className="llm-configuration-table secrets-table" aria-label="Configured secrets">
                      <thead>
                        <tr>
                          <th scope="col">Name</th>
                          <th scope="col">Key ID</th>
                          <th scope="col">Created</th>
                          <th scope="col">Updated</th>
                        </tr>
                      </thead>
                      <tbody>
                        {secrets.map((secret) => (
                          <tr
                            key={secret.name}
                            aria-describedby="secret-row-action-hint"
                            aria-haspopup="dialog"
                            aria-label={`Open secret ${secret.name} configuration`}
                            aria-keyshortcuts="Enter Space"
                            tabIndex={0}
                            onClick={() => openSecretManagement(secret)}
                            onKeyDown={(event) => onConfigurationTableRowKeyDown(event, () => openSecretManagement(secret))}
                          >
                            <th scope="row">{secret.name}</th>
                            <td>{secret.keyId ?? "Server default"}</td>
                            <td>{secret.createdAt}</td>
                            <td>{secret.updatedAt}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </article>
              </div>

              {secretModalOpen ? (
                <ModalShell
                  titleId="secret-dialog-title"
                  title={secretModalTitle}
                  eyebrow="Secret configuration"
                  onClose={closeSecretModal}
                >
                  <form className="admin-form llm-form secrets-form" onSubmit={(event) => void onSaveSecret(event)}>
                    <LlmFeedbackMessages className="llm-modal-feedback" message={secretsMessage} error={secretsError} />
                    <p>Saving requires entering a new value. Stored values are not displayed after save.</p>
                    <div className="field-group">
                      <label htmlFor="secret-name">Secret name</label>
                      <input
                        id="secret-name"
                        value={secretForm.name}
                        onChange={(event) =>
                          setSecretForm((current) => ({ ...current, name: event.target.value }))
                        }
                        autoComplete="off"
                        required
                      />
                    </div>
                    <div className="field-group">
                      <label htmlFor="secret-value">Secret value</label>
                      <input
                        id="secret-value"
                        type="password"
                        value={secretForm.value}
                        onChange={(event) =>
                          setSecretForm((current) => ({ ...current, value: event.target.value }))
                        }
                        autoComplete="new-password"
                        required
                      />
                    </div>
                    <div className="field-group">
                      <label htmlFor="secret-key-id">Key ID</label>
                      <input
                        id="secret-key-id"
                        value={secretForm.keyId}
                        onChange={(event) =>
                          setSecretForm((current) => ({ ...current, keyId: event.target.value }))
                        }
                        autoComplete="off"
                      />
                    </div>
                    <div className="inline-actions">
                      <button type="submit" disabled={secretSavePending}>
                        {selectedSecret ? "Save secret" : "Register secret"}
                      </button>
                      <button type="button" className="secondary-button" onClick={closeSecretModal}>
                        Cancel
                      </button>
                    </div>

                    {selectedSecret ? (
                      <section className="secrets-delete-panel" aria-labelledby="secret-delete-title">
                        <h3 id="secret-delete-title">Delete secret</h3>
                        <p>Type the secret name to confirm deletion.</p>
                        <div className="field-group">
                          <label htmlFor="secret-delete-confirm">Confirm secret name</label>
                          <input
                            id="secret-delete-confirm"
                            value={secretDeleteConfirm}
                            onChange={(event) => setSecretDeleteConfirm(event.target.value)}
                            autoComplete="off"
                          />
                        </div>
                        <button
                          type="button"
                          className="danger-button"
                          disabled={!secretDeleteAllowed || secretDeletePending}
                          onClick={() => void onDeleteSelectedSecret()}
                        >
                          Confirm delete {selectedSecret.name}
                        </button>
                      </section>
                    ) : null}
                  </form>
                </ModalShell>
              ) : null}
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
                <LlmFeedbackMessages
                  message={llmModalOpen ? "" : llmMessage}
                  error={llmModalOpen ? "" : llmError}
                />
              </div>

              <div className="llm-workbench__grid llm-workbench__grid--single">
                <article className="admin-card" aria-labelledby="llm-providers-title">
                  <h2 id="llm-providers-title">Providers</h2>
                  <p id="llm-provider-row-action-hint" className="visually-hidden">
                    Opens the provider edit dialog. Press Enter or Space to activate.
                  </p>
                  {llmProviderEntries.length === 0 ? <p>No providers configured.</p> : null}
                  <div className="llm-table-wrap">
                    <table className="llm-configuration-table" aria-label="LLM providers">
                      <thead>
                        <tr>
                          <th scope="col">Provider ID</th>
                          <th scope="col">Type</th>
                          <th scope="col">Model</th>
                          <th scope="col">Endpoint</th>
                          <th scope="col">Secret</th>
                        </tr>
                      </thead>
                      <tbody>
                      {paginatedLlmProviderEntries.map(([providerId, provider]) => (
                        <tr
                          key={providerId}
                          aria-describedby="llm-provider-row-action-hint"
                          aria-haspopup="dialog"
                          aria-keyshortcuts="Enter Space"
                          aria-label={`Edit provider ${providerId}`}
                          tabIndex={0}
                          onClick={() => onEditLlmProvider(providerId)}
                          onKeyDown={(event) => onConfigurationTableRowKeyDown(event, () => onEditLlmProvider(providerId))}
                        >
                          <th scope="row">{providerId}</th>
                          <td>{provider.type || "Not set"}</td>
                          <td>{provider.model ?? "Not set"}</td>
                          <td>{provider.api_url ?? "Default endpoint"}</td>
                          <td>{provider.api_key_secret ?? "Not set"}</td>
                        </tr>
                      ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="inline-actions llm-card-actions">
                    <button type="button" onClick={openNewLlmProviderModal}>
                      Add provider
                    </button>
                    <button type="button" className="secondary-button" onClick={() => void loadLlmConfig()}>
                      Reload
                    </button>
                  </div>
                  <TablePagination
                    label="LLM providers"
                    page={llmProvidersTablePage}
                    totalItems={llmProviderEntries.length}
                    onPageChange={setLlmProvidersTablePage}
                  />
                </article>
              </div>

              <div className="llm-workbench__grid llm-workbench__grid--single">
                <article className="admin-card" aria-labelledby="llm-profiles-title">
                  <h2 id="llm-profiles-title">Profiles</h2>
                  <p id="llm-profile-row-action-hint" className="visually-hidden">
                    Opens the profile edit dialog. Press Enter or Space to activate.
                  </p>
                  {llmProfileEntries.length === 0 ? <p>No profiles configured.</p> : null}
                  <div className="llm-table-wrap">
                    <table className="llm-configuration-table" aria-label="LLM profiles">
                      <thead>
                        <tr>
                          <th scope="col">Profile ID</th>
                          <th scope="col">Provider</th>
                          <th scope="col">Model</th>
                          <th scope="col">Temperature</th>
                          <th scope="col">Max tokens</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paginatedLlmProfileEntries.map(([profileId, profile]) => (
                          <tr
                            key={profileId}
                            aria-describedby="llm-profile-row-action-hint"
                            aria-haspopup="dialog"
                            aria-keyshortcuts="Enter Space"
                            aria-label={`Edit profile ${profileId}`}
                            tabIndex={0}
                            onClick={() => onEditLlmProfile(profileId)}
                            onKeyDown={(event) => onConfigurationTableRowKeyDown(event, () => onEditLlmProfile(profileId))}
                          >
                            <th scope="row">{profileId}</th>
                            <td>{profile.provider || "Not set"}</td>
                            <td>{profile.model || "Not set"}</td>
                            <td>{profile.temperature ?? "default"}</td>
                            <td>{profile.max_tokens ?? "default"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="inline-actions llm-card-actions">
                    <button type="button" onClick={openNewLlmProfileModal}>
                      Add profile
                    </button>
                  </div>
                  <TablePagination
                    label="LLM profiles"
                    page={llmProfilesTablePage}
                    totalItems={llmProfileEntries.length}
                    onPageChange={setLlmProfilesTablePage}
                  />
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
                    <LlmFeedbackMessages className="llm-modal-feedback" message={llmMessage} error={llmError} />
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
                        onBlur={onFormatLlmProviderExtraHeaders}
                        spellCheck={false}
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
                      <button type="submit">Save</button>
                      {llmProviderEditId ? (
                        <button type="button" className="danger-button" onClick={onDeleteCurrentLlmProvider}>
                          Delete
                        </button>
                      ) : null}
                      <button type="button" className="secondary-button" onClick={closeLlmProviderModal}>
                        Cancel
                      </button>
                    </div>
                  </form>
                </ModalShell>
              ) : null}

              {llmProfileModalOpen ? (
                <ModalShell
                  titleId="llm-profile-dialog-title"
                  title={llmProfileForm.id.trim() ? `Edit profile ${llmProfileForm.id.trim()}` : "Add profile"}
                  eyebrow="Profile"
                  onClose={closeLlmProfileModal}
                >
                  <form className="admin-form llm-form" onSubmit={onSaveLlmProfile}>
                    <LlmFeedbackMessages className="llm-modal-feedback" message={llmMessage} error={llmError} />
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
                        Save
                      </button>
                      {llmProfileEditId ? (
                        <button type="button" className="danger-button" onClick={onDeleteCurrentLlmProfile}>
                          Delete
                        </button>
                      ) : null}
                      <button type="button" className="secondary-button" onClick={closeLlmProfileModal}>
                        Cancel
                      </button>
                    </div>
                  </form>
                </ModalShell>
              ) : null}

              {llmYamlModalOpen ? (
                <ModalShell titleId="llm-yaml-dialog-title" title="Import YAML" eyebrow="YAML migration" onClose={closeLlmYamlModal}>
                  <div className="admin-form llm-form">
                    <LlmFeedbackMessages className="llm-modal-feedback" message={llmMessage} error={llmError} />
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
            <section className="admin-section llm-workbench projects-workbench" aria-label="Projects management">
              <section className="llm-status-panel" aria-label="Projects registry summary">
                <div>
                  <p className="database-kicker">project registry</p>
                  <h2>Project configuration</h2>
                  <p>Rows open the project configuration dialog.</p>
                </div>
                <dl className="llm-status-grid">
                  <div>
                    <dt>Registered</dt>
                    <dd>Projects: {projects.length}</dd>
                  </div>
                  <div>
                    <dt>Allowlists</dt>
                    <dd>Allowlisted: {projectAllowlistCount}</dd>
                  </div>
                  <div>
                    <dt>Hints</dt>
                    <dd>Runtime hints: {projectHintCount}</dd>
                  </div>
                </dl>
              </section>

              <div className="llm-status-stack">
                {projectsLoading ? <p role="status">Loading projects...</p> : null}
                <LlmFeedbackMessages
                  message={projectModalOpen ? "" : projectMessage}
                  error={projectModalOpen ? "" : projectError}
                />
              </div>

              <div className="llm-workbench__grid llm-workbench__grid--single">
                <article className="admin-card" aria-labelledby="projects-table-title">
                  <div className="inline-actions projects-table-actions">
                    <h2 id="projects-table-title">Registered projects</h2>
                    <button type="button" onClick={openNewProjectModal}>
                      New
                    </button>
                    <button type="button" className="secondary-button" onClick={() => void loadProjects()}>
                      Reload
                    </button>
                  </div>
                  <p id="project-row-action-hint" className="visually-hidden">
                    Opens the project configuration dialog. Press Enter or Space to activate.
                  </p>
                  {!projectsLoading && projects.length === 0 ? (
                    <p>No projects yet. Use New to register the first project.</p>
                  ) : null}
                  <div className="llm-table-wrap">
                    <table className="llm-configuration-table projects-configuration-table" aria-label="Registered projects">
                      <thead>
                        <tr>
                          <th scope="col">Project</th>
                          <th scope="col">Slug</th>
                          <th scope="col">Palace</th>
                          <th scope="col">Default</th>
                          <th scope="col">FS root</th>
                          <th scope="col">Allowlist</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paginatedProjects.map((project) => (
                          <tr
                            key={project.id}
                            aria-describedby="project-row-action-hint"
                            aria-haspopup="dialog"
                            aria-keyshortcuts="Enter Space"
                            aria-label={`Open project ${project.name || project.id} configuration`}
                            tabIndex={0}
                            onClick={() => openProjectConfigModal(project)}
                            onKeyDown={(event) =>
                              onConfigurationTableRowKeyDown(event, () => openProjectConfigModal(project))
                            }
                          >
                            <th scope="row">{project.name || project.id}</th>
                            <td>{project.slug || "Not set"}</td>
                            <td>{project.palace || "Not set"}</td>
                            <td>
                              {project.defaultWing || "Not set"}/{project.defaultRoom || "Not set"}
                            </td>
                            <td>{project.fsRoot || "Not set"}</td>
                            <td>
                              {project.fsAllowlist.length > 0
                                ? `${project.fsAllowlist.length} path${project.fsAllowlist.length === 1 ? "" : "s"}`
                                : "No explicit allowlist"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <TablePagination
                    label="Registered projects"
                    page={projectsTablePage}
                    totalItems={projects.length}
                    onPageChange={setProjectsTablePage}
                  />
                </article>
              </div>

              {projectModalOpen ? (
                <ModalShell
                  titleId="project-dialog-title"
                  title={projectModalTitle}
                  eyebrow={projectModalMode === "new" ? "Project onboarding" : "Project configuration"}
                  className={projectModalClassName}
                  onClose={closeProjectModal}
                >
                  <form className="admin-form llm-form project-form" onSubmit={(event) => void onSaveProject(event)}>
                    {projectModalMode === "new" ? (
                      <section className="project-onboarding-hero" aria-label="Project onboarding steps">
                        <div>
                          <p className="database-kicker">guided setup</p>
                          <h3>Register the project boundary</h3>
                          <p>
                            Start with the project identity, confirm runtime defaults, then pin the filesystem root the
                            server can browse.
                          </p>
                        </div>
                        <ol className="project-onboarding-steps">
                          <li>
                            <span>01</span>
                            <strong>Identity</strong>
                            <small>Name, slug, palace</small>
                          </li>
                          <li>
                            <span>02</span>
                            <strong>Defaults</strong>
                            <small>Wing and room</small>
                          </li>
                          <li>
                            <span>03</span>
                            <strong>Filesystem</strong>
                            <small>Root and allowlist</small>
                          </li>
                        </ol>
                      </section>
                    ) : null}

                    <div className="project-onboarding-layout">
                      <div className="project-onboarding-main">
                        <LlmFeedbackMessages className="llm-modal-feedback" message={projectMessage} error={projectError} />

                        <fieldset className="project-fieldset">
                          <legend>Identity</legend>
                          <p>Stable project identifiers used by admin tables, tokens, and metadata records.</p>
                          <div className="project-identity-grid">
                            <div className="field-group project-field-wide">
                              <label htmlFor="project-name">Name</label>
                              <input
                                id="project-name"
                                value={projectName}
                                onChange={(event) => onProjectNameChange(event.target.value)}
                                autoComplete="off"
                                required
                              />
                            </div>
                            <div className="field-group">
                              <label htmlFor="project-slug">Slug</label>
                              <input
                                id="project-slug"
                                value={projectSlug}
                                onChange={(event) => onProjectSlugChange(event.target.value)}
                                autoComplete="off"
                                required
                              />
                              {projectModalMode === "new" && !projectSlugTouched ? (
                                <p className="field-hint">Auto-filled from the project name.</p>
                              ) : null}
                            </div>
                            <div className="field-group">
                              <label htmlFor="project-palace">Palace</label>
                              <input
                                id="project-palace"
                                value={projectPalace}
                                onChange={(event) => onProjectPalaceChange(event.target.value)}
                                disabled={projectModalMode === "edit"}
                                autoComplete="off"
                                required
                              />
                              {projectModalMode === "edit" ? (
                                <p className="field-hint">Palace is immutable after registration.</p>
                              ) : projectModalMode === "new" && !projectPalaceTouched ? (
                                <p className="field-hint">Mirrors the slug until edited.</p>
                              ) : null}
                            </div>
                          </div>
                        </fieldset>

                        <fieldset className="project-fieldset">
                          <legend>Runtime defaults</legend>
                          <p>The initial state namespace used when a workflow does not provide a narrower target.</p>
                          <div className="llm-two-column">
                            <div className="field-group">
                              <label htmlFor="project-default-wing">Default wing</label>
                              <input
                                id="project-default-wing"
                                value={projectDefaultWing}
                                onChange={(event) => setProjectDefaultWing(event.target.value)}
                              />
                            </div>
                            <div className="field-group">
                              <label htmlFor="project-default-room">Default room</label>
                              <input
                                id="project-default-room"
                                value={projectDefaultRoom}
                                onChange={(event) => setProjectDefaultRoom(event.target.value)}
                              />
                            </div>
                          </div>
                        </fieldset>

                        <fieldset className="project-fieldset">
                          <legend>Filesystem boundary</legend>
                          <p>Filesystem scope for workflow source discovery and server-side folder browsing.</p>
                          <div className="field-group">
                            <label htmlFor="project-fs-root">FS root</label>
                            <div className="project-path-control">
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
                          </div>
                          <div className="field-group">
                            <label htmlFor="project-allowlist">Allowlist paths</label>
                            <div className="project-textarea-control">
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
                            </div>
                          </div>
                        </fieldset>

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

                        {projectModalMode === "edit" && projectModalProject ? (
                          <div className="project-runtime-hints" aria-label="Project runtime hints">
                            <p>
                              Watcher default hint:{" "}
                              {projectModalProject.watcherHint ??
                                "Watcher state hints are unavailable from the current projects API response."}
                            </p>
                            <p>
                              Default state hint:{" "}
                              {projectModalProject.defaultStateHint ??
                                "Default state hints are unavailable from the current projects API response."}
                            </p>
                          </div>
                        ) : null}

                        {projectModalMode === "edit" && projectModalProject ? (
                          <section className="project-delete-panel" aria-labelledby="project-delete-title">
                            <h3 id="project-delete-title">Delete project</h3>
                            <p>Type DELETE to confirm removing {projectModalProject.id}.</p>
                            <label htmlFor="project-delete-confirm">Confirm deletion</label>
                            <input
                              id="project-delete-confirm"
                              type="text"
                              value={projectDeleteConfirm}
                              onChange={(event) => setProjectDeleteConfirm(event.target.value)}
                            />
                            <div className="actions-row inline-actions">
                              <ActionButton
                                type="button"
                                variant="danger"
                                disabled={projectDeletePending || projectDeleteConfirm !== "DELETE"}
                                onClick={() => void onConfirmDeleteProject()}
                              >
                                Confirm delete {projectModalProject.id}
                              </ActionButton>
                            </div>
                          </section>
                        ) : null}
                      </div>

                      <aside className="project-onboarding-summary" aria-label="Project onboarding summary">
                        <p className="database-kicker">registration preview</p>
                        <dl>
                          <div>
                            <dt>Project</dt>
                            <dd>{projectSummaryName}</dd>
                          </div>
                          <div>
                            <dt>Slug</dt>
                            <dd>{projectSummarySlug}</dd>
                          </div>
                          <div>
                            <dt>Palace</dt>
                            <dd>{projectSummaryPalace}</dd>
                          </div>
                          <div>
                            <dt>Default</dt>
                            <dd>{projectSummaryDefault}</dd>
                          </div>
                          <div>
                            <dt>FS root</dt>
                            <dd>{projectSummaryFsRoot}</dd>
                          </div>
                          <div>
                            <dt>Allowlist</dt>
                            <dd>
                              {projectSummaryAllowlist.length > 0
                                ? `${projectSummaryAllowlist.length} path${projectSummaryAllowlist.length === 1 ? "" : "s"}`
                                : "No explicit allowlist"}
                            </dd>
                          </div>
                        </dl>
                      </aside>
                    </div>

                    {!pathPickerOpen ? (
                      <div className="inline-actions project-form__actions">
                        <ActionButton type="submit" variant="primary" disabled={projectSavePending}>
                          {projectModalMode === "edit" ? "Save project" : "Register project"}
                        </ActionButton>
                        <ActionButton type="button" variant="secondary" onClick={closeProjectModal}>
                          Cancel
                        </ActionButton>
                      </div>
                    ) : null}
                  </form>
                </ModalShell>
              ) : null}
            </section>
          ) : null}

          {currentPath === "/mcp-clients" ? (
            <section className="admin-section llm-workbench" aria-label="MCP clients management">
              <div className="llm-status-stack">
                {mcpLoading ? <p role="status">Loading MCP clients...</p> : null}
                <LlmFeedbackMessages
                  message={mcpCreateModalOpen || selectedMcpClient ? "" : mcpMessage}
                  error={mcpCreateModalOpen || selectedMcpClient ? "" : mcpError}
                />
                {oneTimeMcpSecret && !selectedMcpClientOneTimeSecret ? (
                  <OneTimeMcpTokenCard secret={oneTimeMcpSecret} />
                ) : null}
              </div>

              <div className="llm-workbench__grid llm-workbench__grid--single">
                <article className="admin-card" aria-labelledby="mcp-clients-table-title">
                  <div className="inline-actions projects-table-actions">
                    <h2 id="mcp-clients-table-title">Registered MCP clients</h2>
                    <button type="button" onClick={openNewMcpClientModal}>
                      New
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      aria-label="Reload MCP clients"
                      onClick={() => void loadMcpClients()}
                    >
                      Reload
                    </button>
                  </div>
                  <p id="mcp-client-row-action-hint" className="visually-hidden">
                    Opens the MCP client detail dialog. Press Enter or Space to activate.
                  </p>
                  {!mcpLoading && mcpClients.length === 0 ? (
                    <p>No MCP clients yet. Use New to issue the first client token.</p>
                  ) : null}
                  <div className="llm-table-wrap">
                    <table className="llm-configuration-table mcp-clients-configuration-table" aria-label="Registered MCP clients">
                      <thead>
                        <tr>
                          <th scope="col">Client</th>
                          <th scope="col">Status</th>
                          <th scope="col">Projects</th>
                          <th scope="col">Created</th>
                          <th scope="col">Last used</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paginatedMcpClients.map((client) => {
                          const statusLabel = client.revokedAt ? "Revoked" : "Active";
                          const projectLabel =
                            client.projectIds.length > 0
                              ? client.projectIds.map(formatProjectReference).join(", ")
                              : "Client-selected";
                          return (
                            <tr
                              key={client.id}
                              aria-describedby="mcp-client-row-action-hint"
                              aria-haspopup="dialog"
                              aria-keyshortcuts="Enter Space"
                              aria-label={`Open MCP client ${client.label || client.id} details ${statusLabel}`}
                              tabIndex={0}
                              onClick={() => openMcpClientDetailModal(client)}
                              onKeyDown={(event) =>
                                onConfigurationTableRowKeyDown(event, () => openMcpClientDetailModal(client))
                              }
                            >
                              <th scope="row">
                                <span className="watchers-project-cell">
                                  <strong>{client.label || client.id}</strong>
                                  <span>{client.id}</span>
                                </span>
                              </th>
                              <td>
                                <StatusBadge tone={client.revokedAt ? "danger" : "success"}>{statusLabel}</StatusBadge>
                              </td>
                              <td>{projectLabel}</td>
                              <td>{client.createdAt || "Unknown"}</td>
                              <td>{client.lastUsedAt ?? "Never"}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <TablePagination
                    label="Registered MCP clients"
                    page={mcpClientsTablePage}
                    totalItems={mcpClients.length}
                    onPageChange={setMcpClientsTablePage}
                  />
                </article>
              </div>

              {mcpCreateModalOpen ? (
                <ModalShell
                  titleId="mcp-client-create-title"
                  title="New MCP client"
                  eyebrow="Client token"
                  onClose={closeNewMcpClientModal}
                >
                  <form className="admin-form" onSubmit={(event) => void onCreateMcpClient(event)}>
                    <LlmFeedbackMessages className="llm-modal-feedback" message={mcpMessage} error={mcpError} />
                    <label htmlFor="mcp-label">Client label</label>
                    <input
                      id="mcp-label"
                      value={mcpLabel}
                      onChange={(event) => setMcpLabel(event.target.value)}
                      required
                    />
                    <ProjectMultiSelect
                      id="mcp-project-ids"
                      label="Project access"
                      projects={projects}
                      selectedIds={mcpSelectedProjectIds}
                      onChange={setMcpSelectedProjectIds}
                      disabled={projectsLoading || mcpCreatePending}
                      helperText={
                        projects.length > 0
                          ? "Leave empty for client-selected project access."
                          : "No registered projects yet. This client can select a project after one exists."
                      }
                    />
                    {projectError ? <p role="alert">{projectError}</p> : null}
                    <div className="inline-actions">
                      <button type="submit" disabled={mcpCreatePending}>
                        Create MCP client
                      </button>
                      <button type="button" className="secondary-button" onClick={closeNewMcpClientModal}>
                        Cancel
                      </button>
                    </div>
                  </form>
                </ModalShell>
              ) : null}

              {selectedMcpClient ? (
                <ModalShell
                  titleId="mcp-client-detail-title"
                  title={`MCP client ${selectedMcpClient.label || selectedMcpClient.id}`}
                  eyebrow="Client access"
                  onClose={closeMcpClientDetailModal}
                >
                  <div className="sync-status-modal">
                    <LlmFeedbackMessages className="llm-modal-feedback" message={mcpMessage} error={mcpError} />
                    {selectedMcpClientOneTimeSecret ? (
                      <OneTimeMcpTokenCard secret={selectedMcpClientOneTimeSecret} />
                    ) : null}
                    <dl className="sync-detail-grid" aria-label="MCP client details">
                      <div>
                        <dt>Client ID</dt>
                        <dd>{selectedMcpClient.id}</dd>
                      </div>
                      <div>
                        <dt>Label</dt>
                        <dd>{selectedMcpClient.label || "Not set"}</dd>
                      </div>
                      <div>
                        <dt>Status</dt>
                        <dd>{selectedMcpClient.revokedAt ? `Revoked at ${selectedMcpClient.revokedAt}` : "Active"}</dd>
                      </div>
                      <div>
                        <dt>Projects</dt>
                        <dd>
                          {selectedMcpClient.projectIds.length > 0
                            ? selectedMcpClient.projectIds.map(formatProjectReference).join(", ")
                            : "Client-selected"}
                        </dd>
                      </div>
                      <div>
                        <dt>Created</dt>
                        <dd>{selectedMcpClient.createdAt || "Unknown"}</dd>
                      </div>
                      <div>
                        <dt>Last used</dt>
                        <dd>{selectedMcpClient.lastUsedAt ?? "Never"}</dd>
                      </div>
                    </dl>
                    <section className="sync-command-panel" aria-labelledby="mcp-client-projects-title">
                      <h3 id="mcp-client-projects-title">Project access</h3>
                      <ProjectMultiSelect
                        id="mcp-client-project-edit"
                        label="Allowed projects"
                        projects={projects}
                        selectedIds={mcpProjectEditIds}
                        onChange={setMcpProjectEditIds}
                        disabled={projectsLoading || mcpMutationPendingId === selectedMcpClient.id}
                        helperText={
                          projects.length > 0
                            ? "Leave empty for client-selected project access."
                            : "No registered projects yet. This client can select a project after one exists."
                        }
                      />
                      <div className="inline-actions">
                        <button
                          type="button"
                          className="secondary-button"
                          disabled={mcpMutationPendingId === selectedMcpClient.id || !mcpProjectEditChanged}
                          onClick={() => void onUpdateMcpClientProjects(selectedMcpClient.id)}
                        >
                          Save project access
                        </button>
                      </div>
                    </section>
                    <section className="sync-command-panel" aria-labelledby="mcp-client-commands-title">
                      <h3 id="mcp-client-commands-title">Commands</h3>
                      <div className="inline-actions">
                        <button
                          type="button"
                          className="secondary-button"
                          disabled={mcpMutationPendingId === selectedMcpClient.id}
                          onClick={() => void onRegenerateMcpClient(selectedMcpClient.id)}
                        >
                          Regenerate
                        </button>
                        <button
                          type="button"
                          disabled={mcpMutationPendingId === selectedMcpClient.id}
                          onClick={() => void onRevokeMcpClient(selectedMcpClient.id)}
                        >
                          Revoke
                        </button>
                        <button
                          type="button"
                          className="danger-button"
                          disabled={mcpMutationPendingId === selectedMcpClient.id}
                          onClick={() => void onDeleteMcpClient(selectedMcpClient.id)}
                        >
                          Delete
                        </button>
                      </div>
                    </section>
                  </div>
                </ModalShell>
              ) : null}
            </section>
          ) : null}

          {currentPath === "/watchers" ? (
            <section className="admin-section status-workbench" aria-label="Watchers dashboard">
              <article className="status-hero-panel">
                <div className="status-hero-panel__copy">
                  <p className="database-kicker">Live watcher status</p>
                  <h2>Project file watchers</h2>
                  <p>
                    Monitor persisted watcher state, queued file changes, and reconciliation needs for every registered project.
                  </p>
                </div>
                <div className="status-toolbar">
                  <ActionButton variant="secondary" onClick={() => void refreshWatchersState()}>
                    Refresh
                  </ActionButton>
                </div>
              </article>

              <dl className="status-stats-grid" aria-label="Watcher summary">
                <div>
                  <dt>Registered</dt>
                  <dd>{watchersRows.length}</dd>
                </div>
                <div>
                  <dt>Enabled</dt>
                  <dd>{watchersEnabledCount}</dd>
                </div>
                <div>
                  <dt>Dirty files</dt>
                  <dd>{watchersDirtyTotal}</dd>
                </div>
                <div>
                  <dt>Needs reconcile</dt>
                  <dd>{watchersReconcileCount}</dd>
                </div>
              </dl>

              <div className="status-status-stack">
                {watchersLoading ? <p role="status">Loading watcher status…</p> : null}
                {!selectedWatcher && watchersMessage ? <p role="status">{watchersMessage}</p> : null}
                {!selectedWatcher && watchersError ? <p role="alert">{watchersError}</p> : null}
              </div>

              {!watchersLoading && watchersRows.length === 0 ? (
                <article className="status-empty-state">
                  <h2>Watcher list is empty</h2>
                  <p>
                    No watchers available yet. <a href="/projects">Create a project</a> to initialize watcher management.
                  </p>
                </article>
              ) : null}

              {watchersRows.length > 0 ? (
                <article className="admin-card" aria-labelledby="watchers-table-title">
                  <div className="inline-actions projects-table-actions">
                    <h2 id="watchers-table-title">Registered watchers</h2>
                    <button type="button" className="secondary-button" onClick={() => void refreshWatchersState()}>
                      Reload
                    </button>
                  </div>
                  <p id="watcher-row-action-hint" className="visually-hidden">
                    Opens the watcher status dialog. Press Enter or Space to activate.
                  </p>
                  <div className="llm-table-wrap">
                    <table className="llm-configuration-table watchers-configuration-table" aria-label="Registered watchers">
                      <thead>
                        <tr>
                          <th scope="col">Project</th>
                          <th scope="col">Status</th>
                          <th scope="col">Dirty</th>
                          <th scope="col">Reconcile</th>
                          <th scope="col">Last event</th>
                          <th scope="col">Updated</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paginatedWatchers.map((row) => {
                          const statusTone = watcherStatusTone(row);
                          const statusLabel = watcherStatusLabel(row);
                          const projectLabel = row.projectName ?? row.projectId;
                          return (
                            <tr
                              key={row.projectId}
                              className={`watchers-table-row watchers-table-row--${statusTone}`}
                              aria-describedby="watcher-row-action-hint"
                              aria-haspopup="dialog"
                              aria-keyshortcuts="Enter Space"
                              aria-label={`Open watcher ${projectLabel} status ${statusLabel} dirty files ${row.dirtyCount}`}
                              tabIndex={0}
                              onClick={() => openWatcherStatusModal(row)}
                              onKeyDown={(event) =>
                                onConfigurationTableRowKeyDown(event, () => openWatcherStatusModal(row))
                              }
                            >
                              <th scope="row">
                                <span className="watchers-project-cell">
                                  <strong>{projectLabel}</strong>
                                  <span>{row.projectId}</span>
                                </span>
                              </th>
                              <td>
                                <StatusBadge tone={statusTone}>{statusLabel}</StatusBadge>
                              </td>
                              <td className="watchers-number-cell">{row.dirtyCount}</td>
                              <td className="watchers-reconcile-cell">{row.requiresReconciliation ? "Yes" : "No"}</td>
                              <td>{row.lastEventAt ?? "Not yet recorded"}</td>
                              <td>{row.updatedAt || "Unavailable"}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <TablePagination
                    label="Registered watchers"
                    page={watchersTablePage}
                    totalItems={watchersRows.length}
                    onPageChange={setWatchersTablePage}
                  />
                </article>
              ) : null}

              {selectedWatcher ? (
                <ModalShell
                  titleId="watcher-dialog-title"
                  title={`Watcher ${selectedWatcherLabel}`}
                  eyebrow="Watcher status"
                  onClose={closeWatcherStatusModal}
                >
                  <div className="watcher-status-modal">
                    <LlmFeedbackMessages className="llm-modal-feedback" message={watchersMessage} error={watchersError} />
                    <dl className="watcher-detail-grid" aria-label="Watcher details">
                      <div>
                        <dt>Project ID</dt>
                        <dd>{selectedWatcher.projectId}</dd>
                      </div>
                      <div>
                        <dt>State</dt>
                        <dd>{formatStatusValue(selectedWatcher.state)}</dd>
                      </div>
                      <div>
                        <dt>Dirty files</dt>
                        <dd>{selectedWatcher.dirtyCount}</dd>
                      </div>
                      <div>
                        <dt>Requires reconciliation</dt>
                        <dd>{selectedWatcher.requiresReconciliation ? "Yes" : "No"}</dd>
                      </div>
                      <div>
                        <dt>Last event</dt>
                        <dd>{selectedWatcher.lastEventAt ?? "Not yet recorded"}</dd>
                      </div>
                      <div>
                        <dt>Updated</dt>
                        <dd>{selectedWatcher.updatedAt || "Unavailable"}</dd>
                      </div>
                    </dl>
                    <section className="watcher-command-panel" aria-labelledby="watcher-command-title">
                      <h3 id="watcher-command-title">Commands</h3>
                      <div className="inline-actions">
                        {selectedWatcher.state === "enabled" || selectedWatcher.state === "unknown" ? (
                          <ActionButton
                            variant="secondary"
                            aria-label={`Pause watcher ${selectedWatcher.projectId}`}
                            disabled={watchersPendingAction !== null}
                            onClick={() => void onWatcherAction(selectedWatcher.projectId, "pause")}
                          >
                            Pause
                          </ActionButton>
                        ) : null}
                        {selectedWatcher.state === "paused" || selectedWatcher.state === "disabled" || selectedWatcher.state === "unknown" ? (
                          <ActionButton
                            variant="secondary"
                            aria-label={`Resume watcher ${selectedWatcher.projectId}`}
                            disabled={watchersPendingAction !== null}
                            onClick={() => void onWatcherAction(selectedWatcher.projectId, "resume")}
                          >
                            Resume
                          </ActionButton>
                        ) : null}
                        {selectedWatcher.state !== "disabled" ? (
                          <ActionButton
                            variant="danger"
                            aria-label={`Disable watcher ${selectedWatcher.projectId}`}
                            disabled={watchersPendingAction !== null}
                            onClick={() => void onWatcherAction(selectedWatcher.projectId, "disable")}
                          >
                            Disable
                          </ActionButton>
                        ) : null}
                        <ActionButton
                          variant="secondary"
                          aria-label={`Refresh watcher ${selectedWatcher.projectId}`}
                          disabled={watchersPendingAction !== null}
                          onClick={() => void refreshWatchersState()}
                        >
                          Refresh
                        </ActionButton>
                      </div>
                    </section>
                  </div>
                </ModalShell>
              ) : null}
            </section>
          ) : null}

          {currentPath === "/sync" ? (
            <section className="admin-section status-workbench" aria-label="Sync dashboard">
              <article className="status-hero-panel">
                <div className="status-hero-panel__copy">
                  <p className="database-kicker">Live sync queue</p>
                  <h2>Project sync operations</h2>
                  <p>
                    Review registered projects, spot queued files, and run targeted sync maintenance without copying project IDs.
                  </p>
                </div>
                <div className="status-toolbar">
                  <ActionButton variant="secondary" onClick={() => void refreshSyncState()}>
                    Refresh
                  </ActionButton>
                </div>
              </article>

              <dl className="status-stats-grid" aria-label="Sync summary">
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

              <div className="status-status-stack">
                {syncLoading ? <p role="status">Loading sync queue status…</p> : null}
                {!selectedSyncRow && syncMessage ? <p role="status">{syncMessage}</p> : null}
                {!selectedSyncRow && syncError ? <p role="alert">{syncError}</p> : null}
              </div>

              {!syncLoading && syncRows.length === 0 ? (
                <article className="status-empty-state">
                  <h2>Queue is empty</h2>
                  <p>
                    No sync queue entries yet. <a href="/projects">Create a project</a> to enqueue files for sync.
                  </p>
                </article>
              ) : null}

              {syncRows.length > 0 ? (
                <article className="admin-card" aria-labelledby="sync-table-title">
                  <div className="inline-actions projects-table-actions">
                    <h2 id="sync-table-title">Registered sync projects</h2>
                    <button type="button" className="secondary-button" onClick={() => void refreshSyncState()}>
                      Reload
                    </button>
                  </div>
                  <p id="sync-row-action-hint" className="visually-hidden">
                    Opens the sync status dialog. Press Enter or Space to activate.
                  </p>
                  <div className="llm-table-wrap">
                    <table className="llm-configuration-table sync-configuration-table" aria-label="Registered sync projects">
                      <thead>
                        <tr>
                          <th scope="col">Project</th>
                          <th scope="col">Status</th>
                          <th scope="col">Dirty</th>
                          <th scope="col">Reconcile</th>
                          <th scope="col">Sync state</th>
                          <th scope="col">Updated</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paginatedSyncRows.map((row) => {
                          const statusTone = syncStatusTone(row);
                          const statusLabel = syncStatusLabel(row);
                          const projectLabel = row.projectName ?? row.projectId;
                          return (
                            <tr
                              key={row.projectId}
                              className={`sync-table-row sync-table-row--${statusTone}`}
                              aria-describedby="sync-row-action-hint"
                              aria-haspopup="dialog"
                              aria-keyshortcuts="Enter Space"
                              aria-label={`Open sync ${projectLabel} status ${statusLabel} dirty files ${row.dirtyCount}`}
                              tabIndex={0}
                              onClick={() => openSyncStatusModal(row)}
                              onKeyDown={(event) =>
                                onConfigurationTableRowKeyDown(event, () => openSyncStatusModal(row))
                              }
                            >
                              <th scope="row">
                                <span className="watchers-project-cell">
                                  <strong>{projectLabel}</strong>
                                  <span>{row.projectId}</span>
                                </span>
                              </th>
                              <td>
                                <StatusBadge tone={statusTone}>{statusLabel}</StatusBadge>
                              </td>
                              <td className="watchers-number-cell">{row.dirtyCount}</td>
                              <td className="watchers-reconcile-cell">{row.requiresReconciliation ? "Yes" : "No"}</td>
                              <td>{formatStatusValue(row.syncState)}</td>
                              <td>{row.updatedAt ?? "Not reported"}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <TablePagination
                    label="Registered sync projects"
                    page={syncTablePage}
                    totalItems={syncRows.length}
                    onPageChange={setSyncTablePage}
                  />
                </article>
              ) : null}

              {selectedSyncRow ? (
                <ModalShell
                  titleId="sync-dialog-title"
                  title={`Sync ${selectedSyncLabel}`}
                  eyebrow="Sync status"
                  onClose={closeSyncStatusModal}
                >
                  <div className="sync-status-modal">
                    <LlmFeedbackMessages className="llm-modal-feedback" message={syncMessage} error={syncError} />
                    <dl className="sync-detail-grid" aria-label="Sync details">
                      <div>
                        <dt>Project ID</dt>
                        <dd>{selectedSyncRow.projectId}</dd>
                      </div>
                      <div>
                        <dt>Dirty files</dt>
                        <dd>{selectedSyncRow.dirtyCount}</dd>
                      </div>
                      <div>
                        <dt>Sync state</dt>
                        <dd>{formatStatusValue(selectedSyncRow.syncState)}</dd>
                      </div>
                      <div>
                        <dt>Reconcile state</dt>
                        <dd>{formatStatusValue(selectedSyncRow.reconcileState)}</dd>
                      </div>
                      <div>
                        <dt>Rebuild state</dt>
                        <dd>{formatStatusValue(selectedSyncRow.rebuildState)}</dd>
                      </div>
                      <div>
                        <dt>Requires reconciliation</dt>
                        <dd>{selectedSyncRow.requiresReconciliation ? "Yes" : "No"}</dd>
                      </div>
                      <div>
                        <dt>Updated</dt>
                        <dd>{selectedSyncRow.updatedAt ?? "Not reported"}</dd>
                      </div>
                      <div>
                        <dt>Lifecycle telemetry</dt>
                        <dd>{selectedSyncRow.lifecycleTelemetry}</dd>
                      </div>
                    </dl>
                    <section className="sync-activity-panel" aria-labelledby="sync-activity-title">
                      <div className="sync-activity-panel__header">
                        <h3 id="sync-activity-title">Recent sync activity</h3>
                        <span>{syncLogs.length} entries</span>
                      </div>
                      {syncLogsLoading ? <p role="status">Loading sync activity...</p> : null}
                      {syncLogsError ? <p role="alert">{syncLogsError}</p> : null}
                      {!syncLogsLoading && !syncLogsError && syncLogs.length === 0 ? (
                        <p>No sync activity recorded for this project yet.</p>
                      ) : null}
                      {syncLogs.length > 0 ? (
                        <div className="sync-activity-table-wrap">
                          <table className="sync-activity-table" aria-label="Recent sync activity entries">
                            <thead>
                              <tr>
                                <th scope="col">Status</th>
                                <th scope="col">Path</th>
                                <th scope="col">Event</th>
                                <th scope="col">Reason</th>
                                <th scope="col">Updated</th>
                              </tr>
                            </thead>
                            <tbody>
                              {syncLogs.map((entry) => {
                                const updatedAt = (entry.processedAt ?? entry.updatedAt) || entry.enqueuedAt;
                                const tone: StatusTone = entry.status === "processed" ? "success" : "info";
                                return (
                                  <tr key={`${entry.id}:${entry.path}:${entry.eventType}`}>
                                    <td>
                                      <StatusBadge tone={tone}>{formatStatusValue(entry.status)}</StatusBadge>
                                    </td>
                                    <td>{entry.path || "."}</td>
                                    <td>{formatLogValue(entry.eventType)}</td>
                                    <td>{formatLogValue(entry.reason)}</td>
                                    <td>{updatedAt || "Not reported"}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      ) : null}
                    </section>
                    <section className="sync-command-panel" aria-labelledby="sync-command-title">
                      <h3 id="sync-command-title">Commands</h3>
                      <div className="inline-actions">
                        <ActionButton
                          variant="secondary"
                          aria-label={`Sync now ${selectedSyncRow.projectId}`}
                          disabled={syncPendingAction !== null}
                          onClick={() => void onSyncAction(selectedSyncRow.projectId, "now")}
                        >
                          Sync now
                        </ActionButton>
                        <ActionButton
                          variant="secondary"
                          aria-label={`Reconcile ${selectedSyncRow.projectId}`}
                          disabled={syncPendingAction !== null}
                          onClick={() => void onSyncAction(selectedSyncRow.projectId, "reconcile")}
                        >
                          Reconcile
                        </ActionButton>
                        <ActionButton
                          variant="danger"
                          aria-label={`Rebuild ${selectedSyncRow.projectId}`}
                          disabled={syncPendingAction !== null}
                          onClick={() => void onSyncAction(selectedSyncRow.projectId, "rebuild")}
                        >
                          Rebuild
                        </ActionButton>
                        <ActionButton
                          variant="secondary"
                          aria-label={`Refresh sync ${selectedSyncRow.projectId}`}
                          disabled={syncPendingAction !== null}
                          onClick={() => void refreshSyncDetails(selectedSyncRow.projectId)}
                        >
                          Refresh
                        </ActionButton>
                      </div>
                    </section>
                  </div>
                </ModalShell>
              ) : null}
            </section>
          ) : null}

          {currentPath === "/workflows" ? (
            <section className="admin-section llm-workbench workflows-workbench" aria-label="Workflows management">
              <section className="llm-status-panel workflow-registry-panel" aria-label="Workflow registry summary">
                <div className="workflow-registry-panel__header">
                  <div className="workflow-registry-panel__copy">
                    <p className="database-kicker">workflow registry</p>
                    <h2>Sources first, workflows inside each directory</h2>
                  </div>
                  <div className="workflow-registry-panel__actions" aria-label="Workflow registry actions">
                    <button type="button" onClick={openNewWorkflowSourceModal}>
                      New
                    </button>
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
                </div>
                <dl className="status-stats-grid workflow-stats-grid">
                  <div>
                    <dt>Sources</dt>
                    <dd>{workflowSources.length}</dd>
                  </div>
                  <div>
                    <dt>Workflows</dt>
                    <dd>{workflowsList.length}</dd>
                  </div>
                  <div>
                    <dt>Loaded</dt>
                    <dd>{loadedWorkflowSourceCount}</dd>
                  </div>
                  <div>
                    <dt>Failed</dt>
                    <dd>{failedWorkflowSourceCount}</dd>
                  </div>
                </dl>
              </section>

              <div className="llm-status-stack">
                {workflowsLoading ? <p role="status">Loading workflows dashboard...</p> : null}
                <LlmFeedbackMessages
                  message={workflowModalOpen ? "" : workflowsMessage}
                  error={workflowModalOpen ? "" : workflowsError}
                />
              </div>

              <div className="llm-workbench__grid llm-workbench__grid--single">
                <article className="admin-card workflow-sources-card" aria-labelledby="workflow-sources-table-title">
                  <div className="workflow-sources-card__header">
                    <div>
                      <h2 id="workflow-sources-table-title">Workflow sources</h2>
                      <p>
                        {workflowSources.length} source{workflowSources.length === 1 ? "" : "s"} tracking {workflowsList.length} workflow
                        {workflowsList.length === 1 ? "" : "s"}.
                      </p>
                    </div>
                  </div>
                  <p id="workflow-row-action-hint" className="visually-hidden">
                    Opens the workflow details dialog. Press Enter or Space to activate.
                  </p>
                  {workflowSources.length === 0 ? (
                    <p>
                      No workflow sources configured.
                      {workflowProjectsCount === 0 ? (
                        <>
                          {" "}
                          <a href="/projects">Create a project first</a>. Once a project exists, use New to add a workflow source path.
                        </>
                      ) : (
                        " Use New to add a workflow source path."
                      )}
                    </p>
                  ) : null}
                  <div className="llm-table-wrap">
                    <table className="llm-configuration-table workflow-sources-configuration-table" aria-label="Workflow sources">
                      <thead>
                        <tr>
                          <th scope="col">Source</th>
                          <th scope="col">Registry state</th>
                          <th scope="col">Last loaded</th>
                          <th scope="col">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paginatedWorkflowSources.map((source) => {
                          const sourceWorkflows = workflowsBySourceId.get(source.sourceId) ?? [];
                          const sourceStatus = workflowSourceStatus(source, sourceWorkflows.length);
                          const sourceStatusLabel = formatStatusValue(sourceStatus);
                          const sourceExpanded = expandedWorkflowSourceIds.has(source.sourceId);
                          return (
                            <Fragment key={source.sourceId}>
                              <tr
                                className="workflow-source-row"
                                aria-label={`Expand workflow source ${source.sourcePath} status ${sourceStatusLabel} workflows ${sourceWorkflows.length}`}
                              >
                                 <th scope="row">
                                   <button
                                     type="button"
                                     className="workflow-source-toggle"
                                     aria-expanded={sourceExpanded}
                                     aria-controls={`workflow-source-${source.sourceId}-workflows`}
                                     aria-label={`${sourceExpanded ? "Collapse" : "Expand"} workflow source ${source.sourcePath}`}
                                     onClick={() => toggleWorkflowSource(source.sourceId)}
                                   >
                                     <span aria-hidden="true">{sourceExpanded ? "v" : ">"}</span>
                                     <span className="workflow-source-toggle__text">
                                       <span className="workflow-source-toggle__path">{source.sourcePath}</span>
                                       {source.isSystem ? (
                                         <span className="workflow-source-system-badge">System</span>
                                       ) : null}
                                     </span>
                                   </button>
                                 </th>
                                 <td>
                                   <div className="workflow-source-state">
                                     <StatusBadge tone={statusToneForValue(sourceStatus)}>{sourceStatusLabel}</StatusBadge>
                                     <span>
                                       {sourceWorkflows.length} workflow{sourceWorkflows.length === 1 ? "" : "s"}
                                     </span>
                                   </div>
                                 </td>
                                 <td className="workflow-source-last-loaded">{source.lastLoadedAt ?? "Never"}</td>
                                 <td className="workflow-source-actions-cell">
                                   {source.isSystem ? null : (
                                     <button
                                       type="button"
                                       className="danger-button workflow-source-delete-button"
                                       disabled={workflowActionPendingId !== null}
                                       onClick={() => void onDeleteWorkflowSource(source.sourceId)}
                                     >
                                       Delete
                                     </button>
                                   )}
                                 </td>
                              </tr>
                              {sourceExpanded ? (
                                <tr className="workflow-source-detail-row">
                                  <td colSpan={4}>
                                    <div id={`workflow-source-${source.sourceId}-workflows`} className="workflow-source-detail-panel">
                                      {source.errorMessage ? (
                                        <p role="alert" className="workflow-source-error">
                                          {source.errorMessage}
                                        </p>
                                      ) : null}
                                      {sourceWorkflows.length === 0 ? (
                                        <p>No workflows loaded from this source.</p>
                                      ) : (
                                        <div className="sync-activity-table-wrap workflow-source-workflows-wrap">
                                          <table
                                            className="sync-activity-table workflow-source-workflows-table"
                                            aria-label={`Workflows loaded from ${source.sourcePath}`}
                                          >
                                            <thead>
                                              <tr>
                                                <th scope="col">Workflow</th>
                                                <th scope="col">Version</th>
                                                <th scope="col">Tags</th>
                                              </tr>
                                            </thead>
                                            <tbody>
                                              {sourceWorkflows.map((workflow) => {
                                                const tagsLabel = workflow.tags.length > 0 ? workflow.tags.join(", ") : "None";
                                                return (
                                                  <tr
                                                    key={workflow.name}
                                                    aria-describedby="workflow-row-action-hint"
                                                    aria-haspopup="dialog"
                                                    aria-keyshortcuts="Enter Space"
                                                    aria-label={`Open workflow ${workflow.name} details ${workflow.version || "Not set"} ${tagsLabel}`}
                                                    tabIndex={0}
                                                    onClick={() => void onViewWorkflowDetails(workflow.name)}
                                                    onKeyDown={(event) =>
                                                      onConfigurationTableRowKeyDown(event, () => void onViewWorkflowDetails(workflow.name))
                                                    }
                                                  >
                                                    <th scope="row">{workflow.name}</th>
                                                    <td>{workflow.version || "Not set"}</td>
                                                    <td>{tagsLabel}</td>
                                                  </tr>
                                                );
                                              })}
                                            </tbody>
                                          </table>
                                        </div>
                                      )}
                                    </div>
                                  </td>
                                </tr>
                              ) : null}
                            </Fragment>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <TablePagination
                    label="Workflow sources"
                    page={workflowSourcesTablePage}
                    totalItems={workflowSources.length}
                    onPageChange={setWorkflowSourcesTablePage}
                  />
                </article>
              </div>

              {workflowSourceModalOpen ? (
                <ModalShell
                  titleId="workflow-source-dialog-title"
                  title="New workflow source"
                  eyebrow="Workflow directory"
                  className="workflow-source-modal"
                  onClose={closeWorkflowSourceModal}
                >
                  <form className="admin-form workflow-source-form" onSubmit={(event) => void onCreateWorkflowSource(event)}>
                    <LlmFeedbackMessages className="llm-modal-feedback" message={workflowsMessage} error={workflowsError} />
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
                    <div className="field-group">
                      <label htmlFor="workflow-source-path">Source path</label>
                      <div className="project-path-control">
                        <input
                          id="workflow-source-path"
                          ref={workflowSourcePathRef}
                          value={workflowSourcePath}
                          onChange={(event) => setWorkflowSourcePath(event.target.value)}
                          placeholder="/workspace/workflows"
                          required
                        />
                        <button
                          type="button"
                          className="secondary-button icon-button"
                          aria-label="Browse workflow source path"
                          title="Browse workflow source path"
                          onClick={() => setWorkflowSourcePathPickerOpen(true)}
                        >
                          <FolderIcon />
                        </button>
                      </div>
                    </div>
                    <div className="field-group">
                      <label htmlFor="workflow-source-checksum">Checksum (optional)</label>
                      <input
                        id="workflow-source-checksum"
                        value={workflowSourceChecksum}
                        onChange={(event) => setWorkflowSourceChecksum(event.target.value)}
                        placeholder="sha256:..."
                      />
                    </div>
                    {workflowProjectsCount === 0 ? (
                      <p>Once a project exists, use New to add a workflow source path.</p>
                    ) : null}
                    {workflowSourcePathPickerOpen ? (
                      <ServerPathPicker
                        title="Browse workflow source folders"
                        selectionMode="folder"
                        startPath={workflowSourcePath.trim() || undefined}
                        listEntries={listServerPathEntries}
                        onSelect={onSelectWorkflowSourcePath}
                        onCancel={closeWorkflowSourcePathPicker}
                      />
                    ) : null}
                    {!workflowSourcePathPickerOpen ? (
                      <div className="inline-actions workflow-source-form__actions">
                        <button
                          type="submit"
                          disabled={
                            workflowSourcePending ||
                            workflowProjectsCount === 0 ||
                            workflowSourceProjectId.trim().length === 0
                          }
                        >
                          Add workflow source
                        </button>
                        <button type="button" className="secondary-button" onClick={closeWorkflowSourceModal}>
                          Cancel
                        </button>
                      </div>
                    ) : null}
                  </form>
                </ModalShell>
              ) : null}

              {selectedWorkflowName ? (
                <ModalShell
                  titleId="workflow-detail-dialog-title"
                  title={`Workflow ${selectedWorkflowName}`}
                  eyebrow="Workflow definition"
                  className="workflow-detail-modal"
                  onClose={closeWorkflowDetailModal}
                >
                  <div className="workflow-detail-body">
                    <LlmFeedbackMessages className="llm-modal-feedback" message="" error={workflowsError} />
                    {workflowDetailLoading ? <p role="status">Loading workflow details...</p> : null}
                    {selectedWorkflowDetail ? (
                      <>
                        <dl className="sync-detail-grid workflow-detail-grid" aria-label="Workflow details">
                          <div>
                            <dt>Version</dt>
                            <dd>{selectedWorkflowDetail.version || "Not set"}</dd>
                          </div>
                          <div>
                            <dt>YAML path</dt>
                            <dd>{selectedWorkflowDetail.yamlPath ?? selectedWorkflowDetail.sourcePath ?? "Unknown"}</dd>
                          </div>
                        </dl>
                        <section className="workflow-detail-section workflow-yaml-section" aria-label="Workflow YAML">
                          <h3>YAML</h3>
                          <YamlCodeBlock yaml={selectedWorkflowDetail.rawYaml ?? "Raw YAML is not available from the current API response."} />
                        </section>
                        <section className="workflow-detail-section" aria-label="Workflow load logs">
                          <h3>Load logs</h3>
                          <pre className="workflow-code-block" tabIndex={0}>{selectedWorkflowLogs.join("\n")}</pre>
                        </section>
                      </>
                    ) : null}
                  </div>
                </ModalShell>
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
            <section className="admin-section runs-console" aria-label="Workflow execution runs">
              <PageHeader
                eyebrow="Workflow runs"
                title="Execution recorder"
                description="Inspect workflow executions from MCP clients and local admin actions, including sync failures, async jobs, pause/resume state, and block-level diagnostics."
              />

              <div className="runs-metrics" aria-label="Run summary">
                <div className="runs-metric" data-tone="danger">
                  <span>Failed</span>
                  <strong>{runsFailureCount}</strong>
                </div>
                <div className="runs-metric" data-tone="warning">
                  <span>Active</span>
                  <strong>{runsActiveCount}</strong>
                </div>
                <div className="runs-metric" data-tone="success">
                  <span>Completed</span>
                  <strong>{runsCompletedCount}</strong>
                </div>
                <div className="runs-metric" data-tone="info">
                  <span>Total match</span>
                  <strong>{runsTotal}</strong>
                </div>
              </div>

              <article className="admin-card runs-filter-card" aria-labelledby="runs-filter-title">
                <h2 id="runs-filter-title">Filters</h2>
                <form className="admin-form llm-form runs-filter-form" onSubmit={(event) => void onApplyRunsFilters(event)}>
                  <div className="runs-filter-grid">
                    <label htmlFor="runs-status-filter">Status filter</label>
                    <select
                      id="runs-status-filter"
                      value={runFilterStatus}
                      onChange={(event) => setRunFilterStatus(event.target.value)}
                    >
                      <option value="">Any</option>
                      <option value="failed">Failed</option>
                      <option value="running">Running</option>
                      <option value="paused">Paused</option>
                      <option value="completed">Completed</option>
                    </select>
                    <label htmlFor="runs-mode-filter">Mode</label>
                    <select id="runs-mode-filter" value={runFilterMode} onChange={(event) => setRunFilterMode(event.target.value)}>
                      <option value="">Any</option>
                      <option value="sync">Sync</option>
                      <option value="async">Async</option>
                      <option value="inline">Inline</option>
                    </select>
                    <label htmlFor="runs-workflow-filter">Workflow</label>
                    <input
                      id="runs-workflow-filter"
                      value={runFilterWorkflow}
                      onChange={(event) => setRunFilterWorkflow(event.target.value)}
                      placeholder="python-ci-pipeline"
                    />
                    <label htmlFor="runs-project-filter">Project ID</label>
                    <input
                      id="runs-project-filter"
                      value={runFilterProjectId}
                      onChange={(event) => setRunFilterProjectId(event.target.value)}
                      placeholder="p1"
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
                  </div>
                  <div className="inline-actions">
                    <button type="submit" disabled={runsLoading || runActionPendingId !== null}>
                      Apply
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => {
                        setRunFilterStatus("");
                        setRunFilterMode("");
                        setRunFilterWorkflow("");
                        setRunFilterProjectId("");
                        setRunFilterOffset("0");
                      }}
                    >
                      Reset
                    </button>
                  </div>
                </form>
              </article>

              <article className="admin-card" aria-labelledby="runs-table-title">
                <div className="inline-actions projects-table-actions">
                  <h2 id="runs-table-title">Execution runs</h2>
                  <button type="button" className="secondary-button" onClick={() => void loadRuns()}>
                    Reload
                  </button>
                </div>

                {runsLoading ? <p role="status" className="runs-state-line">Loading runs...</p> : null}
                {runsMessage ? <p role="status" className="runs-state-line">{runsMessage}</p> : null}
                {runsError ? <p role="alert" className="runs-state-line runs-state-line--error">{runsError}</p> : null}

                <p id="run-row-action-hint" className="visually-hidden">
                  Opens the run detail dialog. Press Enter or Space to activate.
                </p>
                {!runsLoading && runsRows.length === 0 ? (
                  <div className="runs-empty">
                    <h2>No runs match this view</h2>
                    <p>Executed registered workflows will appear here after the MCP server records them in SQLite.</p>
                  </div>
                ) : null}

                {runsRows.length > 0 ? (
                  <div className="llm-table-wrap">
                    <table className="llm-configuration-table projects-configuration-table runs-configuration-table" aria-label="Workflow execution runs">
                      <thead>
                        <tr>
                          <th scope="col">Workflow</th>
                          <th scope="col">Run ID</th>
                          <th scope="col">Status</th>
                          <th scope="col">Mode</th>
                          <th scope="col">Started</th>
                          <th scope="col">Duration</th>
                          <th scope="col">Project</th>
                        </tr>
                      </thead>
                      <tbody>
                        {runsRows.map((run) => (
                          <tr
                            key={run.runId}
                            aria-describedby="run-row-action-hint"
                            aria-haspopup="dialog"
                            aria-keyshortcuts="Enter Space"
                            aria-label={`Open run ${run.workflowName || run.runId} detail`}
                            tabIndex={0}
                            onClick={() => void onViewRunDetail(run.runId)}
                            onKeyDown={(event) =>
                              onConfigurationTableRowKeyDown(event, () => void onViewRunDetail(run.runId))
                            }
                          >
                            <th scope="row">{run.workflowName || "Unknown workflow"}</th>
                            <td>{run.runId || "Unavailable"}</td>
                            <td>
                              <StatusBadge tone={statusToneForValue(run.status)}>{formatStatusValue(run.status)}</StatusBadge>
                            </td>
                            <td>{formatStatusValue(run.executionMode)}</td>
                            <td>{formatRunTimestamp(run.startedAt ?? run.createdAt)}</td>
                            <td>{formatDurationMs(run.durationMs)}</td>
                            <td>{run.projectId ?? "Unbound"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : null}
              </article>

              {runDetail ? (
                <ModalShell
                  titleId="run-detail-dialog-title"
                  title={`Run ${runDetail.workflowName || runDetail.runId}`}
                  eyebrow="Run detail"
                  className="run-detail-modal"
                  onClose={() => setRunDetail(null)}
                >
                  <div className="runs-detail-body">
                    <div className="runs-inspector__header">
                      <p>Run detail</p>
                      <h3>{runDetail.workflowName || "Unknown workflow"}</h3>
                      <StatusBadge tone={statusToneForValue(runDetail.status)}>{formatStatusValue(runDetail.status)}</StatusBadge>
                    </div>
                    <dl className="runs-facts">
                      <div><dt>Run ID</dt><dd>{runDetail.runId}</dd></div>
                      <div><dt>Mode</dt><dd>{formatStatusValue(runDetail.executionMode)}</dd></div>
                      <div><dt>Started</dt><dd>{formatRunTimestamp(runDetail.startedAt)}</dd></div>
                      <div><dt>Finished</dt><dd>{formatRunTimestamp(runDetail.finishedAt)}</dd></div>
                      <div><dt>Duration</dt><dd>{formatDurationMs(runDetail.durationMs)}</dd></div>
                      <div><dt>Token</dt><dd>{runDetail.tokenId ?? "Unbound"}</dd></div>
                    </dl>
                    {(runDetail.cancellable || runCanResume(runDetail)) ? (
                      <div className="inline-actions">
                        {runDetail.cancellable ? (
                          <button type="button" disabled={runActionPendingId !== null} onClick={() => void onCancelRun(runDetail.runId)}>
                            Cancel run
                          </button>
                        ) : null}
                        {runCanResume(runDetail) ? (
                          <button
                            type="button"
                            className="secondary-button"
                            disabled={runActionPendingId !== null}
                            onClick={() => void onResumeRun(runDetail.runId)}
                          >
                            Resume run
                          </button>
                        ) : null}
                      </div>
                    ) : null}
                    {selectedRunSummary ? (
                      <section className="runs-diagnostic">
                        <h3>Diagnostic</h3>
                        <JsonCodeBlock value={selectedRunSummary} parseString />
                      </section>
                    ) : null}
                    <section className="runs-diagnostic">
                      <h3>Inputs</h3>
                      <JsonCodeBlock value={runDetail.inputs} />
                    </section>
                    <section className="runs-timeline" aria-label="Block execution timeline">
                      <h3>Blocks</h3>
                      {runDetail.blocks.length > 0 ? (
                        <ol>
                          {runDetail.blocks.map((block) => (
                            <li key={block.blockId}>
                              <div>
                                <strong>{block.blockId || "Unnamed block"}</strong>
                                <span>{block.blockType ?? "Unknown type"} · {formatDurationMs(block.durationMs)}</span>
                              </div>
                              <StatusBadge tone={statusToneForValue(block.status ?? block.outcome)}>
                                {formatStatusValue(block.status ?? block.outcome ?? "unknown")}
                              </StatusBadge>
                              {block.message ? <p>{block.message}</p> : null}
                            </li>
                          ))}
                        </ol>
                      ) : (
                        <p>No block diagnostics were recorded for this run.</p>
                      )}
                    </section>
                    <details className="runs-technical-json">
                      <summary>Technical JSON</summary>
                      <JsonCodeBlock value={runDetail.technicalJson} parseString />
                    </details>
                  </div>
                </ModalShell>
              ) : null}

              {runResumeTarget ? (
                <ModalShell
                  titleId="run-resume-dialog-title"
                  title={`Resume ${runResumeTarget.runId}`}
                  eyebrow="Paused workflow"
                  className="run-resume-modal"
                  onClose={() => setRunResumeTarget(null)}
                >
                  <form className="admin-form" onSubmit={(event) => void onSubmitRunResume(event)}>
                    <label htmlFor="run-resume-response">Response</label>
                    <textarea
                      id="run-resume-response"
                      value={runResumeResponse}
                      onChange={(event) => setRunResumeResponse(event.target.value)}
                      rows={5}
                      placeholder="approval, rejection, or other prompt response"
                    />
                    <div className="inline-actions">
                      <button type="submit" disabled={runActionPendingId !== null}>
                        Submit resume
                      </button>
                      <button type="button" className="secondary-button" onClick={() => setRunResumeTarget(null)}>
                        Cancel
                      </button>
                    </div>
                  </form>
                </ModalShell>
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
