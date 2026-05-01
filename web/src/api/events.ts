export class EventStreamUrlPolicyError extends Error {
  readonly name = "EventStreamUrlPolicyError";
  readonly inputUrl: string;

  constructor(message: string, inputUrl: string) {
    super(message);
    this.inputUrl = inputUrl;
  }
}

export class EventStreamParseError extends Error {
  readonly name = "EventStreamParseError";
  readonly rawData: string;

  constructor(message: string, rawData: string, cause?: unknown) {
    super(message, { cause });
    this.rawData = rawData;
  }
}

export interface CreateEventStreamOptions<TPayload> {
  url: string;
  baseUrl?: string;
  eventName?: string;
  onMessage: (payload: TPayload) => void;
  onError: (error: Error) => void;
  eventSourceFactory?: (url: string, init: EventSourceInit) => EventSource;
}

export interface EventStreamHandle {
  stop: () => void;
}

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

function resolveAndValidateEventUrl(baseUrl: string | undefined, inputUrl: string): string {
  if (isProtocolRelativeUrl(inputUrl)) {
    throw new EventStreamUrlPolicyError("Protocol-relative URLs are not allowed", inputUrl);
  }

  if (!baseUrl) {
    if (isRelativeUrl(inputUrl)) {
      return inputUrl;
    }

    const locationOrigin = getLocationOrigin();
    if (!locationOrigin) {
      throw new EventStreamUrlPolicyError(
        "Absolute URLs are not allowed without a configured baseUrl",
        inputUrl,
      );
    }

    const resolved = new URL(inputUrl);
    if (resolved.origin !== locationOrigin) {
      throw new EventStreamUrlPolicyError("Cross-origin URLs are not allowed", inputUrl);
    }
    return resolved.toString();
  }

  const baseOrigin = new URL(baseUrl).origin;
  const resolved = new URL(inputUrl, baseUrl);
  if (resolved.origin !== baseOrigin) {
    throw new EventStreamUrlPolicyError("Cross-origin URLs are not allowed", inputUrl);
  }
  return resolved.toString();
}

export function createEventStream<TPayload>(
  options: CreateEventStreamOptions<TPayload>,
): EventStreamHandle {
  const resolvedUrl = resolveAndValidateEventUrl(options.baseUrl, options.url);
  const factory =
    options.eventSourceFactory ??
    ((url: string, init: EventSourceInit): EventSource => new EventSource(url, init));

  const source = factory(resolvedUrl, { withCredentials: true });

  const onSseMessage = (event: MessageEvent<string>): void => {
    try {
      const payload = JSON.parse(event.data) as TPayload;
      options.onMessage(payload);
    } catch (error) {
      options.onError(new EventStreamParseError("Invalid SSE JSON payload", event.data, error));
    }
  };

  if (options.eventName) {
    source.addEventListener(options.eventName, onSseMessage as EventListener);
  } else {
    source.onmessage = onSseMessage;
  }

  source.onerror = () => {
    options.onError(new Error(`SSE stream error for ${resolvedUrl}`));
  };

  return {
    stop: () => {
      source.close();
    },
  };
}

export interface CreatePollingFallbackOptions<TState> {
  intervalMs: number;
  fetcher: () => Promise<TState>;
  onData: (state: TState) => void;
  onError: (error: Error) => void;
}

export interface PollingFallbackHandle {
  start: () => void;
  stop: () => void;
}

export interface WatcherStateItem {
  project_id: string;
  state: string;
  dirty_count: number;
  requires_reconciliation: boolean;
  last_event_at: string | null;
  updated_at: string;
}

export interface WatcherStatePayload {
  version: number;
  items: WatcherStateItem[];
}

export interface SyncStateItem {
  project_id: string;
  dirty_count: number;
  requires_reconciliation: boolean;
  sync_state?: string;
  reconcile_state?: string;
  rebuild_state?: string;
  updated_at?: string | null;
}

export interface SyncStatePayload {
  version: number;
  items: SyncStateItem[];
}

function normalizeError(error: unknown, fallback: string): Error {
  if (error instanceof Error) return error;
  return new Error(fallback);
}

async function parseJsonOrThrow<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) {
    throw new Error(`${fallback} (HTTP ${response.status})`);
  }
  return (await response.json()) as T;
}

export async function fetchWatcherState(fetchImpl: typeof fetch = fetch): Promise<WatcherStatePayload> {
  try {
    const response = await fetchImpl("/api/events/v1/watchers/state", { credentials: "include" });
    return await parseJsonOrThrow<WatcherStatePayload>(response, "Unable to load watcher status state");
  } catch (error) {
    throw normalizeError(error, "Unable to load watcher status state");
  }
}

export async function fetchSyncState(fetchImpl: typeof fetch = fetch): Promise<SyncStatePayload> {
  try {
    const response = await fetchImpl("/api/events/v1/sync/state", { credentials: "include" });
    return await parseJsonOrThrow<SyncStatePayload>(response, "Unable to load sync status state");
  } catch (error) {
    throw normalizeError(error, "Unable to load sync status state");
  }
}

export function createPollingFallback<TState>(
  options: CreatePollingFallbackOptions<TState>,
): PollingFallbackHandle {
  let intervalId: ReturnType<typeof setInterval> | undefined;
  let running = false;
  let generation = 0;

  const tick = async (runGeneration: number): Promise<void> => {
    try {
      const state = await options.fetcher();
      if (!running || runGeneration !== generation) {
        return;
      }
      options.onData(state);
    } catch (error) {
      if (!running || runGeneration !== generation) {
        return;
      }
      options.onError(error instanceof Error ? error : new Error("Polling fetcher failed"));
    }
  };

  return {
    start: () => {
      if (running) {
        return;
      }

      running = true;
      generation += 1;
      const runGeneration = generation;

      void tick(runGeneration);
      intervalId = setInterval(() => {
        void tick(runGeneration);
      }, options.intervalMs);
    },
    stop: () => {
      running = false;
      generation += 1;

      if (intervalId !== undefined) {
        clearInterval(intervalId);
        intervalId = undefined;
      }
    },
  };
}
