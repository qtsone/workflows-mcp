import { afterEach, describe, expect, it, vi } from "vitest";

import {
  EventStreamUrlPolicyError,
  createEventStream,
  createPollingFallback,
  fetchSyncState,
  fetchWatcherState,
} from "./events";

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;
  readonly withCredentials: boolean;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  private readonly listeners = new Map<string, Set<(event: MessageEvent<string>) => void>>();
  close = vi.fn();

  constructor(url: string, init?: EventSourceInit) {
    this.url = url;
    this.withCredentials = init?.withCredentials ?? false;
    FakeEventSource.instances.push(this);
  }

  addEventListener(eventName: string, listener: (event: MessageEvent<string>) => void): void {
    const existing = this.listeners.get(eventName) ?? new Set<(event: MessageEvent<string>) => void>();
    existing.add(listener);
    this.listeners.set(eventName, existing);
  }

  emitNamed(eventName: string, data: string): void {
    const handlers = this.listeners.get(eventName);
    if (!handlers) {
      return;
    }

    for (const handler of handlers) {
      handler({ data } as MessageEvent<string>);
    }
  }
}

describe("createEventStream", () => {
  afterEach(() => {
    FakeEventSource.instances = [];
  });

  it("creates EventSource with same-origin URL and withCredentials true", () => {
    const onMessage = vi.fn<(payload: { state: string }) => void>();
    const onError = vi.fn<(error: Error) => void>();

    const stream = createEventStream<{ state: string }>({
      url: "/api/events/v1/workflows",
      baseUrl: "https://admin.example.test",
      eventSourceFactory: (url, init) => new FakeEventSource(url, init) as unknown as EventSource,
      onMessage,
      onError,
    });

    expect(FakeEventSource.instances).toHaveLength(1);
    const [instance] = FakeEventSource.instances;
    expect(instance.url).toBe("https://admin.example.test/api/events/v1/workflows");
    expect(instance.withCredentials).toBe(true);

    stream.stop();
    expect(instance.close).toHaveBeenCalledTimes(1);
  });

  it("rejects protocol-relative and cross-origin URLs before EventSource construction", () => {
    const onMessage = vi.fn<(payload: { state: string }) => void>();
    const onError = vi.fn<(error: Error) => void>();

    expect(() =>
      createEventStream<{ state: string }>({
        url: "//evil.example.test/stream",
        baseUrl: "https://admin.example.test",
        eventSourceFactory: (url, init) => new FakeEventSource(url, init) as unknown as EventSource,
        onMessage,
        onError,
      }),
    ).toThrowError(EventStreamUrlPolicyError);

    expect(() =>
      createEventStream<{ state: string }>({
        url: "https://evil.example.test/stream",
        baseUrl: "https://admin.example.test",
        eventSourceFactory: (url, init) => new FakeEventSource(url, init) as unknown as EventSource,
        onMessage,
        onError,
      }),
    ).toThrowError(EventStreamUrlPolicyError);

    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("parses JSON messages to typed payload and routes parse failures to error callback", () => {
    const onMessage = vi.fn<(payload: { state: string }) => void>();
    const onError = vi.fn<(error: Error) => void>();

    createEventStream<{ state: string }>({
      url: "/api/events/v1/workflows",
      baseUrl: "https://admin.example.test",
      eventSourceFactory: (url, init) => new FakeEventSource(url, init) as unknown as EventSource,
      onMessage,
      onError,
    });

    const [instance] = FakeEventSource.instances;
    instance.onmessage?.({ data: "{\"state\":\"ok\"}" } as MessageEvent<string>);

    expect(onMessage).toHaveBeenCalledTimes(1);
    expect(onMessage).toHaveBeenCalledWith({ state: "ok" });

    instance.onmessage?.({ data: "not-json" } as MessageEvent<string>);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0]?.[0]).toBeInstanceOf(Error);
  });

  it("supports named SSE events and parses named payloads", () => {
    const onMessage = vi.fn<(payload: { state: string }) => void>();
    const onError = vi.fn<(error: Error) => void>();

    createEventStream<{ state: string }>({
      url: "/api/events/v1/workflows",
      baseUrl: "https://admin.example.test",
      eventSourceFactory: (url, init) => new FakeEventSource(url, init) as unknown as EventSource,
      eventName: "sync.status",
      onMessage,
      onError,
    });

    const [instance] = FakeEventSource.instances;
    instance.emitNamed("sync.status", '{"state":"queued"}');

    expect(onMessage).toHaveBeenCalledTimes(1);
    expect(onMessage).toHaveBeenCalledWith({ state: "queued" });
    expect(onError).not.toHaveBeenCalled();
  });

  it("routes named-event JSON parse failures to error callback", () => {
    const onMessage = vi.fn<(payload: { state: string }) => void>();
    const onError = vi.fn<(error: Error) => void>();

    createEventStream<{ state: string }>({
      url: "/api/events/v1/workflows",
      baseUrl: "https://admin.example.test",
      eventSourceFactory: (url, init) => new FakeEventSource(url, init) as unknown as EventSource,
      eventName: "watcher.status",
      onMessage,
      onError,
    });

    const [instance] = FakeEventSource.instances;
    instance.emitNamed("watcher.status", "not-json");

    expect(onMessage).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0]?.[0]).toBeInstanceOf(Error);
  });
});

describe("createPollingFallback", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("invokes fetcher on interval and stops cleanly", async () => {
    vi.useFakeTimers();
    const onData = vi.fn<(payload: { value: number }) => void>();
    const onError = vi.fn<(error: Error) => void>();

    let callCount = 0;
    const fetcher = vi.fn(async () => {
      callCount += 1;
      return { value: callCount };
    });

    const poller = createPollingFallback<{ value: number }>({
      intervalMs: 100,
      fetcher,
      onData,
      onError,
    });

    poller.start();
    await vi.advanceTimersByTimeAsync(350);

    expect(fetcher).toHaveBeenCalledTimes(4);
    expect(onData).toHaveBeenLastCalledWith({ value: 4 });

    poller.stop();
    await vi.advanceTimersByTimeAsync(500);
    expect(fetcher).toHaveBeenCalledTimes(4);
    expect(onError).not.toHaveBeenCalled();
  });
});

describe("state fetch helpers", () => {
  it("fetches watcher state payload from events endpoint", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ version: 1, items: [{ project_id: "p1", state: "enabled", dirty_count: 0, requires_reconciliation: false, last_event_at: null, updated_at: "2026-04-30T00:00:00Z" }] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const payload = await fetchWatcherState(fetchMock as unknown as typeof fetch);
    expect(payload.items[0]?.project_id).toBe("p1");
    expect(fetchMock).toHaveBeenCalledWith("/api/events/v1/watchers/state", { credentials: "include" });
  });

  it("fetches sync state payload from events endpoint", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ version: 1, items: [{ project_id: "p1", dirty_count: 3, requires_reconciliation: true }] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const payload = await fetchSyncState(fetchMock as unknown as typeof fetch);
    expect(payload.items[0]?.dirty_count).toBe(3);
    expect(fetchMock).toHaveBeenCalledWith("/api/events/v1/sync/state", { credentials: "include" });
  });
});
