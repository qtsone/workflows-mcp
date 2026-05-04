import { describe, expect, it, vi } from "vitest";

import { ApiHttpError, ApiUrlPolicyError, createApiClient, createApiFetch } from "./client";

describe("createApiFetch", () => {
  it("rejects protocol-relative URLs before fetch", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 200 }));
    const apiFetch = createApiFetch({
      fetchImpl: fetchMock,
      baseUrl: "https://admin.example.test",
      csrfTokenProvider: () => "csrf-token-abc",
    });

    await expect(apiFetch("//evil.test/path", { method: "POST" })).rejects.toBeInstanceOf(
      ApiUrlPolicyError,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects absolute cross-origin URL when baseUrl origin differs", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 200 }));
    const apiFetch = createApiFetch({
      fetchImpl: fetchMock,
      baseUrl: "https://admin.example.test",
      csrfTokenProvider: () => "csrf-token-abc",
    });

    await expect(
      apiFetch("https://evil.example.test/private", { method: "POST" }),
    ).rejects.toBeInstanceOf(ApiUrlPolicyError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("allows normal relative paths", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 200 }));
    const apiFetch = createApiFetch({
      fetchImpl: fetchMock,
      baseUrl: "https://admin.example.test",
    });

    await apiFetch("/api/workflows", { method: "GET" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://admin.example.test/api/workflows");
  });

  it("allows absolute same-origin URLs when baseUrl is configured", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 200 }));
    const apiFetch = createApiFetch({
      fetchImpl: fetchMock,
      baseUrl: "https://admin.example.test",
    });

    await apiFetch("https://admin.example.test/api/workflows", { method: "GET" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://admin.example.test/api/workflows");
  });

  it("uses credentials include and preserves caller headers", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 200 }));
    const apiFetch = createApiFetch({
      fetchImpl: fetchMock,
      baseUrl: "https://example.test",
    });

    await apiFetch("/admin/workflows", {
      method: "GET",
      headers: {
        "X-Request-ID": "req-123",
      },
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://example.test/admin/workflows");
    expect(init.credentials).toBe("include");

    const headers = new Headers(init.headers);
    expect(headers.get("X-Request-ID")).toBe("req-123");
  });

  it("attaches CSRF token for POST but not GET", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 200 }));
    const apiFetch = createApiFetch({
      fetchImpl: fetchMock,
      csrfTokenProvider: () => "csrf-token-abc",
    });

    await apiFetch("/mutating", { method: "POST" });
    await apiFetch("/safe", { method: "GET" });

    const [, postInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const [, getInit] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];

    const postHeaders = new Headers(postInit.headers);
    const getHeaders = new Headers(getInit.headers);
    expect(postHeaders.get("X-CSRF-Token")).toBe("csrf-token-abc");
    expect(getHeaders.get("X-CSRF-Token")).toBeNull();
  });

  it("throws normalized JSON error with status code message and body", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({
          code: "invalid_request",
          message: "Bad workflow input",
          detail: { field: "name" },
        }),
        {
          status: 400,
          statusText: "Bad Request",
          headers: { "content-type": "application/json" },
        },
      ),
    );
    const apiFetch = createApiFetch({ fetchImpl: fetchMock });

    await expect(apiFetch("/fail", { method: "POST" })).rejects.toEqual(
      expect.objectContaining<ApiHttpError>({
        name: "ApiHttpError",
        status: 400,
        statusText: "Bad Request",
        code: "invalid_request",
        message: "Bad workflow input",
        body: {
          code: "invalid_request",
          message: "Bad workflow input",
          detail: { field: "name" },
        },
      }),
    );
  });

  it("normalizes FastAPI detail envelope object with code/message", async () => {
    const body = {
      detail: {
        code: "filesystem_browsing_root_missing",
        message: "Server folder browsing requires WORKFLOWS_SCAN_ROOT to be configured.",
      },
    };
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(body), {
        status: 409,
        statusText: "Conflict",
        headers: { "content-type": "application/json" },
      }),
    );
    const apiFetch = createApiFetch({ fetchImpl: fetchMock });

    await expect(apiFetch("/fail", { method: "GET" })).rejects.toEqual(
      expect.objectContaining<ApiHttpError>({
        name: "ApiHttpError",
        status: 409,
        statusText: "Conflict",
        code: "filesystem_browsing_root_missing",
        message: "Server folder browsing requires WORKFLOWS_SCAN_ROOT to be configured.",
        body,
      }),
    );
  });

  it("normalizes FastAPI detail string into message", async () => {
    const body = { detail: "Plain FastAPI error" };
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(body), {
        status: 400,
        statusText: "Bad Request",
        headers: { "content-type": "application/json" },
      }),
    );
    const apiFetch = createApiFetch({ fetchImpl: fetchMock });

    await expect(apiFetch("/fail", { method: "POST" })).rejects.toEqual(
      expect.objectContaining<ApiHttpError>({
        name: "ApiHttpError",
        status: 400,
        statusText: "Bad Request",
        code: undefined,
        message: "Plain FastAPI error",
        body,
      }),
    );
  });

  it("normalizes FastAPI detail array into stable validation message", async () => {
    const body = {
      detail: [{ loc: ["query", "path"], msg: "Field required", type: "missing" }],
    };
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(body), {
        status: 422,
        statusText: "Unprocessable Entity",
        headers: { "content-type": "application/json" },
      }),
    );
    const apiFetch = createApiFetch({ fetchImpl: fetchMock });

    await expect(apiFetch("/fail", { method: "GET" })).rejects.toEqual(
      expect.objectContaining<ApiHttpError>({
        name: "ApiHttpError",
        status: 422,
        statusText: "Unprocessable Entity",
        code: undefined,
        message: "Request validation failed.",
        body,
      }),
    );
  });

  it("preserves top-level message for non-envelope detail metadata object", async () => {
    const body = {
      message: "Top-level",
      detail: { reason: "metadata" },
    };
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(body), {
        status: 400,
        statusText: "Bad Request",
        headers: { "content-type": "application/json" },
      }),
    );
    const apiFetch = createApiFetch({ fetchImpl: fetchMock });

    await expect(apiFetch("/fail", { method: "POST" })).rejects.toEqual(
      expect.objectContaining<ApiHttpError>({
        name: "ApiHttpError",
        status: 400,
        statusText: "Bad Request",
        code: undefined,
        message: "Top-level",
        body,
      }),
    );
  });

  it("invokes unauthorized callback for HTTP 401", async () => {
    const fetchMock = vi.fn(async () =>
      new Response("Unauthorized", { status: 401, statusText: "Unauthorized" }),
    );
    const onUnauthorized = vi.fn();

    const apiFetch = createApiFetch({
      fetchImpl: fetchMock,
      onUnauthorized,
    });

    await expect(apiFetch("/private", { method: "GET" })).rejects.toBeInstanceOf(
      ApiHttpError,
    );
    expect(onUnauthorized).toHaveBeenCalledTimes(1);
  });
});

describe("createApiClient admin helpers", () => {
  it("fetches csrf first when creating project without a stored token", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
        return new Response(JSON.stringify({ csrf_token: "csrf-bootstrap" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/projects") && method === "POST") {
        return new Response(JSON.stringify({ id: "p1" }), {
          status: 201,
          headers: { "content-type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.createProject({ name: "P1" });

    const csrfCallIndex = fetchMock.mock.calls.findIndex((call) =>
      String(call[0]).includes("/api/admin/v1/auth/csrf"),
    );
    const projectCallIndex = fetchMock.mock.calls.findIndex((call) => {
      const init = call[1] as RequestInit | undefined;
      return String(call[0]).includes("/api/admin/v1/projects") && init?.method === "POST";
    });

    expect(csrfCallIndex).toBeGreaterThanOrEqual(0);
    expect(projectCallIndex).toBeGreaterThanOrEqual(0);
    expect(csrfCallIndex).toBeLessThan(projectCallIndex);

    const projectHeaders = new Headers((fetchMock.mock.calls[projectCallIndex]?.[1] as RequestInit | undefined)?.headers);
    expect(projectHeaders.get("X-CSRF-Token")).toBe("csrf-bootstrap");
  });

  it("fetches csrf first when creating an MCP client without a stored token", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
        return new Response(JSON.stringify({ csrf_token: "csrf-bootstrap" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/mcp-clients") && method === "POST") {
        return new Response(JSON.stringify({ id: "m1", label: "ci", token: "secret" }), {
          status: 201,
          headers: { "content-type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.createMcpClient({ label: "ci", project_ids: ["p1"] });

    const csrfCallIndex = fetchMock.mock.calls.findIndex((call) =>
      String(call[0]).includes("/api/admin/v1/auth/csrf"),
    );
    const createCallIndex = fetchMock.mock.calls.findIndex((call) => {
      const init = call[1] as RequestInit | undefined;
      return String(call[0]).includes("/api/admin/v1/mcp-clients") && init?.method === "POST";
    });

    expect(csrfCallIndex).toBeGreaterThanOrEqual(0);
    expect(createCallIndex).toBeGreaterThanOrEqual(0);
    expect(csrfCallIndex).toBeLessThan(createCallIndex);

    const createHeaders = new Headers((fetchMock.mock.calls[createCallIndex]?.[1] as RequestInit | undefined)?.headers);
    expect(createHeaders.get("X-CSRF-Token")).toBe("csrf-bootstrap");
  });

  it("fetches csrf first when updating MCP client project access", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
        return new Response(JSON.stringify({ csrf_token: "csrf-bootstrap" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/mcp-clients/m1") && method === "PATCH") {
        return new Response(JSON.stringify({ id: "m1", label: "ci", project_ids: [] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.updateMcpClient("m1", { project_ids: [] });

    const updateCallIndex = fetchMock.mock.calls.findIndex((call) => {
      const init = call[1] as RequestInit | undefined;
      return String(call[0]).endsWith("/api/admin/v1/mcp-clients/m1") && init?.method === "PATCH";
    });

    expect(updateCallIndex).toBeGreaterThanOrEqual(0);
    const updateInit = fetchMock.mock.calls[updateCallIndex]?.[1] as RequestInit | undefined;
    const updateHeaders = new Headers(updateInit?.headers);
    expect(updateHeaders.get("X-CSRF-Token")).toBe("csrf-bootstrap");
    expect(JSON.parse(String(updateInit?.body ?? "{}"))).toEqual({ project_ids: [] });
  });

  it("fetches csrf first when revoking and deleting MCP clients", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
        return new Response(JSON.stringify({ csrf_token: "csrf-bootstrap" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/mcp-clients/m1") && method === "DELETE") {
        return new Response(JSON.stringify({ revoked: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/mcp-clients/m1/registration") && method === "DELETE") {
        return new Response(JSON.stringify({ deleted: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.revokeMcpClient("m1");
    await client.deleteMcpClient("m1");

    const csrfCallIndex = fetchMock.mock.calls.findIndex((call) =>
      String(call[0]).includes("/api/admin/v1/auth/csrf"),
    );
    const revokeCallIndex = fetchMock.mock.calls.findIndex((call) => {
      const init = call[1] as RequestInit | undefined;
      return String(call[0]).endsWith("/api/admin/v1/mcp-clients/m1") && init?.method === "DELETE";
    });
    const deleteCallIndex = fetchMock.mock.calls.findIndex((call) => {
      const init = call[1] as RequestInit | undefined;
      return String(call[0]).endsWith("/api/admin/v1/mcp-clients/m1/registration") && init?.method === "DELETE";
    });

    expect(csrfCallIndex).toBeGreaterThanOrEqual(0);
    expect(revokeCallIndex).toBeGreaterThanOrEqual(0);
    expect(deleteCallIndex).toBeGreaterThanOrEqual(0);
    expect(csrfCallIndex).toBeLessThan(revokeCallIndex);

    const revokeHeaders = new Headers((fetchMock.mock.calls[revokeCallIndex]?.[1] as RequestInit | undefined)?.headers);
    const deleteHeaders = new Headers((fetchMock.mock.calls[deleteCallIndex]?.[1] as RequestInit | undefined)?.headers);
    expect(revokeHeaders.get("X-CSRF-Token")).toBe("csrf-bootstrap");
    expect(deleteHeaders.get("X-CSRF-Token")).toBe("csrf-bootstrap");
  });

  it("lists server path entries without requiring csrf for default request", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/admin/v1/filesystem/entries") && method === "GET") {
        return new Response(
          JSON.stringify({
            root: "/",
            path: "/",
            parent: null,
            can_go_up: false,
            entries: [{ name: "tmp", path: "/tmp", type: "directory", selectable: true }],
          }),
          {
            status: 200,
            headers: { "content-type": "application/json" },
          },
        );
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.listServerPathEntries();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [requestUrl, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit | undefined];
    expect(requestUrl).toBe("https://admin.example.test/api/admin/v1/filesystem/entries");
    expect((init?.method ?? "GET").toUpperCase()).toBe("GET");
    expect(fetchMock.mock.calls.some((call) => String(call[0]).includes("/api/admin/v1/auth/csrf"))).toBe(false);
  });

  it("encodes path selection type and deterministic extensions for listServerPathEntries", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.includes("/api/admin/v1/filesystem/entries") && method === "GET") {
        return new Response(
          JSON.stringify({
            root: "/",
            path: "/tmp",
            parent: "/",
            can_go_up: true,
            entries: [],
          }),
          {
            status: 200,
            headers: { "content-type": "application/json" },
          },
        );
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.listServerPathEntries({
      path: "a/b c",
      selectionType: "file",
      extensions: [" .yaml ", "", "  ", ".yml", " .json"],
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [requestUrl, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit | undefined];
    expect(requestUrl).toBe(
      "https://admin.example.test/api/admin/v1/filesystem/entries?path=a%2Fb+c&selection_type=file&extensions=.yaml%2C.yml%2C.json",
    );
    expect((init?.method ?? "GET").toUpperCase()).toBe("GET");
  });

  it("exposes helper methods across admin/public endpoint families", () => {
    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: vi.fn(async () => new Response(null, { status: 200 })),
    }) as unknown as Record<string, unknown>;

    expect(typeof client.login).toBe("function");
    expect(typeof client.getSession).toBe("function");
    expect(typeof client.getCsrf).toBe("function");
    expect(typeof client.logout).toBe("function");
    expect(typeof client.getSystemStatus).toBe("function");
    expect(typeof client.getSetupStatus).toBe("function");
    expect(typeof client.getDatabaseSettings).toBe("function");
    expect(typeof client.saveDatabaseSettings).toBe("function");
    expect(typeof client.testDatabaseConnection).toBe("function");
    expect(typeof client.listProjects).toBe("function");
    expect(typeof client.listServerPathEntries).toBe("function");
    expect(typeof client.createProject).toBe("function");
    expect(typeof client.getProject).toBe("function");
    expect(typeof client.updateProject).toBe("function");
    expect(typeof client.deleteProject).toBe("function");
    expect(typeof client.listWatchers).toBe("function");
    expect(typeof client.getWatcher).toBe("function");
    expect(typeof client.pauseWatcher).toBe("function");
    expect(typeof client.resumeWatcher).toBe("function");
    expect(typeof client.disableWatcher).toBe("function");
    expect(typeof client.listSyncQueue).toBe("function");
    expect(typeof client.syncNow).toBe("function");
    expect(typeof client.reconcileSync).toBe("function");
    expect(typeof client.rebuildSync).toBe("function");
    expect(typeof client.listMcpClients).toBe("function");
    expect(typeof client.createMcpClient).toBe("function");
    expect(typeof client.updateMcpClient).toBe("function");
    expect(typeof client.revokeMcpClient).toBe("function");
    expect(typeof client.deleteMcpClient).toBe("function");
    expect(typeof client.regenerateMcpClient).toBe("function");
    expect(typeof client.listWorkflows).toBe("function");
    expect(typeof client.getWorkflowDetail).toBe("function");
    expect(typeof client.getWorkflowSchema).toBe("function");
    expect(typeof client.listWorkflowSources).toBe("function");
    expect(typeof client.createWorkflowSource).toBe("function");
    expect(typeof client.deleteWorkflowSource).toBe("function");
    expect(typeof client.validateWorkflowSource).toBe("function");
    expect(typeof client.listRuns).toBe("function");
    expect(typeof client.getRunDetail).toBe("function");
    expect(typeof client.cancelRun).toBe("function");
    expect(typeof client.resumeRun).toBe("function");
    expect(typeof client.getLlmConfig).toBe("function");
    expect(typeof client.updateLlmConfig).toBe("function");
    expect(typeof client.previewLlmConfig).toBe("function");
    expect(typeof client.importLlmConfig).toBe("function");
    expect(typeof client.exportLlmConfig).toBe("function");
    expect(typeof client.listSecrets).toBe("function");
    expect(typeof client.upsertSecret).toBe("function");
    expect(typeof client.deleteSecret).toBe("function");
    expect(typeof client.reloadWorkflows).toBe("function");
  });

  it("sends csrf header on reloadWorkflows after login retrieves csrf", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/admin/v1/auth/login") && method === "POST") {
        return new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/auth/session") && method === "GET") {
        return new Response(JSON.stringify({ authenticated: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
        return new Response(JSON.stringify({ csrf_token: "csrf-login-token" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/workflows/reload") && method === "POST") {
        return new Response(
          JSON.stringify({
            status: "ok",
            source_count: 1,
            total: 1,
            workflow_names: ["a"],
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.login("admin-pass");
    await client.reloadWorkflows();

    const reloadCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/workflows/reload") && init?.method === "POST";
    });

    expect(reloadCall).toBeTruthy();
    const reloadInit = reloadCall?.[1] as RequestInit | undefined;
    const headers = new Headers(reloadInit?.headers);
    expect(headers.get("X-CSRF-Token")).toBe("csrf-login-token");
  });

  it("covers representative family helpers with credentials, csrf, and endpoint shape", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/public/v1/system/status") && method === "GET") {
        return new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/auth/login") && method === "POST") {
        return new Response(JSON.stringify({ authenticated: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/auth/session") && method === "GET") {
        return new Response(JSON.stringify({ authenticated: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
        return new Response(JSON.stringify({ csrf_token: "csrf-token-1" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/database/settings") && method === "GET") {
        return new Response(JSON.stringify({ enabled: true, configured: true, updated_at: "t" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/database/settings") && method === "PUT") {
        return new Response(JSON.stringify({ enabled: true, configured: true, updated_at: "t2" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/projects") && method === "GET") {
        return new Response(JSON.stringify({ projects: [] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/watchers/proj-1/pause") && method === "POST") {
        return new Response(JSON.stringify({ project_id: "proj-1", state: "paused" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/sync/proj-1/reconcile") && method === "POST") {
        return new Response(JSON.stringify({ project_id: "proj-1", dirty_count: 1 }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/mcp-clients") && method === "POST") {
        return new Response(JSON.stringify({ id: "m1", label: "x", token: "t" }), {
          status: 201,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/mcp-clients/m1") && method === "PATCH") {
        return new Response(JSON.stringify({ id: "m1", label: "x", project_ids: [] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/workflows/sources/source-1/validate") && method === "POST") {
        return new Response(JSON.stringify({ valid: true, workflow_names: ["wf"], total: 1 }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.includes("/api/admin/v1/runs") && url.endsWith("status=paused") && method === "GET") {
        return new Response(JSON.stringify({ runs: [] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/llm/config") && method === "GET") {
        return new Response(JSON.stringify({ version: "1.0", providers: {}, profiles: {} }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/secrets") && method === "POST") {
        return new Response(JSON.stringify({ name: "OPENAI_API_KEY", key_id: null }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.login("admin-pass");
    await client.getSystemStatus();
    await client.getDatabaseSettings();
    await client.saveDatabaseSettings({
      enabled: true,
      host: "127.0.0.1",
      port: 5432,
      database: "workflows",
      username: "wf_user",
      password: "typed-secret",
      password_clear: false,
      ssl_mode: "disable",
      extra_params: "connect_timeout=5&application_name=workflows-mcp",
      container_name: "workflows-postgres",
      container_image: "pgvector/pgvector:pg17",
      container_host_port: 5432,
      volume_name: "workflows-postgres-data",
      dsn_import: null,
    });
    await client.listProjects();
    await client.pauseWatcher("proj-1");
    await client.reconcileSync("proj-1");
    await client.createMcpClient({ label: "x", project_ids: ["proj-1"] });
    await client.updateMcpClient("m1", { project_ids: [] });
    await client.validateWorkflowSource("source-1");
    await client.listRuns({ status: "paused" });
    await client.getLlmConfig();
    await client.upsertSecret({ name: "OPENAI_API_KEY", value: "secret" });

    const settingsSaveCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/database/settings") && init?.method === "PUT";
    });
    expect(settingsSaveCall).toBeTruthy();
    const settingsSaveHeaders = new Headers((settingsSaveCall?.[1] as RequestInit | undefined)?.headers);
    expect(settingsSaveHeaders.get("X-CSRF-Token")).toBe("csrf-token-1");
    const settingsSaveBody = JSON.parse(
      String((settingsSaveCall?.[1] as RequestInit | undefined)?.body ?? "{}"),
    ) as Record<string, unknown>;
    expect(settingsSaveBody).toEqual({
      enabled: true,
      host: "127.0.0.1",
      port: 5432,
      database: "workflows",
      username: "wf_user",
      password: "typed-secret",
      password_clear: false,
      ssl_mode: "disable",
      extra_params: "connect_timeout=5&application_name=workflows-mcp",
      container_name: "workflows-postgres",
      container_image: "pgvector/pgvector:pg17",
      container_host_port: 5432,
      volume_name: "workflows-postgres-data",
      dsn_import: null,
    });

    const publicStatusCall = fetchMock.mock.calls.find((call) =>
      String(call[0]).includes("/api/public/v1/system/status"),
    );
    expect(publicStatusCall).toBeTruthy();
    expect((publicStatusCall?.[1] as RequestInit | undefined)?.credentials).toBe("include");
  });

  it("keeps 401 detectable for helper calls", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/api/admin/v1/auth/session")) {
        return new Response(JSON.stringify({ message: "Unauthorized" }), {
          status: 401,
          statusText: "Unauthorized",
          headers: { "content-type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ message: "ok" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await expect(client.getSession()).rejects.toBeInstanceOf(ApiHttpError);
    await expect(client.getSession()).rejects.toMatchObject({
      status: 401,
      statusText: "Unauthorized",
    });
  });

  it("preserves explicit password null when saving database settings", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
        return new Response(JSON.stringify({ csrf_token: "csrf-token-2" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/database/settings") && method === "PUT") {
        return new Response(JSON.stringify({ enabled: true, configured: true, updated_at: "t2" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.saveDatabaseSettings({
      enabled: true,
      host: "127.0.0.1",
      port: 5432,
      database: "workflows",
      username: "wf_user",
      password: null,
      password_clear: false,
      ssl_mode: "disable",
      extra_params: "",
      container_name: "workflows-postgres",
      container_image: "pgvector/pgvector:pg17",
      container_host_port: 5432,
      volume_name: "workflows-postgres-data",
      dsn_import: null,
    });

    const settingsSaveCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/database/settings") && init?.method === "PUT";
    });
    expect(settingsSaveCall).toBeTruthy();

    const settingsSaveBody = JSON.parse(
      String((settingsSaveCall?.[1] as RequestInit | undefined)?.body ?? "{}"),
    ) as Record<string, unknown>;
    expect(settingsSaveBody.password).toBeNull();
  });

  it("sends full LLM config updates with csrf and JSON body", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
        return new Response(JSON.stringify({ csrf_token: "csrf-llm-token" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/admin/v1/llm/config") && method === "PUT") {
        return new Response(String(init?.body ?? "{}"), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });
    const payload = {
      version: "1.0",
      providers: { openai: { type: "openai", model: "gpt-4.1-mini" } },
      profiles: { default: { provider: "openai", model: "gpt-4.1-mini" } },
      default_profile: "default",
    };

    await client.updateLlmConfig(payload);

    const updateCall = fetchMock.mock.calls.find((call) => {
      const url = String(call[0]);
      const init = call[1] as RequestInit | undefined;
      return url.includes("/api/admin/v1/llm/config") && init?.method === "PUT";
    });
    expect(updateCall).toBeTruthy();

    const headers = new Headers((updateCall?.[1] as RequestInit | undefined)?.headers);
    expect(headers.get("X-CSRF-Token")).toBe("csrf-llm-token");
    expect(headers.get("content-type")).toBe("application/json");
    expect(JSON.parse(String((updateCall?.[1] as RequestInit | undefined)?.body ?? "{}"))).toEqual(payload);
  });

  it("sends LLM YAML preview and import mutations with csrf and raw_yaml body", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = typeof input === "string" ? input : input.toString();

      if (url.endsWith("/api/admin/v1/auth/csrf") && method === "GET") {
        return new Response(JSON.stringify({ csrf_token: "csrf-llm-token" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if ((url.endsWith("/api/admin/v1/llm/preview") || url.endsWith("/api/admin/v1/llm/import")) && method === "POST") {
        return new Response(JSON.stringify({ version: "1.0", providers: {}, profiles: {} }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ message: `unexpected ${method} ${url}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    });

    const client = createApiClient({
      baseUrl: "https://admin.example.test",
      fetchImpl: fetchMock,
    });

    await client.previewLlmConfig("version: '1.0'\n");
    await client.importLlmConfig("providers: {}\n");

    for (const endpoint of ["/api/admin/v1/llm/preview", "/api/admin/v1/llm/import"]) {
      const call = fetchMock.mock.calls.find((entry) => {
        const url = String(entry[0]);
        const init = entry[1] as RequestInit | undefined;
        return url.includes(endpoint) && init?.method === "POST";
      });
      expect(call).toBeTruthy();
      const headers = new Headers((call?.[1] as RequestInit | undefined)?.headers);
      expect(headers.get("X-CSRF-Token")).toBe("csrf-llm-token");
      expect(headers.get("content-type")).toBe("application/json");
    }

    const previewCall = fetchMock.mock.calls.find((entry) => String(entry[0]).includes("/api/admin/v1/llm/preview"));
    expect(JSON.parse(String((previewCall?.[1] as RequestInit | undefined)?.body ?? "{}"))).toEqual({
      raw_yaml: "version: '1.0'\n",
    });

    const importCall = fetchMock.mock.calls.find((entry) => String(entry[0]).includes("/api/admin/v1/llm/import"));
    expect(JSON.parse(String((importCall?.[1] as RequestInit | undefined)?.body ?? "{}"))).toEqual({
      raw_yaml: "providers: {}\n",
    });
  });
});
