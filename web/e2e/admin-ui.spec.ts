import { expect, test, type Page } from "@playwright/test";

const LONG_WATCHER_PROJECT_ID = "0eeb6dcb-64d5-4383-b81a-127d09346240";
const LONG_SYNC_PROJECT_ID = "31cc7ff8-a614-4fd5-b381-e2f6ff2f98a3";

async function routeWatcherDashboard(page: Page): Promise<void> {
  await page.route("**/api/admin/v1/projects", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        projects: [
          {
            id: LONG_WATCHER_PROJECT_ID,
            name: "Main Project",
            slug: "main",
            palace: "x",
            default_wing: "w",
            default_room: "r",
            fs_root: "/tmp",
            fs_allowlist: [],
          },
        ],
      }),
    });
  });
  await page.route("**/api/events/v1/watchers/state", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        version: 1,
        items: [
          {
            project_id: LONG_WATCHER_PROJECT_ID,
            state: "enabled",
            dirty_count: 12,
            requires_reconciliation: true,
            last_event_at: "2026-04-30T00:04:00Z",
            updated_at: "2026-04-30T00:05:00Z",
          },
        ],
      }),
    });
  });
  await page.route("**/api/events/v1/watchers", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ message: "SSE unavailable in layout test" }),
    });
  });
}

async function routeSyncDashboard(page: Page): Promise<void> {
  await page.route("**/api/admin/v1/projects", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        projects: [
          {
            id: LONG_SYNC_PROJECT_ID,
            name: "Main Project",
            slug: "main",
            palace: "x",
            default_wing: "w",
            default_room: "r",
            fs_root: "/tmp",
            fs_allowlist: [],
          },
        ],
      }),
    });
  });
  await page.route("**/api/events/v1/sync/state", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        version: 1,
        items: [
          {
            project_id: LONG_SYNC_PROJECT_ID,
            dirty_count: 12,
            requires_reconciliation: true,
            sync_state: "queued",
            reconcile_state: "required",
            rebuild_state: "available as manual action",
            updated_at: "2026-04-30T00:05:00Z",
          },
        ],
      }),
    });
  });
  await page.route("**/api/events/v1/sync", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ message: "SSE unavailable in layout test" }),
    });
  });
}

test("admin shell mounts, assets load, login is interactive, and workflow reload sends CSRF", async ({
  page,
}) => {
  const apiRequests: Array<{ url: string; method: string; csrf: string | null }> = [];

  await page.route("**/api/admin/v1/auth/login", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ok: true }),
    });
  });
  await page.route("**/api/admin/v1/auth/session", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ user: "admin" }),
    });
  });
  await page.route("**/api/admin/v1/auth/csrf", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ csrf_token: "csrf-test-token" }),
    });
  });
  await page.route("**/api/public/v1/system/status", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok" }),
    });
  });
  await page.route("**/api/admin/v1/database/settings", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        enabled: false,
        configured: false,
        updated_at: "2026-04-30T00:00:00Z",
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
      }),
    });
  });
  await page.route("**/api/admin/v1/llm/config", async (route) => {
    const request = route.request();
    if (request.method() === "PUT") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: request.postData() ?? JSON.stringify({ version: "1.0", providers: {}, profiles: {}, default_profile: null }),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ version: "1.0", providers: {}, profiles: {}, default_profile: null }),
    });
  });
  await page.route("**/api/admin/v1/llm/export", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ raw_yaml: "version: '1.0'\nproviders: {}\nprofiles: {}\n" }),
    });
  });
  await page.route("**/api/admin/v1/projects", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ projects: [] }),
    });
  });
  await page.route("**/api/admin/v1/mcp-clients", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ mcp_clients: [] }),
    });
  });
  await page.route("**/api/admin/v1/workflows", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ workflows: [] }),
    });
  });
  await page.route("**/api/admin/v1/workflows/sources", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ sources: [] }),
    });
  });
  await page.route("**/api/admin/v1/workflows/reload", async (route) => {
    const request = route.request();
    const csrfHeader = await request.headerValue("x-csrf-token");
    apiRequests.push({
      url: request.url(),
      method: request.method(),
      csrf: csrfHeader,
    });
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        source_count: 1,
        total: 1,
        workflow_names: ["example"],
      }),
    });
  });

  const scriptResponsePromise = page.waitForResponse((resp) => /\/assets\/.+\.js($|\?)/.test(resp.url()));
  const cssResponsePromise = page.waitForResponse((resp) => /\/assets\/.+\.css($|\?)/.test(resp.url()));
  await page.goto("/workflows");
  const [scriptResponse, cssResponse] = await Promise.all([scriptResponsePromise, cssResponsePromise]);
  expect(scriptResponse.status()).toBe(200);
  expect(cssResponse.status()).toBe(200);

  await expect(page.locator("#root > .admin-shell")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Workflows", exact: true })).toBeVisible();

  await page.goto("/login");
  const passwordInput = page.getByLabel("Password");
  await expect(passwordInput).toBeEnabled();
  await passwordInput.fill("secret");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/setup$/);
  await expect(page.getByRole("heading", { name: "Setup", exact: true })).toBeVisible();

  await page.goto("/llm");
  await expect(page.getByRole("heading", { name: "LLM configuration" })).toBeVisible();
  await expect(page.getByText("Source of truth: SQLite-backed.")).toBeVisible();

  await page.goto("/workflows");
  await page.getByRole("button", { name: "Reload workflows" }).click();
  await expect(page.getByText("Workflow registry reloaded: 1 workflows from 1 source(s).")).toBeVisible();

  expect(apiRequests).toHaveLength(1);
  expect(apiRequests[0]).toMatchObject({
    method: "POST",
    csrf: "csrf-test-token",
  });
});

test("watchers page keeps long project ids inside responsive table and modal", async ({ page }) => {
  await routeWatcherDashboard(page);

  for (const viewport of [
    { width: 1242, height: 1246 },
    { width: 390, height: 900 },
  ]) {
    await page.setViewportSize(viewport);
    await page.goto("/watchers");

    const watcherTable = page.getByRole("table", { name: "Registered watchers" });
    await expect(watcherTable).toBeVisible();
    const watcherRow = watcherTable.getByRole("row", {
      name: /open watcher main project status needs reconcile dirty files 12/i,
    });
    await expect(watcherRow).toBeVisible();
    await expect(watcherRow.getByText(LONG_WATCHER_PROJECT_ID)).toBeVisible();

    await watcherRow.click();
    const watcherDialog = page.getByRole("dialog", { name: /watcher main project/i });
    await expect(watcherDialog).toBeVisible();
    await expect(watcherDialog.getByRole("button", { name: `Pause watcher ${LONG_WATCHER_PROJECT_ID}` })).toHaveText(
      "Pause",
    );
    await expect(watcherDialog.getByRole("button", { name: `Disable watcher ${LONG_WATCHER_PROJECT_ID}` })).toHaveText(
      "Disable",
    );

    const overflow = await page.locator("body").evaluate(() => {
      const tolerance = 1;
      const pageOverflow = document.documentElement.scrollWidth - document.documentElement.clientWidth;
      const elements = Array.from(
        document.querySelectorAll<HTMLElement>(
          ".watchers-project-cell, .watcher-command-panel .action-button, .llm-modal",
        ),
      ).map((element) => ({
        className: element.className,
        overflow: element.scrollWidth - element.clientWidth,
      }));
      return {
        pageOverflow,
        overflowingElements: elements.filter((element) => element.overflow > tolerance),
      };
    });

    expect(overflow.pageOverflow).toBeLessThanOrEqual(1);
    expect(overflow.overflowingElements).toEqual([]);
  }
});

test("sync page keeps long project ids inside responsive table and modal", async ({ page }) => {
  await routeSyncDashboard(page);

  for (const viewport of [
    { width: 1242, height: 1246 },
    { width: 390, height: 900 },
  ]) {
    await page.setViewportSize(viewport);
    await page.goto("/sync");

    const syncTable = page.getByRole("table", { name: "Registered sync projects" });
    await expect(syncTable).toBeVisible();
    const syncRow = syncTable.getByRole("row", {
      name: /open sync main project status needs reconcile dirty files 12/i,
    });
    await expect(syncRow).toBeVisible();
    await expect(syncRow.getByText(LONG_SYNC_PROJECT_ID)).toBeVisible();

    await syncRow.click();
    const syncDialog = page.getByRole("dialog", { name: /sync main project/i });
    await expect(syncDialog).toBeVisible();
    await expect(syncDialog.getByRole("button", { name: `Sync now ${LONG_SYNC_PROJECT_ID}` })).toHaveText(
      "Sync now",
    );
    await expect(syncDialog.getByRole("button", { name: `Reconcile ${LONG_SYNC_PROJECT_ID}` })).toHaveText(
      "Reconcile",
    );
    await expect(syncDialog.getByRole("button", { name: `Rebuild ${LONG_SYNC_PROJECT_ID}` })).toHaveText("Rebuild");

    const overflow = await page.locator("body").evaluate(() => {
      const tolerance = 1;
      const pageOverflow = document.documentElement.scrollWidth - document.documentElement.clientWidth;
      const elements = Array.from(
        document.querySelectorAll<HTMLElement>(
          ".watchers-project-cell, .sync-command-panel .action-button, .llm-modal",
        ),
      ).map((element) => ({
        className: element.className,
        overflow: element.scrollWidth - element.clientWidth,
      }));
      return {
        pageOverflow,
        overflowingElements: elements.filter((element) => element.overflow > tolerance),
      };
    });

    expect(overflow.pageOverflow).toBeLessThanOrEqual(1);
    expect(overflow.overflowingElements).toEqual([]);
  }
});
