import { expect, test } from "@playwright/test";

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
  await page.route("**/api/admin/v1/workflows", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ workflows: [] }),
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
  await expect(page.getByRole("heading", { name: "Workflows" })).toBeVisible();

  await page.goto("/login");
  const passwordInput = page.getByLabel("Password");
  await expect(passwordInput).toBeEnabled();
  await passwordInput.fill("secret");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("status")).toContainText("Signed in");

  await page.goto("/workflows");
  await page.getByRole("button", { name: "Reload workflows" }).click();
  await expect(page.getByRole("status")).toContainText("Reloaded 1 workflows.");

  expect(apiRequests).toHaveLength(1);
  expect(apiRequests[0]).toMatchObject({
    method: "POST",
    csrf: "csrf-test-token",
  });
});
