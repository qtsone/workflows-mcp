import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "../..");
const generatedSchemaPathFromRepoRoot = "web/src/api/generated/schema.d.ts";
const isCi = process.env.CI === "true";

const trackedResult = spawnSync(
  "git",
  ["-C", repoRoot, "ls-files", "--error-unmatch", "--", generatedSchemaPathFromRepoRoot],
  { stdio: "ignore" },
);

if (trackedResult.error) {
  console.error(`Unable to determine tracking status for ${generatedSchemaPathFromRepoRoot}.`);
  console.error(trackedResult.error.message);
  process.exit(1);
}

if (trackedResult.status !== 0) {
  const message =
    `Generated client schema is not tracked by git: ${generatedSchemaPathFromRepoRoot}. ` +
    "Track this file before merging so CI can enforce generated drift checks.";

  if (isCi) {
    console.error(message);
    process.exit(1);
  }

  console.warn(`Warning: ${message}`);
  process.exit(0);
}

const diffResult = spawnSync(
  "git",
  ["-C", repoRoot, "diff", "--exit-code", "--", generatedSchemaPathFromRepoRoot],
  { stdio: "inherit" },
);

if (diffResult.error) {
  console.error(`Unable to run git diff for ${generatedSchemaPathFromRepoRoot}.`);
  console.error(diffResult.error.message);
  process.exit(1);
}

if (diffResult.status !== 0) {
  console.error(
    `Generated client schema is out of date. Run \`npm run generate:client\` and commit updated ${generatedSchemaPathFromRepoRoot}.`,
  );
  process.exit(diffResult.status ?? 1);
}
