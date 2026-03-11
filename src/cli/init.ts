#!/usr/bin/env node
/**
 * npx querymode init — scaffold a QueryMode project in the current directory.
 *
 * Creates:
 *   - wrangler.toml (Worker + DOs + R2 binding)
 *   - src/worker.ts (re-exports QueryMode Worker)
 *
 * Then prints next steps: `wrangler r2 bucket create`, `wrangler deploy`.
 */

import { writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const cwd = process.cwd();
const projectName = cwd.split("/").pop() ?? "my-querymode";

// ── wrangler.toml ─────────────────────────────────────────────────────

const wranglerPath = join(cwd, "wrangler.toml");
if (existsSync(wranglerPath)) {
  console.log("wrangler.toml already exists — skipping.");
} else {
  const wrangler = `name = "${projectName}"
main = "src/worker.ts"
compatibility_date = "${new Date().toISOString().split("T")[0]}"
compatibility_flags = ["nodejs_compat"]

[[r2_buckets]]
binding = "DATA_BUCKET"
bucket_name = "${projectName}-data"

[durable_objects]
bindings = [
  { name = "MASTER_DO", class_name = "MasterDO" },
  { name = "QUERY_DO", class_name = "QueryDO" },
  { name = "FRAGMENT_DO", class_name = "FragmentDO" }
]

[[migrations]]
tag = "v1"
new_classes = ["MasterDO", "QueryDO"]

[[migrations]]
tag = "v2"
new_classes = ["FragmentDO"]

[[rules]]
type = "CompiledWasm"
globs = ["**/*.wasm"]
fallthrough = false
`;
  writeFileSync(wranglerPath, wrangler);
  console.log("Created wrangler.toml");
}

// ── src/worker.ts ─────────────────────────────────────────────────────

const srcDir = join(cwd, "src");
const workerPath = join(srcDir, "worker.ts");
if (existsSync(workerPath)) {
  console.log("src/worker.ts already exists — skipping.");
} else {
  mkdirSync(srcDir, { recursive: true });
  const worker = `// Re-export QueryMode Worker + Durable Objects
export { default, MasterDO, QueryDO, FragmentDO } from "querymode";
`;
  writeFileSync(workerPath, worker);
  console.log("Created src/worker.ts");
}

// ── Next steps ────────────────────────────────────────────────────────

console.log(`
Setup complete! Next steps:

  1. Install querymode:
     pnpm add querymode

  2. Create R2 bucket:
     wrangler r2 bucket create ${projectName}-data

  3. Deploy:
     wrangler deploy

  4. Upload data:
     curl -X POST https://${projectName}.<your-subdomain>.workers.dev/upload?key=events.parquet \\
       --data-binary @events.parquet

  5. Query:
     curl -X POST https://${projectName}.<your-subdomain>.workers.dev/query \\
       -H "Content-Type: application/json" \\
       -d '{"table":"events","filters":[],"projections":[],"limit":10}'
`);
