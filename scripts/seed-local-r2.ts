#!/usr/bin/env npx tsx
/**
 * Seed local R2 (Miniflare) with test fixtures for benchmarking.
 *
 * Uploads Lance and Parquet files from wasm/tests/fixtures/ to local R2
 * via the wrangler dev Worker's /write endpoint.
 *
 * Prerequisites: `pnpm dev` must be running on localhost:8787
 *
 * Usage: npx tsx scripts/seed-local-r2.ts
 */

import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, basename, relative } from "node:path";

const BASE_URL = process.env.WORKER_URL ?? "http://localhost:8787";
const FIXTURES = join(import.meta.dirname, "../wasm/tests/fixtures");

async function uploadFile(r2Key: string, data: Uint8Array): Promise<void> {
  // Use wrangler's local R2 API directly via unstable_dev or
  // PUT through the Worker. Since the Worker doesn't expose raw R2 PUT,
  // we'll use the /write endpoint which reads from R2.
  // But first we need the data IN R2. For local dev, Miniflare stores
  // R2 objects in .wrangler/state/. We write files there directly.

  // Actually, the simplest approach: use wrangler CLI to put objects.
  const { execSync } = await import("node:child_process");
  const tmpFile = `/tmp/qm-seed-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const { writeFileSync, unlinkSync } = await import("node:fs");
  writeFileSync(tmpFile, data);

  try {
    execSync(
      `npx wrangler r2 object put querymode-data/${r2Key} --file=${tmpFile} --local`,
      { cwd: join(import.meta.dirname, ".."), stdio: "pipe" },
    );
  } finally {
    unlinkSync(tmpFile);
  }
}

async function seedLanceDataset(dir: string): Promise<void> {
  const name = basename(dir); // e.g. "simple_int64.lance"
  console.log(`\nSeeding Lance dataset: ${name}`);

  // Upload all files in the dataset directory recursively
  const walkDir = (d: string): string[] => {
    const entries: string[] = [];
    for (const entry of readdirSync(d, { withFileTypes: true })) {
      const full = join(d, entry.name);
      if (entry.isDirectory()) entries.push(...walkDir(full));
      else entries.push(full);
    }
    return entries;
  };

  const files = walkDir(dir);
  for (const file of files) {
    const relPath = relative(join(dir, ".."), file);
    const r2Key = relPath;
    const data = readFileSync(file);
    console.log(`  PUT ${r2Key} (${data.length} bytes)`);
    await uploadFile(r2Key, data);
  }

  // Tell Master DO about the dataset
  const resp = await fetch(`${BASE_URL}/write`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ r2Key: `${name}/` }),
  });
  const result = await resp.json();
  console.log(`  Master response:`, result);
}

async function seedParquetFile(filePath: string): Promise<void> {
  const name = basename(filePath);
  console.log(`\nSeeding Parquet file: ${name}`);

  const data = readFileSync(filePath);
  const r2Key = name;
  console.log(`  PUT ${r2Key} (${data.length} bytes)`);
  await uploadFile(r2Key, data);

  // Tell Master DO about the file
  try {
    const resp = await fetch(`${BASE_URL}/write`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ r2Key: name }),
    });
    if (resp.ok) {
      const result = await resp.json();
      console.log(`  Master response:`, result);
    } else {
      console.log(`  Master error: ${resp.status} (file too large? Will lazy-load on first query)`);
    }
  } catch (err) {
    console.log(`  Master error:`, String(err).slice(0, 100));
  }
}

async function main(): Promise<void> {
  // Check Worker is running
  try {
    const health = await fetch(`${BASE_URL}/health`);
    if (!health.ok) throw new Error(`Health check failed: ${health.status}`);
    console.log("Worker is running:", await health.json());
  } catch (err) {
    console.error(`Cannot reach Worker at ${BASE_URL}. Is 'pnpm dev' running?`);
    process.exit(1);
  }

  if (!existsSync(FIXTURES)) {
    console.error(`Fixtures not found at ${FIXTURES}`);
    process.exit(1);
  }

  // Seed Lance datasets
  const lanceDatasets = readdirSync(FIXTURES)
    .filter(f => f.endsWith(".lance"))
    .map(f => join(FIXTURES, f))
    .filter(f => statSync(f).isDirectory());

  for (const dir of lanceDatasets) {
    await seedLanceDataset(dir);
  }

  // Seed Parquet files
  const parquetFiles = readdirSync(FIXTURES)
    .filter(f => f.endsWith(".parquet"))
    .map(f => join(FIXTURES, f));

  for (const file of parquetFiles) {
    await seedParquetFile(file);
  }

  // Force Query DO to register with Master by hitting /tables
  console.log("\nRegistering Query DO with Master...");
  await fetch(`${BASE_URL}/tables`);
  // Wait for async registration to complete
  await new Promise(r => setTimeout(r, 1000));

  // Re-notify Master for all datasets so broadcasts reach the now-registered Query DO
  console.log("Re-broadcasting footers to Query DOs...");
  for (const dir of lanceDatasets) {
    const name = basename(dir);
    await fetch(`${BASE_URL}/write`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ r2Key: `${name}/` }),
    });
  }
  for (const file of parquetFiles) {
    const name = basename(file);
    await fetch(`${BASE_URL}/write`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ r2Key: name }),
    });
  }
  await new Promise(r => setTimeout(r, 500));

  // Verify tables are registered
  console.log("\n--- Verifying tables ---");
  const tablesResp = await fetch(`${BASE_URL}/tables`);
  const tables = await tablesResp.json();
  console.log("Registered tables:", tables);

  console.log("\nSeeding complete!");
}

main().catch(console.error);
