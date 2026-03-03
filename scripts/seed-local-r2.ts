#!/usr/bin/env npx tsx
/**
 * Seed local R2 (Miniflare) with test fixtures for benchmarking.
 *
 * Uploads Lance, Parquet, and Iceberg data to local R2 via wrangler CLI.
 *
 * Prerequisites:
 *   1. `pnpm dev` must be running on localhost:8787
 *   2. `npx tsx scripts/generate-bench-data.ts` to create large fixtures
 *
 * Usage: npx tsx scripts/seed-local-r2.ts
 */

import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, basename, relative } from "node:path";
import { execSync } from "node:child_process";
import { writeFileSync, unlinkSync } from "node:fs";

const BASE_URL = process.env.WORKER_URL ?? "http://localhost:8787";
const FIXTURES = join(import.meta.dirname, "../wasm/tests/fixtures");
const GENERATED = join(FIXTURES, "generated");
const ROOT = join(import.meta.dirname, "..");

function uploadFile(r2Key: string, data: Uint8Array): void {
  const tmpFile = `/tmp/qm-seed-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  writeFileSync(tmpFile, data);
  try {
    execSync(
      `npx wrangler r2 object put querymode-data/${r2Key} --file=${tmpFile} --local`,
      { cwd: ROOT, stdio: "pipe" },
    );
  } finally {
    unlinkSync(tmpFile);
  }
}

function walkDir(d: string): string[] {
  const entries: string[] = [];
  for (const entry of readdirSync(d, { withFileTypes: true })) {
    const full = join(d, entry.name);
    if (entry.isDirectory()) entries.push(...walkDir(full));
    else entries.push(full);
  }
  return entries;
}

async function seedLanceDataset(dir: string): Promise<void> {
  const name = basename(dir);
  console.log(`\nSeeding Lance dataset: ${name}`);
  for (const file of walkDir(dir)) {
    const r2Key = relative(join(dir, ".."), file);
    const data = readFileSync(file);
    console.log(`  PUT ${r2Key} (${data.length} bytes)`);
    uploadFile(r2Key, data);
  }
  const resp = await fetch(`${BASE_URL}/write`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ r2Key: `${name}/` }),
  });
  console.log(`  Master:`, await resp.json());
}

async function seedParquetFile(filePath: string, r2Key?: string): Promise<void> {
  const name = r2Key ?? basename(filePath);
  const data = readFileSync(filePath);
  console.log(`\nSeeding Parquet: ${name} (${(data.length / 1024 / 1024).toFixed(1)} MB)`);
  uploadFile(name, data);
  try {
    const resp = await fetch(`${BASE_URL}/write`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ r2Key: name }),
    });
    if (resp.ok) console.log(`  Master:`, await resp.json());
    else console.log(`  Master: ${resp.status} (will lazy-load)`);
  } catch (err) {
    console.log(`  Master error:`, String(err).slice(0, 100));
  }
}

async function seedIcebergTable(dir: string): Promise<void> {
  const name = basename(dir);
  console.log(`\nSeeding Iceberg table: ${name}`);
  for (const file of walkDir(dir)) {
    const relPath = relative(dir, file);
    const r2Key = `${name}/${relPath}`;
    const data = readFileSync(file);
    console.log(`  PUT ${r2Key} (${(data.length / 1024).toFixed(0)} KB)`);
    uploadFile(r2Key, data);
  }
  // Iceberg tables are auto-discovered by Query DO via loadIcebergFromR2
  console.log(`  (Iceberg tables lazy-load on first query)`);
}

async function main(): Promise<void> {
  // Check Worker is running
  try {
    const health = await fetch(`${BASE_URL}/health`);
    if (!health.ok) throw new Error(`${health.status}`);
    console.log("Worker is running:", await health.json());
  } catch {
    console.error(`Cannot reach Worker at ${BASE_URL}. Is 'pnpm dev' running?`);
    process.exit(1);
  }

  if (!existsSync(FIXTURES)) {
    console.error(`Fixtures not found at ${FIXTURES}`);
    process.exit(1);
  }

  // --- 1. Seed Lance datasets ---
  const lanceDatasets = readdirSync(FIXTURES)
    .filter(f => f.endsWith(".lance"))
    .map(f => join(FIXTURES, f))
    .filter(f => statSync(f).isDirectory());

  for (const dir of lanceDatasets) {
    await seedLanceDataset(dir);
  }

  // --- 2. Seed fixture Parquet files ---
  const parquetFiles = readdirSync(FIXTURES)
    .filter(f => f.endsWith(".parquet"))
    .map(f => join(FIXTURES, f));

  for (const file of parquetFiles) {
    await seedParquetFile(file);
  }

  // --- 3. Seed generated benchmark data ---
  if (existsSync(GENERATED)) {
    const genParquets = readdirSync(GENERATED)
      .filter(f => f.endsWith(".parquet"))
      .map(f => join(GENERATED, f));
    for (const file of genParquets) {
      await seedParquetFile(file);
    }

    // Iceberg tables
    const icebergDirs = readdirSync(GENERATED)
      .filter(f => f.startsWith("bench_iceberg"))
      .map(f => join(GENERATED, f))
      .filter(f => statSync(f).isDirectory());
    for (const dir of icebergDirs) {
      await seedIcebergTable(dir);
    }
  } else {
    console.log("\nNo generated data found. Run: npx tsx scripts/generate-bench-data.ts");
  }

  // --- 4. Register Query DO and re-broadcast ---
  console.log("\nRegistering Query DO with Master...");
  await fetch(`${BASE_URL}/tables`);
  await new Promise(r => setTimeout(r, 1000));

  console.log("Re-broadcasting footers...");
  for (const dir of lanceDatasets) {
    const name = basename(dir);
    await fetch(`${BASE_URL}/write`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ r2Key: `${name}/` }),
    });
  }
  const allParquets = [...parquetFiles.map(f => basename(f))];
  if (existsSync(GENERATED)) {
    allParquets.push(...readdirSync(GENERATED).filter(f => f.endsWith(".parquet")));
  }
  for (const name of allParquets) {
    await fetch(`${BASE_URL}/write`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ r2Key: name }),
    });
  }
  await new Promise(r => setTimeout(r, 500));

  // Verify
  const tablesResp = await fetch(`${BASE_URL}/tables`);
  const { tables } = await tablesResp.json() as { tables: { name: string; totalRows: number }[] };
  console.log(`\nRegistered ${tables.length} tables:`);
  for (const t of tables) console.log(`  ${t.name} (${t.totalRows} rows)`);

  console.log("\nSeeding complete!");
}

main().catch(console.error);
