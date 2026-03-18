/**
 * Formatters for query results and explain plans.
 * Human-readable output for observability and debugging.
 */

import type { QueryResult, ExplainResult } from "./types.js";

/**
 * Timing breakdown from local executor phases.
 * Attached as extra properties on QueryResult when running locally.
 */
export interface LocalTimingInfo {
  /** Time spent reading/decoding pages in ScanOperator (ms). */
  scanMs: number;
  /** Time spent in the operator pipeline after scan (ms). */
  pipelineMs: number;
  /** Time spent loading metadata/footer (ms). */
  metaMs: number;
}

/** Format bytes into a human-readable string. */
export function formatBytes(n: number): string {
  if (n === 0) return "0B";
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)}KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)}MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)}GB`;
}

/**
 * One-line summary of a query result.
 *
 * Example: "20 rows in 3.2ms | 847 pages skipped | 12KB read"
 */
export function formatResultSummary(result: QueryResult & Partial<LocalTimingInfo>): string {
  const parts: string[] = [];

  parts.push(`${result.rowCount} row${result.rowCount !== 1 ? "s" : ""} in ${result.durationMs.toFixed(1)}ms`);

  if (result.pagesSkipped > 0) {
    parts.push(`${result.pagesSkipped} pages skipped`);
  }

  if (result.bytesRead > 0) {
    parts.push(`${formatBytes(result.bytesRead)} read`);
  }

  if (result.cacheHit) {
    parts.push("cache hit");
  }

  if (result.r2ReadMs !== undefined) {
    parts.push(`r2: ${result.r2ReadMs.toFixed(1)}ms`);
  }

  if (result.wasmExecMs !== undefined) {
    parts.push(`wasm: ${result.wasmExecMs.toFixed(1)}ms`);
  }

  if ((result.cacheHits ?? 0) + (result.cacheMisses ?? 0) > 0) {
    parts.push(`L1: ${result.cacheHits ?? 0}/${(result.cacheHits ?? 0) + (result.cacheMisses ?? 0)} hits`);
  }

  if ((result.edgeCacheHits ?? 0) + (result.edgeCacheMisses ?? 0) > 0) {
    parts.push(`L2: ${result.edgeCacheHits ?? 0}/${(result.edgeCacheHits ?? 0) + (result.edgeCacheMisses ?? 0)} hits`);
  }

  if (result.spillBytesWritten && result.spillBytesWritten > 0) {
    parts.push(`${formatBytes(result.spillBytesWritten)} spilled`);
  }

  if (result.scanMs !== undefined) {
    parts.push(`scan: ${result.scanMs.toFixed(1)}ms`);
  }
  if (result.pipelineMs !== undefined) {
    parts.push(`pipeline: ${result.pipelineMs.toFixed(1)}ms`);
  }
  if (result.metaMs !== undefined) {
    parts.push(`meta: ${result.metaMs.toFixed(1)}ms`);
  }

  return parts.join(" | ");
}

/**
 * Format an explain plan into a multi-line readable string.
 */
export function formatExplain(plan: ExplainResult): string {
  const lines: string[] = [];

  lines.push(`Table: ${plan.table} (${plan.format})`);

  const totalRowsFormatted = plan.totalRows.toLocaleString();
  const fragParts = [`Fragments: ${plan.fragments}`];
  if (plan.fragmentsSkipped) fragParts.push(`${plan.fragmentsSkipped} skipped`);
  if (plan.fragmentsScanned != null) fragParts.push(`${plan.fragmentsScanned} scanned`);
  lines.push(`Total rows: ${totalRowsFormatted} | ${fragParts.join(", ")}`);

  lines.push(`Columns: ${plan.columns.length} scanned`);
  for (const col of plan.columns) {
    const name = col.name.padEnd(20);
    const dtype = (col.dtype as string).padEnd(10);
    const pages = `${col.pages} page${col.pages !== 1 ? "s" : ""}`.padEnd(12);
    const bytes = formatBytes(col.bytes);
    lines.push(`  ${name} ${dtype} ${pages} ${bytes}`);
  }

  if (plan.filters.length > 0) {
    lines.push(`Filters: ${plan.filters.length}`);
    for (const f of plan.filters) {
      const pushable = f.pushable ? "[pushable]" : "[not pushable]";
      lines.push(`  ${f.column} ${f.op}  ${pushable}`);
    }
  }

  const skipPct = plan.pagesTotal > 0
    ? ((plan.pagesSkipped / plan.pagesTotal) * 100).toFixed(1)
    : "0.0";
  lines.push(`Pages: ${plan.pagesTotal} total, ${plan.pagesSkipped} skipped (${skipPct}%), ${plan.pagesScanned} scanned`);

  lines.push(`Estimated: ${formatBytes(plan.estimatedBytes)} across ${plan.estimatedR2Reads} read${plan.estimatedR2Reads !== 1 ? "s" : ""}`);

  if (plan.estimatedRows !== plan.totalRows) {
    lines.push(`Estimated rows after pruning: ${plan.estimatedRows.toLocaleString()}`);
  }

  if (plan.partitionCatalog) {
    lines.push(`Partition catalog: ${plan.partitionCatalog.column} (${plan.partitionCatalog.partitionValues} values)`);
  }

  if (plan.fanOut) {
    lines.push(`Fan-out: yes${plan.hierarchicalReduction ? ` (${plan.reducerTiers} reducer tiers)` : ""}`);
  }

  if (plan.metaCached) {
    lines.push("Meta: cached");
  }

  return lines.join("\n");
}
