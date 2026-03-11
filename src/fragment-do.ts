import { DurableObject } from "cloudflare:workers";
import type { Env, TableMeta, QueryResult, Row } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { canSkipPage } from "./decode.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";
import { coalesceRanges, autoCoalesceGap, fetchBounded, withRetry, withTimeout } from "./coalesce.js";
import { R2SpillBackend } from "./r2-spill.js";
import {
  type Operator, type RowBatch,
  buildEdgePipeline, drainPipeline,
} from "./operators.js";
import { mergeQueryResults } from "./merge.js";
import { resolveBucket } from "./bucket.js";
import wasmModule from "./wasm-module.js";

const R2_TIMEOUT_MS = 10_000;

interface ScanRequest {
  fragments: { r2Key: string; meta: TableMeta }[];
  query: QueryDescriptor;
}

/**
 * Fragment DO — pooled per-datacenter scanner for TB-scale tables.
 * Named deterministically (frag-{region}-slot-{N}) so they are reused
 * across queries and datasets. Footer cache persists across scans,
 * and the DO hibernates when idle — next query wakes it with cache warm.
 */
export class FragmentDO extends DurableObject<Env> {
  private wasmEngine!: WasmEngine;
  /** Footer cache keyed by r2Key — survives across scans while the DO is alive. */
  private footerCache = new Map<string, TableMeta>();
  private initialized = false;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
  }

  private log(level: "info" | "warn" | "error", msg: string, data?: Record<string, unknown>): void {
    console[level === "error" ? "error" : level === "warn" ? "warn" : "log"](
      JSON.stringify({ ts: new Date().toISOString(), level, msg, ...data }),
    );
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    // Restore footer cache from SQLite (survives hibernation)
    const stored = await this.ctx.storage.list<TableMeta>({ prefix: "frag:" });
    for (const [key, meta] of stored) this.footerCache.set(key.replace("frag:", ""), meta);

    this.wasmEngine = await instantiateWasm(wasmModule);
  }

  /** Core scan logic used by RPC. */
  private async executeScan(fragments: ScanRequest["fragments"], query: QueryDescriptor): Promise<QueryResult> {
    const t0 = Date.now();
    let totalBytesRead = 0;
    let totalPagesSkipped = 0;
    let totalCacheHits = 0;
    let totalCacheMisses = 0;
    let totalR2ReadMs = 0;
    let totalWasmExecMs = 0;
    const allRows: Row[] = [];

    for (const { r2Key, meta } of fragments) {
      // Use cached footer if available, otherwise cache what Query DO sent us
      const cachedMeta = this.footerCache.get(r2Key);
      const effectiveMeta = (cachedMeta && cachedMeta.updatedAt >= meta.updatedAt) ? cachedMeta : meta;
      if (!cachedMeta || cachedMeta.updatedAt < meta.updatedAt) {
        this.footerCache.set(r2Key, meta);
        this.ctx.storage.put(`frag:${r2Key}`, meta);
      }

      let cols = query.projections.length > 0
        ? effectiveMeta.columns.filter(c => query.projections.includes(c.name))
        : effectiveMeta.columns;

      // Ensure vector search column is included even if not in projections
      if (query.vectorSearch) {
        const vc = query.vectorSearch.column;
        if (!cols.some(c => c.name === vc)) {
          const ec = effectiveMeta.columns.find(c => c.name === vc);
          if (ec) cols = [...cols, ec];
        }
      }

      // Build byte ranges for each page, skipping uniformly across all columns.
      // A page is skipped only if any AND filter eliminates it — same for all columns.
      const maxPages = cols.reduce((m, c) => Math.max(m, c.pages.length), 0);
      const keptPageIndices: number[] = [];
      for (let pi = 0; pi < maxPages; pi++) {
        let skip = false;
        if (!query.vectorSearch) {
          for (const f of query.filters) {
            const col = cols.find(c => c.name === f.column);
            if (!col) continue;
            const page = col.pages[pi];
            if (!page) continue;
            if (canSkipPage(page, [f], f.column)) { skip = true; break; }
          }
        }
        if (skip) { totalPagesSkipped += cols.length; continue; }
        keptPageIndices.push(pi);
      }

      const ranges: { column: string; offset: number; length: number }[] = [];
      for (const col of cols) {
        for (const pi of keptPageIndices) {
          const page = col.pages[pi];
          if (!page) continue;
          ranges.push({ column: col.name, offset: Number(page.byteOffset), length: page.byteLength });
        }
      }

      // Cache-before-fetch: check WASM buffer pool for each range.
      // Track results by index to preserve page ordering when cache is partially warm.
      const rangeResults = new Array<ArrayBuffer | null>(ranges.length).fill(null);
      const uncachedRanges: { range: typeof ranges[0]; idx: number }[] = [];

      for (let ri = 0; ri < ranges.length; ri++) {
        const r = ranges[ri];
        const cacheKey = `${r2Key}:${r.offset}:${r.length}`;
        const cached = this.wasmEngine.cacheGet(cacheKey);
        if (cached) {
          totalCacheHits++;
          rangeResults[ri] = cached;
        } else {
          totalCacheMisses++;
          uncachedRanges.push({ range: r, idx: ri });
        }
      }

      // Fetch uncached ranges from R2 with retry + timeout
      const r2Start = Date.now();
      if (uncachedRanges.length > 0) {
        const fetchRanges = uncachedRanges.map(u => u.range);
        const coalesced = coalesceRanges(fetchRanges, autoCoalesceGap(fetchRanges));
        const fetched = await fetchBounded(
          coalesced.map(c => () =>
            withRetry(() =>
              withTimeout(
                (async () => {
                  const obj = await resolveBucket(this.env, r2Key).get(r2Key, {
                    range: { offset: c.offset, length: c.length },
                  });
                  return obj ? { ...c, data: await obj.arrayBuffer() } : null;
                })(),
                R2_TIMEOUT_MS,
              ),
            ),
          ),
          8,
        );
        // Build a lookup from (column, offset, length) → fetched slice
        const fetchedMap = new Map<string, ArrayBuffer>();
        for (const f of fetched) {
          if (!f) continue;
          totalBytesRead += f.data.byteLength;
          for (const sub of f.ranges) {
            const slice = f.data.slice(sub.offset - f.offset, sub.offset - f.offset + sub.length);
            fetchedMap.set(`${sub.column}:${sub.offset}:${sub.length}`, slice);
            const cacheKey = `${r2Key}:${sub.offset}:${sub.length}`;
            this.wasmEngine.cacheSet(cacheKey, slice);
          }
        }
        // Fill in uncached slots in order
        for (const u of uncachedRanges) {
          const key = `${u.range.column}:${u.range.offset}:${u.range.length}`;
          rangeResults[u.idx] = fetchedMap.get(key) ?? null;
        }
      }

      // Assemble columnData in correct page order
      const columnData = new Map<string, ArrayBuffer[]>();
      for (let ri = 0; ri < ranges.length; ri++) {
        const buf = rangeResults[ri];
        if (!buf) continue;
        const arr = columnData.get(ranges[ri].column) ?? [];
        arr.push(buf);
        columnData.set(ranges[ri].column, arr);
      }
      totalR2ReadMs += Date.now() - r2Start;

      // Zero-copy WASM path: register raw page data per-column, execute SQL in WASM
      const wasmStart = Date.now();
      const fragTable = `__frag_${r2Key}`;
      this.wasmEngine.exports.resetHeap();
      const fragColEntries = cols
        .filter(col => columnData.get(col.name)?.length)
        .map(col => ({
          name: col.name, dtype: col.dtype, listDim: col.listDimension,
          pages: columnData.get(col.name)!,
          pageInfos: keptPageIndices.map(pi => col.pages[pi]).filter(Boolean),
        }));
      if (!this.wasmEngine.registerColumns(fragTable, fragColEntries)) {
        throw new Error(`WASM OOM: failed to register columns for fragment "${r2Key}"`);
      }

      const fragQuery = { ...query, table: fragTable };
      const rows = this.wasmEngine.executeQuery(fragQuery);
      if (!rows) throw new Error(`WASM query execution failed for fragment "${r2Key}"`);
      this.wasmEngine.clearTable(fragTable);
      for (let ri = 0; ri < rows.length; ri++) allRows.push(rows[ri]);
      totalWasmExecMs += Date.now() - wasmStart;
    }

    // If query needs cross-page orchestration (ORDER BY without LIMIT),
    // run through operator pipeline with R2 spill using streaming batches
    const needsPipeline = query.sortColumn && query.limit === undefined;
    let finalRows = allRows;
    let spillBytesWritten: number | undefined;
    let spillBytesRead: number | undefined;

    if (needsPipeline && allRows.length > 0) {
      const spillBucket = fragments.length > 0 ? resolveBucket(this.env, fragments[0].r2Key) : resolveBucket(this.env, "");
      const spill = new R2SpillBackend(spillBucket, `__spill/${crypto.randomUUID()}`);
      try {
        // Yield allRows in bounded batches to avoid holding 2x copies
        const BATCH_SIZE = 4096;
        let batchIdx = 0;
        const batchSource: Operator = {
          async next(): Promise<RowBatch | null> {
            if (batchIdx >= allRows.length) return null;
            const end = Math.min(batchIdx + BATCH_SIZE, allRows.length);
            const batch = allRows.slice(batchIdx, end);
            batchIdx = end;
            return batch;
          },
          async close() {},
        };

        const outputColumns = query.projections.length > 0
          ? query.projections
          : (fragments[0]?.meta.columns.map(c => c.name) ?? []);

        const FRAG_MEMORY_BUDGET = 32 * 1024 * 1024;
        const pipeline = buildEdgePipeline(batchSource, query, outputColumns, {
          memoryBudgetBytes: FRAG_MEMORY_BUDGET,
          spill,
        });
        finalRows = await drainPipeline(pipeline);
        spillBytesWritten = spill.bytesWritten || undefined;
        spillBytesRead = spill.bytesRead || undefined;
      } finally {
        await spill.cleanup();
      }
    }

    return {
      rows: finalRows,
      rowCount: finalRows.length,
      columns: query.projections.length > 0
        ? query.projections
        : (fragments[0]?.meta.columns.map(c => c.name) ?? []),
      bytesRead: totalBytesRead,
      pagesSkipped: totalPagesSkipped,
      durationMs: Date.now() - t0,
      r2ReadMs: totalR2ReadMs,
      wasmExecMs: totalWasmExecMs,
      cacheHits: totalCacheHits,
      cacheMisses: totalCacheMisses,
      spillBytesWritten,
      spillBytesRead,
    };
  }

  /** RPC: Scan fragments — zero-serialization call from QueryDO. */
  async scanRpc(fragments: ScanRequest["fragments"], query: QueryDescriptor): Promise<QueryResult> {
    await this.ensureInitialized();
    return this.executeScan(fragments, query);
  }

  /** RPC: Reduce (merge) partial results from other Fragment DOs.
   *  Used by hierarchical reduction — QueryDO fans out scans to leaf DOs,
   *  then groups their results and sends each group to a reducer DO. */
  async reduceRpc(partials: QueryResult[], query: QueryDescriptor): Promise<QueryResult> {
    return mergeQueryResults(partials, query);
  }
}
