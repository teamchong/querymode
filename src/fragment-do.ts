import type { Env, TableMeta, QueryResult, Row } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { canSkipPage, bigIntReplacer } from "./decode.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";
import { coalesceRanges, fetchBounded, withRetry, withTimeout } from "./coalesce.js";
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
export class FragmentDO implements DurableObject {
  private state: DurableObjectState;
  private env: Env;
  private wasmEngine!: WasmEngine;
  /** Footer cache keyed by r2Key — survives across scans while the DO is alive. */
  private footerCache = new Map<string, TableMeta>();
  private initialized = false;

  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
  }

  private log(level: "info" | "warn" | "error", msg: string, data?: Record<string, unknown>): void {
    console[level === "error" ? "error" : level === "warn" ? "warn" : "log"](
      JSON.stringify({ ts: new Date().toISOString(), level, msg, ...data }),
    );
  }

  async fetch(request: Request): Promise<Response> {
    await this.ensureInitialized();

    const url = new URL(request.url);
    if (url.pathname === "/scan") return this.handleScan(request);
    return new Response("Not found", { status: 404 });
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    // Restore footer cache from SQLite (survives hibernation)
    const stored = await this.state.storage.list<TableMeta>({ prefix: "frag:" });
    for (const [key, meta] of stored) this.footerCache.set(key.replace("frag:", ""), meta);

    this.wasmEngine = await instantiateWasm(wasmModule);
  }

  private json(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body, bigIntReplacer), {
      status, headers: { "content-type": "application/json" },
    });
  }

  private async handleScan(request: Request): Promise<Response> {
    const { fragments, query } = (await request.json()) as ScanRequest;
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
        this.state.storage.put(`frag:${r2Key}`, meta);
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

      // Build byte ranges for each page, skipping via min/max stats
      const ranges: { column: string; offset: number; length: number }[] = [];
      for (const col of cols) {
        for (const page of col.pages) {
          if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) {
            totalPagesSkipped++;
            continue;
          }
          ranges.push({ column: col.name, offset: Number(page.byteOffset), length: page.byteLength });
        }
      }

      // Cache-before-fetch: check WASM buffer pool for each range
      const columnData = new Map<string, ArrayBuffer[]>();
      const uncachedRanges: typeof ranges = [];

      for (const r of ranges) {
        const cacheKey = `${r2Key}:${r.offset}:${r.length}`;
        const cached = this.wasmEngine.cacheGet(cacheKey);
        if (cached) {
          totalCacheHits++;
          const arr = columnData.get(r.column) ?? [];
          arr.push(cached);
          columnData.set(r.column, arr);
        } else {
          totalCacheMisses++;
          uncachedRanges.push(r);
        }
      }

      // Fetch uncached ranges from R2 with retry + timeout
      const r2Start = Date.now();
      if (uncachedRanges.length > 0) {
        const coalesced = coalesceRanges(uncachedRanges, 64 * 1024);
        const fetched = await fetchBounded(
          coalesced.map(c => () =>
            withRetry(() =>
              withTimeout(
                (async () => {
                  const obj = await this.env.DATA_BUCKET.get(r2Key, {
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
        for (const f of fetched) {
          if (!f) continue;
          totalBytesRead += f.data.byteLength;
          for (const sub of f.ranges) {
            const slice = f.data.slice(sub.offset - f.offset, sub.offset - f.offset + sub.length);
            const arr = columnData.get(sub.column) ?? [];
            arr.push(slice);
            columnData.set(sub.column, arr);

            // Populate cache for next time
            const cacheKey = `${r2Key}:${sub.offset}:${sub.length}`;
            this.wasmEngine.cacheSet(cacheKey, slice);
          }
        }
      }
      totalR2ReadMs += Date.now() - r2Start;

      // Zero-copy WASM path: register raw page data per-column, execute SQL in WASM
      const wasmStart = Date.now();
      const fragTable = `__frag_${r2Key}`;
      this.wasmEngine.exports.resetHeap();
      for (const col of cols) {
        const pages = columnData.get(col.name);
        if (!pages?.length) continue;
        if (!this.wasmEngine.registerColumn(fragTable, col.name, col.dtype, pages, col.pages, col.listDimension)) {
          throw new Error(`WASM OOM: failed to register column "${col.name}" for fragment "${r2Key}"`);
        }
      }

      const fragQuery = { ...query, table: fragTable };
      const rows = this.wasmEngine.executeQuery(fragQuery);
      if (!rows) throw new Error(`WASM query execution failed for fragment "${r2Key}"`);
      this.wasmEngine.clearTable(fragTable);
      allRows.push(...rows);
      totalWasmExecMs += Date.now() - wasmStart;
    }

    const result: QueryResult = {
      rows: allRows,
      rowCount: allRows.length,
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
    };

    this.log("info", "scan_complete", {
      fragmentCount: fragments.length, rowCount: result.rowCount,
      bytesRead: totalBytesRead, durationMs: result.durationMs,
      r2ReadMs: totalR2ReadMs, wasmExecMs: totalWasmExecMs,
      cacheHits: totalCacheHits, cacheMisses: totalCacheMisses,
    });

    return this.json(result);
  }
}
