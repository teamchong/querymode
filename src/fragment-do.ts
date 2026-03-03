import type { ColumnMeta, Env, Footer, TableMeta, QueryResult, Row } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { parseFooter, parseColumnMetaFromProtobuf } from "./footer.js";
import { assembleRows, canSkipPage, bigIntReplacer } from "./decode.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";
import { computePartialAgg, type PartialAgg } from "./partial-agg.js";
import { coalesceRanges } from "./coalesce.js";

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

    this.wasmEngine = await instantiateWasm(this.env.LANCEQL_WASM);
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

      // Coalesce nearby ranges into fewer R2 reads (max 64KB gap to merge)
      const coalesced = coalesceRanges(ranges, 64 * 1024);
      const columnData = new Map<string, ArrayBuffer[]>();
      const fetched = await Promise.all(coalesced.map(async c => {
        const obj = await this.env.DATA_BUCKET.get(r2Key, {
          range: { offset: c.offset, length: c.length },
        });
        return obj ? { ...c, data: await obj.arrayBuffer() } : null;
      }));
      for (const f of fetched) {
        if (!f) continue;
        totalBytesRead += f.data.byteLength;
        for (const sub of f.ranges) {
          const slice = f.data.slice(sub.offset - f.offset, sub.offset - f.offset + sub.length);
          const arr = columnData.get(sub.column) ?? [];
          arr.push(slice);
          columnData.set(sub.column, arr);
        }
      }

      const rows = assembleRows(columnData, cols, query, this.wasmEngine);
      allRows.push(...rows);
    }

    // If query has aggregates, compute partial and return
    if (query.aggregates?.length) {
      const partial = computePartialAgg(allRows, query);
      return this.json({
        partial,
        bytesRead: totalBytesRead,
        pagesSkipped: totalPagesSkipped,
        durationMs: Date.now() - t0,
      });
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
    };
    return this.json(result);
  }
}
