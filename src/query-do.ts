import type { ColumnMeta, Env, Footer, Row, TableMeta, QueryResult } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { parseFooter, parseColumnMetaFromProtobuf } from "./footer.js";
import { assembleRows, canSkipPage, bigIntReplacer } from "./decode.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";

const MAX_WASM_FILE = 64 * 1024 * 1024; // 64MB — stay within 128MB DO limit

/**
 * Query DO — per-region reader with cached footers.
 * WASM (Zig SIMD) handles all compute; TS handles orchestration.
 * Falls back to TS decode if WASM unavailable.
 */
export class QueryDO implements DurableObject {
  private state: DurableObjectState;
  private env: Env;
  private footerCache = new Map<string, TableMeta>();
  private wasmEngine: WasmEngine | null = null;
  private wasmLoaded = new Map<string, number>(); // table → updatedAt
  private queryCount = new Map<string, number>(); // table → hit count (for VIP pinning)
  private initialized = false;
  private registeredWithMaster = false;

  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
  }

  private json(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body, bigIntReplacer), {
      status, headers: { "content-type": "application/json" },
    });
  }

  async fetch(request: Request): Promise<Response> {
    const region = request.headers.get("x-edgeq-region");
    if (region) {
      const stored = await this.state.storage.get<string>("region");
      if (!stored) await this.state.storage.put("region", region);
    }
    await this.ensureInitialized();
    if (!this.registeredWithMaster) this.registerWithMaster();

    switch (new URL(request.url).pathname) {
      case "/invalidate": return this.handleInvalidation(request);
      case "/query":      return this.handleQuery(request);
      case "/tables":     return this.handleListTables();
      case "/meta":       return this.handleGetMeta(request);
      default:            return new Response("Not found", { status: 404 });
    }
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    const stored = await this.state.storage.list<TableMeta>({ prefix: "table:" });
    for (const [key, meta] of stored) this.footerCache.set(key.replace("table:", ""), meta);

    if (this.env.LANCEQL_WASM) {
      try { this.wasmEngine = await instantiateWasm(this.env.LANCEQL_WASM); }
      catch { /* TS fallback */ }
    }

    // Register with Master for invalidation broadcasts
    this.registerWithMaster();
  }

  private registerWithMaster(): void {
    const master = this.env.MASTER_DO.get(this.env.MASTER_DO.idFromName("master"));
    this.state.storage.get<string>("region").then(region => {
      master.fetch(new Request("http://internal/register", {
        method: "POST",
        body: JSON.stringify({ queryDoId: this.state.id.toString(), region: region ?? "unknown" }),
        headers: { "content-type": "application/json" },
      })).then(() => {
        this.registeredWithMaster = true;
      }).catch(() => {
        console.warn("Failed to register with Master DO, will retry on next request");
      });
    });
  }

  private async handleInvalidation(request: Request): Promise<Response> {
    const body = (await request.json()) as {
      table: string; r2Key: string; footerBytes: number[];
      columns?: ColumnMeta[]; fileSize?: string; timestamp: number;
    };

    const footer = parseFooter(new Uint8Array(body.footerBytes).buffer);
    if (!footer) return this.json({ error: "Invalid footer" }, 400);

    const columns = body.columns ?? await this.readColumnMeta(body.r2Key, footer);
    const meta: TableMeta = {
      name: body.table, footer, columns,
      totalRows: columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0,
      fileSize: body.fileSize ? BigInt(body.fileSize) : 0n,
      r2Key: body.r2Key, updatedAt: body.timestamp,
    };

    this.footerCache.set(body.table, meta);
    if (this.wasmEngine && this.wasmLoaded.has(body.table)) {
      this.wasmEngine.clearTable(body.table);
      this.wasmLoaded.delete(body.table);
    }
    await this.state.storage.put(`table:${body.table}`, meta);
    return this.json({ updated: true, table: body.table });
  }

  private async handleQuery(request: Request): Promise<Response> {
    const query = (await request.json()) as QueryDescriptor;
    const t0 = Date.now();

    let meta: TableMeta | undefined = this.footerCache.get(query.table);
    if (!meta) {
      meta = (await this.loadTableFromR2(query.table)) ?? undefined;
      if (!meta) return this.json({ error: `Table "${query.table}" not found` }, 404);
    }

    // Track query frequency for VIP pinning (zell-inspired)
    this.queryCount.set(query.table, (this.queryCount.get(query.table) ?? 0) + 1);

    // WASM path — skip R2 fetch if table already loaded at current version
    if (this.wasmEngine && Number(meta.fileSize) <= MAX_WASM_FILE) {
      try {
        const rows = await this.executeWasm(query, meta);
        if (rows) {
          return this.json({
            rows, rowCount: rows.length,
            columns: query.projections.length > 0 ? query.projections : meta.columns.map(c => c.name),
            bytesRead: Number(meta.fileSize), pagesSkipped: 0, durationMs: Date.now() - t0,
          });
        }
      } catch (err) {
        const wasmErr = this.wasmEngine.getLastError();
        console.warn("WASM failed:", wasmErr ?? err);
        this.wasmEngine.reset();
      }
    }

    // TS fallback
    return this.json(await this.executeTsFallback(query, meta, t0));
  }

  private async executeWasm(query: QueryDescriptor, meta: TableMeta): Promise<Row[] | null> {
    if (this.wasmLoaded.get(query.table) !== meta.updatedAt) {
      // Evict coldest table if we're loading a new one and have many loaded
      if (this.wasmLoaded.size >= 8) this.evictColdestTable(query.table);

      const fileSize = Number(meta.fileSize);
      const obj = fileSize > 0
        ? await this.env.DATA_BUCKET.get(meta.r2Key, { range: { offset: 0, length: fileSize } })
        : await this.env.DATA_BUCKET.get(meta.r2Key);
      if (!obj) return null;
      if (!this.wasmEngine!.loadTable(query.table, await obj.arrayBuffer())) return null;
      this.wasmLoaded.set(query.table, meta.updatedAt);
    }
    return this.wasmEngine!.executeQuery(query);
  }

  /** Evict the least-queried table from WASM memory (VIP pinning — zell-inspired). */
  private evictColdestTable(exclude: string): void {
    let coldest: string | null = null;
    let minHits = Infinity;
    for (const [table] of this.wasmLoaded) {
      if (table === exclude) continue;
      const hits = this.queryCount.get(table) ?? 0;
      if (hits < minHits) { coldest = table; minHits = hits; }
    }
    if (coldest) {
      this.wasmEngine!.clearTable(coldest);
      this.wasmLoaded.delete(coldest);
    }
  }

  private async executeTsFallback(query: QueryDescriptor, meta: TableMeta, t0: number): Promise<QueryResult> {
    let cols = query.projections.length > 0
      ? meta.columns.filter(c => query.projections.includes(c.name))
      : meta.columns;

    if (query.vectorSearch) {
      const vc = query.vectorSearch.column;
      if (!cols.some(c => c.name === vc)) {
        const ec = meta.columns.find(c => c.name === vc);
        if (ec) cols = [...cols, ec];
      }
    }

    const ranges: { column: string; offset: number; length: number }[] = [];
    let pagesSkipped = 0;
    for (const col of cols) {
      for (const page of col.pages) {
        if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) { pagesSkipped++; continue; }
        ranges.push({ column: col.name, offset: Number(page.byteOffset), length: page.byteLength });
      }
    }

    // Coalesce nearby ranges into fewer R2 reads (max 64KB gap to merge)
    const coalesced = coalesceRanges(ranges, 64 * 1024);
    const fetched = await Promise.all(coalesced.map(async c => {
      const obj = await this.env.DATA_BUCKET.get(meta.r2Key, { range: { offset: c.offset, length: c.length } });
      return obj ? { ...c, data: await obj.arrayBuffer() } : null;
    }));

    const columnData = new Map<string, ArrayBuffer[]>();
    let bytesRead = 0;
    for (const f of fetched) {
      if (!f) continue;
      bytesRead += f.data.byteLength;
      // Slice coalesced buffer back into per-range pieces
      for (const sub of f.ranges) {
        const slice = f.data.slice(sub.offset - f.offset, sub.offset - f.offset + sub.length);
        const arr = columnData.get(sub.column) ?? [];
        arr.push(slice);
        columnData.set(sub.column, arr);
      }
    }

    const rows = assembleRows(columnData, cols, query, this.wasmEngine);
    return {
      rows, rowCount: rows.length, columns: cols.map(c => c.name),
      bytesRead, pagesSkipped, durationMs: Date.now() - t0,
    };
  }

  private handleListTables(): Response {
    const tables = [...this.footerCache.entries()].map(([name, meta]) => ({
      name, columns: meta.columns.map(c => c.name), totalRows: meta.totalRows, updatedAt: meta.updatedAt,
    }));
    return this.json({ tables });
  }

  private handleGetMeta(request: Request): Response {
    const table = new URL(request.url).searchParams.get("table");
    if (!table) return this.json({ error: "Missing ?table= parameter" }, 400);
    const meta = this.footerCache.get(table);
    if (!meta) return this.json({ error: `Table "${table}" not found` }, 404);
    return this.json(meta);
  }

  private async readColumnMeta(r2Key: string, footer: Footer): Promise<ColumnMeta[]> {
    const len = Number(footer.columnMetaOffsetsStart) - Number(footer.columnMetaStart);
    if (len <= 0) return [];
    const obj = await this.env.DATA_BUCKET.get(r2Key, { range: { offset: Number(footer.columnMetaStart), length: len } });
    if (!obj) return [];
    return parseColumnMetaFromProtobuf(await obj.arrayBuffer(), footer.numColumns);
  }

  private async loadTableFromR2(tableName: string): Promise<TableMeta | null> {
    for (const r2Key of [`${tableName}.lance`, tableName, `data/${tableName}.lance`, `data/${tableName}`]) {
      const head = await this.env.DATA_BUCKET.head(r2Key);
      if (!head) continue;

      const fileSize = BigInt(head.size);
      const obj = await this.env.DATA_BUCKET.get(r2Key, { range: { offset: Number(fileSize) - 40, length: 40 } });
      if (!obj) continue;

      const footer = parseFooter(await obj.arrayBuffer());
      if (!footer) continue;

      const columns = await this.readColumnMeta(r2Key, footer);
      const meta: TableMeta = {
        name: tableName, footer, columns,
        totalRows: columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0,
        fileSize, r2Key, updatedAt: Date.now(),
      };
      this.footerCache.set(tableName, meta);
      await this.state.storage.put(`table:${tableName}`, meta);
      return meta;
    }
    return null;
  }
}

type Range = { column: string; offset: number; length: number };
type CoalescedRange = { offset: number; length: number; ranges: Range[] };

/** Merge nearby byte ranges into fewer R2 reads. Sorts by offset, merges if gap <= maxGap. */
function coalesceRanges(ranges: Range[], maxGap: number): CoalescedRange[] {
  if (ranges.length === 0) return [];
  const sorted = [...ranges].sort((a, b) => a.offset - b.offset);
  const result: CoalescedRange[] = [];
  let cur: CoalescedRange = { offset: sorted[0].offset, length: sorted[0].length, ranges: [sorted[0]] };

  for (let i = 1; i < sorted.length; i++) {
    const r = sorted[i];
    const curEnd = cur.offset + cur.length;
    if (r.offset <= curEnd + maxGap) {
      cur.length = Math.max(curEnd, r.offset + r.length) - cur.offset;
      cur.ranges.push(r);
    } else {
      result.push(cur);
      cur = { offset: r.offset, length: r.length, ranges: [r] };
    }
  }
  result.push(cur);
  return result;
}
