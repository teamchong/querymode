import type { ColumnMeta, Env, Footer, Row, TableMeta, DatasetMeta, QueryResult } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { parseFooter, parseColumnMetaFromProtobuf } from "./footer.js";
import { parseManifest } from "./manifest.js";
import { assembleRows, canSkipPage, bigIntReplacer } from "./decode.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";
import { mergeQueryResults } from "./merge.js";
import { coalesceRanges } from "./coalesce.js";

const FRAGMENT_POOL_MAX = 20; // Max Fragment DO slots per datacenter (idle slots cost nothing)

/**
 * Query DO — per-region reader with cached footers.
 * WASM (Zig SIMD) handles all compute; TS handles I/O orchestration only.
 */
export class QueryDO implements DurableObject {
  private state: DurableObjectState;
  private env: Env;
  private footerCache = new Map<string, TableMeta>();
  private datasetCache = new Map<string, DatasetMeta>();
  private wasmEngine!: WasmEngine;
  private queryCount = new Map<string, number>(); // table → hit count (for VIP pinning)
  private activeFragmentSlots = new Set<number>(); // slots currently scanning
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
      case "/invalidate":   return this.handleInvalidation(request);
      case "/query":        return this.handleQuery(request);
      case "/query/stream": return this.handleQueryStream(request);
      case "/tables":       return this.handleListTables();
      case "/meta":         return this.handleGetMeta(request);
      default:              return new Response("Not found", { status: 404 });
    }
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    const stored = await this.state.storage.list<TableMeta>({ prefix: "table:" });
    for (const [key, meta] of stored) this.footerCache.set(key.replace("table:", ""), meta);

    this.wasmEngine = await instantiateWasm(this.env.LANCEQL_WASM);

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
      })).then(async (resp) => {
        this.registeredWithMaster = true;
        // Master returns current table timestamps — refresh any stale entries
        // This closes the gap where broadcasts were missed during hibernation
        const body = await resp.json() as {
          registered: boolean; region: string;
          tableVersions?: Record<string, { r2Key: string; updatedAt: number }>;
        };
        if (body.tableVersions) {
          await this.refreshStaleTables(body.tableVersions);
        }
      }).catch(() => {
        console.warn("Failed to register with Master DO, will retry on next request");
      });
    });
  }

  /** Compare Master's table timestamps against local cache, refresh any that are stale. */
  private async refreshStaleTables(
    masterVersions: Record<string, { r2Key: string; updatedAt: number }>,
  ): Promise<void> {
    for (const [table, { r2Key, updatedAt }] of Object.entries(masterVersions)) {
      const cached = this.footerCache.get(table);
      if (cached && cached.updatedAt >= updatedAt) continue;

      // Stale or missing — re-read footer from R2
      const head = await this.env.DATA_BUCKET.head(r2Key);
      if (!head) continue;

      const fileSize = BigInt(head.size);
      const obj = await this.env.DATA_BUCKET.get(r2Key, { range: { offset: Number(fileSize) - 40, length: 40 } });
      if (!obj) continue;

      const footer = parseFooter(await obj.arrayBuffer());
      if (!footer) continue;

      const columns = await this.readColumnMeta(r2Key, footer);
      const meta: TableMeta = {
        name: table, footer, columns,
        totalRows: columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0,
        fileSize, r2Key, updatedAt,
      };
      this.footerCache.set(table, meta);

      this.wasmEngine.clearTable(table);
      await this.state.storage.put(`table:${table}`, meta);
    }
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
    this.wasmEngine.clearTable(body.table);
    await this.state.storage.put(`table:${body.table}`, meta);
    return this.json({ updated: true, table: body.table });
  }

  private async handleQuery(request: Request): Promise<Response> {
    const query = (await request.json()) as QueryDescriptor;
    return this.json(await this.executeQuery(query));
  }

  /** Execute query using page-level R2 Range reads + WASM compute. Never downloads full files. */
  private async executeQuery(query: QueryDescriptor): Promise<QueryResult> {
    const t0 = Date.now();

    // Multi-fragment dataset path
    const dataset = this.datasetCache.get(query.table);
    if (dataset) return this.executeMultiFragment(query, dataset, t0);

    let meta: TableMeta | undefined = this.footerCache.get(query.table);
    if (!meta) {
      meta = (await this.loadTableFromR2(query.table)) ?? undefined;
      if (!meta) throw new Error(`Table "${query.table}" not found`);
    }

    this.queryCount.set(query.table, (this.queryCount.get(query.table) ?? 0) + 1);
    return this.scanPages(query, meta, t0);
  }

  /** Scan only the needed pages from R2 via coalesced Range reads. */
  private async scanPages(query: QueryDescriptor, meta: TableMeta, t0: number): Promise<QueryResult> {
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

    // Try as Lance dataset directory (has _versions/ with manifests)
    const datasetMeta = await this.loadDatasetFromR2(tableName);
    if (datasetMeta) return datasetMeta.fragmentMetas.values().next().value ?? null;
    return null;
  }

  /** Discover a multi-fragment Lance dataset in R2 by listing _versions/ manifests. */
  private async loadDatasetFromR2(tableName: string): Promise<DatasetMeta | null> {
    for (const prefix of [`${tableName}.lance/`, `${tableName}/`, `data/${tableName}.lance/`, `data/${tableName}/`]) {
      const listed = await this.env.DATA_BUCKET.list({ prefix: `${prefix}_versions/`, limit: 100 });
      const manifestKeys = listed.objects
        .filter(o => o.key.endsWith(".manifest"))
        .sort((a, b) => a.key.localeCompare(b.key));
      if (manifestKeys.length === 0) continue;

      // Read latest manifest
      const latestKey = manifestKeys[manifestKeys.length - 1].key;
      const manifestObj = await this.env.DATA_BUCKET.get(latestKey);
      if (!manifestObj) continue;

      const manifest = parseManifest(await manifestObj.arrayBuffer());
      if (!manifest) continue;

      // Read footer + columns for each fragment
      const fragmentMetas = new Map<number, TableMeta>();
      for (const frag of manifest.fragments) {
        const fragKey = `${prefix}${frag.filePath}`;
        const head = await this.env.DATA_BUCKET.head(fragKey);
        if (!head) continue;

        const fileSize = BigInt(head.size);
        const footerObj = await this.env.DATA_BUCKET.get(fragKey, { range: { offset: Number(fileSize) - 40, length: 40 } });
        if (!footerObj) continue;

        const footer = parseFooter(await footerObj.arrayBuffer());
        if (!footer) continue;

        const columns = await this.readColumnMeta(fragKey, footer);
        fragmentMetas.set(frag.id, {
          name: frag.filePath, footer, columns,
          totalRows: frag.physicalRows,
          fileSize, r2Key: fragKey, updatedAt: Date.now(),
        });
      }

      if (fragmentMetas.size === 0) continue;

      const dataset: DatasetMeta = {
        name: tableName, r2Prefix: prefix, manifest,
        fragmentMetas, totalRows: manifest.totalRows, updatedAt: Date.now(),
      };
      this.datasetCache.set(tableName, dataset);
      return dataset;
    }
    return null;
  }

  /** Execute a query across multiple fragments of a dataset. */
  private async executeMultiFragment(query: QueryDescriptor, dataset: DatasetMeta, t0: number): Promise<QueryResult> {
    const fragments = [...dataset.fragmentMetas.values()];
    // Large datasets (>4 fragments): fan out to Fragment DOs
    if (fragments.length > 4) {
      return this.executeWithFragmentDOs(query, dataset, t0);
    }

    // Small datasets: scan locally
    const partials: QueryResult[] = [];
    for (const meta of fragments) {
      partials.push(await this.scanPages(query, meta, t0));
    }
    return mergeQueryResults(partials, query);
  }

  /** Claim N available slots from the Fragment DO pool. Returns slot indices.
   *  Slots are per-datacenter, named frag-{region}-slot-{N}.
   *  Idle slots cost nothing (hibernated DOs). Active tracking prevents
   *  concurrent queries from queueing behind each other on the same slot. */
  private claimSlots(needed: number): number[] | null {
    const slots: number[] = [];
    for (let i = 0; i < FRAGMENT_POOL_MAX && slots.length < needed; i++) {
      if (!this.activeFragmentSlots.has(i)) slots.push(i);
    }
    if (slots.length < needed) return null; // not enough capacity
    for (const s of slots) this.activeFragmentSlots.add(s);
    return slots;
  }

  private releaseSlots(slots: number[]): void {
    for (const s of slots) this.activeFragmentSlots.delete(s);
  }

  /** Fan-out query to pooled Fragment DOs for parallel scanning.
   *  Dynamically claims available slots to avoid queueing behind concurrent queries.
   *  Pool is per-datacenter (frag-{region}-slot-{N}), max FRAGMENT_POOL_MAX slots. */
  private async executeWithFragmentDOs(query: QueryDescriptor, dataset: DatasetMeta, t0: number): Promise<QueryResult> {
    const fragments = [...dataset.fragmentMetas.entries()];
    const slotsNeeded = Math.min(fragments.length, FRAGMENT_POOL_MAX - this.activeFragmentSlots.size);

    if (slotsNeeded <= 0) {
      // All slots busy — scan locally rather than queueing
      const partials: QueryResult[] = [];
      for (const [, meta] of fragments) {
        partials.push(await this.scanPages(query, meta, t0));
      }
      return mergeQueryResults(partials, query);
    }

    const slots = this.claimSlots(slotsNeeded)!;
    const chunkSize = Math.ceil(fragments.length / slots.length);

    // Partition fragments across claimed slots
    const groups: { r2Key: string; meta: TableMeta }[][] = [];
    for (let i = 0; i < fragments.length; i += chunkSize) {
      groups.push(fragments.slice(i, i + chunkSize).map(([, meta]) => ({
        r2Key: meta.r2Key, meta,
      })));
    }

    const region = (await this.state.storage.get<string>("region")) ?? "default";

    try {
      const results = await Promise.all(groups.map(async (group, idx) => {
        const doName = `frag-${region}-slot-${slots[idx]}`;
        const doId = this.env.FRAGMENT_DO.idFromName(doName);
        const fragmentDo = this.env.FRAGMENT_DO.get(doId);

        const resp = await fragmentDo.fetch(new Request("http://internal/scan", {
          method: "POST",
          body: JSON.stringify({ fragments: group, query }, bigIntReplacer),
          headers: { "content-type": "application/json" },
        }));

        return resp.json() as Promise<QueryResult>;
      }));

      const merged = mergeQueryResults(results, query);
      merged.durationMs = Date.now() - t0;
      return merged;
    } finally {
      this.releaseSlots(slots);
    }
  }

  /** Stream query results as NDJSON. */
  private async handleQueryStream(request: Request): Promise<Response> {
    const query = (await request.json()) as QueryDescriptor;
    const result = await this.executeQuery(query);

    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();

    (async () => {
      try {
        for (const row of result.rows) {
          await writer.write(encoder.encode(JSON.stringify(row, bigIntReplacer) + "\n"));
        }
      } finally {
        await writer.close();
      }
    })();

    return new Response(readable, {
      headers: { "content-type": "application/x-ndjson" },
    });
  }

}

