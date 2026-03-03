import { TableQuery } from "./client.js";
import type { QueryDescriptor, QueryExecutor } from "./client.js";
import type { AppendResult, ColumnMeta, Env, QueryResult, Row, TableMeta, DatasetMeta } from "./types.js";
import { parseFooter, parseColumnMetaFromProtobuf, FOOTER_SIZE } from "./footer.js";
import { parseManifest, type ManifestInfo } from "./manifest.js";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";
import { assembleRows, canSkipPage, bigIntReplacer } from "./decode.js";
import { instantiateWasm, WasmEngine } from "./wasm-engine.js";

export { MasterDO } from "./master-do.js";
export { QueryDO } from "./query-do.js";
export { FragmentDO } from "./fragment-do.js";
export { TableQuery } from "./client.js";
export type { QueryExecutor, QueryDescriptor } from "./client.js";
export type {
  Env,
  Footer,
  TableMeta,
  ColumnMeta,
  PageInfo,
  PageEncoding,
  DataType,
  DatasetMeta,
  ManifestInfo,
  FragmentInfo,
  FilterOp,
  AggregateOp,
  QueryResult,
  Row,
  VectorSearchParams,
  IcebergSchema,
  IcebergDatasetMeta,
  AppendResult,
  VectorIndexInfo,
} from "./types.js";

/**
 * EdgeQ — serverless columnar query engine.
 *
 * Usage:
 *   // Edge mode (Durable Objects + R2)
 *   const eq = EdgeQ.remote(env.QUERY_DO, { region: "SJC" })
 *   const results = await eq.table("users").filter("age", "gt", 25).exec()
 *
 *   // Local mode (Node/Bun + filesystem)
 *   const eq = EdgeQ.local()
 *   const results = await eq.table("./data/users.lance").select("name").exec()
 */
export class EdgeQ {
  private executor: QueryExecutor;

  private constructor(executor: QueryExecutor) {
    this.executor = executor;
  }

  /**
   * Create an EdgeQ client backed by a regional Query DO.
   * @param region - Datacenter code (e.g., "SJC", "NRT"). Must match worker.ts naming.
   *                 Defaults to "default" for direct SDK use.
   * @param locationHint - Cloudflare locationHint for DO placement.
   */
  static remote(
    queryDoNamespace: DurableObjectNamespace,
    options?: { region?: string; locationHint?: string; masterDoNamespace?: DurableObjectNamespace },
  ): EdgeQ {
    const executor = new RemoteExecutor(
      queryDoNamespace,
      options?.region ?? "default",
      options?.locationHint,
      options?.masterDoNamespace,
    );
    return new EdgeQ(executor);
  }

  /** Create an EdgeQ client for local use (Node/Bun, reads files from disk or URLs). */
  static local(wasmModule?: WebAssembly.Module): EdgeQ {
    const executor = new LocalExecutor(wasmModule);
    return new EdgeQ(executor);
  }

  /** Start building a query against a table. */
  table(name: string): TableQuery {
    return new TableQuery(name, this.executor);
  }

  /**
   * Execute a multi-table query with explicit orchestration.
   * Use this for JOINs — write the join logic in code, not SQL.
   */
  async query<T>(fn: () => Promise<T>): Promise<T> {
    return fn();
  }
}

/**
 * Executor that sends queries to a regional Query DO.
 * The DO has cached footers — no metadata round-trip needed.
 */
class RemoteExecutor implements QueryExecutor {
  private namespace: DurableObjectNamespace;
  private masterNamespace?: DurableObjectNamespace;
  private region: string;
  private locationHint?: string;

  constructor(namespace: DurableObjectNamespace, region: string, locationHint?: string, masterNamespace?: DurableObjectNamespace) {
    this.namespace = namespace;
    this.region = region;
    this.locationHint = locationHint;
    this.masterNamespace = masterNamespace;
  }

  private getQueryDo() {
    const doName = `query-${this.region}`;
    const id = this.namespace.idFromName(doName);
    return this.locationHint
      ? this.namespace.get(id, { locationHint: this.locationHint as DurableObjectLocationHint })
      : this.namespace.get(id);
  }

  async execute(query: QueryDescriptor): Promise<QueryResult> {
    const queryDo = this.getQueryDo();

    const response = await queryDo.fetch(new Request("http://internal/query", {
      method: "POST",
      body: JSON.stringify(query, bigIntReplacer),
      headers: { "content-type": "application/json" },
    }));

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`EdgeQ query failed: ${error}`);
    }

    return response.json() as Promise<QueryResult>;
  }

  async append(table: string, rows: Record<string, unknown>[]): Promise<AppendResult> {
    if (!this.masterNamespace) {
      throw new Error("append() requires masterDoNamespace — pass it via EdgeQ.remote(queryDO, { masterDO })");
    }
    // Append goes to Master DO (single writer)
    const id = this.masterNamespace.idFromName("master");
    const masterDo = this.masterNamespace.get(id);

    const response = await masterDo.fetch(new Request("http://internal/append", {
      method: "POST",
      body: JSON.stringify({ table, rows }, bigIntReplacer),
      headers: { "content-type": "application/json" },
    }));

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`EdgeQ append failed: ${error}`);
    }

    return response.json() as Promise<AppendResult>;
  }

  async executeStream(query: QueryDescriptor): Promise<ReadableStream<Row>> {
    const queryDo = this.getQueryDo();

    const response = await queryDo.fetch(new Request("http://internal/query/stream", {
      method: "POST",
      body: JSON.stringify(query, bigIntReplacer),
      headers: { "content-type": "application/json" },
    }));

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`EdgeQ stream failed: ${error}`);
    }

    if (!response.body) throw new Error("No response body for stream");

    const decoder = new TextDecoder();
    let buffer = "";

    return new ReadableStream<Row>({
      async start(controller) {
        const reader = response.body!.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop()!;
            for (const line of lines) {
              if (line.trim()) controller.enqueue(JSON.parse(line) as Row);
            }
          }
          if (buffer.trim()) controller.enqueue(JSON.parse(buffer) as Row);
          controller.close();
        } catch (err) {
          controller.error(err);
        }
      },
    });
  }
}

/**
 * Executor for local mode (Node/Bun).
 * Reads Lance files directly from the filesystem or via HTTP.
 * Footer is parsed on first access and cached in-process.
 */
class LocalExecutor implements QueryExecutor {
  private metaCache: Map<string, { columns: ColumnMeta[]; fileSize: number }> = new Map();
  private datasetCache: Map<string, DatasetMeta> = new Map();
  private wasmModule?: WebAssembly.Module;
  private wasmEngine?: WasmEngine;

  constructor(wasmModule?: WebAssembly.Module) {
    this.wasmModule = wasmModule;
  }

  private async getWasm(): Promise<WasmEngine> {
    if (this.wasmEngine) return this.wasmEngine;
    if (!this.wasmModule) {
      // Load WASM from file system (Node/Bun)
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      const wasmPath = path.join(import.meta.dirname ?? ".", "wasm", "edgeq.wasm");
      const wasmBytes = await fs.readFile(wasmPath);
      // WebAssembly.compile not in Cloudflare types but available in Node/Bun
      this.wasmModule = await (WebAssembly as unknown as { compile(b: BufferSource): Promise<WebAssembly.Module> }).compile(wasmBytes);
    }
    this.wasmEngine = await instantiateWasm(this.wasmModule!);
    return this.wasmEngine;
  }

  /** Append rows to a local Lance dataset. Builds fragment via WASM and writes to disk. */
  async append(tablePath: string, rows: Record<string, unknown>[]): Promise<AppendResult> {
    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");
    const wasm = await this.getWasm();

    // Ensure dataset directory structure exists
    const dataDir = pathMod.join(tablePath, "data");
    const versionsDir = pathMod.join(tablePath, "_versions");
    await fs.mkdir(dataDir, { recursive: true });
    await fs.mkdir(versionsDir, { recursive: true });

    // Convert rows to column arrays
    const columnNames = Object.keys(rows[0]);
    const columnArrays: { name: string; dtype: string; values: ArrayBufferLike }[] = [];

    for (const colName of columnNames) {
      const sample = rows.find(r => r[colName] != null)?.[colName];
      if (sample === undefined) continue;

      if (typeof sample === "number") {
        if (Number.isInteger(sample)) {
          const arr = new BigInt64Array(rows.length);
          for (let i = 0; i < rows.length; i++) arr[i] = BigInt(rows[i][colName] as number);
          columnArrays.push({ name: colName, dtype: "int64", values: arr.buffer });
        } else {
          const arr = new Float64Array(rows.length);
          for (let i = 0; i < rows.length; i++) arr[i] = rows[i][colName] as number;
          columnArrays.push({ name: colName, dtype: "float64", values: arr.buffer });
        }
      } else if (typeof sample === "bigint") {
        const arr = new BigInt64Array(rows.length);
        for (let i = 0; i < rows.length; i++) arr[i] = rows[i][colName] as bigint;
        columnArrays.push({ name: colName, dtype: "int64", values: arr.buffer });
      } else if (typeof sample === "string") {
        const enc = new TextEncoder();
        const parts: Uint8Array[] = [];
        let totalLen = 0;
        for (const row of rows) {
          const str = enc.encode(String(row[colName] ?? ""));
          const header = new Uint8Array(4);
          new DataView(header.buffer).setUint32(0, str.length, true);
          parts.push(header, str);
          totalLen += 4 + str.length;
        }
        const buf = new Uint8Array(totalLen);
        let off = 0;
        for (const p of parts) { buf.set(p, off); off += p.length; }
        columnArrays.push({ name: colName, dtype: "utf8", values: buf.buffer });
      }
    }

    // Build fragment via WASM
    const fragmentBytes = wasm.buildFragment(columnArrays);

    // Write data file
    const uuid = crypto.randomUUID();
    const dataFilePath = `data/${uuid}.lance`;
    const fullDataPath = pathMod.join(tablePath, dataFilePath);
    await fs.writeFile(fullDataPath, fragmentBytes);

    // Read current version
    const latestPath = pathMod.join(versionsDir, "_latest");
    let currentVersion = 0;
    try {
      const text = await fs.readFile(latestPath, "utf8");
      currentVersion = parseInt(text.trim(), 10) || 0;
    } catch { /* file doesn't exist yet */ }

    const newVersion = currentVersion + 1;

    // Write _latest
    await fs.writeFile(latestPath, String(newVersion));

    // Invalidate meta cache
    this.metaCache.delete(tablePath);
    this.datasetCache.delete(tablePath);

    return {
      version: newVersion,
      dataFilePath,
      retries: 0,
      rowsWritten: rows.length,
    };
  }

  async execute(query: QueryDescriptor): Promise<QueryResult> {
    const startTime = Date.now();
    const isUrl = query.table.startsWith("http://") || query.table.startsWith("https://");

    // Check if this is a Lance dataset directory (has _versions/ subdir)
    if (!isUrl) {
      try {
        const fs = await import("node:fs/promises");
        const stat = await fs.stat(query.table).catch(() => null);
        if (stat?.isDirectory()) {
          return this.executeDatasetQuery(query, startTime);
        }
      } catch { /* stat failed — not a directory, fall through to single-file path */ }
    }

    // Step 1: Get or cache table metadata (footer + column meta)
    let cached = this.metaCache.get(query.table);
    if (!cached) {
      cached = isUrl
        ? await this.loadMetaFromUrl(query.table)
        : await this.loadMetaFromFile(query.table);
      this.metaCache.set(query.table, cached);
    }

    const { columns, fileSize } = cached;

    // Step 2: Determine projected columns
    const projectedColumns =
      query.projections.length > 0
        ? columns.filter((c) => query.projections.includes(c.name))
        : columns;

    // Step 3: Determine which pages to fetch using filter pushdown
    const pageRanges: { column: string; offset: bigint; length: number }[] = [];
    let pagesSkipped = 0;

    for (const col of projectedColumns) {
      for (const page of col.pages) {
        if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) {
          pagesSkipped++;
          continue;
        }
        pageRanges.push({ column: col.name, offset: page.byteOffset, length: page.byteLength });
      }
    }

    // Step 4: Read page data
    const columnData = new Map<string, ArrayBuffer[]>();
    let bytesRead = 0;

    if (isUrl) {
      // Parallel HTTP Range reads
      const pageBuffers = await Promise.all(
        pageRanges.map(async (range) => {
          const start = Number(range.offset);
          const end = start + range.length - 1;
          const resp = await fetch(query.table, {
            headers: { Range: `bytes=${start}-${end}` },
          });
          if (!resp.ok && resp.status !== 206) return null;
          return { column: range.column, data: await resp.arrayBuffer() };
        }),
      );
      for (const buf of pageBuffers) {
        if (!buf) continue;
        bytesRead += buf.data.byteLength;
        const existing = columnData.get(buf.column) ?? [];
        existing.push(buf.data);
        columnData.set(buf.column, existing);
      }
    } else {
      // Sequential reads from local filesystem
      const fs = await import("node:fs/promises");
      const handle = await fs.open(query.table, "r");
      try {
        for (const range of pageRanges) {
          const buf = Buffer.alloc(range.length);
          await handle.read(buf, 0, range.length, Number(range.offset));
          bytesRead += range.length;
          const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          const existing = columnData.get(range.column) ?? [];
          existing.push(ab);
          columnData.set(range.column, existing);
        }
      } finally {
        await handle.close();
      }
    }

    // Step 5: Decode and assemble rows
    const wasm = await this.getWasm();
    const rows = assembleRows(columnData, projectedColumns, query, wasm);

    return {
      rows,
      rowCount: rows.length,
      columns: projectedColumns.map((c) => c.name),
      bytesRead,
      pagesSkipped,
      durationMs: Date.now() - startTime,
    };
  }

  /** Execute a query against a multi-fragment Lance dataset directory. */
  private async executeDatasetQuery(query: QueryDescriptor, startTime: number): Promise<QueryResult> {
    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");

    // Discover or reuse cached dataset metadata
    let dataset = this.datasetCache.get(query.table);
    if (!dataset) {
      // Find latest manifest
      const versionsDir = pathMod.join(query.table, "_versions");
      const entries = await fs.readdir(versionsDir).catch(() => [] as string[]);
      const manifests = entries.filter(e => e.endsWith(".manifest")).sort();
      if (manifests.length === 0) {
        throw new Error(`No manifests found in ${versionsDir}`);
      }

      const latestManifest = manifests[manifests.length - 1];
      const manifestBuf = await fs.readFile(pathMod.join(versionsDir, latestManifest));
      const ab = manifestBuf.buffer.slice(manifestBuf.byteOffset, manifestBuf.byteOffset + manifestBuf.byteLength);
      const manifest = parseManifest(ab);
      if (!manifest) throw new Error(`Failed to parse manifest ${latestManifest}`);

      // Read footer + column metadata for each fragment
      const fragmentMetas = new Map<number, TableMeta>();
      for (const frag of manifest.fragments) {
        const fragPath = pathMod.join(query.table, frag.filePath);
        try {
          const cached = await this.loadMetaFromFile(fragPath);
          // Read the actual footer from the fragment file
          const fragStat = await fs.stat(fragPath);
          const fragHandle = await fs.open(fragPath, "r");
          let footer: import("./types.js").Footer;
          try {
            const footerBuf = Buffer.alloc(FOOTER_SIZE);
            await fragHandle.read(footerBuf, 0, FOOTER_SIZE, fragStat.size - FOOTER_SIZE);
            const footerAb = footerBuf.buffer.slice(footerBuf.byteOffset, footerBuf.byteOffset + footerBuf.byteLength);
            const parsed = parseFooter(footerAb);
            if (!parsed) continue;
            footer = parsed;
          } finally {
            await fragHandle.close();
          }
          fragmentMetas.set(frag.id, {
            name: frag.filePath,
            footer,
            columns: cached.columns,
            totalRows: frag.physicalRows,
            fileSize: BigInt(cached.fileSize),
            r2Key: fragPath,
            updatedAt: Date.now(),
          });
        } catch (err) {
          throw new Error(`Failed to load fragment ${frag.filePath}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }

      dataset = {
        name: query.table,
        r2Prefix: query.table + "/",
        manifest,
        fragmentMetas,
        totalRows: manifest.totalRows,
        updatedAt: Date.now(),
      };
      this.datasetCache.set(query.table, dataset);
    }

    // Execute query across all fragments
    const fsMod = await import("node:fs/promises");
    let allRows: import("./types.js").Row[] = [];
    let totalBytesRead = 0;
    let totalPagesSkipped = 0;

    for (const [, meta] of dataset.fragmentMetas) {
      const projectedColumns = query.projections.length > 0
        ? meta.columns.filter(c => query.projections.includes(c.name))
        : meta.columns;

      const pageRanges: { column: string; offset: bigint; length: number }[] = [];
      for (const col of projectedColumns) {
        for (const page of col.pages) {
          if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) {
            totalPagesSkipped++;
            continue;
          }
          pageRanges.push({ column: col.name, offset: page.byteOffset, length: page.byteLength });
        }
      }

      const columnData = new Map<string, ArrayBuffer[]>();
      const handle = await fsMod.open(meta.r2Key, "r");
      try {
        for (const range of pageRanges) {
          const buf = Buffer.alloc(range.length);
          await handle.read(buf, 0, range.length, Number(range.offset));
          totalBytesRead += range.length;
          const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          const existing = columnData.get(range.column) ?? [];
          existing.push(ab);
          columnData.set(range.column, existing);
        }
      } finally {
        await handle.close();
      }

      const wasm = await this.getWasm();
      const rows = assembleRows(columnData, projectedColumns, query, wasm);
      allRows.push(...rows);
    }

    // Apply limit if no sort (sorted results already limited per-fragment)
    if (query.limit && !query.sortColumn && allRows.length > query.limit) {
      allRows = allRows.slice(0, query.limit);
    }

    return {
      rows: allRows,
      rowCount: allRows.length,
      columns: query.projections.length > 0
        ? query.projections
        : (dataset.fragmentMetas.values().next().value?.columns.map((c: ColumnMeta) => c.name) ?? []),
      bytesRead: totalBytesRead,
      pagesSkipped: totalPagesSkipped,
      durationMs: Date.now() - startTime,
    };
  }

  /** Load footer + column metadata from a local file */
  private async loadMetaFromFile(path: string): Promise<{ columns: ColumnMeta[]; fileSize: number }> {
    const fs = await import("node:fs/promises");
    const stat = await fs.stat(path);
    const fileSize = Number(stat.size);

    const handle = await fs.open(path, "r");
    try {
      // Read last 40 bytes for format detection
      const tailSize = Math.min(fileSize, FOOTER_SIZE);
      const tailBuf = Buffer.alloc(tailSize);
      await handle.read(tailBuf, 0, tailSize, fileSize - tailSize);
      const tailAb = tailBuf.buffer.slice(tailBuf.byteOffset, tailBuf.byteOffset + tailBuf.byteLength);

      const fmt = detectFormat(tailAb);

      if (fmt === "parquet") {
        const footerLen = getParquetFooterLength(tailAb);
        if (!footerLen) throw new Error(`Invalid Parquet file: cannot read footer length in ${path}`);

        const parquetFooterBuf = Buffer.alloc(footerLen);
        await handle.read(parquetFooterBuf, 0, footerLen, fileSize - footerLen - 8);
        const parquetFooterAb = parquetFooterBuf.buffer.slice(
          parquetFooterBuf.byteOffset, parquetFooterBuf.byteOffset + parquetFooterBuf.byteLength,
        );

        const parquetMeta = parseParquetFooter(parquetFooterAb);
        if (!parquetMeta) throw new Error(`Failed to parse Parquet footer in ${path}`);

        const tableMeta = parquetMetaToTableMeta(parquetMeta, path, BigInt(fileSize));
        return { columns: tableMeta.columns, fileSize };
      }

      // Lance format (default)
      const footer = parseFooter(tailAb);
      if (!footer) throw new Error(`Invalid file format: unrecognized magic in ${path}`);

      // Read column metadata (protobuf region)
      const metaStart = Number(footer.columnMetaStart);
      const metaEnd = Number(footer.columnMetaOffsetsStart);
      const metaLength = metaEnd - metaStart;

      if (metaLength <= 0) return { columns: [], fileSize };

      const metaBuf = Buffer.alloc(metaLength);
      await handle.read(metaBuf, 0, metaLength, metaStart);
      const metaAb = metaBuf.buffer.slice(metaBuf.byteOffset, metaBuf.byteOffset + metaBuf.byteLength);

      const columns = parseColumnMetaFromProtobuf(metaAb, footer.numColumns);
      return { columns, fileSize };
    } finally {
      await handle.close();
    }
  }

  /** Load footer + column metadata via HTTP Range requests */
  private async loadMetaFromUrl(url: string): Promise<{ columns: ColumnMeta[]; fileSize: number }> {
    // Get file size
    const headResp = await fetch(url, { method: "HEAD" });
    const fileSize = Number(headResp.headers.get("content-length") ?? 0);
    if (fileSize < 8) throw new Error(`File too small: ${url}`);

    // Read tail for format detection (last 40 bytes)
    const tailSize = Math.min(fileSize, FOOTER_SIZE);
    const tailStart = fileSize - tailSize;
    const tailResp = await fetch(url, {
      headers: { Range: `bytes=${tailStart}-${fileSize - 1}` },
    });
    const tailAb = await tailResp.arrayBuffer();
    const fmt = detectFormat(tailAb);

    if (fmt === "parquet") {
      const footerLen = getParquetFooterLength(tailAb);
      if (!footerLen) throw new Error(`Invalid Parquet file: cannot read footer length in ${url}`);

      const footerStart = fileSize - footerLen - 8;
      const footerResp = await fetch(url, {
        headers: { Range: `bytes=${footerStart}-${footerStart + footerLen - 1}` },
      });
      const footerBuf = await footerResp.arrayBuffer();
      const parquetMeta = parseParquetFooter(footerBuf);
      if (!parquetMeta) throw new Error(`Failed to parse Parquet footer in ${url}`);

      const tableMeta = parquetMetaToTableMeta(parquetMeta, url, BigInt(fileSize));
      return { columns: tableMeta.columns, fileSize };
    }

    // Lance format
    const footer = parseFooter(tailAb);
    if (!footer) throw new Error(`Invalid file format: unrecognized magic in ${url}`);

    const metaStart = Number(footer.columnMetaStart);
    const metaEnd = Number(footer.columnMetaOffsetsStart);
    const metaLength = metaEnd - metaStart;

    if (metaLength <= 0) return { columns: [], fileSize };

    const metaResp = await fetch(url, {
      headers: { Range: `bytes=${metaStart}-${metaStart + metaLength - 1}` },
    });
    const metaAb = await metaResp.arrayBuffer();
    const columns = parseColumnMetaFromProtobuf(metaAb, footer.numColumns);

    return { columns, fileSize };
  }
}
