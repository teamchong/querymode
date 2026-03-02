import { TableQuery } from "./client.js";
import type { QueryDescriptor, QueryExecutor } from "./client.js";
import type { ColumnMeta, Env, QueryResult } from "./types.js";
import { parseFooter, parseColumnMetaFromProtobuf, FOOTER_SIZE } from "./footer.js";
import { assembleRows, canSkipPage, bigIntReplacer } from "./decode.js";

export { MasterDO } from "./master-do.js";
export { QueryDO } from "./query-do.js";
export { TableQuery } from "./client.js";
export type { QueryExecutor, QueryDescriptor } from "./client.js";
export type {
  Env,
  Footer,
  TableMeta,
  ColumnMeta,
  PageInfo,
  DataType,
  FilterOp,
  AggregateOp,
  QueryResult,
  Row,
  VectorSearchParams,
  FooterInvalidation,
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
    options?: { region?: string; locationHint?: string },
  ): EdgeQ {
    const executor = new RemoteExecutor(
      queryDoNamespace,
      options?.region ?? "default",
      options?.locationHint,
    );
    return new EdgeQ(executor);
  }

  /** Create an EdgeQ client for local use (Node/Bun, reads files from disk or URLs). */
  static local(): EdgeQ {
    const executor = new LocalExecutor();
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
  private region: string;
  private locationHint?: string;

  constructor(namespace: DurableObjectNamespace, region: string, locationHint?: string) {
    this.namespace = namespace;
    this.region = region;
    this.locationHint = locationHint;
  }

  async execute(query: QueryDescriptor): Promise<QueryResult> {
    // FIX: Use query-${region} naming to match worker.ts DO routing.
    // Previously hardcoded "regional-query" which never matched worker.ts's "query-{datacenter}".
    const doName = `query-${this.region}`;
    const id = this.namespace.idFromName(doName);
    const queryDo = this.locationHint
      ? this.namespace.get(id, { locationHint: this.locationHint as DurableObjectLocationHint })
      : this.namespace.get(id);

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
}

/**
 * Executor for local mode (Node/Bun).
 * Reads Lance files directly from the filesystem or via HTTP.
 * Footer is parsed on first access and cached in-process.
 */
class LocalExecutor implements QueryExecutor {
  private metaCache: Map<string, { columns: ColumnMeta[]; fileSize: number }> = new Map();

  async execute(query: QueryDescriptor): Promise<QueryResult> {
    const startTime = Date.now();
    const isUrl = query.table.startsWith("http://") || query.table.startsWith("https://");

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
    const rows = assembleRows(columnData, projectedColumns, query);

    return {
      rows,
      rowCount: rows.length,
      columns: projectedColumns.map((c) => c.name),
      bytesRead,
      pagesSkipped,
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
      // Read footer (last 40 bytes)
      const footerBuf = Buffer.alloc(FOOTER_SIZE);
      await handle.read(footerBuf, 0, FOOTER_SIZE, fileSize - FOOTER_SIZE);
      const footerAb = footerBuf.buffer.slice(footerBuf.byteOffset, footerBuf.byteOffset + footerBuf.byteLength);

      const footer = parseFooter(footerAb);
      if (!footer) throw new Error(`Invalid Lance file: bad magic number in ${path}`);

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
    if (fileSize < FOOTER_SIZE) throw new Error(`File too small: ${url}`);

    // Read footer
    const footerStart = fileSize - FOOTER_SIZE;
    const footerResp = await fetch(url, {
      headers: { Range: `bytes=${footerStart}-${fileSize - 1}` },
    });
    const footerAb = await footerResp.arrayBuffer();
    const footer = parseFooter(footerAb);
    if (!footer) throw new Error(`Invalid Lance file: bad magic number in ${url}`);

    // Read column metadata
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
