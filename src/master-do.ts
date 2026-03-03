import type { ColumnMeta, Env, Footer, TableMeta, DatasetMeta } from "./types.js";
import { parseFooter, parseColumnMetaFromProtobuf, FOOTER_SIZE } from "./footer.js";
import { parseManifest } from "./manifest.js";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";

/** Master DO — single writer, reads footers, broadcasts invalidations. */
export class MasterDO implements DurableObject {
  private state: DurableObjectState;
  private env: Env;
  private broadcastFailures = new Map<string, number>(); // region → consecutive failure count

  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
  }

  private json(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), {
      status, headers: { "content-type": "application/json" },
    });
  }

  async fetch(request: Request): Promise<Response> {
    switch (new URL(request.url).pathname) {
      case "/register": return this.handleRegister(request);
      case "/write":    return this.handleWrite(request);
      case "/refresh":  return this.handleRefresh(request);
      case "/tables":   return this.handleListTables();
      default:          return new Response("Not found", { status: 404 });
    }
  }

  private async handleRegister(request: Request): Promise<Response> {
    const { queryDoId, region } = (await request.json()) as { queryDoId: string; region: string };
    const regions = (await this.state.storage.get<Record<string, string>>("regions")) ?? {};
    regions[region] = queryDoId;
    await this.state.storage.put("regions", regions);

    // Return current table timestamps so waking Query DOs can detect stale caches
    const tables = await this.state.storage.list<TableMeta>({ prefix: "table:" });
    const tableVersions: Record<string, { r2Key: string; updatedAt: number }> = {};
    for (const [key, meta] of tables) {
      const name = key.replace("table:", "");
      tableVersions[name] = { r2Key: meta.r2Key ?? name, updatedAt: meta.updatedAt ?? 0 };
    }
    return this.json({ registered: true, region, tableVersions });
  }

  private async handleWrite(request: Request): Promise<Response> {
    const { r2Key } = (await request.json()) as { r2Key: string };

    // Check if this is a dataset directory (ends with / or .lance/)
    if (r2Key.endsWith("/") || r2Key.endsWith(".lance/")) {
      return this.handleDatasetWrite(r2Key);
    }

    const result = await this.readFooterAndColumns(r2Key);
    if (!result) return this.json({ error: "Failed to read footer" }, 500);

    const tableName = r2Key.replace(/\.(lance|parquet)$/, "").split("/").pop() ?? r2Key;
    const meta: TableMeta = {
      name: tableName, footer: result.parsed, format: result.format, columns: result.columns,
      totalRows: result.columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0,
      fileSize: result.fileSize, r2Key, updatedAt: Date.now(),
    };
    await this.state.storage.put(`table:${tableName}`, meta);
    await this.broadcast(tableName, r2Key, result);
    return this.json({ success: true, table: tableName });
  }

  /** Handle write notification for a multi-fragment dataset directory. */
  private async handleDatasetWrite(r2Prefix: string): Promise<Response> {
    const tableName = r2Prefix.replace(/\/$/, "").replace(/\.lance$/, "").split("/").pop() ?? r2Prefix;

    // Find latest manifest
    const listed = await this.env.DATA_BUCKET.list({ prefix: `${r2Prefix}_versions/`, limit: 100 });
    const manifestKeys = listed.objects
      .filter(o => o.key.endsWith(".manifest"))
      .sort((a, b) => a.key.localeCompare(b.key));
    if (manifestKeys.length === 0) return this.json({ error: "No manifests found" }, 404);

    const latestKey = manifestKeys[manifestKeys.length - 1].key;
    const manifestObj = await this.env.DATA_BUCKET.get(latestKey);
    if (!manifestObj) return this.json({ error: "Failed to read manifest" }, 500);

    const manifest = parseManifest(await manifestObj.arrayBuffer());
    if (!manifest) return this.json({ error: "Failed to parse manifest" }, 500);

    // Read first fragment's footer to broadcast (Query DOs will discover the rest)
    if (manifest.fragments.length > 0) {
      const firstFrag = manifest.fragments[0];
      const fragKey = `${r2Prefix}${firstFrag.filePath}`;
      const result = await this.readFooterAndColumns(fragKey);
      if (result) {
        await this.broadcast(tableName, fragKey, result);
      }
    }

    await this.state.storage.put(`table:${tableName}`, {
      name: tableName, r2Prefix, manifest, totalRows: manifest.totalRows, updatedAt: Date.now(),
    });

    return this.json({ success: true, table: tableName, fragments: manifest.fragments.length, totalRows: manifest.totalRows });
  }

  private async handleRefresh(request: Request): Promise<Response> {
    const { r2Key } = (await request.json()) as { r2Key: string };
    const result = await this.readFooterAndColumns(r2Key);
    if (!result) return this.json({ error: "Failed to read footer" }, 500);

    const tableName = r2Key.replace(/\.lance$/, "").split("/").pop() ?? r2Key;
    await this.broadcast(tableName, r2Key, result);
    return this.json({ refreshed: true, table: tableName });
  }

  private async handleListTables(): Promise<Response> {
    const tables = await this.state.storage.list<TableMeta>({ prefix: "table:" });
    return this.json({ tables: [...tables.keys()].map(k => k.replace("table:", "")) });
  }

  /** Read footer + column metadata from R2 (2 range reads, done once by Master). */
  private async readFooterAndColumns(r2Key: string): Promise<{
    parsed?: Footer; raw: ArrayBuffer; fileSize: bigint; columns: ColumnMeta[];
    format: "lance" | "parquet";
  } | null> {
    const head = await this.env.DATA_BUCKET.head(r2Key);
    if (!head) return null;

    const fileSize = BigInt(head.size);
    // Read last 40 bytes — enough for Lance footer or Parquet tail detection
    const tailSize = Math.min(Number(fileSize), FOOTER_SIZE);
    const obj = await this.env.DATA_BUCKET.get(r2Key, {
      range: { offset: Number(fileSize) - tailSize, length: tailSize },
    });
    if (!obj) return null;

    const raw = await obj.arrayBuffer();
    const format = detectFormat(raw);

    if (format === "parquet") {
      const footerLen = getParquetFooterLength(raw);
      if (!footerLen) return null;

      // Fetch full Parquet Thrift footer
      const footerOffset = Number(fileSize) - footerLen - 8;
      const footerObj = await this.env.DATA_BUCKET.get(r2Key, {
        range: { offset: footerOffset, length: footerLen },
      });
      if (!footerObj) return null;

      const footerBuf = await footerObj.arrayBuffer();
      const parquetMeta = parseParquetFooter(footerBuf);
      if (!parquetMeta) return null;

      const tableMeta = parquetMetaToTableMeta(parquetMeta, r2Key, fileSize);
      return { raw, fileSize, columns: tableMeta.columns, format: "parquet" };
    }

    // Lance format
    const parsed = parseFooter(raw);
    if (!parsed) return null;

    let columns: ColumnMeta[] = [];
    const metaLen = Number(parsed.columnMetaOffsetsStart) - Number(parsed.columnMetaStart);
    if (metaLen > 0) {
      const metaObj = await this.env.DATA_BUCKET.get(r2Key, {
        range: { offset: Number(parsed.columnMetaStart), length: metaLen },
      });
      if (metaObj) columns = parseColumnMetaFromProtobuf(await metaObj.arrayBuffer(), parsed.numColumns);
    }

    return { parsed, raw, fileSize, columns, format: "lance" };
  }

  /** Broadcast invalidation with pre-parsed columns to all Query DOs. */
  private async broadcast(
    table: string, r2Key: string,
    footer: { raw: ArrayBuffer; fileSize: bigint; columns: ColumnMeta[]; format?: "lance" | "parquet" },
  ): Promise<void> {
    const regions = (await this.state.storage.get<Record<string, string>>("regions")) ?? {};
    const payload = JSON.stringify({
      table, r2Key, columns: footer.columns, format: footer.format ?? "lance",
      footerBytes: Array.from(new Uint8Array(footer.raw)),
      fileSize: footer.fileSize.toString(), timestamp: Date.now(),
    });

    const deadRegions: string[] = [];
    await Promise.allSettled(Object.entries(regions).map(async ([region, doId]) => {
      try {
        const queryDo = this.env.QUERY_DO.get(this.env.QUERY_DO.idFromString(doId));
        await queryDo.fetch(new Request("http://internal/invalidate", {
          method: "POST", body: payload, headers: { "content-type": "application/json" },
        }));
        this.broadcastFailures.delete(region);
      } catch {
        const count = (this.broadcastFailures.get(region) ?? 0) + 1;
        this.broadcastFailures.set(region, count);
        if (count >= 3) {
          console.warn(`Broadcast to ${region} failed ${count} times, removing`);
          deadRegions.push(region);
        } else {
          console.warn(`Broadcast to ${region} failed (${count}/3)`);
        }
      }
    }));

    if (deadRegions.length > 0) {
      for (const r of deadRegions) {
        delete regions[r];
        this.broadcastFailures.delete(r);
      }
      await this.state.storage.put("regions", regions);
    }
  }
}
