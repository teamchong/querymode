import type { ColumnMeta, Env, Footer, TableMeta } from "./types.js";
import { parseFooter, parseColumnMetaFromProtobuf, FOOTER_SIZE } from "./footer.js";

/** Master DO — single writer, reads footers, broadcasts invalidations. */
export class MasterDO implements DurableObject {
  private state: DurableObjectState;
  private env: Env;

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
    return this.json({ registered: true, region });
  }

  private async handleWrite(request: Request): Promise<Response> {
    const { r2Key } = (await request.json()) as { r2Key: string };
    const result = await this.readFooterAndColumns(r2Key);
    if (!result) return this.json({ error: "Failed to read footer" }, 500);

    const tableName = r2Key.replace(/\.lance$/, "").split("/").pop() ?? r2Key;
    const meta: TableMeta = {
      name: tableName, footer: result.parsed, columns: result.columns,
      totalRows: result.columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0,
      fileSize: result.fileSize, r2Key, updatedAt: Date.now(),
    };
    await this.state.storage.put(`table:${tableName}`, meta);
    await this.broadcast(tableName, r2Key, result);
    return this.json({ success: true, table: tableName });
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
    parsed: Footer; raw: ArrayBuffer; fileSize: bigint; columns: ColumnMeta[];
  } | null> {
    const head = await this.env.DATA_BUCKET.head(r2Key);
    if (!head) return null;

    const fileSize = BigInt(head.size);
    const obj = await this.env.DATA_BUCKET.get(r2Key, {
      range: { offset: Number(fileSize) - FOOTER_SIZE, length: FOOTER_SIZE },
    });
    if (!obj) return null;

    const raw = await obj.arrayBuffer();
    const parsed = parseFooter(raw);
    if (!parsed) return null;

    // Read column metadata once — saves N redundant R2 reads from Query DOs
    let columns: ColumnMeta[] = [];
    const metaLen = Number(parsed.columnMetaOffsetsStart) - Number(parsed.columnMetaStart);
    if (metaLen > 0) {
      const metaObj = await this.env.DATA_BUCKET.get(r2Key, {
        range: { offset: Number(parsed.columnMetaStart), length: metaLen },
      });
      if (metaObj) columns = parseColumnMetaFromProtobuf(await metaObj.arrayBuffer(), parsed.numColumns);
    }

    return { parsed, raw, fileSize, columns };
  }

  /** Broadcast invalidation with pre-parsed columns to all Query DOs. */
  private async broadcast(
    table: string, r2Key: string,
    footer: { raw: ArrayBuffer; fileSize: bigint; columns: ColumnMeta[] },
  ): Promise<void> {
    const regions = (await this.state.storage.get<Record<string, string>>("regions")) ?? {};
    const payload = JSON.stringify({
      table, r2Key, columns: footer.columns,
      footerBytes: Array.from(new Uint8Array(footer.raw)),
      fileSize: footer.fileSize.toString(), timestamp: Date.now(),
    });

    await Promise.allSettled(Object.entries(regions).map(async ([region, doId]) => {
      try {
        const queryDo = this.env.QUERY_DO.get(this.env.QUERY_DO.idFromString(doId));
        await queryDo.fetch(new Request("http://internal/invalidate", {
          method: "POST", body: payload, headers: { "content-type": "application/json" },
        }));
      } catch {
        console.warn(`Broadcast to ${region} failed, will resync`);
      }
    }));
  }
}
