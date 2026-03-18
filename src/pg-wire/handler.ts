/**
 * Postgres wire protocol handler — translates pg queries to QueryMode execution.
 *
 * Handles the connection lifecycle:
 *   1. Startup handshake (SSL reject, auth ok, parameter status)
 *   2. Simple Query protocol (parse SQL → compile → execute → format as DataRow)
 *   3. Terminate
 */

import type { QueryExecutor } from "../client.js";
import type { Row } from "../types.js";
import { buildSqlDataFrame } from "../sql/index.js";
import {
  parseStartupMessage,
  parseFrontendMessage,
  sslRefused,
  authenticationOk,
  parameterStatus,
  backendKeyData,
  readyForQuery,
  rowDescription,
  dataRow,
  commandComplete,
  errorResponse,
  type FrontendMessage,
} from "./protocol.js";

export interface PgConnectionOptions {
  /** QueryMode executor to run queries against */
  executor: QueryExecutor;
  /** Called when data should be sent to the client */
  send: (data: Uint8Array) => void;
}

export class PgConnectionHandler {
  private executor: QueryExecutor;
  private send: (data: Uint8Array) => void;
  private startupDone = false;
  private buffer = new Uint8Array(0);

  constructor(opts: PgConnectionOptions) {
    this.executor = opts.executor;
    this.send = opts.send;
  }

  /** Feed incoming bytes from the client. May trigger responses via send(). */
  async onData(chunk: Uint8Array): Promise<void> {
    // Append to buffer
    const combined = new Uint8Array(this.buffer.length + chunk.length);
    combined.set(this.buffer);
    combined.set(chunk, this.buffer.length);
    this.buffer = combined;

    if (!this.startupDone) {
      await this.handleStartup();
      return;
    }

    // Parse regular messages
    while (this.buffer.length >= 5) {
      const result = parseFrontendMessage(this.buffer);
      if (!result) break;
      const [msg, consumed] = result;
      this.buffer = this.buffer.subarray(consumed);
      await this.handleMessage(msg);
    }
  }

  private async handleStartup(): Promise<void> {
    const msg = parseStartupMessage(this.buffer);
    if (!msg) return;

    if (msg.type === "ssl_request") {
      // Reject SSL, client will retry without
      this.send(sslRefused());
      this.buffer = this.buffer.subarray(8);
      return;
    }

    if (msg.type === "startup") {
      // Consume startup message
      const dv = new DataView(this.buffer.buffer, this.buffer.byteOffset);
      const len = dv.getInt32(0);
      this.buffer = this.buffer.subarray(len);
      this.startupDone = true;

      // Send auth ok + server params + ready
      this.send(authenticationOk());
      this.send(parameterStatus("server_version", "15.0 (QueryMode)"));
      this.send(parameterStatus("server_encoding", "UTF8"));
      this.send(parameterStatus("client_encoding", "UTF8"));
      this.send(parameterStatus("DateStyle", "ISO, MDY"));
      this.send(backendKeyData(1, 0));
      this.send(readyForQuery());
    }
  }

  private async handleMessage(msg: FrontendMessage): Promise<void> {
    if (msg.type === "terminate" || msg.type === "skip") return;

    if (msg.type === "query") {
      await this.handleQuery(msg.sql);
    }
  }

  private async handleQuery(sql: string): Promise<void> {
    const trimmed = sql.trim().replace(/;$/, "").trim();

    // Handle empty query
    if (!trimmed) {
      this.send(commandComplete("SELECT 0"));
      this.send(readyForQuery());
      return;
    }

    // Handle SET/RESET/DISCARD — just acknowledge
    const upper = trimmed.toUpperCase();
    if (upper.startsWith("SET ") || upper.startsWith("RESET ") || upper.startsWith("DISCARD ")) {
      this.send(commandComplete("SET"));
      this.send(readyForQuery());
      return;
    }

    // Handle SHOW — return a fake value for compatibility
    if (upper.startsWith("SHOW ")) {
      const param = trimmed.slice(5).trim().toLowerCase();
      const cols = [{ name: param, oid: 25 }];
      this.send(rowDescription(cols));
      this.send(dataRow(["on"]));
      this.send(commandComplete("SHOW"));
      this.send(readyForQuery());
      return;
    }

    try {
      const df = buildSqlDataFrame(trimmed, this.executor);
      const result = await df.collect();

      // Build column descriptors
      const colNames = result.columns.length > 0
        ? result.columns
        : result.rows.length > 0
          ? Object.keys(result.rows[0])
          : [];

      const cols = colNames.map(name => ({
        name,
        oid: this.inferOid(name, result.rows),
      }));

      // Send RowDescription
      this.send(rowDescription(cols));

      // Send DataRows
      for (const row of result.rows) {
        const values = colNames.map(col => formatValue(row[col]));
        this.send(dataRow(values));
      }

      // Send CommandComplete
      this.send(commandComplete(`SELECT ${result.rowCount}`));
      this.send(readyForQuery());
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.send(errorResponse(msg));
      this.send(readyForQuery());
    }
  }

  private inferOid(colName: string, rows: Row[]): number {
    // Find first non-null value to infer type
    for (const row of rows) {
      const val = row[colName];
      if (val === null || val === undefined) continue;
      if (typeof val === "number") return Number.isInteger(val) ? 23 : 701;
      if (typeof val === "bigint") return 20;
      if (typeof val === "boolean") return 16;
      if (val instanceof Float32Array) return 25; // vectors as text
      return 25; // string
    }
    return 25; // default to text
  }
}

function formatValue(val: unknown): string | null {
  if (val === null || val === undefined) return null;
  if (typeof val === "bigint") return val.toString();
  if (val instanceof Float32Array) {
    return "[" + Array.from(val).join(",") + "]";
  }
  return String(val);
}
