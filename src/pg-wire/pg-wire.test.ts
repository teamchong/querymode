import { describe, it, expect } from "vitest";
import { QueryMode } from "../local.js";
import { MaterializedExecutor } from "../client.js";
import { PgConnectionHandler } from "./handler.js";
import {
  parseStartupMessage,
  parseFrontendMessage,
  authenticationOk,
  readyForQuery,
  rowDescription,
  dataRow,
  commandComplete,
  errorResponse,
  parameterStatus,
  sslRefused,
  dtypeToOid,
} from "./protocol.js";

const SAMPLE_DATA = [
  { id: 1, name: "Alice", age: 30, dept: "eng" },
  { id: 2, name: "Bob", age: 25, dept: "eng" },
  { id: 3, name: "Charlie", age: 35, dept: "sales" },
];

function makeExecutor() {
  return new MaterializedExecutor({
    rows: SAMPLE_DATA,
    columns: ["id", "name", "age", "dept"],
    rowCount: SAMPLE_DATA.length,
    totalRows: SAMPLE_DATA.length,
    scannedBytes: 0,
    elapsedMs: 0,
  });
}

/** Build a startup message (protocol v3) */
function buildStartup(params: Record<string, string>): Uint8Array {
  const parts: Uint8Array[] = [];
  const enc = new TextEncoder();
  for (const [k, v] of Object.entries(params)) {
    parts.push(enc.encode(k), new Uint8Array([0]), enc.encode(v), new Uint8Array([0]));
  }
  parts.push(new Uint8Array([0])); // terminator

  let bodyLen = 0;
  for (const p of parts) bodyLen += p.length;
  const totalLen = 4 + 4 + bodyLen; // length(4) + version(4) + params

  const buf = new Uint8Array(totalLen);
  const dv = new DataView(buf.buffer);
  dv.setInt32(0, totalLen);
  dv.setInt32(4, 196608); // v3.0
  let pos = 8;
  for (const p of parts) { buf.set(p, pos); pos += p.length; }
  return buf;
}

/** Build a Simple Query message ('Q') */
function buildQuery(sql: string): Uint8Array {
  const enc = new TextEncoder();
  const sqlBytes = enc.encode(sql);
  const len = 4 + sqlBytes.length + 1;
  const buf = new Uint8Array(1 + len);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x51; // 'Q'
  dv.setInt32(1, len);
  buf.set(sqlBytes, 5);
  buf[5 + sqlBytes.length] = 0;
  return buf;
}

/** Build a Terminate message ('X') */
function buildTerminate(): Uint8Array {
  const buf = new Uint8Array(5);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x58; // 'X'
  dv.setInt32(1, 4);
  return buf;
}

/** Helper: collect all messages sent by handler */
function createCollector() {
  const sent: Uint8Array[] = [];
  return {
    send: (data: Uint8Array) => sent.push(data.slice()),
    sent,
  };
}

/** Parse a backend message type byte */
function messageType(buf: Uint8Array): string {
  return String.fromCharCode(buf[0]);
}

// ── Protocol unit tests ─────────────────────────────────────────────────

describe("pg-wire protocol", () => {
  it("parses startup message", () => {
    const startup = buildStartup({ user: "test", database: "mydb" });
    const msg = parseStartupMessage(startup);
    expect(msg).not.toBeNull();
    expect(msg!.type).toBe("startup");
    if (msg!.type === "startup") {
      expect(msg!.params.get("user")).toBe("test");
      expect(msg!.params.get("database")).toBe("mydb");
    }
  });

  it("parses SSL request", () => {
    const buf = new Uint8Array(8);
    const dv = new DataView(buf.buffer);
    dv.setInt32(0, 8);
    dv.setInt32(4, 80877103); // SSL request code
    const msg = parseStartupMessage(buf);
    expect(msg).not.toBeNull();
    expect(msg!.type).toBe("ssl_request");
  });

  it("parses query message", () => {
    const qbuf = buildQuery("SELECT 1");
    const result = parseFrontendMessage(qbuf);
    expect(result).not.toBeNull();
    const [msg, consumed] = result!;
    expect(msg.type).toBe("query");
    if (msg.type === "query") {
      expect(msg.sql).toBe("SELECT 1");
    }
    expect(consumed).toBe(qbuf.length);
  });

  it("serializes authenticationOk", () => {
    const buf = authenticationOk();
    expect(buf[0]).toBe(0x52); // 'R'
    const dv = new DataView(buf.buffer);
    expect(dv.getInt32(5)).toBe(0); // auth ok
  });

  it("serializes readyForQuery", () => {
    const buf = readyForQuery();
    expect(buf[0]).toBe(0x5a); // 'Z'
    expect(buf[5]).toBe("I".charCodeAt(0));
  });

  it("serializes rowDescription", () => {
    const buf = rowDescription([
      { name: "id", oid: 23 },
      { name: "name", oid: 25 },
    ]);
    expect(buf[0]).toBe(0x54); // 'T'
    const dv = new DataView(buf.buffer);
    expect(dv.getInt16(5)).toBe(2); // 2 columns
  });

  it("serializes dataRow with null", () => {
    const buf = dataRow(["hello", null, "42"]);
    expect(buf[0]).toBe(0x44); // 'D'
    const dv = new DataView(buf.buffer);
    expect(dv.getInt16(5)).toBe(3); // 3 fields
  });

  it("maps dtypes to OIDs", () => {
    expect(dtypeToOid("int64")).toBe(20);
    expect(dtypeToOid("float64")).toBe(701);
    expect(dtypeToOid("utf8")).toBe(25);
    expect(dtypeToOid("bool")).toBe(16);
  });
});

// ── Handler integration tests ───────────────────────────────────────────

describe("pg-wire handler", () => {
  it("completes startup handshake", async () => {
    const { send, sent } = createCollector();
    const handler = new PgConnectionHandler({ executor: makeExecutor(), send });

    await handler.onData(buildStartup({ user: "test", database: "data" }));

    // Should send: AuthOk + ParameterStatus(s) + BackendKeyData + ReadyForQuery
    const types = sent.map(b => messageType(b));
    expect(types).toContain("R"); // AuthenticationOk
    expect(types).toContain("S"); // ParameterStatus
    expect(types).toContain("K"); // BackendKeyData
    expect(types[types.length - 1]).toBe("Z"); // ReadyForQuery last
  });

  it("handles SSL request then startup", async () => {
    const { send, sent } = createCollector();
    const handler = new PgConnectionHandler({ executor: makeExecutor(), send });

    // Send SSL request
    const sslBuf = new Uint8Array(8);
    const dv = new DataView(sslBuf.buffer);
    dv.setInt32(0, 8);
    dv.setInt32(4, 80877103);
    await handler.onData(sslBuf);
    expect(sent[0][0]).toBe(0x4e); // 'N' — SSL refused

    // Then normal startup
    await handler.onData(buildStartup({ user: "test" }));
    const types = sent.map(b => messageType(b));
    expect(types).toContain("R"); // Auth ok
  });

  it("executes SELECT * and returns rows", async () => {
    const { send, sent } = createCollector();
    const handler = new PgConnectionHandler({ executor: makeExecutor(), send });

    // Startup
    await handler.onData(buildStartup({ user: "test" }));
    sent.length = 0; // clear startup messages

    // Query
    await handler.onData(buildQuery("SELECT * FROM data"));

    const types = sent.map(b => messageType(b));
    expect(types[0]).toBe("T"); // RowDescription
    // 3 DataRows (Alice, Bob, Charlie)
    const dataRows = types.filter(t => t === "D");
    expect(dataRows.length).toBe(3);
    expect(types).toContain("C"); // CommandComplete
    expect(types[types.length - 1]).toBe("Z"); // ReadyForQuery
  });

  it("executes filtered query", async () => {
    const { send, sent } = createCollector();
    const handler = new PgConnectionHandler({ executor: makeExecutor(), send });

    await handler.onData(buildStartup({ user: "test" }));
    sent.length = 0;

    await handler.onData(buildQuery("SELECT name FROM data WHERE dept = 'eng'"));

    const types = sent.map(b => messageType(b));
    const dataRows = types.filter(t => t === "D");
    expect(dataRows.length).toBe(2); // Alice, Bob (eng, age<=30 for filter pushdown test)
  });

  it("executes aggregate query", async () => {
    const { send, sent } = createCollector();
    const handler = new PgConnectionHandler({ executor: makeExecutor(), send });

    await handler.onData(buildStartup({ user: "test" }));
    sent.length = 0;

    await handler.onData(buildQuery("SELECT count(*) AS cnt FROM data"));

    const types = sent.map(b => messageType(b));
    expect(types[0]).toBe("T"); // RowDescription
    expect(types.filter(t => t === "D").length).toBe(1); // one result row
  });

  it("handles SET commands", async () => {
    const { send, sent } = createCollector();
    const handler = new PgConnectionHandler({ executor: makeExecutor(), send });

    await handler.onData(buildStartup({ user: "test" }));
    sent.length = 0;

    await handler.onData(buildQuery("SET client_encoding TO 'UTF8'"));
    const types = sent.map(b => messageType(b));
    expect(types).toContain("C"); // CommandComplete
    expect(types[types.length - 1]).toBe("Z");
  });

  it("returns error for bad SQL", async () => {
    const { send, sent } = createCollector();
    const handler = new PgConnectionHandler({ executor: makeExecutor(), send });

    await handler.onData(buildStartup({ user: "test" }));
    sent.length = 0;

    await handler.onData(buildQuery("INVALID SQL QUERY HERE"));
    const types = sent.map(b => messageType(b));
    expect(types[0]).toBe("E"); // ErrorResponse
    expect(types[types.length - 1]).toBe("Z"); // still ready after error
  });

  it("handles multiple queries in sequence", async () => {
    const { send, sent } = createCollector();
    const handler = new PgConnectionHandler({ executor: makeExecutor(), send });

    await handler.onData(buildStartup({ user: "test" }));
    sent.length = 0;

    await handler.onData(buildQuery("SELECT * FROM data LIMIT 1"));
    const firstBatch = sent.length;
    expect(sent.filter(b => messageType(b) === "D").length).toBe(1);

    await handler.onData(buildQuery("SELECT * FROM data"));
    const dataRowsTotal = sent.filter(b => messageType(b) === "D").length;
    expect(dataRowsTotal).toBe(4); // 1 from first + 3 from second
  });

  it("handles CTE via pg wire", async () => {
    const { send, sent } = createCollector();
    const handler = new PgConnectionHandler({ executor: makeExecutor(), send });

    await handler.onData(buildStartup({ user: "test" }));
    sent.length = 0;

    await handler.onData(buildQuery(
      "WITH engineers AS (SELECT * FROM data WHERE dept = 'eng') SELECT * FROM engineers"
    ));

    const types = sent.map(b => messageType(b));
    const dataRows = types.filter(t => t === "D");
    expect(dataRows.length).toBe(2); // Alice, Bob
  });
});
