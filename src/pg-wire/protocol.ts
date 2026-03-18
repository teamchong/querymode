/**
 * Postgres wire protocol v3 — message parsing and serialization.
 *
 * Implements the Simple Query protocol subset:
 *   Client → StartupMessage, Query, Terminate
 *   Server → AuthenticationOk, ReadyForQuery, RowDescription, DataRow,
 *            CommandComplete, ErrorResponse, ParameterStatus
 *
 * Reference: https://www.postgresql.org/docs/current/protocol-message-formats.html
 */

// ── Frontend (client → server) messages ─────────────────────────────────

export interface StartupMessage {
  type: "startup";
  protocolVersion: number;
  params: Map<string, string>; // user, database, client_encoding, etc.
}

export interface QueryMessage {
  type: "query";
  sql: string;
}

export interface TerminateMessage {
  type: "terminate";
}

export interface SSLRequest {
  type: "ssl_request";
}

export type FrontendMessage = StartupMessage | QueryMessage | TerminateMessage | SSLRequest;

// ── Parsing ─────────────────────────────────────────────────────────────

const SSL_REQUEST_CODE = 80877103;
const PROTOCOL_VERSION_3 = 196608; // 3.0

/**
 * Parse the initial startup message (no type byte — length-prefixed only).
 * Returns null if more data is needed.
 */
export function parseStartupMessage(buf: Uint8Array): FrontendMessage | null {
  if (buf.length < 8) return null;
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const len = dv.getInt32(0);
  if (buf.length < len) return null;

  const code = dv.getInt32(4);

  if (code === SSL_REQUEST_CODE) {
    return { type: "ssl_request" };
  }

  if (code !== PROTOCOL_VERSION_3) {
    throw new Error(`Unsupported protocol version: ${code}`);
  }

  const params = new Map<string, string>();
  let pos = 8;
  while (pos < len - 1) {
    const keyEnd = buf.indexOf(0, pos);
    if (keyEnd === -1 || keyEnd >= len) break;
    const key = textDecoder.decode(buf.subarray(pos, keyEnd));
    pos = keyEnd + 1;
    const valEnd = buf.indexOf(0, pos);
    if (valEnd === -1 || valEnd >= len) break;
    const val = textDecoder.decode(buf.subarray(pos, valEnd));
    pos = valEnd + 1;
    params.set(key, val);
  }

  return { type: "startup", protocolVersion: code, params };
}

/**
 * Parse a regular frontend message (type byte + length + payload).
 * Returns [message, bytesConsumed] or null if more data is needed.
 */
export function parseFrontendMessage(buf: Uint8Array): [FrontendMessage, number] | null {
  if (buf.length < 5) return null;
  const type = buf[0];
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const len = dv.getInt32(1); // length includes self but not type byte
  const totalLen = 1 + len;
  if (buf.length < totalLen) return null;

  switch (type) {
    case 0x51: { // 'Q' — Simple Query
      // SQL string is null-terminated
      const sql = textDecoder.decode(buf.subarray(5, totalLen - 1));
      return [{ type: "query", sql }, totalLen];
    }
    case 0x58: // 'X' — Terminate
      return [{ type: "terminate" }, totalLen];
    default:
      // Skip unknown messages
      return [{ type: "terminate" }, totalLen];
  }
}

// ── Backend (server → client) messages ──────────────────────────────────

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

/** 'N' — SSL not supported (single byte response to SSLRequest) */
export function sslRefused(): Uint8Array {
  return new Uint8Array([0x4e]); // 'N'
}

/** 'R' — AuthenticationOk */
export function authenticationOk(): Uint8Array {
  const buf = new Uint8Array(9);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x52; // 'R'
  dv.setInt32(1, 8); // length
  dv.setInt32(5, 0); // auth type 0 = ok
  return buf;
}

/** 'S' — ParameterStatus (key=value) */
export function parameterStatus(key: string, value: string): Uint8Array {
  const keyBytes = textEncoder.encode(key);
  const valBytes = textEncoder.encode(value);
  const len = 4 + keyBytes.length + 1 + valBytes.length + 1;
  const buf = new Uint8Array(1 + len);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x53; // 'S'
  dv.setInt32(1, len);
  let pos = 5;
  buf.set(keyBytes, pos); pos += keyBytes.length; buf[pos++] = 0;
  buf.set(valBytes, pos); pos += valBytes.length; buf[pos++] = 0;
  return buf;
}

/** 'K' — BackendKeyData (process ID + secret key for cancel) */
export function backendKeyData(pid: number, secretKey: number): Uint8Array {
  const buf = new Uint8Array(13);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x4b; // 'K'
  dv.setInt32(1, 12);
  dv.setInt32(5, pid);
  dv.setInt32(9, secretKey);
  return buf;
}

/** 'Z' — ReadyForQuery (transaction status: 'I' = idle) */
export function readyForQuery(status: "I" | "T" | "E" = "I"): Uint8Array {
  const buf = new Uint8Array(6);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x5a; // 'Z'
  dv.setInt32(1, 5);
  buf[5] = status.charCodeAt(0);
  return buf;
}

/** 'T' — RowDescription */
export function rowDescription(columns: { name: string; oid: number }[]): Uint8Array {
  // Calculate total size
  let bodyLen = 2; // field count (Int16)
  for (const col of columns) {
    bodyLen += textEncoder.encode(col.name).length + 1 + 18;
    // name + null + tableOid(4) + colAttrNum(2) + typeOid(4) + typeLen(2) + typeMod(4) + format(2)
  }
  const buf = new Uint8Array(1 + 4 + bodyLen);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x54; // 'T'
  dv.setInt32(1, 4 + bodyLen);
  dv.setInt16(5, columns.length);

  let pos = 7;
  for (const col of columns) {
    const nameBytes = textEncoder.encode(col.name);
    buf.set(nameBytes, pos); pos += nameBytes.length; buf[pos++] = 0;
    dv.setInt32(pos, 0); pos += 4;      // table OID
    dv.setInt16(pos, 0); pos += 2;      // column attr number
    dv.setInt32(pos, col.oid); pos += 4; // type OID
    dv.setInt16(pos, -1); pos += 2;     // type length (-1 = variable)
    dv.setInt32(pos, -1); pos += 4;     // type modifier
    dv.setInt16(pos, 0); pos += 2;      // format code (0 = text)
  }
  return buf;
}

/** 'D' — DataRow */
export function dataRow(values: (string | null)[]): Uint8Array {
  // Pre-encode to avoid double textEncoder.encode() per value
  const encoded = new Array<Uint8Array | null>(values.length);
  let bodyLen = 2; // field count
  for (let i = 0; i < values.length; i++) {
    bodyLen += 4; // length prefix per field
    if (values[i] !== null) {
      const bytes = textEncoder.encode(values[i]!);
      encoded[i] = bytes;
      bodyLen += bytes.length;
    } else {
      encoded[i] = null;
    }
  }
  const buf = new Uint8Array(1 + 4 + bodyLen);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x44; // 'D'
  dv.setInt32(1, 4 + bodyLen);
  dv.setInt16(5, values.length);

  let pos = 7;
  for (let i = 0; i < values.length; i++) {
    const bytes = encoded[i];
    if (bytes === null) {
      dv.setInt32(pos, -1); pos += 4; // NULL
    } else {
      dv.setInt32(pos, bytes.length); pos += 4;
      buf.set(bytes, pos); pos += bytes.length;
    }
  }
  return buf;
}

/** 'C' — CommandComplete */
export function commandComplete(tag: string): Uint8Array {
  const tagBytes = textEncoder.encode(tag);
  const len = 4 + tagBytes.length + 1;
  const buf = new Uint8Array(1 + len);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x43; // 'C'
  dv.setInt32(1, len);
  buf.set(tagBytes, 5);
  buf[5 + tagBytes.length] = 0;
  return buf;
}

/** 'E' — ErrorResponse */
export function errorResponse(message: string, code = "42000"): Uint8Array {
  const parts: Uint8Array[] = [];
  // Severity
  parts.push(field(0x53, "ERROR")); // 'S'
  // SQLSTATE code
  parts.push(field(0x43, code));     // 'C'
  // Message
  parts.push(field(0x4d, message));  // 'M'
  // Terminator
  parts.push(new Uint8Array([0]));

  let bodyLen = 0;
  for (const p of parts) bodyLen += p.length;

  const buf = new Uint8Array(1 + 4 + bodyLen);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x45; // 'E'
  dv.setInt32(1, 4 + bodyLen);
  let pos = 5;
  for (const p of parts) { buf.set(p, pos); pos += p.length; }
  return buf;
}

function field(type: number, value: string): Uint8Array {
  const bytes = textEncoder.encode(value);
  const buf = new Uint8Array(1 + bytes.length + 1);
  buf[0] = type;
  buf.set(bytes, 1);
  buf[1 + bytes.length] = 0;
  return buf;
}

// ── Type OID mapping ────────────────────────────────────────────────────

/** Map QueryMode dtypes to Postgres type OIDs */
export function dtypeToOid(dtype: string): number {
  switch (dtype) {
    case "int8": case "int16": case "uint8": case "uint16": return 21;  // INT2
    case "int32": case "uint32": return 23;                              // INT4
    case "int64": case "uint64": return 20;                              // INT8
    case "float16": case "float32": return 700;                          // FLOAT4
    case "float64": return 701;                                          // FLOAT8
    case "utf8": case "string": return 25;                               // TEXT
    case "bool": case "boolean": return 16;                              // BOOL
    case "binary": case "blob": return 17;                               // BYTEA
    default: return 25;                                                  // TEXT fallback
  }
}
