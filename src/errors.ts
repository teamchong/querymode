/**
 * QueryModeError — user-friendly errors with structured context.
 *
 * Wraps low-level errors (ENOENT, parse failures) into actionable messages.
 */

export type ErrorCode =
  | "TABLE_NOT_FOUND"
  | "COLUMN_NOT_FOUND"
  | "INVALID_FORMAT"
  | "SCHEMA_MISMATCH"
  | "INVALID_FILTER"
  | "INVALID_AGGREGATE"
  | "MEMORY_EXCEEDED"
  | "NETWORK_TIMEOUT"
  | "QUERY_TIMEOUT"
  | "QUERY_FAILED";

export class QueryModeError extends Error {
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string, cause?: unknown) {
    super(message);
    this.name = "QueryModeError";
    this.code = code;
    if (cause) this.cause = cause;
  }

  /** Wrap a low-level error into a QueryModeError with context. */
  static from(err: unknown, context: { table?: string; operation?: string } = {}): QueryModeError {
    if (err instanceof QueryModeError) return err;

    const raw = err instanceof Error ? err : new Error(String(err));
    const { table, operation } = context;
    const msg = raw.message;

    // ENOENT — table/file not found
    if ("code" in raw && (raw as NodeJS.ErrnoException).code === "ENOENT") {
      return new QueryModeError(
        "TABLE_NOT_FOUND",
        `Table not found: ${table ?? (raw as NodeJS.ErrnoException).path ?? "unknown"}. Check the path exists, use QueryMode.remote() for edge, or fromJSON()/fromCSV() for in-memory data.`,
        raw,
      );
    }

    // Parse failures
    if (msg.includes("footer") || msg.includes("Invalid file") || msg.includes("Failed to parse")) {
      return new QueryModeError(
        "INVALID_FORMAT",
        `Invalid table format${table ? `: ${table}` : ""}. Supported formats: .lance, .parquet, .csv, .tsv, .json, .ndjson, .jsonl, .arrow, .ipc, .feather`,
        raw,
      );
    }

    // Column/schema mismatches
    if (msg.includes("column") && (msg.includes("not found") || msg.includes("does not exist"))) {
      return new QueryModeError(
        "SCHEMA_MISMATCH",
        `Column not found${table ? ` in ${table}` : ""}: ${msg}`,
        raw,
      );
    }

    // Memory exceeded
    const msgLwr = msg.toLowerCase();
    if (msg.includes("OOM") || msgLwr.includes("memory") || msgLwr.includes("budget")) {
      return new QueryModeError(
        "MEMORY_EXCEEDED",
        `Memory budget exceeded${table ? ` querying ${table}` : ""}. Try adding filters, reducing projections, or increasing memoryBudgetBytes.`,
        raw,
      );
    }

    // Timeouts — case-insensitive (withTimeout throws "Timeout after ...")
    const msgLower = msg.toLowerCase();
    if (msgLower.includes("timeout") || msgLower.includes("timed out")) {
      const code = msgLower.includes("network") || msgLower.includes("r2") ? "NETWORK_TIMEOUT" : "QUERY_TIMEOUT";
      return new QueryModeError(
        code,
        `${code === "NETWORK_TIMEOUT" ? "Network" : "Query"} timeout${table ? ` on ${table}` : ""}: ${msg}`,
        raw,
      );
    }

    // Generic wrapper
    return new QueryModeError(
      "QUERY_FAILED",
      `${operation ?? "Query"} failed${table ? ` on ${table}` : ""}: ${msg}`,
      raw,
    );
  }
}
