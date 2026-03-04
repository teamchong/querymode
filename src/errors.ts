/**
 * QueryModeError — user-friendly errors with structured context.
 *
 * Wraps low-level errors (ENOENT, parse failures) into actionable messages.
 */
export class QueryModeError extends Error {
  readonly code: string;

  constructor(code: string, message: string, cause?: unknown) {
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

    // ENOENT — table/file not found
    if ("code" in raw && (raw as NodeJS.ErrnoException).code === "ENOENT") {
      return new QueryModeError(
        "TABLE_NOT_FOUND",
        `Table not found: ${table ?? (raw as NodeJS.ErrnoException).path ?? "unknown"}`,
        raw,
      );
    }

    // Parse failures
    if (raw.message.includes("footer") || raw.message.includes("Invalid file")) {
      return new QueryModeError(
        "INVALID_FORMAT",
        `Invalid table format${table ? `: ${table}` : ""}. Expected Lance or Parquet file.`,
        raw,
      );
    }

    // Generic wrapper
    return new QueryModeError(
      "QUERY_FAILED",
      `${operation ?? "Query"} failed${table ? ` on ${table}` : ""}: ${raw.message}`,
      raw,
    );
  }
}
