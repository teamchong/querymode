/**
 * R2-backed spill storage for edge operator pipeline.
 *
 * Used by ExternalSortOperator and HashJoinOperator when running on
 * Cloudflare Workers/DOs where no filesystem is available.
 * Spill objects are stored under `__spill/{queryId}/` and cleaned up
 * in a finally block — no orphaned objects.
 */

import type { Row } from "./types.js";
import { withRetry, withTimeout } from "./coalesce.js";

const R2_SPILL_TIMEOUT_MS = 10_000;

/** Generic spill backend for sort/join operators. */
export interface SpillBackend {
  /** Write a sorted run of rows. Returns a spill ID for later streaming. */
  writeRun(rows: Row[]): Promise<string>;
  /** Stream rows back from a previously written run. */
  streamRun(spillId: string): AsyncGenerator<Row>;
  /** Total bytes written so far. */
  bytesWritten: number;
  /** Total bytes read so far. */
  bytesRead: number;
  /** Delete all spill data. Safe to call multiple times. */
  cleanup(): Promise<void>;
}

/** JSON replacer for bigint values in NDJSON spill runs. */
function spillJsonReplacer(_key: string, value: unknown): unknown {
  return typeof value === "bigint" ? `__bigint__${value.toString()}` : value;
}

/** Parse a single NDJSON line, restoring bigint values. */
function parseSpillRow(line: string): Row {
  return JSON.parse(line, (_key, value) => {
    if (typeof value === "string" && value.startsWith("__bigint__")) {
      return BigInt(value.slice(10));
    }
    return value;
  }) as Row;
}

/**
 * R2-backed spill backend for Cloudflare Workers edge.
 * Stores NDJSON files under `__spill/{prefix}/{runIndex}.ndjson`.
 */
export class R2SpillBackend implements SpillBackend {
  private bucket: R2Bucket;
  private prefix: string;
  private runCount = 0;
  private spillKeys: string[] = [];
  bytesWritten = 0;
  bytesRead = 0;

  constructor(bucket: R2Bucket, prefix: string) {
    this.bucket = bucket;
    this.prefix = prefix;
  }

  async writeRun(rows: Row[]): Promise<string> {
    const key = `${this.prefix}/${this.runCount++}.ndjson`;
    const encoder = new TextEncoder();

    // Stream rows one at a time to avoid building the entire body in memory.
    // Use a ReadableStream that encodes row-by-row.
    let rowIdx = 0;
    let totalBytes = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull: (controller) => {
        if (rowIdx >= rows.length) {
          controller.close();
          return;
        }
        const line = JSON.stringify(rows[rowIdx++], spillJsonReplacer) + "\n";
        const chunk = encoder.encode(line);
        totalBytes += chunk.byteLength;
        controller.enqueue(chunk);
      },
    });

    await withRetry(() =>
      withTimeout(this.bucket.put(key, stream), R2_SPILL_TIMEOUT_MS),
    );

    this.bytesWritten += totalBytes;
    this.spillKeys.push(key);
    return key;
  }

  async *streamRun(spillId: string): AsyncGenerator<Row> {
    const obj = await withRetry(() =>
      withTimeout(this.bucket.get(spillId), R2_SPILL_TIMEOUT_MS),
    );
    if (!obj) throw new Error(`Spill object not found: ${spillId}`);

    // Stream the body chunk-by-chunk to avoid loading entire spill file into memory
    const reader = obj.body!.getReader();
    const decoder = new TextDecoder();
    let partial = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        this.bytesRead += value.byteLength;
        partial += decoder.decode(value, { stream: true });

        // Yield complete lines
        const lines = partial.split("\n");
        partial = lines.pop()!; // last element is incomplete or empty
        for (const line of lines) {
          if (line.length === 0) continue;
          yield parseSpillRow(line);
        }
      }

      // Handle any remaining partial line
      partial += decoder.decode();
      if (partial.length > 0) {
        yield parseSpillRow(partial);
      }
    } finally {
      reader.releaseLock();
    }
  }

  async cleanup(): Promise<void> {
    const keys = this.spillKeys.splice(0);
    await Promise.all(
      keys.map(key => this.bucket.delete(key).catch(() => {})),
    );
  }
}
