import type { Row } from "./types.js";
import { encodeColumnarRun, decodeColumnarRun } from "./r2-spill.js";
import { QueryModeError } from "./errors.js";

/** A partition of data stored in R2 */
export interface R2Partition {
  key: string;           // R2 object key
  rowCount: number;
  bytesWritten: number;
}

/** Partitioner — streams data into R2 partitions by hash key */
export class R2Partitioner {
  private bucket: R2Bucket;
  private prefix: string;
  private partitionCount: number;
  private buckets: Row[][];
  private bucketBytes: number[];
  private flushThreshold: number;
  private partitionKeys: string[];
  private partitionRowCounts: number[];
  private partitionBytesWritten: number[];

  constructor(bucket: R2Bucket, prefix: string, partitionCount: number) {
    if (partitionCount < 1) throw new QueryModeError("QUERY_FAILED", "R2Partitioner: partitionCount must be >= 1");
    this.bucket = bucket;
    this.prefix = prefix;
    this.partitionCount = partitionCount;
    this.buckets = Array.from({ length: partitionCount }, () => []);
    this.bucketBytes = new Array(partitionCount).fill(0);
    this.flushThreshold = 4 * 1024 * 1024; // 4MB per partition bucket
    this.partitionKeys = Array.from({ length: partitionCount }, (_, i) => `${prefix}/part-${String(i).padStart(4, '0')}.bin`);
    this.partitionRowCounts = new Array(partitionCount).fill(0);
    this.partitionBytesWritten = new Array(partitionCount).fill(0);
  }

  private hashKey(key: string): number {
    let h = 0;
    for (let i = 0; i < key.length; i++) {
      h = ((h << 5) - h + key.charCodeAt(i)) | 0;
    }
    return ((h % this.partitionCount) + this.partitionCount) % this.partitionCount;
  }

  /** Stream rows into partitions (bounded memory — flushes per bucket when full) */
  async addBatch(rows: Row[], partitionKeyFn: (row: Row) => string): Promise<void> {
    for (const row of rows) {
      const key = partitionKeyFn(row);
      const pi = this.hashKey(key);
      this.buckets[pi].push(row);
      // Rough size estimate
      let rowSize = 64;
      for (const k in row) {
        const val = row[k];
        if (typeof val === "string") rowSize += 40 + val.length * 2;
        else if (val instanceof Float32Array) rowSize += 40 + val.byteLength;
        else rowSize += 16;
      }
      this.bucketBytes[pi] += rowSize;

      if (this.bucketBytes[pi] >= this.flushThreshold) {
        await this.flushBucket(pi);
      }
    }
  }

  private async flushBucket(pi: number): Promise<void> {
    if (this.buckets[pi].length === 0) return;
    const buf = encodeColumnarRun(this.buckets[pi]);
    const chunkKey = `${this.prefix}/part-${String(pi).padStart(4, '0')}-${this.partitionRowCounts[pi]}.bin`;
    await this.bucket.put(chunkKey, buf);
    this.partitionRowCounts[pi] += this.buckets[pi].length;
    this.partitionBytesWritten[pi] += buf.byteLength;
    this.buckets[pi] = [];
    this.bucketBytes[pi] = 0;
  }

  /** Flush remaining buffers, return partition metadata */
  async finalize(): Promise<R2Partition[]> {
    for (let pi = 0; pi < this.partitionCount; pi++) {
      await this.flushBucket(pi);
    }

    const partitions: R2Partition[] = [];
    for (let pi = 0; pi < this.partitionCount; pi++) {
      if (this.partitionRowCounts[pi] > 0) {
        partitions.push({
          key: this.partitionKeys[pi],
          rowCount: this.partitionRowCounts[pi],
          bytesWritten: this.partitionBytesWritten[pi],
        });
      }
    }
    return partitions;
  }

  /** Clean up all partition data */
  async cleanup(): Promise<void> {
    // List and delete all objects under prefix
    const listed = await this.bucket.list({ prefix: this.prefix + "/" });
    await Promise.all(
      listed.objects.map(obj => this.bucket.delete(obj.key).catch(() => {})),
    );
  }
}

/** Tree fan-out coordinator */
export class WorkerPool {
  private bucket: R2Bucket;
  private doNamespace: DurableObjectNamespace;
  private maxFanOut: number;

  constructor(opts: {
    bucket: R2Bucket;
    doNamespace: DurableObjectNamespace;
    maxFanOut?: number;
  }) {
    this.bucket = opts.bucket;
    this.doNamespace = opts.doNamespace;
    const fanOut = opts.maxFanOut ?? 50;
    if (fanOut < 1) throw new QueryModeError("QUERY_FAILED", "WorkerPool: maxFanOut must be >= 1");
    this.maxFanOut = fanOut;
  }

  /**
   * Fan out partitions to worker DOs via tree.
   * - <= maxFanOut partitions: direct fan-out (1 level)
   * - <= maxFanOut^2 partitions: coordinator -> worker (2 levels)
   * - > maxFanOut^2: coordinator -> coordinator -> worker (3 levels)
   *
   * Each worker DO calls `taskName` RPC with its partition keys.
   * Workers read from R2, process, write results to R2.
   */
  async execute(
    partitions: R2Partition[],
    taskName: string,
    taskParams: Record<string, unknown>,
  ): Promise<R2Partition[]> {
    if (partitions.length === 0) return [];

    if (partitions.length <= this.maxFanOut) {
      // Direct fan-out: call each worker DO directly
      return this.directFanOut(partitions, taskName, taskParams);
    }

    if (partitions.length <= this.maxFanOut * this.maxFanOut) {
      // Two-level tree: coordinators then workers
      return this.twoLevelFanOut(partitions, taskName, taskParams);
    }

    // Three-level tree: coordinators then coordinators then workers
    return this.threeLevelFanOut(partitions, taskName, taskParams);
  }

  private async directFanOut(
    partitions: R2Partition[],
    taskName: string,
    taskParams: Record<string, unknown>,
  ): Promise<R2Partition[]> {
    const results = await Promise.all(
      partitions.map(async (partition, i) => {
        const workerId = `worker-${taskName}-${i}-${Date.now()}`;
        const id = this.doNamespace.idFromName(workerId);
        const handle = this.doNamespace.get(id) as unknown as WorkerDORpc;
        return handle.executeTask({
          taskName,
          partitionKey: partition.key,
          resultKeyPrefix: `__shuffle/${taskName}/result`,
          ...taskParams,
        });
      }),
    );

    return results.flat();
  }

  private async twoLevelFanOut(
    partitions: R2Partition[],
    taskName: string,
    taskParams: Record<string, unknown>,
  ): Promise<R2Partition[]> {
    // Split partitions into chunks of maxFanOut
    const chunks: R2Partition[][] = [];
    for (let i = 0; i < partitions.length; i += this.maxFanOut) {
      chunks.push(partitions.slice(i, i + this.maxFanOut));
    }

    const results = await Promise.all(
      chunks.map(async (chunk, ci) => {
        const coordId = `coord-${taskName}-${ci}-${Date.now()}`;
        const id = this.doNamespace.idFromName(coordId);
        const handle = this.doNamespace.get(id) as unknown as WorkerDORpc;
        return handle.executeCoordinator({
          taskName,
          partitions: chunk,
          taskParams,
          resultKeyPrefix: `__shuffle/${taskName}/result`,
          doNamespaceName: "WORKER_DO",
        });
      }),
    );

    return results.flat();
  }

  private async threeLevelFanOut(
    partitions: R2Partition[],
    taskName: string,
    taskParams: Record<string, unknown>,
  ): Promise<R2Partition[]> {
    // Split into maxFanOut groups of maxFanOut chunks each
    const chunkSize = this.maxFanOut;
    const chunks: R2Partition[][] = [];
    for (let i = 0; i < partitions.length; i += chunkSize) {
      chunks.push(partitions.slice(i, i + chunkSize));
    }

    const superChunks: R2Partition[][][] = [];
    for (let i = 0; i < chunks.length; i += this.maxFanOut) {
      superChunks.push(chunks.slice(i, i + this.maxFanOut));
    }

    const results = await Promise.all(
      superChunks.map(async (superChunk, sci) => {
        const superCoordId = `supercoord-${taskName}-${sci}-${Date.now()}`;
        const id = this.doNamespace.idFromName(superCoordId);
        const handle = this.doNamespace.get(id) as unknown as WorkerDORpc;
        return handle.executeSuperCoordinator({
          taskName,
          partitionGroups: superChunk,
          taskParams,
          resultKeyPrefix: `__shuffle/${taskName}/result`,
          doNamespaceName: "WORKER_DO",
        });
      }),
    );

    return results.flat();
  }
}

/** RPC interface for Worker DOs */
export interface WorkerDORpc {
  executeTask(params: Record<string, unknown>): Promise<R2Partition[]>;
  executeCoordinator(params: Record<string, unknown>): Promise<R2Partition[]>;
  executeSuperCoordinator(params: Record<string, unknown>): Promise<R2Partition[]>;
}
