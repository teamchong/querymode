import { DurableObject } from "cloudflare:workers";
import type { R2Partition, WorkerDORpc } from "./worker-pool.js";
import { decodeColumnarRun, encodeColumnarRun } from "./r2-spill.js";
import type { Row } from "./types.js";
import { rowComparator } from "./types.js";

interface WorkerEnv {
  DATA_BUCKET: R2Bucket;
  WORKER_DO: DurableObjectNamespace;
}

/**
 * Generic Worker Durable Object for distributed query execution.
 *
 * Receives task parameters via RPC, reads partitions from R2,
 * processes them, and writes results back to R2.
 *
 * Task handlers are registered at construction time.
 * Built-in tasks: hash_join_partition, sort_partition, window_partition,
 * distinct_partition, set_op_partition.
 */
export class WorkerDO extends DurableObject<WorkerEnv> implements WorkerDORpc {
  private bucket: R2Bucket;

  constructor(ctx: DurableObjectState, env: WorkerEnv) {
    super(ctx, env);
    this.bucket = env.DATA_BUCKET;
  }

  /** Read all rows from an R2 partition key */
  private async readPartition(key: string): Promise<Row[]> {
    const obj = await this.bucket.get(key);
    if (!obj) return [];
    const buf = await obj.arrayBuffer();
    const rows: Row[] = [];
    for (const row of decodeColumnarRun(buf)) {
      rows.push(row);
    }
    return rows;
  }

  /** Write rows to R2 and return partition metadata */
  private async writePartition(key: string, rows: Row[]): Promise<R2Partition> {
    const buf = encodeColumnarRun(rows);
    await this.bucket.put(key, buf);
    return {
      key,
      rowCount: rows.length,
      bytesWritten: buf.byteLength,
    };
  }

  /** Execute a single task on a partition */
  async executeTask(params: Record<string, unknown>): Promise<R2Partition[]> {
    const taskName = params.taskName as string;
    const partitionKey = params.partitionKey as string;
    const resultKeyPrefix = params.resultKeyPrefix as string;

    switch (taskName) {
      case "hash_join_partition":
        return this.hashJoinPartition(params, resultKeyPrefix);
      case "sort_partition":
        return this.sortPartition(partitionKey, params, resultKeyPrefix);
      case "distinct_partition":
        return this.distinctPartition(partitionKey, params, resultKeyPrefix);
      default:
        throw new Error(`Unknown task: ${taskName}`);
    }
  }

  /** Coordinator: fan out partitions to worker DOs */
  async executeCoordinator(params: Record<string, unknown>): Promise<R2Partition[]> {
    const partitions = params.partitions as R2Partition[];
    const taskName = params.taskName as string;
    const taskParams = params.taskParams as Record<string, unknown>;
    const resultKeyPrefix = params.resultKeyPrefix as string;

    const results = await Promise.all(
      partitions.map(async (partition, i) => {
        const workerId = `worker-${taskName}-${i}-${Date.now()}`;
        const id = this.env.WORKER_DO.idFromName(workerId);
        const handle = this.env.WORKER_DO.get(id) as unknown as WorkerDORpc;
        return handle.executeTask({
          taskName,
          partitionKey: partition.key,
          resultKeyPrefix,
          ...taskParams,
        });
      }),
    );

    return results.flat();
  }

  /** Super-coordinator: fan out to coordinators */
  async executeSuperCoordinator(params: Record<string, unknown>): Promise<R2Partition[]> {
    const partitionGroups = params.partitionGroups as R2Partition[][];
    const taskName = params.taskName as string;
    const taskParams = params.taskParams as Record<string, unknown>;
    const resultKeyPrefix = params.resultKeyPrefix as string;

    const results = await Promise.all(
      partitionGroups.map(async (group, ci) => {
        const coordId = `coord-${taskName}-${ci}-${Date.now()}`;
        const id = this.env.WORKER_DO.idFromName(coordId);
        const handle = this.env.WORKER_DO.get(id) as unknown as WorkerDORpc;
        return handle.executeCoordinator({
          taskName,
          partitions: group,
          taskParams,
          resultKeyPrefix,
        });
      }),
    );

    return results.flat();
  }

  // --- Built-in task handlers ---

  private async hashJoinPartition(
    params: Record<string, unknown>,
    resultKeyPrefix: string,
  ): Promise<R2Partition[]> {
    const leftKey = params.leftPartitionKey as string;
    const rightKey = params.rightPartitionKey as string;
    const joinKey = params.joinKey as string;
    const joinType = (params.joinType as string) ?? "inner";
    const resultKey = (params.resultKey as string) ?? `${resultKeyPrefix}/${crypto.randomUUID()}.bin`;

    const leftRows = await this.readPartition(leftKey);
    const rightRows = await this.readPartition(rightKey);

    // Build hash map from right side
    const rightMap = new Map<string, Row[]>();
    for (const row of rightRows) {
      const key = String(row[joinKey] ?? "__null__");
      const existing = rightMap.get(key);
      if (existing) existing.push(row);
      else rightMap.set(key, [row]);
    }

    // Probe with left side
    const result: Row[] = [];
    for (const leftRow of leftRows) {
      const key = String(leftRow[joinKey] ?? "__null__");
      const matches = rightMap.get(key);
      if (matches) {
        for (const rightRow of matches) {
          const merged: Row = { ...leftRow };
          for (const k in rightRow) {
            if (k === joinKey) continue;
            merged[k in merged ? `right_${k}` : k] = rightRow[k];
          }
          result.push(merged);
        }
      } else if (joinType === "left" || joinType === "full") {
        result.push({ ...leftRow });
      }
    }

    // Emit unmatched right rows for full join
    if (joinType === "full") {
      const matchedRightKeys = new Set<string>();
      for (const leftRow of leftRows) {
        const key = String(leftRow[joinKey] ?? "__null__");
        if (rightMap.has(key)) matchedRightKeys.add(key);
      }
      for (const [key, rows] of rightMap) {
        if (!matchedRightKeys.has(key)) {
          for (const row of rows) result.push(row);
        }
      }
    }

    if (result.length === 0) return [];
    return [await this.writePartition(resultKey, result)];
  }

  private async sortPartition(
    partitionKey: string,
    params: Record<string, unknown>,
    resultKeyPrefix: string,
  ): Promise<R2Partition[]> {
    const sortColumn = params.sortColumn as string;
    const sortDesc = (params.sortDirection === "desc");
    const resultKey = `${resultKeyPrefix}/${crypto.randomUUID()}.bin`;

    const rows = await this.readPartition(partitionKey);
    rows.sort(rowComparator(sortColumn, sortDesc));

    return [await this.writePartition(resultKey, rows)];
  }

  private async distinctPartition(
    partitionKey: string,
    params: Record<string, unknown>,
    resultKeyPrefix: string,
  ): Promise<R2Partition[]> {
    const columns = params.columns as string[] | undefined;
    const resultKey = `${resultKeyPrefix}/${crypto.randomUUID()}.bin`;

    const rows = await this.readPartition(partitionKey);
    const seen = new Set<string>();
    const unique: Row[] = [];

    for (const row of rows) {
      const key = columns
        ? columns.map(c => String(row[c] ?? "")).join("\x00")
        : Object.values(row).map(v => String(v ?? "")).join("\x00");
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(row);
      }
    }

    return [await this.writePartition(resultKey, unique)];
  }
}
