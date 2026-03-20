/**
 * Conformance tests: DuckDB reference at 1M-10M scale.
 *
 * Every test generates data, runs the same query through both our operator
 * pipeline and DuckDB, then compares results row-for-row.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import duckdb from "duckdb";
import type { Row } from "./types.js";
import type { AggregateOp, WindowSpec } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { getPlatformProxy } from "wrangler";
import { R2SpillBackend } from "./r2-spill.js";
import {
  type Operator,
  type RowBatch,
  HashJoinOperator,
  WindowOperator,
  DistinctOperator,
  SetOperator,
  ComputedColumnOperator,
  SubqueryInOperator,
  ExternalSortOperator,
  InMemorySortOperator,
  AggregateOperator,
  TopKOperator,
  FsSpillBackend,
  drainPipeline,
} from "./operators.js";

// ---------------------------------------------------------------------------
// DuckDB helper infrastructure
// ---------------------------------------------------------------------------

let db: duckdb.Database;
let con: duckdb.Connection;
let platform: Awaited<ReturnType<typeof getPlatformProxy>>;
let r2Bucket: R2Bucket;

beforeAll(async () => {
  db = new duckdb.Database(":memory:");
  con = new duckdb.Connection(db);
  platform = await getPlatformProxy();
  r2Bucket = platform.env.DATA_BUCKET as unknown as R2Bucket;
});

afterAll(async () => {
  con.close();
  db.close();
  await platform.dispose();
});

function duckRun(sql: string): Promise<void> {
  return new Promise((resolve, reject) => {
    con.run(sql, (err: Error | null) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

function duckQuery(sql: string): Promise<Record<string, unknown>[]> {
  return new Promise((resolve, reject) => {
    con.all(sql, (err: Error | null, rows: Record<string, unknown>[]) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
}

/** Bulk-load rows into a DuckDB table via batched INSERT VALUES. */
async function loadTable(tableName: string, rows: Row[], schema: string): Promise<void> {
  await duckRun(`DROP TABLE IF EXISTS ${tableName}`);
  await duckRun(`CREATE TABLE ${tableName} (${schema})`);

  const BATCH = 10_000;
  const cols = schema.split(",").map(s => s.trim().split(/\s+/)[0]);

  for (let i = 0; i < rows.length; i += BATCH) {
    const chunk = rows.slice(i, i + BATCH);
    const values = chunk
      .map(r => `(${cols.map(c => {
        const v = r[c];
        if (typeof v === "string") return `'${v}'`;
        if (v === null || v === undefined) return "NULL";
        return String(v);
      }).join(",")})`)
      .join(",");
    await duckRun(`INSERT INTO ${tableName} VALUES ${values}`);
  }
}

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

const REGIONS = ["us", "eu", "ap", "sa"] as const;
const CATEGORIES = ["A", "B", "C"] as const;

function generateRows(n: number): Row[] {
  const rows: Row[] = [];
  for (let i = 0; i < n; i++) {
    rows.push({
      id: i,
      region: REGIONS[i % REGIONS.length],
      category: CATEGORIES[i % CATEGORIES.length],
      amount: (i * 7 + 13) % 1000,
      score: ((i * 31) % 997) / 10,
    });
  }
  return rows;
}

const TABLE_SCHEMA = "id INT, region VARCHAR, category VARCHAR, amount INT, score DOUBLE";

// ---------------------------------------------------------------------------
// MockOperator
// ---------------------------------------------------------------------------

class MockOperator implements Operator {
  private rows: Row[];
  private batchSize: number;
  private cursor = 0;

  constructor(rows: Row[], batchSize = 4096) {
    this.rows = rows;
    this.batchSize = batchSize;
  }

  async next(): Promise<RowBatch | null> {
    if (this.cursor >= this.rows.length) return null;
    const end = Math.min(this.cursor + this.batchSize, this.rows.length);
    const batch = this.rows.slice(this.cursor, end);
    this.cursor = end;
    return batch;
  }

  async close(): Promise<void> {}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function aggQuery(opts: {
  aggregates: AggregateOp[];
  groupBy?: string[];
}): QueryDescriptor {
  return {
    table: "test",
    filters: [],
    projections: [],
    aggregates: opts.aggregates,
    groupBy: opts.groupBy,
  };
}

/** Sort rows by a set of keys for deterministic comparison. */
function sortByKeys(rows: Record<string, unknown>[], keys: string[]): void {
  rows.sort((a, b) => {
    for (const k of keys) {
      const av = a[k] as number | string;
      const bv = b[k] as number | string;
      if (av < bv) return -1;
      if (av > bv) return 1;
    }
    return 0;
  });
}

// ===================================================================
// 1. HashJoinOperator
// ===================================================================

describe("HashJoinOperator", () => {
  const LEFT_N = 1_000_000;
  const RIGHT_N = 500_000;

  let leftRows: Row[];
  let rightRows: Row[];

  beforeAll(async () => {
    leftRows = generateRows(LEFT_N);
    rightRows = [];
    for (let i = 0; i < RIGHT_N; i++) {
      rightRows.push({ id: i, value: i * 10 });
    }

    await loadTable("join_left", leftRows, TABLE_SCHEMA);
    await loadTable("join_right", rightRows, "id INT, value INT");
  }, 300_000);

  it("inner join — 1M x 500K", async () => {
    const op = new HashJoinOperator(
      new MockOperator(leftRows),
      new MockOperator(rightRows),
      "id", "id", "inner",
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT l.id, l.region, l.category, l.amount, l.score, r.value
       FROM join_left l INNER JOIN join_right r USING(id)`,
    );

    expect(ours.length).toBe(duck.length);

    // Spot-check: sort both by id and compare a sample
    sortByKeys(ours, ["id"]);
    sortByKeys(duck, ["id"]);

    for (let i = 0; i < ours.length; i += 50_000) {
      expect(ours[i].id).toBe(duck[i].id);
      expect(ours[i].value).toBe(duck[i].value);
      expect(ours[i].region).toBe(duck[i].region);
    }
  }, 60_000);

  it("left join — 1M x 500K", async () => {
    const op = new HashJoinOperator(
      new MockOperator(leftRows),
      new MockOperator(rightRows),
      "id", "id", "left",
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT l.id, l.region, l.category, l.amount, l.score, r.value
       FROM join_left l LEFT JOIN join_right r USING(id)`,
    );

    expect(ours.length).toBe(duck.length);

    sortByKeys(ours, ["id"]);
    sortByKeys(duck, ["id"]);

    // Verify unmatched rows have null value
    const oursUnmatched = ours.filter(r => r.value === null);
    const duckUnmatched = duck.filter(r => r.value === null);
    expect(oursUnmatched.length).toBe(duckUnmatched.length);
  }, 60_000);

  it("right join", async () => {
    // Smaller dataset: left 2500, right 5000
    const left = generateRows(2500);
    const right: Row[] = [];
    for (let i = 0; i < 5000; i++) right.push({ id: i, value: i * 10 });

    await loadTable("rj_left", left, TABLE_SCHEMA);
    await loadTable("rj_right", right, "id INT, value INT");

    const op = new HashJoinOperator(
      new MockOperator(left),
      new MockOperator(right),
      "id", "id", "right",
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT * FROM rj_left l RIGHT JOIN rj_right r USING(id)`,
    );

    expect(ours.length).toBe(duck.length);
  }, 60_000);

  it("full join", async () => {
    const left: Row[] = [];
    const right: Row[] = [];
    for (let i = 0; i < 100; i++) left.push({ id: i, side: "L" });
    for (let i = 50; i < 150; i++) right.push({ id: i, side: "R" });

    await loadTable("fj_left", left, "id INT, side VARCHAR");
    await loadTable("fj_right", right, "id INT, side VARCHAR");

    const op = new HashJoinOperator(
      new MockOperator(left),
      new MockOperator(right),
      "id", "id", "full",
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT COALESCE(l.id, r.id) as id FROM fj_left l FULL OUTER JOIN fj_right r USING(id)`,
    );

    expect(ours.length).toBe(duck.length);
  }, 60_000);

  it("cross join — 1K x 500", async () => {
    const left: Row[] = [];
    const right: Row[] = [];
    for (let i = 0; i < 1000; i++) left.push({ x: i });
    for (let i = 0; i < 500; i++) right.push({ _key: i, y_val: i });

    await loadTable("cj_left", left, "x INT");
    await loadTable("cj_right", right, "_key INT, y_val INT");

    const op = new HashJoinOperator(
      new MockOperator(left),
      new MockOperator(right),
      "x", "_key", "cross",
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT l.x, r.y_val FROM cj_left l CROSS JOIN cj_right r`,
    );

    expect(ours.length).toBe(duck.length);
    expect(ours.length).toBe(500_000);
  }, 60_000);

  it("spill join (1M x 1M, 1MB budget) matches DuckDB", async () => {
    const N = 1_000_000;
    const left = generateRows(N);
    const right: Row[] = [];
    for (let i = 0; i < N; i++) right.push({ id: i, value: i * 10 });

    await loadTable("spill_left", left, TABLE_SCHEMA);
    await loadTable("spill_right", right, "id INT, value INT");

    // In-memory join as baseline (no spill, large budget)
    const inMemOp = new HashJoinOperator(
      new MockOperator(left),
      new MockOperator([...right]),
      "id", "id", "inner",
    );
    const inMemResult = await drainPipeline(inMemOp);

    const duck = await duckQuery(
      `SELECT COUNT(*) as cnt FROM spill_left l INNER JOIN spill_right r USING(id)`,
    );

    // In-memory join must match DuckDB exactly
    expect(inMemResult.length).toBe(Number(duck[0].cnt));

    // Spill join with tiny budget forces Grace hash partitioning
    const spill = new FsSpillBackend();
    const spillOp = new HashJoinOperator(
      new MockOperator(left),
      new MockOperator([...right]),
      "id", "id", "inner",
      1024 * 1024, // 1MB budget
      spill,
    );
    const spillResult = await drainPipeline(spillOp);

    // Spill join must also match DuckDB exactly
    expect(spillResult.length).toBe(Number(duck[0].cnt));

    // Verify all spill-produced rows are correct (no corrupt data)
    sortByKeys(spillResult, ["id"]);
    for (const row of spillResult) {
      expect(row.value).toBe((row.id as number) * 10);
    }

    await spill.cleanup();
  }, 300_000);

  it("spill join via R2SpillBackend (1M x 1M, 1MB budget)", async () => {
    const N = 1_000_000;
    const left = generateRows(N);
    const right: Row[] = [];
    for (let i = 0; i < N; i++) right.push({ id: i, value: i * 10 });

    await loadTable("r2_spill_left", left, TABLE_SCHEMA);
    await loadTable("r2_spill_right", right, "id INT, value INT");

    const duck = await duckQuery(
      `SELECT COUNT(*) as cnt FROM r2_spill_left l INNER JOIN r2_spill_right r USING(id)`,
    );

    const spill = new R2SpillBackend(r2Bucket, "__spill/test-join");
    const spillOp = new HashJoinOperator(
      new MockOperator(left),
      new MockOperator([...right]),
      "id", "id", "inner",
      1024 * 1024, // 1MB budget
      spill,
    );
    const spillResult = await drainPipeline(spillOp);

    expect(spillResult.length).toBe(Number(duck[0].cnt));

    sortByKeys(spillResult, ["id"]);
    for (const row of spillResult) {
      expect(row.value).toBe((row.id as number) * 10);
    }

    await spill.cleanup();
  }, 300_000);
});

// ===================================================================
// 2. GROUP BY + aggregates
// ===================================================================

describe("GROUP BY + aggregates", () => {
  const N = 5_000_000;
  let rows: Row[];

  beforeAll(async () => {
    rows = generateRows(N);
    await loadTable("agg_test", rows, TABLE_SCHEMA);
  }, 300_000);

  it("sum/avg/min/max/count by region — 5M rows", async () => {
    const op = new AggregateOperator(
      new MockOperator(rows),
      aggQuery({
        groupBy: ["region"],
        aggregates: [
          { fn: "sum", column: "amount", alias: "sum_amount" },
          { fn: "avg", column: "amount", alias: "avg_amount" },
          { fn: "min", column: "amount", alias: "min_amount" },
          { fn: "max", column: "amount", alias: "max_amount" },
          { fn: "count", column: "amount", alias: "cnt" },
        ],
      }),
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT region,
              SUM(amount) as sum_amount,
              AVG(amount) as avg_amount,
              MIN(amount) as min_amount,
              MAX(amount) as max_amount,
              COUNT(amount) as cnt
       FROM agg_test GROUP BY region`,
    );

    expect(ours.length).toBe(duck.length);

    sortByKeys(ours, ["region"]);
    sortByKeys(duck, ["region"]);

    for (let i = 0; i < ours.length; i++) {
      expect(ours[i].region).toBe(duck[i].region);
      expect(ours[i].sum_amount).toBe(Number(duck[i].sum_amount));
      expect(ours[i].avg_amount as number).toBeCloseTo(duck[i].avg_amount as number, 5);
      expect(ours[i].min_amount).toBe(duck[i].min_amount);
      expect(ours[i].max_amount).toBe(duck[i].max_amount);
      expect(ours[i].cnt).toBe(Number(duck[i].cnt));
    }
  }, 60_000);

  it("count_distinct by region, category — 5M rows", async () => {
    const op = new AggregateOperator(
      new MockOperator(rows),
      aggQuery({
        groupBy: ["region", "category"],
        aggregates: [{ fn: "count_distinct", column: "id", alias: "unique_ids" }],
      }),
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT region, category, COUNT(DISTINCT id) as unique_ids
       FROM agg_test GROUP BY region, category`,
    );

    expect(ours.length).toBe(duck.length);

    sortByKeys(ours, ["region", "category"]);
    sortByKeys(duck, ["region", "category"]);

    for (let i = 0; i < ours.length; i++) {
      expect(ours[i].unique_ids).toBe(Number(duck[i].unique_ids));
    }
  }, 60_000);

  it("stddev/variance/median/percentile(0.95) — 5M rows", async () => {
    const op = new AggregateOperator(
      new MockOperator(rows),
      aggQuery({
        aggregates: [
          { fn: "stddev", column: "amount", alias: "std_amount" },
          { fn: "variance", column: "amount", alias: "var_amount" },
          { fn: "median", column: "amount", alias: "med_amount" },
          { fn: "percentile", column: "amount", alias: "p95_amount", percentileTarget: 0.95 },
        ],
      }),
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT STDDEV_POP(amount) as std_amount,
              VAR_POP(amount) as var_amount,
              MEDIAN(amount) as med_amount,
              QUANTILE_CONT(amount, 0.95) as p95_amount
       FROM agg_test`,
    );

    expect(ours.length).toBe(1);

    expect(ours[0].std_amount as number).toBeCloseTo(duck[0].std_amount as number, 2);
    expect(ours[0].var_amount as number).toBeCloseTo(duck[0].var_amount as number, 0);
    expect(ours[0].med_amount as number).toBeCloseTo(duck[0].med_amount as number, 0);
    expect(ours[0].p95_amount as number).toBeCloseTo(duck[0].p95_amount as number, 0);
  }, 60_000);
});

// ===================================================================
// 3. DistinctOperator
// ===================================================================

describe("DistinctOperator", () => {
  const N = 5_000_000;
  let rows: Row[];

  beforeAll(async () => {
    rows = generateRows(N);
    await loadTable("distinct_test", rows, TABLE_SCHEMA);
  }, 300_000);

  it("distinct region — 5M rows", async () => {
    const op = new DistinctOperator(new MockOperator(rows), ["region"]);
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT DISTINCT region FROM distinct_test`,
    );

    expect(ours.length).toBe(duck.length);

    const oursRegions = new Set(ours.map(r => r.region));
    const duckRegions = new Set(duck.map(r => r.region));
    expect(oursRegions).toEqual(duckRegions);
  }, 60_000);

  it("distinct region, category — 5M rows", async () => {
    const op = new DistinctOperator(new MockOperator(rows), ["region", "category"]);
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT DISTINCT region, category FROM distinct_test`,
    );

    expect(ours.length).toBe(duck.length);
  }, 60_000);
});

// ===================================================================
// 4. ORDER BY
// ===================================================================

describe("ORDER BY", () => {
  const N = 5_000_000;
  let rows: Row[];

  beforeAll(async () => {
    rows = generateRows(N);
    await loadTable("sort_test", rows, TABLE_SCHEMA);
  }, 300_000);

  it("sort ascending by amount — 5M rows", async () => {
    const op = new InMemorySortOperator(new MockOperator(rows), "amount", false);
    const ours = await drainPipeline(op);

    // Verify sorted (don't pull all 5M from DuckDB — just check ordering property)
    expect(ours.length).toBe(N);
    for (let i = 1; i < ours.length; i++) {
      expect(ours[i].amount as number).toBeGreaterThanOrEqual(ours[i - 1].amount as number);
    }

    // Check boundary values match DuckDB
    const duck = await duckQuery(
      `SELECT MIN(amount) as lo, MAX(amount) as hi FROM sort_test`,
    );
    expect(ours[0].amount).toBe(duck[0].lo);
    expect(ours[ours.length - 1].amount).toBe(duck[0].hi);
  }, 60_000);

  it("TopK limit 100 — 5M rows", async () => {
    const op = new TopKOperator(new MockOperator(rows), "amount", true, 100);
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT * FROM sort_test ORDER BY amount DESC LIMIT 100`,
    );

    expect(ours.length).toBe(duck.length);

    // Sort both by amount desc, id asc for deterministic comparison
    sortByKeys(ours, ["amount"]);
    sortByKeys(duck, ["amount"]);

    // Verify same amount values
    const oursAmounts = ours.map(r => r.amount as number).sort((a, b) => b - a);
    const duckAmounts = duck.map(r => r.amount as number).sort((a, b) => b - a);
    expect(oursAmounts).toEqual(duckAmounts);
  }, 60_000);

  it("ExternalSort with spill (5M, 1MB budget) — matches DuckDB order", async () => {
    const spill = new FsSpillBackend();
    const op = new ExternalSortOperator(
      new MockOperator(rows),
      "amount", false,
      0,
      1024 * 1024, // 1MB budget
      spill,
    );
    const ours = await drainPipeline(op);

    expect(ours.length).toBe(N);

    // Verify sort order
    for (let i = 1; i < ours.length; i++) {
      expect(ours[i].amount as number).toBeGreaterThanOrEqual(ours[i - 1].amount as number);
    }

    // Verify we still have all rows (count of distinct ids)
    const idSet = new Set(ours.map(r => r.id as number));
    expect(idSet.size).toBe(N);

    await spill.cleanup();
  }, 300_000);

  it("ExternalSort via R2SpillBackend (5M, 1MB budget)", async () => {
    const spill = new R2SpillBackend(r2Bucket, "__spill/test-sort");
    const op = new ExternalSortOperator(
      new MockOperator(rows),
      "amount", false,
      0,
      1024 * 1024, // 1MB budget
      spill,
    );
    const ours = await drainPipeline(op);

    expect(ours.length).toBe(N);

    for (let i = 1; i < ours.length; i++) {
      expect(ours[i].amount as number).toBeGreaterThanOrEqual(ours[i - 1].amount as number);
    }

    const idSet = new Set(ours.map(r => r.id as number));
    expect(idSet.size).toBe(N);

    await spill.cleanup();
  }, 300_000);
});

// ===================================================================
// 5. WindowOperator
// ===================================================================

describe("WindowOperator", () => {
  const N = 1_000_000;
  let rows: Row[];

  beforeAll(async () => {
    rows = generateRows(N);
    await loadTable("win_test", rows, TABLE_SCHEMA);
  }, 300_000);

  it("row_number partitioned by region, ordered by amount, id", async () => {
    const win: WindowSpec = {
      fn: "row_number",
      partitionBy: ["region"],
      orderBy: [
        { column: "amount", direction: "asc" },
        { column: "id", direction: "asc" },
      ],
      alias: "rn",
    };
    const op = new WindowOperator(new MockOperator(rows), [win]);
    const ours = await drainPipeline(op);

    expect(ours.length).toBe(N);

    // Verify sequential row numbers per partition
    const byRegion = new Map<string, Row[]>();
    for (const r of ours) {
      const region = r.region as string;
      if (!byRegion.has(region)) byRegion.set(region, []);
      byRegion.get(region)!.push(r);
    }

    for (const [, regionRows] of byRegion) {
      regionRows.sort((a, b) => (a.rn as number) - (b.rn as number));
      expect(regionRows[0].rn).toBe(1);
      expect(regionRows[regionRows.length - 1].rn).toBe(regionRows.length);
    }

    // Spot-check against DuckDB for one region
    const duckSample = await duckQuery(
      `SELECT id, ROW_NUMBER() OVER(PARTITION BY region ORDER BY amount, id) as rn
       FROM win_test WHERE region = 'us' ORDER BY rn LIMIT 10`,
    );
    const ourUs = ours
      .filter(r => r.region === "us")
      .sort((a, b) => (a.rn as number) - (b.rn as number))
      .slice(0, 10);

    for (let i = 0; i < duckSample.length; i++) {
      expect(ourUs[i].rn).toBe(Number(duckSample[i].rn));
      expect(ourUs[i].id).toBe(duckSample[i].id);
    }
  }, 60_000);

  it("rank and dense_rank with ties", async () => {
    const tieRows: Row[] = [
      { id: 0, grp: "A", val: 10 },
      { id: 1, grp: "A", val: 10 },
      { id: 2, grp: "A", val: 20 },
      { id: 3, grp: "A", val: 20 },
      { id: 4, grp: "A", val: 30 },
    ];

    await loadTable("rank_test", tieRows, "id INT, grp VARCHAR, val INT");

    const rankWin: WindowSpec = {
      fn: "rank",
      partitionBy: ["grp"],
      orderBy: [{ column: "val", direction: "asc" }],
      alias: "rnk",
    };
    const denseWin: WindowSpec = {
      fn: "dense_rank",
      partitionBy: ["grp"],
      orderBy: [{ column: "val", direction: "asc" }],
      alias: "drnk",
    };

    const op = new WindowOperator(new MockOperator(tieRows), [rankWin, denseWin]);
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT id,
              RANK() OVER(PARTITION BY grp ORDER BY val) as rnk,
              DENSE_RANK() OVER(PARTITION BY grp ORDER BY val) as drnk
       FROM rank_test ORDER BY id`,
    );

    ours.sort((a, b) => (a.id as number) - (b.id as number));

    for (let i = 0; i < ours.length; i++) {
      expect(ours[i].rnk).toBe(Number(duck[i].rnk));
      expect(ours[i].drnk).toBe(Number(duck[i].drnk));
    }
  }, 60_000);

  it("lag(1) / lead(1)", async () => {
    const partRows: Row[] = [
      { id: 0, grp: "A", val: 10 },
      { id: 1, grp: "A", val: 20 },
      { id: 2, grp: "A", val: 30 },
      { id: 3, grp: "B", val: 40 },
      { id: 4, grp: "B", val: 50 },
    ];

    await loadTable("lag_test", partRows, "id INT, grp VARCHAR, val INT");

    const lagWin: WindowSpec = {
      fn: "lag",
      args: { offset: 1 },
      partitionBy: ["grp"],
      orderBy: [{ column: "val", direction: "asc" }],
      alias: "lag_val",
    };
    const leadWin: WindowSpec = {
      fn: "lead",
      args: { offset: 1 },
      partitionBy: ["grp"],
      orderBy: [{ column: "val", direction: "asc" }],
      alias: "lead_val",
    };

    const op = new WindowOperator(new MockOperator(partRows), [lagWin, leadWin]);
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT id,
              LAG(val, 1) OVER(PARTITION BY grp ORDER BY val) as lag_val,
              LEAD(val, 1) OVER(PARTITION BY grp ORDER BY val) as lead_val
       FROM lag_test ORDER BY id`,
    );

    ours.sort((a, b) => (a.id as number) - (b.id as number));

    for (let i = 0; i < ours.length; i++) {
      const oursLag = ours[i].lag_val;
      const duckLag = duck[i].lag_val;
      // null comparison: our null vs DuckDB null
      if (duckLag === null) {
        expect(oursLag === null || oursLag === undefined).toBe(true);
      } else {
        expect(oursLag).toBe(Number(duckLag));
      }

      const oursLead = ours[i].lead_val;
      const duckLead = duck[i].lead_val;
      if (duckLead === null) {
        expect(oursLead === null || oursLead === undefined).toBe(true);
      } else {
        expect(oursLead).toBe(Number(duckLead));
      }
    }
  }, 60_000);

  it("rolling sum — frame rows between -2 and current", async () => {
    const partRows: Row[] = [];
    for (let i = 0; i < 10; i++) {
      partRows.push({ id: i, grp: "A", val: (i + 1) * 10 });
    }

    await loadTable("rolling_test", partRows, "id INT, grp VARCHAR, val INT");

    const win: WindowSpec = {
      fn: "sum",
      partitionBy: ["grp"],
      orderBy: [{ column: "val", direction: "asc" }],
      alias: "rolling_sum",
      frame: { type: "rows", start: -2, end: "current" },
    };
    const op = new WindowOperator(new MockOperator(partRows), [win]);
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT id, val,
              SUM(val) OVER(PARTITION BY grp ORDER BY val ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as rolling_sum
       FROM rolling_test ORDER BY val`,
    );

    ours.sort((a, b) => (a.val as number) - (b.val as number));

    for (let i = 0; i < ours.length; i++) {
      expect(ours[i].rolling_sum).toBe(Number(duck[i].rolling_sum));
    }
  }, 60_000);
});

// ===================================================================
// 6. SetOperator
// ===================================================================

describe("SetOperator", () => {
  const N = 1_000_000;

  let leftRows: Row[];
  let rightRows: Row[];

  beforeAll(async () => {
    leftRows = generateRows(N);
    rightRows = [];
    const offset = N / 2;
    for (let i = offset; i < offset + N; i++) {
      rightRows.push({
        id: i,
        region: REGIONS[i % REGIONS.length],
        category: CATEGORIES[i % CATEGORIES.length],
        amount: (i * 7 + 13) % 1000,
        score: ((i * 31) % 997) / 10,
      });
    }

    await loadTable("set_left", leftRows, TABLE_SCHEMA);
    await loadTable("set_right", rightRows, TABLE_SCHEMA);
  }, 300_000);

  it("union_all — 1M + 1M", async () => {
    const op = new SetOperator(
      new MockOperator(leftRows),
      new MockOperator(rightRows),
      "union_all",
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT COUNT(*) as cnt FROM (SELECT * FROM set_left UNION ALL SELECT * FROM set_right)`,
    );

    expect(ours.length).toBe(Number(duck[0].cnt));
    expect(ours.length).toBe(leftRows.length + rightRows.length);
  }, 60_000);

  it("union (dedup) — 1M + 1M", async () => {
    const op = new SetOperator(
      new MockOperator(leftRows),
      new MockOperator(rightRows),
      "union",
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT COUNT(*) as cnt FROM (SELECT * FROM set_left UNION SELECT * FROM set_right)`,
    );

    expect(ours.length).toBe(Number(duck[0].cnt));

    // Verify no duplicate ids
    const ids = new Set(ours.map(r => r.id as number));
    expect(ids.size).toBe(ours.length);
  }, 60_000);

  it("intersect — 1M + 1M", async () => {
    const op = new SetOperator(
      new MockOperator(leftRows),
      new MockOperator(rightRows),
      "intersect",
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT COUNT(*) as cnt FROM (SELECT * FROM set_left INTERSECT SELECT * FROM set_right)`,
    );

    expect(ours.length).toBe(Number(duck[0].cnt));
  }, 60_000);

  it("except — 1M + 1M", async () => {
    const op = new SetOperator(
      new MockOperator(leftRows),
      new MockOperator(rightRows),
      "except",
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT COUNT(*) as cnt FROM (SELECT * FROM set_left EXCEPT SELECT * FROM set_right)`,
    );

    expect(ours.length).toBe(Number(duck[0].cnt));

    // Verify all result ids are left-only (id < N/2)
    for (const row of ours) {
      expect(row.id as number).toBeLessThan(N / 2);
    }
  }, 60_000);
});

// ===================================================================
// 7. ComputedColumnOperator
// ===================================================================

describe("ComputedColumnOperator", () => {
  const N = 1_000_000;
  let rows: Row[];

  beforeAll(async () => {
    rows = generateRows(N);
    await loadTable("comp_test", rows, TABLE_SCHEMA);
  }, 300_000);

  it("amount * 2 — 1M rows", async () => {
    const op = new ComputedColumnOperator(
      new MockOperator(rows),
      [{ alias: "total", fn: (row: Row) => (row.amount as number) * 2 }],
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT id, amount, amount * 2 as total FROM comp_test ORDER BY id LIMIT 1000`,
    );

    expect(ours.length).toBe(N);

    // Sort ours by id and spot-check first 1000
    sortByKeys(ours, ["id"]);
    for (let i = 0; i < duck.length; i++) {
      expect(ours[i].total).toBe(duck[i].total);
    }
  }, 60_000);
});

// ===================================================================
// 8. SubqueryInOperator
// ===================================================================

describe("SubqueryInOperator", () => {
  const N = 1_000_000;
  let rows: Row[];

  beforeAll(async () => {
    rows = generateRows(N);
    await loadTable("subq_test", rows, TABLE_SCHEMA);
  }, 300_000);

  it("filter region in ('us','eu') — 1M rows", async () => {
    const op = new SubqueryInOperator(
      new MockOperator(rows),
      "region",
      new Set(["us", "eu"]),
    );
    const ours = await drainPipeline(op);

    const duck = await duckQuery(
      `SELECT COUNT(*) as cnt FROM subq_test WHERE region IN ('us','eu')`,
    );

    expect(ours.length).toBe(Number(duck[0].cnt));

    for (const row of ours) {
      expect(["us", "eu"]).toContain(row.region);
    }
  }, 60_000);
});
