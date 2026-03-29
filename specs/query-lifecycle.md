# Spec: Query Lifecycle

Status: **Implemented** — documents the current execution paths.

Every query follows one of three paths. This spec defines the exact sequence of operations for each.

---

## Path 1: Local Mode

**When**: `QueryMode.local()` — Node/Bun, reads from filesystem or URLs.

```
User: df.filter("age", "gt", 25).select("name").limit(10).collect()
  │
  ▼
1. DataFrame.collect() builds QueryDescriptor:
   { table: "./users.lance", filters: [{column:"age", op:"gt", value:25}],
     projections: ["name"], limit: 10 }
  │
  ▼
2. LocalExecutor.execute(descriptor)
   a. Detect format (Lance/Parquet/Iceberg/CSV/JSON/Arrow) from extension or magic bytes
   b. Parse footer → ColumnMeta[] with page-level min/max stats
   c. For datasets: parse manifest → discover all fragments
  │
  ▼
3. buildPipeline(descriptor, columns, options)
   a. ScanOperator created with pages + filter columns
   b. canSkipPageMultiCol() marks pages to skip (all 14 ops checked)
   c. Pipeline assembled: Scan → [Filter] → [Agg] → [Sort] → [Window] → [Distinct] → [Project] → [Limit]
      Note: ScanOperator applies filters via WASM SIMD, so FilterOperator is usually skipped
  │
  ▼
4. drainPipeline(pipeline)
   a. Pull loop: call pipeline.next() repeatedly
   b. Each next() triggers:
      - ScanOperator: fetch next non-skipped page from disk
      - Register column data in WASM (zero-copy typed array views)
      - WASM executeQueryColumnar() → QMCB result
      - Decode QMCB → RowBatch
   c. Downstream operators consume RowBatch, apply transforms
   d. Collect rows until pipeline returns null
  │
  ▼
5. Return QueryResult { rows, rowCount, bytesRead, pagesSkipped, durationMs }
```

**Key property**: Single-threaded, no parallelism, no caching between queries.

---

## Path 2: Edge Mode — Small Query (no fan-out)

**When**: `QueryMode.remote(env.QUERY_DO)` with estimated scan < 100K rows or < 2 fragments.

```
User: df.filter("status", "eq", "active").limit(50).exec()
  │
  ▼
1. RemoteExecutor.execute(descriptor)
   → RPC to regional QueryDO (zero-serialization via structured clone)
  │
  ▼
2. QueryDO.queryRpc(descriptor)
   a. Check result cache (queryCacheKey hash → VipCache)
      → If hit: return cached result immediately
   b. Load TableMeta (footer cache → durable storage → R2)
   c. For datasets: load manifest → fragment footers (parallel fetch)
   d. Partition catalog: O(1) fragment lookup if partition filter matches
   e. Fragment pruning: canSkipFragment() on min/max stats
  │
  ▼
3. Fan-out decision:
   estimated_rows = fragments × avg_rows × selectivity
   If estimated_rows < 100K OR fragments < 2: execute locally in QueryDO
  │
  ▼
4. Local execution (same as Path 1 steps 3-4, but with R2 I/O):
   a. buildEdgePipeline() — same operators, but ScanOperator uses:
      - L1 cache: WASM buffer pool (in-process, 64MB)
      - L2 cache: caches.default (per-datacenter, shared)
      - L3: R2 range requests
   b. Two-phase materialization:
      - Phase 1: fetch + decode filter columns only
      - Phase 2: if rows pass, fetch projection columns
   c. drainPipeline() → rows
  │
  ▼
5. Cache result → return QueryResult
```

**Key property**: Single DO, 3-tier caching, two-phase I/O.

---

## Path 3: Edge Mode — Large Query (fan-out)

**When**: `QueryMode.remote(env.QUERY_DO)` with estimated scan ≥ 100K rows AND ≥ 2 fragments.

```
User: df.aggregate([{fn:"sum", column:"amount"}]).groupBy("region").exec()
  │
  ▼
1-2. Same as Path 2 steps 1-2
  │
  ▼
3. Fan-out decision: estimated_rows ≥ 100K AND fragments ≥ 2
   → Fan out to Fragment DOs
  │
  ▼
4. Fragment assignment:
   a. One Fragment DO per fragment (via FRAGMENT_DO namespace)
   b. Each DO gets: fragment R2 key, query descriptor, schema
   c. All Fragment DO RPCs launched in parallel

   If fragments > 25 (REDUCER_TIER_THRESHOLD):
     → Hierarchical reduction: group DOs into batches of 25
     → Each batch gets a Reducer DO that merges partials
     → QueryDO merges reducer outputs (not fragment outputs)
  │
  ▼
5. Fragment DO execution (per fragment):
   a. Load fragment footer (durable storage cache → R2)
   b. Page pruning: canSkipPageMultiCol()
   c. Two-phase scan → WASM execution → QMCB result
   d. If aggregation: compute partial aggregates (partial-agg.ts)
   e. Return QMCB columnar binary (zero-copy RPC transfer)
  │
  ▼
6. QueryDO merge:
   a. Collect all QMCB partials
   b. mergeQueryResults():
      - Re-group GROUP BY keys across fragments
      - Finalize partial aggregates (SUM sums, COUNT sums, AVG = sum/count)
      - Re-apply DISTINCT (dedup across fragments)
      - Re-apply WINDOW functions (need full result set)
      - Final ORDER BY + LIMIT + OFFSET
   c. columnarBatchToRows() → Row[] (only at API boundary)
  │
  ▼
7. Cache result → return QueryResult
```

**Key property**: Parallel fragment scans, partial aggregation, tree merge for large fan-outs.

---

## Path 4: Vector Search

**When**: Query includes `vectorSearch` parameter.

```
User: df.nearestTo("embedding", queryVec, 10, { metric: "cosine" }).exec()
  │
  ▼
1. QueryDO receives descriptor with vectorSearch
  │
  ▼
2. Index check:
   a. Check TableMeta.vectorIndexes[] for IVF-PQ index on the column
   b. If no metadata: check convention path {r2Key}.ivf_pq.index
  │
  ▼
3a. With IVF-PQ index:
    - Load index from R2 (cached in WASM buffer pool)
    - WASM searchIvfPq(handle, queryVector, topK, nprobe)
    - Returns sorted (index, distance) pairs
    - Fetch corresponding rows by index

3b. Without index (flat scan):
    - Register all vectors in WASM
    - WASM batchCosineSimilarity() with SIMD
    - O(n) full scan, but SIMD-accelerated
    - Returns top-K by distance
  │
  ▼
4. Return rows with special columns: _index, _distance, _score
```

---

## Path 5: SQL

**When**: `QueryMode.sql("SELECT ...")` or PG wire query.

```
User: qm.sql("SELECT region, SUM(amount) FROM sales WHERE year > 2023 GROUP BY region")
  │
  ▼
1. SQL frontend:
   a. lexer.ts: tokenize → Token[]
   b. parser.ts: parse → AST (SelectStatement)
   c. compiler.ts: compileFull() → { descriptor, whereExpr, havingExpr, computedExprs, allOrderBy }
      - WHERE predicates flattened to filters[] / filterGroups[][]
      - OR decomposed into filterGroups
      - Aggregates extracted to descriptor.aggregates
  │
  ▼
2. SqlWrappingExecutor wraps the DataFrame executor:
   - Base execution: same pipeline as Path 1/2/3
   - Post-pipeline: HAVING filter, multi-column ORDER BY, CASE/CAST/arithmetic
   - evaluator.ts handles row-level expression evaluation
  │
  ▼
3. Returns QueryResult (same as all other paths)
```

---

## Streaming Path

**When**: `df.stream()` in edge mode.

```
User: for await (const row of await df.stream()) { ... }
  │
  ▼
1. RemoteExecutor.executeStream(descriptor)
   → RPC streamRpc() → ReadableStream<Uint8Array>
  │
  ▼
2. QueryDO produces length-prefixed QMCB frames:
   [4-byte frame length][QMCB binary]
   [4-byte frame length][QMCB binary]
   ...
  │
  ▼
3. Client decodes frames:
   a. Read 4-byte length prefix
   b. Read frame body
   c. decodeColumnarRun() → Row[]
   d. Enqueue rows to ReadableStream<Row>
```

---

## Write Path

**When**: `df.append(rows)` or HTTP POST `/append`.

```
1. RemoteExecutor.append(table, rows, options)
   → RPC to MasterDO
  │
  ▼
2. MasterDO.appendRpc(table, rows, options)
   a. If partitionBy: split rows by partition column value
   b. rowsToColumnArrays(rows) → FragmentColumn[]
   c. WasmEngine.buildFragment(columns) → Lance binary (Uint8Array)
   d. Write fragment to R2: data/{uuid}.lance
  │
  ▼
3. Manifest CAS loop (up to 10 retries):
   a. Read current version from _versions/_latest (with ETag)
   b. Read previous manifest
   c. Append new fragment entry
   d. buildManifestBinary() → protobuf
   e. Write _versions/{version}.manifest
   f. Write _versions/_latest with ETag condition
   g. If ETag mismatch → retry
  │
  ▼
4. Post-write:
   a. Parse new fragment's footer
   b. Broadcast invalidation to all QueryDOs (footer + columns)
   c. Cache table metadata in DO durable storage
   d. Update partition catalog if partitioned
  │
  ▼
5. Return AppendResult { version, dataFilePath, retries, rowsWritten }
```

---

## Cache Hierarchy

All caches and their invalidation triggers:

| Cache | Location | Scope | TTL | Invalidated by |
|-------|----------|-------|-----|---------------|
| Result cache | QueryDO memory | Per-DO | LRU (200 entries) | Write broadcast from MasterDO |
| Footer cache | QueryDO memory | Per-DO | LRU (1000 entries) | Write broadcast |
| Dataset cache | QueryDO memory | Per-DO | LRU (100 entries) | Write broadcast |
| WASM buffer pool | QueryDO/FragmentDO | Per-DO | LRU (64MB) | Capacity eviction |
| Edge cache | caches.default | Per-datacenter | CF default | TTL expiry |
| Footer durable | FragmentDO storage | Per-DO | Persistent | LRU (500 entries) |
| Partition catalog | MasterDO storage | Per-table | Persistent | Append with partitionBy |
