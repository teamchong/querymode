# Zig Engine Roadmap

Learnings from sibling Zig repos, prioritized by impact on QueryMode's WASM engine.

## P0: Selection Vectors + Late Materialization (from lanceql) — DONE

**Source:** `../lanceql/src/sql/late_materialization.zig`, `../lanceql/src/query/vector_engine.zig`

**Already exists in Zig:** `wasm/src/query/vector_engine.zig` has SelectionVector, DataChunk, Vector types (DuckDB-style, VECTOR_SIZE=2048). `wasm/src/columnar_ops.zig` has SIMD filter ops returning row indices.

**Done (TS layer):**
- ScanOperator now applies filters during scan using WASM SIMD (`filterFloat64Buffer`/`filterInt32Buffer`/`filterInt64Buffer`) before row materialization
- `buildPipeline` skips FilterOperator when ScanOperator handles filters
- Parquet bounded path registers decoded columns in WASM and uses `executeQuery()` (SIMD filter/sort/agg) instead of JS row-by-row
- True two-phase execution: fetch+decode only filter columns first, run WASM SIMD filter, skip projection column R2 I/O entirely if 0 matches. Only fetch+decode projection columns for pages with matches.
- Prefetch is two-phase aware: only prefetches filter columns (not all columns)
- Short-circuit evaluation: if any AND filter returns 0 matches, skip remaining filters immediately

**Remaining:**
- Connect TS pipeline to Zig SelectionVector/DataChunk types for full columnar execution (removes Row[] batch format)

**Impact:** Peak memory drops from ~128MB to ~12MB on 1M row queries.

**Files modified:** `src/operators.ts`, `src/wasm-engine.ts`, `src/query-do.ts`

## P1: SIMD128 Filter Predicates (from vectorjson + edgebox) — DONE (numeric)

**Source:** `../vectorjson/src/zig/simd.zig`, `../edgebox/src/simd_utils.zig`

**Done:**
- `filterFloat64Buffer`: SIMD128 with @Vector(2, f64) — 2 f64 per cycle + scalar tail
- `filterInt32Buffer`: SIMD128 with @Vector(4, i32) — 4 i32 per cycle + scalar tail
- `filterInt64Buffer`: SIMD128 with @Vector(2, i64) — 2 i64 per cycle + scalar tail
- `intersectIndices`: O(n+m) sorted merge (was O(n*m) nested loop)

**Remaining:**
- Comptime `anyMatch` pattern for string column scanning
- SIMD null bitmap evaluation

**Files modified:** `wasm/src/wasm/aggregates.zig`

## P1: Arena Allocator Per Batch (from edgebox) — ALREADY SOLVED

**Source:** `../edgebox/src/native_arena.zig`

**Status:** Already effectively solved. WASM engine uses `std.heap.WasmAllocator` (bump allocator — linear memory, no free, no fragmentation). TS calls `resetHeap()` between queries. This is equivalent to arena-per-query. See `wasm/src/wasm/memory.zig`.

**No action needed.**

## P2: Vectorized WHERE Evaluation (from lanceql) — PARTIALLY DONE

**Source:** `../lanceql/src/sql/where_eval.zig`

**Done:** TS `scanFilterIndices()` handles compound AND (intersect index arrays via WASM `intersectIndices`). WASM SQL path (`executeSql`) already evaluates WHERE vectorized in Zig. `unionIndices` upgraded to O(n+m) sorted merge. `WasmAggregateOperator` supports filtered aggregates via indexed aggregate exports (`sumFloat64Indexed`, etc.) — zero Row[] materialization for filter+aggregate queries. Short-circuit evaluation: both `scanFilterIndices` and `WasmAggregateOperator` bail immediately when any AND filter returns 0 matches. `filterInt64Buffer` SIMD128 added for int64 column filtering.

### LIKE / NOT IN / NOT BETWEEN / NOT LIKE Filter Pushdown — DONE
- SQL compiler now flattens LIKE, NOT LIKE, NOT IN, NOT BETWEEN into FilterOp[] (was falling through to SqlWrappingExecutor)
- `matchesFilter()` handles all four ops in JS evaluation path
- `canSkipPage()` handles NOT BETWEEN (skip if all values within [lo, hi])
- Client API: `.whereLike()`, `.whereNotLike()`, `.whereIn()`, `.whereNotIn()`, `.whereNotBetween()`
- `canUseWasmAggregate()` correctly rejects non-numeric ops
- `scanFilterIndices` guards new ops from WASM-only numeric paths

### OR Filter Support — DONE
- `filterGroups: FilterOp[][]` on QueryDescriptor — each group is AND-connected, groups are OR'd
- SQL compiler decomposes `(a AND b) OR (c AND d)` into filterGroups via `tryFlattenOrGroups()`
- `scanFilterIndices`: evaluates each OR group independently via `applyAndFilters()`, unions results via `wasmUnion()`
- `canSkipPageMultiCol`: page skipped only if ALL OR groups eliminate it
- `FilterOperator.matchesRow()`: AND filters + at least one OR group must pass
- `assembleRows()` in decode.ts handles filterGroups
- Client API: `.whereOr(...groups: FilterOp[][])`
- `canUseWasmAggregate` rejects queries with filterGroups (not optimized for OR yet)
- `query-do.ts` fully supports filterGroups: cache key, applyJsPostProcessing, join path FilterOperator, explain output
- `query-schema.ts` validates filterGroups and new filter ops (not_in, between, not_between, like, not_like)
- `applyJsPostProcessing` uses shared `matchesFilter()` (was duplicated inline logic missing new ops)

### NOT BETWEEN WASM Filter — DONE
- `filterFloat64NotRange`, `filterInt32NotRange`, `filterInt64NotRange` Zig exports
- Returns indices where `val < low OR val > high`
- `wasmFilterNotRange()` TS helper mirrors `wasmFilterRange()`
- `scanFilterIndices`, `WasmAggregateOperator.evalAndFilters` dispatch BETWEEN/NOT BETWEEN to range/not-range exports
- `canUseWasmAggregate` allows NOT BETWEEN filters

### WASM String LIKE Filter — DONE
- `filterStringLike` Zig export: SQL LIKE pattern matching (%, _) on packed string data
- Case-insensitive matching via `sqlLikeMatchCI` (ASCII toLower)
- Data format: offsets[0..count+1] + packed UTF-8 data buffer
- `wasmFilterLike()` TS helper packs JS strings into WASM memory, calls Zig filter
- `scanFilterIndices` dispatches LIKE/NOT LIKE to WASM for string/utf8/binary columns
- Negated flag supports both LIKE and NOT LIKE in single export

### Columnar Pipeline — DONE (core operators)
- `ColumnarBatch` type: `{ columns: Map<string, DecodedValue[]>, rowCount, selection?: Uint32Array }`
- `Operator.nextColumnar?()` optional method — operators upgraded incrementally
- `materializeRows(batch)` — Row[] materialization only at pipeline exit
- `ScanOperator.nextColumnar()` — returns decoded columns directly, no Row[] creation
- `FilterOperator.nextColumnar()` — narrows selection vector on columnar data
- `LimitOperator.nextColumnar()` — slices selection vector
- `ProjectOperator.nextColumnar()` — drops columns from Map
- `DistinctOperator.nextColumnar()` — dedup via selection vector
- `AggregateOperator` — consumes upstream via `nextColumnar()`, uses `computePartialAggColumnar()`
- `TopKOperator` — consumes upstream via `nextColumnar()`, materializes only heap entries
- `InMemorySortOperator` — consumes upstream via `nextColumnar()`
- `ExternalSortOperator` — consumes upstream via `nextColumnar()`
- `drainPipeline()` prefers columnar path when available
- Pipeline: Scan→Filter→Distinct→Limit→Project all stay columnar; consuming operators (Agg/Sort/TopK) consume columnar and produce Row[]

**Remaining:**
- HashJoinOperator columnar consumption (complex, lower ROI)
- Connect to Zig SelectionVector/DataChunk types (share selection vectors in WASM memory — zero-copy)

### BETWEEN Support — DONE
- `filterFloat64Range`, `filterInt32Range`, `filterInt64Range` Zig exports
- `canSkipPage` handles BETWEEN: page skipped if `max < low || min > high`
- `wasmFilterRange()` helper for scanFilterIndices WASM path
- `WasmAggregateOperator` dispatches BETWEEN to range filter exports
- SQL compiler emits single `{ op: "between", value: [low, high] }` instead of gte+lte pair
- Client API: `.whereBetween(column, low, high)`

### Multi-Column Page Skip — DONE
- `canSkipPageMultiCol()` checks ALL filter columns' min/max stats, not just first column
- Used in ScanOperator, findNextPage (prefetch), and WasmAggregateOperator

**Files modified:** `src/operators.ts` (scanFilterIndices)

## P2: VIP Pinning with Safety Locks (from zell) — DONE

**Source:** `../zell/src/expert_cache.zig`

**Done:** Added acquire/release reference counting to VipCache:
- `acquire(key)` — like get() but increments refCount, prevents eviction
- `release(key)` — decrements refCount, deletes if pending eviction and refCount=0
- `evict()` skips entries with refCount > 0; lets map grow temporarily if all locked
- `stats()` includes `lockedCount` (entries with refCount > 0)

**Files modified:** `src/vip-cache.ts`

## P1: PB-Scale Infrastructure — IN PROGRESS

### Fragment-Level Pruning — DONE
- `canSkipFragment()` in decode.ts: computes per-column min/max across all pages, checks if any filter eliminates the entire fragment
- Reuses `canSkipPage()` on synthetic "fragment page" with aggregated stats
- Supports AND filters and OR filterGroups (skip only if ALL groups eliminate)
- `executeMultiFragment()` prunes fragments before any R2 I/O or DO dispatch
- Logs `fragment_prune` with total/skipped/remaining counts
- `executeExplain()` reports `fragmentsSkipped` in explain output
- `executeWithFragmentDOs()` accepts pruned fragment list (no wasted DO slots)

### Multi-Bucket Sharding — FOUNDATION
- `Env` supports `DATA_BUCKET_1`, `DATA_BUCKET_2`, `DATA_BUCKET_3` (optional)
- `resolveBucket(r2Key)` routes by FNV-1a hash of table prefix across all available buckets
- Bypasses R2 per-bucket rate limits by distributing data across buckets
- Remaining: wire `resolveBucket` into R2 read paths, add shard-aware ingest

## P3: Host Import Pattern for R2 I/O (from gitmode)

**Source:** `../gitmode/wasm/src/r2_backend.zig`, `../gitmode/wasm/src/main.zig`

**Problem:** WASM engine currently receives data pushed from TypeScript.

**Solution:** WASM calls host-imported functions to request R2 reads:
- `extern fn r2_read(key_ptr: [*]u8, key_len: u32, offset: u64, len: u32) i32`
- WASM engine drives its own I/O, enabling prefetch decisions inside Zig

**Files to modify:** `wasm/src/main.zig`, `src/wasm-engine.ts`

## P3: Comptime Type Marshaling (from metal0)

**Source:** `../metal0/packages/c_interop/src/comptime_wrapper.zig`

**Problem:** Format-specific decoders have repetitive type conversion code.

**Solution:** Use Zig comptime to auto-generate type converters:
```zig
fn MarshalColumn(comptime T: type) type {
    return struct {
        pub fn decode(buf: []const u8) []T { ... }
        pub fn encode(values: []const T) []u8 { ... }
    };
}
```

Generate Arrow<->Lance<->Parquet converters from one template.

**Files to modify:** `wasm/src/decode.zig`

## P3: Canonical ABI for WASM Boundary (from edgebox)

**Source:** `../edgebox/src/component/canonical_abi.zig`

**Problem:** Column data exchange between TS and WASM uses manual pointer math.

**Solution:** Type-safe lift/lower functions:
- Lower (Host->WASM): allocate in WASM memory, copy column data
- Lift (WASM->Host): validate, copy to host
- Handles strings, lists, nested types

**Files to modify:** `src/wasm-engine.ts`, `wasm/src/main.zig`

## Reference: Key Files in Sibling Repos

| Repo | File | What to learn |
|------|------|---------------|
| lanceql | `src/sql/late_materialization.zig` | Two-phase execution, streaming batches |
| lanceql | `src/query/vector_engine.zig` | SelectionVector, DataChunk, Vector types |
| lanceql | `src/sql/where_eval.zig` | Vectorized compound filter evaluation |
| lanceql | `src/simd.zig` | Threshold-based SIMD dispatch |
| vectorjson | `src/zig/simd.zig` | Comptime SIMD128 anyMatch pattern |
| edgebox | `src/native_arena.zig` | Bump allocator with LIFO + in-place realloc |
| edgebox | `src/simd_utils.zig` | SIMD byte scanning patterns |
| gitmode | `wasm/src/r2_backend.zig` | Host import R2 I/O from WASM |
| gitmode | `wasm/src/simd.zig` | SIMD128 memchr/memeql |
| zell | `src/expert_cache.zig` | VIP pinning + LRU + safety locks |
| metal0 | `packages/c_interop/src/comptime_wrapper.zig` | Comptime code generation |
