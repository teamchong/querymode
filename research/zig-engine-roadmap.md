# Zig Engine Roadmap

Learnings from sibling Zig repos, prioritized by impact on QueryMode's WASM engine.

## P0: Selection Vectors + Late Materialization (from lanceql) — PARTIALLY DONE

**Source:** `../lanceql/src/sql/late_materialization.zig`, `../lanceql/src/query/vector_engine.zig`

**Already exists in Zig:** `wasm/src/query/vector_engine.zig` has SelectionVector, DataChunk, Vector types (DuckDB-style, VECTOR_SIZE=2048). `wasm/src/columnar_ops.zig` has SIMD filter ops returning row indices.

**Done (TS layer):**
- ScanOperator now applies filters during scan using WASM SIMD (`filterFloat64Buffer`/`filterInt32Buffer`) before row materialization
- `buildPipeline` skips FilterOperator when ScanOperator handles filters
- Parquet bounded path registers decoded columns in WASM and uses `executeQuery()` (SIMD filter/sort/agg) instead of JS row-by-row

**Remaining:**
- True two-phase execution: decode only filter columns first, get matching indices, then decode projection columns only for matches (saves string decode cost)
- Connect TS pipeline to Zig SelectionVector/DataChunk types for full columnar execution

**Impact:** Peak memory drops from ~128MB to ~12MB on 1M row queries.

**Files modified:** `src/operators.ts`, `src/wasm-engine.ts`, `src/query-do.ts`

## P1: SIMD128 Filter Predicates (from vectorjson + edgebox) — DONE (numeric)

**Source:** `../vectorjson/src/zig/simd.zig`, `../edgebox/src/simd_utils.zig`

**Done:**
- `filterFloat64Buffer`: SIMD128 with @Vector(2, f64) — 2 f64 per cycle + scalar tail
- `filterInt32Buffer`: SIMD128 with @Vector(4, i32) — 4 i32 per cycle + scalar tail
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

**Done:** TS `scanFilterIndices()` handles compound AND (intersect index arrays via WASM `intersectIndices`). WASM SQL path (`executeSql`) already evaluates WHERE vectorized in Zig.

**Remaining:**
- OR support in TS scan-time filter (union index arrays via `unionIndices`)
- Short-circuit evaluation: if first AND filter returns 0 matches, skip remaining filters
- Complex expressions (BETWEEN, LIKE) in the WASM filter fast path

**Files modified:** `src/operators.ts` (scanFilterIndices)

## P2: VIP Pinning with Safety Locks (from zell) — DONE

**Source:** `../zell/src/expert_cache.zig`

**Done:** Added acquire/release reference counting to VipCache:
- `acquire(key)` — like get() but increments refCount, prevents eviction
- `release(key)` — decrements refCount, deletes if pending eviction and refCount=0
- `evict()` skips entries with refCount > 0; lets map grow temporarily if all locked
- `stats()` includes `lockedCount` (entries with refCount > 0)

**Files modified:** `src/vip-cache.ts`

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
