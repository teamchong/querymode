# Spec: Invariants

Status: **Living** — updated whenever a new invariant is discovered or an existing one changes.

These are rules that must never break. Every code change should be checked against this list. Violations are bugs, not trade-offs.

---

## 1. Null Handling

**I-1.1** NULL is never equal to anything, including NULL. `NULL = NULL` → `NULL` (three-valued logic).

**I-1.2** `NULL_SENTINEL` (`\x01NULL\x01`) is used in GROUP BY / DISTINCT keys to distinguish null from empty string. No user data may contain this sentinel.

**I-1.3** Group key separator is `\0`. Columns containing null bytes in values may collide (known LOW-risk limitation).

**I-1.4** NaN sorts last (same position as NULL) in all comparators. Four sites enforce this: `rowComparator`, `ExternalSortOperator`, `InMemorySortOperator`, `columnarKWayMerge`.

**I-1.5** Empty-group aggregates return: COUNT → 0, SUM → null, AVG/MIN/MAX → null. Never undefined.

**I-1.6** BigInt64Array rejects null/undefined. All append paths (local-executor, master-do) must guard null before writing to bigint or float64 typed arrays.

---

## 2. WASM Bridge

**I-2.1** All WASM exports expect **byte addresses**. Never divide pointers by element size (`ptr >> 3`, `ptr / 4`). This applies to all 150+ exports.

**I-2.2** `safeBigInt(v)` must be used at all 15 sites where a JS number is converted to BigInt. NaN/Infinity → `0n`, not a RangeError crash.

**I-2.3** `WasmEngine.resetHeap()` is safe to call only between queries, never during a query. Documents in the function.

**I-2.4** String data registered in WASM uses `subarray(0, dataLen)`, not the full allocated buffer. Overflow otherwise.

**I-2.5** `fragTable` name passed to Zig SQL parser must be sanitized: `.replace(/[^a-zA-Z0-9_]/g, "_")`. Dots in R2 keys break the parser.

---

## 3. Storage

**I-3.1** MasterDO is the **single writer**. All mutations (append, drop, schema evolution) go through one MasterDO instance per table. No concurrent manifest writers.

**I-3.2** Manifest updates use **CAS with ETag** on `_versions/_latest`. Retry up to 10 times on ETag mismatch.

**I-3.3** `resolveBucket(env, r2Key)` must be used at **all** R2 call sites (32 as of last audit). Direct `env.DATA_BUCKET` access bypasses multi-bucket sharding.

**I-3.4** Fragment files are immutable after write. Never overwrite a `data/{uuid}.lance` file. Append-only.

**I-3.5** Lance footer is the last 40 bytes of the file. Footer parsing must read from `fileSize - 40`.

**I-3.6** Lance v2 nullable pages have alignment padding between null bitmap and data. Use `dataOffsetInPage` (not `Math.ceil(rowCount / 8)`) when present.

---

## 4. Query Execution

**I-4.1** Pull-based pipeline: operators call `next()` on their child. Never push. Never buffer beyond one batch unless explicitly spilling.

**I-4.2** Page-level pruning (`canSkipPageMultiCol`) must be applied at **all 4 call sites**: fragment-do, query-do×2, local-executor. Missing a site = scanning pages that could be skipped.

**I-4.3** All 14 filter ops must be covered in page skip logic: eq, neq, gt, gte, lt, lte, in, not_in, between, not_between, like, not_like, is_null, is_not_null.

**I-4.4** Two-phase materialization: filter columns fetched first, projection columns fetched only if rows survive filtering. Violating this wastes R2 reads.

**I-4.5** Fragment DO partial aggregation must NOT re-aggregate. Each fragment computes partials once; QueryDO merges. Double-aggregation corrupts results.

**I-4.6** `queryCacheKey()` must include **all** query fields. Missing a field → cache collisions → wrong results. Currently covers: table, version, filters, filterGroups, projections, sort, limit, offset, aggregates, groupBy, distinct, windows, computedColumns, setOperation, subqueryIn, join, vectorSearch.

**I-4.7** Stack safety: never use `array.push(...largeArray)`. Use loops. Push with spread on arrays >64K elements causes stack overflow.

---

## 5. Columnar Format (QMCB)

**I-5.1** QMCB magic: `0x42434D51` ("QMCB" little-endian). First 4 bytes of every QMCB buffer.

**I-5.2** Dtype tags are stable (part of the wire format):
  - 0=f64, 1=i64, 2=utf8, 3=bool, 4=f32vec, 5=null, 6=i32, 7=f32

**I-5.3** String columns in QMCB use `Uint32Array` offsets (rowCount+1 entries). First offset is always 0.

**I-5.4** WASM result auto-detection: check for "LANC" magic at end of buffer to distinguish Lance v2 fragment format from aggregate result format.

---

## 6. Concurrency

**I-6.1** R2 operations must be wrapped with `withTimeout(10_000)` to prevent hanging on bucket.delete() or slow reads.

**I-6.2** Fragment DOs are stateless scanners. They cache footers in durable storage but hold no query state across requests.

**I-6.3** QueryDO result cache is invalidated on any write broadcast from MasterDO. Stale cache = wrong results.

**I-6.4** Partition catalog CAS: concurrent `putPartitionCatalog()` calls must retry, not silently overwrite.

---

## 7. Type Safety

**I-7.1** BigInt values from int64 columns must stay as BigInt through the entire pipeline. Converting to Number loses precision for values > 2^53.

**I-7.2** `canSkipPage` for IN/NOT_IN with bigint values must compare as BigInt, not Number. Precision loss otherwise.

**I-7.3** Type promotion in WASM registration: int8/16/32 → int64, float16/32 → float64. WASM SQL engine only handles int64 + float64 + string + bool.

---

## 8. Error Handling

**I-8.1** All production errors use `QueryModeError` with typed `ErrorCode`. Never throw raw strings or generic Error.

**I-8.2** Error codes: QUERY_FAILED, MEMORY_EXCEEDED, SCHEMA_MISMATCH, NOT_FOUND, TIMEOUT, INVALID_INPUT, FORMAT_ERROR, SPILL_ERROR.

**I-8.3** Memory check in errors.ts uses `.toLowerCase()` for case-insensitive matching. "Memory" and "memory" both trigger MEMORY_EXCEEDED.
