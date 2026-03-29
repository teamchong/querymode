# src/ — TypeScript Orchestration Layer

> 65K lines. Coordinates I/O, caching, R2 storage, and Durable Objects.
> All compute pushes down to the Zig WASM engine (`wasm/`).
>
> **For behavior specs, see `specs/`** — this README is a navigation map, not a contract.

## Module Map

The flat files in `src/` group into 6 logical modules:

```
┌─────────────────────────────────────────────────────────┐
│                    PUBLIC API                            │
│  index.ts          Entry point, QueryMode class         │
│  client.ts         DataFrame builder, QueryExecutor     │
│  convenience.ts    fromJSON(), fromCSV(), demo()        │
│  descriptor-to-code.ts  QueryDescriptor → code string   │
│  format.ts         Human-readable result formatting     │
│  lazy.ts           MaterializationCache for lazy eval   │
├─────────────────────────────────────────────────────────┤
│                   QUERY EXECUTION                        │
│  operators.ts      Pull-based operator pipeline (3.2K)  │
│                    Scan → Filter → Agg → Sort → Project │
│  partial-agg.ts    Partial aggregation + finalization    │
│  merge.ts          Cross-fragment result merging         │
│  columnar.ts       QMCB binary format encode/decode     │
│  query-schema.ts   Query validation + descriptor types  │
│  coalesce.ts       Range request coalescing for R2      │
├─────────────────────────────────────────────────────────┤
│                  DURABLE OBJECTS                         │
│  master-do.ts      Single-writer: append, drop, CAS     │
│  query-do.ts       Regional coordinator: plan + fan-out │
│  fragment-do.ts    Pooled scanner: one per fragment      │
│  worker-do.ts      Ingestion worker DO                  │
│  worker-pool.ts    Fan-out + partition routing           │
│  worker.ts         CF Worker entry (HTTP → DO routing)  │
├─────────────────────────────────────────────────────────┤
│                  FORMAT DECODERS                         │
│  footer.ts         Lance footer parser (last 40 bytes)  │
│  lance-v2.ts       Lance v2 page decode + stats         │
│  parquet.ts        Parquet file format reader            │
│  parquet-decode.ts Parquet page decompression            │
│  iceberg.ts        Iceberg table format support          │
│  decode.ts         Column value decoding + LIKE regex    │
│  manifest.ts       Lance dataset manifest (versions)     │
│  readers/          CSV, JSON, Arrow ingestion readers    │
├─────────────────────────────────────────────────────────┤
│                  INFRASTRUCTURE                          │
│  types.ts          All shared types, Footer, ColumnMeta  │
│  errors.ts         QueryModeError with typed ErrorCode   │
│  bucket.ts         Multi-bucket R2 sharding resolver     │
│  r2-spill.ts       Spill-to-R2 for large sorts/joins    │
│  vip-cache.ts      LRU cache with VIP pinning            │
│  partition-catalog.ts  O(1) partition pruning catalog    │
│  wasm-engine.ts    TS↔Zig WASM bridge (150+ exports)    │
│  wasm-module.ts    WASM module loader                    │
│  cloudflare-workers-polyfill.ts  Node compat shim       │
├─────────────────────────────────────────────────────────┤
│                  SUB-MODULES (own READMEs)               │
│  sql/              SQL parser → compiler → executor      │
│  pg-wire/          PostgreSQL wire protocol (psql/BI)    │
│  readers/          CSV, JSON, Arrow ingestion            │
│  cli/              CLI tooling (init)                    │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

```
User code
  │
  ▼
QueryMode.table("t").filter(...).exec()     ← client.ts (DataFrame)
  │
  ▼
QueryDescriptor                              ← types.ts
  │
  ├── Local mode ──► LocalExecutor           ← local-executor.ts
  │                    │
  │                    ▼
  │                  buildPipeline()          ← operators.ts
  │                    │
  │                    ▼
  │                  ScanOperator → WASM      ← wasm-engine.ts
  │
  └── Edge mode ──► RemoteExecutor (RPC)     ← index.ts
                     │
                     ▼
                   QueryDO                    ← query-do.ts
                     │
                     ├── Small: scan locally
                     │
                     └── Large: fan out to FragmentDOs  ← fragment-do.ts
                           │
                           ▼
                         Partial results (QMCB)
                           │
                           ▼
                         merge + finalize     ← merge.ts, partial-agg.ts
```

## Key Design Rules

1. **TS = orchestration, WASM = compute.** Never do math in TS if WASM can do it.
2. **Pull-based pipeline.** Each operator calls `next()` on its child. No push, no buffering.
3. **Page-level pruning before I/O.** `canSkipPageMultiCol()` checks min/max stats before any R2 GET.
4. **Two-phase materialization.** Fetch filter columns first. Only fetch projection columns if rows pass.
5. **Fragment-level parallelism.** Each fragment gets its own DO. Idle DOs hibernate (zero cost).
6. **Columnar everywhere.** QMCB binary format between DOs. Rows materialized only at API boundary.

## File Size Guide (where complexity lives)

| File | Lines | Why it's big |
|------|-------|-------------|
| operators.ts | 3,179 | 13 operator types + page pruning + pipeline assembly |
| query-do.ts | 2,116 | Query planning, fan-out, caching, streaming, explain |
| wasm-engine.ts | 1,427 | 150+ WASM export bindings + buildFragment |
| client.ts | 1,304 | DataFrame fluent API + MaterializedExecutor |
| columnar.ts | 1,121 | QMCB encode/decode/merge/slice |
| local-executor.ts | 1,080 | Local-mode pipeline: file I/O → scan → exec |

Everything else is under 700 lines.
