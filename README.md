# QueryMode

> **Experimental** — early prototype, not production-ready. Architecture and API will change.

A pluggable columnar query library — not a query engine you push data to, but a query capability your code uses directly. No data materialization, no engine boundary, no SQL transpilation.

**[Docs](https://teamchong.github.io/querymode/)** · **[Why QueryMode?](https://teamchong.github.io/querymode/why-querymode/)** · **[Architecture](https://teamchong.github.io/querymode/architecture/)**

## Quickstart

```bash
git clone https://github.com/teamchong/querymode.git
cd querymode && pnpm install
```

```typescript
import { QueryMode } from "querymode/local"

// Zero-config: demo data, no files needed
const top5 = await QueryMode.demo()
  .filter("category", "eq", "Electronics")
  .sort("amount", "desc")
  .limit(5)
  .collect()

// Query your own files — Parquet, Lance, CSV, JSON
const qm = QueryMode.local()
const result = await qm
  .table("./data/events.parquet")
  .filter("status", "eq", "active")
  .select("id", "amount", "region")
  .sort("amount", "desc")
  .limit(20)
  .collect()

// SQL works too — same operator pipeline underneath
const sql = await qm
  .sql("SELECT region, SUM(amount) AS total FROM orders GROUP BY region ORDER BY total DESC")
  .collect()

// Edge mode — same API, WASM runs inside regional DOs
const edge = QueryMode.remote(env.QUERY_DO, { region: "SJC" })
```

## What it is

Operators are composable building blocks, not a fixed query plan. Your code assembles the pipeline, controls the memory budget, decides when to spill. The query engine isn't a service you call — it's a library your code composes.

| Layer | What |
|-------|------|
| **Zig WASM engine** | Column decoding, SIMD filtering, SQL execution, vector search |
| **TypeScript orchestration** | DO lifecycle, R2 range reads, footer caching, request routing |
| **Code-first API** | `.table().filter().sort().exec()` or `.sql("SELECT ...")` |
| **Edge runtime** | Master/Query/Fragment DOs, R2 spill, multi-bucket sharding |

14 operators (filter, project, aggregate, sort, join, window, distinct, set ops, limit, sample, computed columns, subquery-in, top-K, vector search), all pull-based with the same `next() → RowBatch | null` interface.

## Build

```bash
pnpm build:ts         # typecheck (pre-built WASM included)
pnpm test:node        # node tests (~2 min)
pnpm test:workers     # workerd tests
pnpm test             # all tests (~8 min)
pnpm dev              # local dev with wrangler

# Rebuild WASM from Zig source (requires zig toolchain)
pnpm wasm
```

## What exists

- 600+ tests, 110+ conformance tests validated against DuckDB at 1M-5M row scale
- CI benchmarks: QueryMode (Miniflare) vs DuckDB (native) on every push
- Multi-format: Lance, Parquet, Iceberg, CSV, JSON
- Memory-bounded operators with R2 spill (sort, join, aggregate)
- IVF-PQ vector search with flat SIMD fallback
- Zero-copy columnar pipeline (QMCB binary format, no Row[] until response boundary)
- Local mode (Node/Bun) and edge mode (Cloudflare Workers)

## What doesn't exist yet

- No deployed instance
- No browser mode
- No npm package published (install from source)

## License

MIT
