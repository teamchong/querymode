# QueryMode

> **Experimental** — early prototype, not production-ready. Architecture and API will change.

A pluggable columnar query library — not a query engine you push data to, but a query capability your code uses directly. No data materialization, no engine boundary, no SQL transpilation.

## Why "mode" not "engine"

Every query engine — Spark, DataFusion, DuckDB, Polars — has a boundary between your code and the engine:

```
Traditional engine:

  Your Code                      Engine
  ─────────                      ──────
  filter(age > 25)  ──────►   translate to internal plan
                               materialize data into Arrow/DataFrame
                               run engine's fixed operators
                               serialize results
                    ◄──────   return results to your code

  Your code CANNOT cross the boundary.
  Custom business logic? Pull data out, process in your code, push back in.
  That round-trip IS data materialization.
```

LINQ and ORMs look like code-first but they're transpiling expressions to SQL strings sent to a separate database. The database still materializes your data into its format, runs its fixed operators, and sends results back.

QueryMode has no boundary:

```typescript
// This IS the execution — not a description translated to SQL
const orders = await qm.table("orders").filter("amount", "gt", 100).exec()
const userIds = orders.rows.map(r => r.user_id)  // your code, zero materialization
const users = await qm.table("users").filter("id", "in", userIds).exec()

// JOIN logic, business rules, ML scoring — all your code
// WASM handles byte-level column decode + SIMD filtering
// But orchestration is YOUR code, not a query planner's fixed operators
```

Your app code IS the query execution. The WASM engine is a library function your code calls — column decoding, SIMD filtering, vector search happen in-process, on raw bytes, zero-copy. There's no "register UDF → engine materializes data → calls your function → collects results" boundary.

**What this means in practice:**
- No data materialization — data stays in R2/disk, only the exact matching bytes are read
- No engine boundary — your business logic runs directly, not as a registered UDF
- No SQL transpilation — the API calls ARE the execution, not a description sent elsewhere
- No fixed operator set — your code can do anything between query steps
- Same binary everywhere — browser, Node/Bun, Cloudflare DO

## What exists

- **TypeScript orchestration** — Durable Object lifecycle, R2 range reads, footer caching, request routing
- **Zig WASM engine** (`wasm/`) — column decoding, SIMD ops, SQL execution, vector search, fragment writing, compiles to `querymode.wasm`
- **Code-first query API** — `.table().filter().select().sort().limit().exec()`, no SQL
- **Write path** — `TableQuery.append(rows)` with CAS-based manifest coordination via Master DO
- **Master/Query DO split** — single-writer Master broadcasts footer invalidations to per-region Query DOs
- **Footer caching** — table footers (~4KB each) cached in DO memory with VIP eviction (hot tables protected from eviction)
- **Bounded prefetch pipeline** — R2 range fetches overlap I/O (fetch page N+1 while WASM processes page N)
- **IVF-PQ vector search** — index-aware routing in Query DO, falls back to flat SIMD search when no index present
- **Multi-format support** — Lance, Parquet, and Iceberg tables
- **Local mode** — same API reads Lance/Parquet files from disk or HTTP (Node/Bun)
- **Fragment DO pool** — fan-out parallel scanning for multi-fragment datasets (max 20 slots per datacenter)
- **112 unit tests + 20 integration tests** — footer parsing, column decoding, Parquet/Thrift, merging, aggregates, VIP cache, WASM integration (skipped without binary)

## What doesn't exist yet

- No deployed instance
- No browser mode
- No benchmarks against real data
- No npm package published

## Architecture
![querymode-architecture](docs/architecture/querymode-architecture.svg)

## Build

```bash
pnpm install          # install dependencies
pnpm build            # typecheck (tsc)
pnpm test             # run vitest
pnpm dev              # local dev with wrangler

# build WASM from Zig source (requires zig)
pnpm wasm             # cd wasm && zig build wasm && cp to src/wasm/
```

## Query API

```typescript
import { QueryMode } from "querymode"

// Local mode — query files directly where they sit
const qm = QueryMode.local()
const results = await qm
  .table("./data/users.lance")
  .filter("age", "gt", 25)
  .select("name", "email")
  .exec()

// Edge mode — same API, WASM runs inside regional DOs
const qm = QueryMode.remote(env.QUERY_DO, { region: "SJC" })
const results = await qm
  .table("users")
  .filter("age", "gt", 25)
  .select("name", "email")
  .sort("age", "desc")
  .limit(100)
  .exec()

// JOINs are code, not SQL — your logic, zero materialization
const orders = await qm.table("orders").filter("amount", "gt", 100).exec()
const userIds = orders.rows.map(r => r.user_id)
const users = await qm.table("users").filter("id", "in", userIds).exec()
const enriched = orders.rows.map(o => ({
  ...o,
  user: users.rows.find(u => u.id === o.user_id)
}))

// Write path (append rows)
await qm.table("users").append([
  { id: 1, name: "Alice", age: 30 },
  { id: 2, name: "Bob", age: 25 },
])

// Vector search (flat or IVF-PQ accelerated)
const similar = await qm
  .table("images")
  .vector("embedding", queryVec, 10)
  .exec()
```

## How it works

```
Traditional engine:  fetch metadata (RTT) → plan → fetch ALL data (RTT) → materialize → execute → serialize → return
QueryMode:           plan instantly (footer cached) → fetch ONLY matching byte ranges (RTT) → WASM decode zero-copy → done
```

1. **Footer cache** — every table's metadata (~4KB) is cached in DO memory. Query planning is instant, no round-trip.
2. **Page-level skip** — min/max stats per page mean non-matching pages are never read, never downloaded, never allocated.
3. **Coalesced Range reads** — nearby byte ranges merged within 64KB gaps into fewer R2 requests.
4. **Zero-copy WASM** — raw bytes from R2 are passed directly to Zig SIMD. No Arrow conversion, no DataFrame construction.
5. **VIP eviction** — frequently-accessed table footers are protected from cache eviction by cold one-off accesses.
6. **Bounded prefetch** — up to 8 R2 reads in-flight simultaneously, overlapping I/O with compute.

## License

MIT
