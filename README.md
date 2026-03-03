# EdgeQ

> **Experimental** — early prototype, not production-ready. Architecture and API will change.

Serverless columnar query engine on Cloudflare Durable Objects. Queries Lance and Parquet files in R2 with cached footers and parallel range reads.

## What exists

- **TypeScript orchestration** — Durable Object lifecycle, R2 range reads, footer caching, request routing
- **Zig WASM engine** (`wasm/`) — column decoding, SIMD ops, SQL execution, vector search, fragment writing, compiles to `edgeq.wasm`
- **Code-first query API** — `.table().filter().select().sort().limit().exec()`, no SQL
- **Write path** — `TableQuery.append(rows)` with CAS-based manifest coordination via Master DO
- **Master/Query DO split** — single-writer Master broadcasts footer invalidations to per-region Query DOs
- **Footer caching** — table footers (~4KB each) cached in DO memory + SQLite, eliminating the metadata round-trip
- **IVF-PQ vector search** — index-aware routing in Query DO, falls back to flat SIMD search when no index present
- **Multi-format support** — Lance, Parquet, and Iceberg tables
- **Local mode** — same API reads Lance/Parquet files from disk or HTTP (Node/Bun)
- **Fragment DO pool** — fan-out parallel scanning for multi-fragment datasets
- **105 unit tests + 20 integration tests** — footer parsing, column decoding, Parquet/Thrift, merging, aggregates, WASM integration (skipped without binary)

## What doesn't exist yet

- No deployed instance
- No browser mode
- No benchmarks against real data
- No npm package published

## Architecture
![edgeq-architecture](.docs/architecture/edgeq-architecture.svg)

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
import { EdgeQ } from "edgeq"

// Edge mode (Durable Objects + R2)
const eq = EdgeQ.remote(env.QUERY_DO, { region: "SJC" })
const results = await eq
  .table("users")
  .filter("age", "gt", 25)
  .select("name", "email")
  .sort("age", "desc")
  .limit(100)
  .exec()

// Local mode (Node/Bun + filesystem)
const local = EdgeQ.local()
const results = await local
  .table("./data/users.lance")
  .select("name")
  .exec()

// Write path (append rows)
await eq.table("users").append([
  { id: 1, name: "Alice", age: 30 },
  { id: 2, name: "Bob", age: 25 },
])

// Vector search (flat or IVF-PQ accelerated)
const similar = await eq
  .table("images")
  .vector("embedding", queryVec, 10)
  .exec()
```

## License

MIT
