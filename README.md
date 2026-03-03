# EdgeQ

> **Experimental** — early prototype, not production-ready. Architecture and API will change.

Serverless columnar query engine on Cloudflare Durable Objects. Queries Lance files in R2 with cached footers and parallel range reads.

## What exists

- **TypeScript orchestration** (~1200 lines) — Durable Object lifecycle, R2 range reads, footer caching, request routing
- **Zig WASM engine** (163 .zig files in `wasm/`) — column decoding, SIMD ops, vector search, compiles to `lanceql.wasm`
- **Code-first query API** — `.table().filter().select().sort().limit().exec()`, no SQL
- **Master/Query DO split** — single-writer Master broadcasts footer invalidations to per-region Query DOs
- **Footer caching** — table footers (~4KB each) cached in DO memory + SQLite, eliminating the metadata round-trip
- **Local mode** — same API reads Lance files from disk or HTTP (Node/Bun)
- **35 unit tests passing** — footer parsing and column decoding

## What doesn't exist yet

- No end-to-end tests against R2 or deployed Durable Objects
- No deployed instance
- No browser mode
- WASM query path (`executeSql`) not integration-tested from TS side
- Vector search API defined but not tested end-to-end
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
```

## License

MIT
