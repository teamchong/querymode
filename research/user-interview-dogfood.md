# User Interview: Dogfooding QueryMode

**Date:** 2026-03-06
**Method:** Hands-on usage audit + competitive analysis
**Persona:** Developer evaluating QueryMode for edge analytics

---

## Executive Summary

QueryMode has a **unique and defensible position** as the only columnar query engine that runs natively inside Cloudflare Durable Objects with zero-copy R2 reads. The DataFrame API is well-designed and the operator pipeline is impressive. However, **onboarding friction, documentation bugs, and missing table-stakes features** would cause most evaluators to bounce before seeing the value.

**Verdict:** Strong engine, weak front door.

---

## Part 1: Issues Encountered (as a first-time user)

### P0 - Blockers (would stop evaluation)

| # | Issue | Detail |
|---|-------|--------|
| 1 | **`pnpm add querymode` fails** | README line 8 says to install, but package isn't published to npm. Line 242 buries this. User tries to install, fails, leaves. |
| 2 | **`whereNotNull()` crashes** | Documented in `dataframe-api.mdx:14` and used in `examples/local-quickstart.ts:18`, but the method does not exist in `client.ts`. Example literally crashes. |
| 3 | **`.join()` docs show wrong signature** | Docs: `orders.join(users, "user_id", "id", "inner")`. Actual: `orders.join(users, { left: "user_id", right: "id" }, "inner")`. User code won't compile. |
| 4 | **Build requires Zig (undocumented)** | `pnpm build` runs `build:wasm` which needs Zig toolchain. No mention in README or docs. User clones repo, can't build. |

### P1 - Significant Friction

| # | Issue | Detail |
|---|-------|--------|
| 5 | **`.compute()` vs `.computed()` mismatch** | Docs say `compute()`, code says `computed()` (`client.ts:276`). |
| 6 | **`.unionAll()` doesn't exist** | Docs say `df1.unionAll(df2)`, actual API is `df1.union(df2, true)`. |
| 7 | **`head()` return type wrong in docs** | Docs say returns `QueryResult`, actually returns `Promise<T[]>`. |
| 8 | **`fromCSV()` async but `fromJSON()` sync** | `QueryMode.fromCSV()` returns `Promise<DataFrame>`, `fromJSON()` returns `DataFrame`. User writes `const df = QueryMode.fromCSV(...)` then calls `.filter()` on a Promise. |
| 9 | **No fast test command** | `pnpm test` takes ~8 minutes. No `test:quick` or `test:unit`. Kills iteration speed. |
| 10 | **Two vitest configs, no explanation** | `vitest.config.ts` and `vitest.workers.config.ts` with no docs on which to use. |

### P2 - Paper Cuts

| # | Issue | Detail |
|---|-------|--------|
| 11 | **Only 3 error codes** | `TABLE_NOT_FOUND`, `INVALID_FORMAT`, `QUERY_FAILED`. No codes for: schema mismatch, timeout, memory exceeded, invalid filter op. |
| 12 | **`query()` method is a no-op** | `QueryMode.query(fn)` literally does `return fn()`. Misleading — implies transaction semantics. |
| 13 | **Type safety lost on method chains** | `select()`, `aggregate()`, `join()`, `computed()` all return unparameterized `DataFrame`, losing generic `<T>`. |
| 14 | **Filter on non-existent column is silent** | `df.filter("typo_col", "eq", 5)` returns empty results with no warning. |
| 15 | **`count_*` alias is ugly** | Default alias for `count("*")` is `"count_*"` — easy to mistype. |
| 16 | **No `.orderBy()` alias** | Most data tools use `orderBy`. QueryMode only has `sort()`. |
| 17 | **Single-column sort only** | No multi-column sort support. Common need. |
| 18 | **Wrangler warnings about Fragment DO** | `wrangler.toml` binds `FragmentDO` but worker entry doesn't always export it, causing confusing warnings. |

---

## Part 2: Missing Features (what users expect)

| Feature | Status | Competitors Have It? |
|---------|--------|---------------------|
| `whereNotNull()` / `whereNull()` | Not implemented | Polars, DuckDB, Pandas |
| `rename()` columns | Missing | Polars, Pandas, DuckDB |
| `drop()` columns (inverse of select) | Missing | Polars, Pandas |
| `sample(n)` random sampling | README says "planned" | Polars, DuckDB |
| `tail(n)` | Missing | Polars, Pandas |
| `fillna()` / null handling | Missing | Polars, Pandas |
| `pivot()` / `unpivot()` | Missing | DuckDB, Pandas |
| `toJSON()` / `toCSV()` export | Missing | Polars, DuckDB, Pandas |
| Multi-column sort | Missing | All competitors |
| Full-text search (BM25) | Missing | ParadeDB, ClickHouse |
| Materialized views | Missing | ClickHouse, StarRocks |
| SQL mode | Missing | DuckDB, ClickHouse, Athena, BigQuery |
| Browser/WASM mode | Missing | DuckDB-WASM |
| Delta Lake / Hudi support | Missing | Athena, StarRocks |

---

## Part 3: Competitive Landscape

### Direct Competitors

| | QueryMode | DuckDB-WASM | LanceDB | MotherDuck | Turbopuffer |
|---|---|---|---|---|---|
| **Runs at edge** | Yes (CF DOs) | Yes (browser) | No | No | No |
| **Serverless** | Yes | Client-only | Yes | Yes | Yes |
| **Cold start** | ~50ms (DO wake) | ~200ms (WASM init) | N/A | ~500ms | ~500ms |
| **Formats** | Lance+Parquet+Iceberg | Parquet+CSV | Lance | Parquet+CSV | Proprietary |
| **Vector search** | HNSW built-in | No | Yes (core) | No | Yes (core) |
| **SQL** | No (DataFrame API) | Full SQL | No | Full SQL | No |
| **Memory-bounded** | Yes (R2 spill) | No (browser RAM) | N/A | Yes | N/A |
| **Multi-region** | Yes (DO per-datacenter) | N/A | No | No | Yes |
| **Pricing** | CF Workers pricing | Free | Open source | $0.00375/GiB | $1/mo per 1M vectors |

### QueryMode's Unique Position (what nobody else does)

1. **Edge-native columnar pipeline** — Pull-based operators running inside Durable Objects. No other system puts a full query engine at the edge with bounded memory.
2. **Zero-copy R2 reads with WASM SIMD** — Range reads directly from R2 into WASM memory. No intermediate materialization.
3. **Multi-format + vector in one binary** — Parquet, Lance v1/v2, and Iceberg with HNSW vector search, all in one edge-deployed WASM module.
4. **Fragment DO fan-out** — Parallel scan across pooled Durable Objects per-datacenter. Scales horizontally without provisioning.
5. **No engine boundary** — Business logic runs between pipeline stages. No serialization between app code and query engine.

### Where QueryMode Overlaps (competitors do it better today)

| Area | Better Alternative | Why |
|------|-------------------|-----|
| Ad-hoc SQL analytics | DuckDB / MotherDuck | Full SQL, mature optimizer, huge ecosystem |
| Petabyte scans | Athena / BigQuery | Proven at scale, managed infrastructure |
| Real-time OLAP dashboard | ClickHouse / StarRocks | Sub-100ms on pre-aggregated data |
| Python data science | Polars / DuckDB | Rich DataFrame API with type inference |
| Full-text search | ParadeDB / ClickHouse | BM25 built-in, inverted indexes |

### Where QueryMode Wins (nobody else does this)

| Use Case | Why QueryMode |
|----------|---------------|
| Edge analytics dashboard | Sub-50ms p50 from any datacenter. No cold start penalty. |
| Multi-tenant SaaS metrics | One Query DO per tenant/region. Isolated, hibernating, pay-per-use. |
| Vector search at the edge | HNSW in WASM + columnar filters. No separate vector DB needed. |
| Lakehouse over R2 | Native Lance/Parquet/Iceberg. No ETL into a separate DB. |
| Real-time writes + reads | Append via Master DO, instant invalidation to Query DOs. |

---

## Part 4: Recommendations

### Tier 1: Fix the Front Door (week 1)

These are killing the first-5-minutes experience:

1. **Publish to npm** — Even a `0.1.0-alpha` with a disclaimer. `pnpm add querymode` must work.
2. **Fix all doc/code mismatches** — `whereNotNull`, `compute`/`computed`, `unionAll`, `join` signature, `head` return type. Every wrong example is a lost user.
3. **Add `whereNotNull()` and `whereNull()`** — It's in the docs, users expect it, trivial to implement as sugar over filter.
4. **Add a working quickstart** — Clone, install, 5-line script, see results. Test this on a fresh machine.
5. **Document Zig requirement** — Or better: ship pre-built WASM so `pnpm install` is enough.

### Tier 2: Close the DX Gap (weeks 2-4)

These make the library feel production-ready:

6. **Make `fromCSV()` sync** (or make `fromJSON()` async) — Pick one, be consistent.
7. **Add multi-column sort** — `.sort([{ column: "region", direction: "asc" }, { column: "amount", direction: "desc" }])`
8. **Add `rename()`, `drop()`, `toJSON()`, `toCSV()`** — Table-stakes DataFrame methods.
9. **Expand error codes** — `SCHEMA_MISMATCH`, `INVALID_FILTER_OP`, `MEMORY_EXCEEDED`, `NETWORK_TIMEOUT`.
10. **Add `pnpm test:quick`** — Run unit tests only, skip conformance suite. Target <10s.
11. **Fix type safety** — Preserve `DataFrame<T>` generic through `select()`, `aggregate()`, etc.
12. **Validate column names at query build time** — Compare against schema when available (local mode has it).

### Tier 3: Strategic Differentiation (month 2+)

These widen the moat:

13. **SQL mode** — Even a subset (SELECT/WHERE/GROUP BY/ORDER BY). Lowers adoption barrier massively. DuckDB compatibility for migration stories.
14. **Browser mode** — Ship a `querymode/browser` export that runs the WASM engine client-side against fetch() or Cache API. Compete directly with DuckDB-WASM.
15. **Materialized views** — Pre-aggregate hot queries in DO storage. Instant reads for dashboard use cases.
16. **Full-text search** — BM25 inverted index in WASM. Combine with vector search for hybrid retrieval.
17. **Dashboard SDK** — React hooks (`useQuery`, `useStream`) that connect to QueryMode over RPC. The "Supabase for analytics" story.

---

## Part 5: Positioning Recommendation

### Current: "Serverless columnar query engine on Cloudflare"
**Problem:** Too technical, doesn't convey the "why".

### Proposed: "The analytics database that runs at the edge"
**Tagline:** "Query your data lake from every Cloudflare datacenter. Sub-50ms. Zero servers."

### Target personas (in order):
1. **Cloudflare-native teams** already using R2 + Workers. Lowest friction adoption.
2. **Multi-tenant SaaS** needing per-tenant analytics without provisioning databases.
3. **AI/ML teams** needing vector search + columnar queries in one system.

### Anti-personas (don't target yet):
- Data scientists (need SQL + notebooks)
- Petabyte-scale analytics (need Athena/BigQuery scale)
- Teams committed to ClickHouse/StarRocks (need migration tooling first)
