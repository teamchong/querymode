# QueryMode Roadmap

## Completed

### 1. CTE in SQL compiler ✓
SQL `WITH` clause → inlines CTE filters into main query descriptor. 3 parser tests + 5 integration tests.

### 2. RBAC middleware example ✓
Worker middleware that injects row-level filters and strips column-level projections based on JWT. Example in `examples/rbac-middleware.ts`.

### 3. Postgres wire protocol ✓
TCP server implementing pg wire v3 Simple Query protocol. Connect with psql, Tableau, Metabase, Grafana, dbt. 17 tests. `src/pg-wire/`.

### 4. Schema evolution ✓
`addColumn(name, dtype, default)` and `dropColumn(name)` — no data rewrite. Works on MaterializedExecutor, same pattern for Lance manifests. 5 tests.

### 5. Partitioned writes ✓
Fan out writes by partition key to separate MasterDOs. Each DO handles its own CAS loop — no contention across partitions. Bio-cell model: more write pressure → more cells.

### 6. `npx querymode init` ✓
One-command project scaffold: generates wrangler.toml + src/worker.ts. Idempotent. Prints next steps.

### 7. Hierarchical reduction ✓
Tree merge when fragment count exceeds threshold (default 50). Leaf Fragment DOs scan → reducer Fragment DOs merge groups of 25 → QueryDO merges final set. Keeps QueryDO memory bounded at any scale. `reduceRpc` on Fragment DO, `hierarchicalReduce` on Query DO. Explain output shows `hierarchicalReduction` and `reducerTiers`.

### 8. `materializeAs()` multi-stage pipeline ✓
`df.materializeAs("table_name")` executes stage, writes results to a named table, returns new DataFrame for next stage. TypeScript is the DAG scheduler — chain stages through the executor's write path. 2 tests.

## Future

### `.pipe()` DAG scheduler (Spark replacement)
Build on `materializeAs()`: automatic intermediate table naming, cleanup after pipeline completion, lineage tracking via `AppendOptions.metadata`. TypeScript orchestrates the DAG — no Spark/YARN needed.
