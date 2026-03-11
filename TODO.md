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

## Future

### `.pipe()` stage chaining (Spark replacement)
Multi-stage MapReduce: each `.pipe()` writes intermediate results to R2, next stage reads from it. TypeScript is the DAG scheduler — no Spark/YARN needed.

### Hierarchical reduction
When fragment count exceeds single-QueryDO capacity (~1000), add a reducer tier. Not needed at current scale — pruning keeps fan-out small.
