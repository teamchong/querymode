# QueryMode Roadmap

## From experiment to Snowflake/Spark killer

### 1. CTE in SQL compiler
SQL `WITH` clause → compiles to chained QueryDescriptors. Runtime already handles it (DataFrame variables). Just syntax support in `src/sql/parser.ts` + `compiler.ts`.

### 2. RBAC middleware example
Worker middleware that injects filters (row-level security) and restricts projections (column-level security) based on JWT/CF Access. Example only — no runtime changes.

### 3. Postgres wire protocol
TCP Worker that translates pg wire messages → SQL → QueryDescriptor → pipeline → pg DataRow responses. One adapter = every BI tool (Tableau, Metabase, Grafana, dbt, psql). Lives in `src/pg-wire/`.

### 4. Schema evolution
Add/drop columns via Lance manifest update — no data rewrite. Old fragments return null for new columns. Manifest-only operation in MasterDO.

### 5. Partitioned writes (fan-out writes)
Route writes by partition key hash → multiple MasterDOs. Each owns a partition range, no contention. Bio-cell model: more pressure → more cells. Router Worker picks MasterDO by hash.

### 6. `npx querymode init`
One-command deploy: scaffolds wrangler.toml, creates R2 bucket, configures DOs, deploys Worker. Zero-to-running in 60 seconds.

### 7. `.pipe()` stage chaining (Spark replacement)
Multi-stage MapReduce: each `.pipe()` writes intermediate results to R2, next stage reads from it. TypeScript is the DAG scheduler — no Spark/YARN needed. Each step fans out to Fragment DOs, reduces back, feeds the next step.
