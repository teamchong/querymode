# sql/ — SQL Frontend

> Parses SQL strings into QueryDescriptors that the pipeline executes.
> Follows LanceDB syntax. 755+ tests.

## Pipeline

```
SQL string
  │
  ▼
lexer.ts ──► Token[]           Tokenizes keywords, identifiers, literals, operators
  │
  ▼
parser.ts ──► AST              Recursive-descent parser → SelectStatement, Expression, etc.
  │
  ▼
compiler.ts ──► QueryDescriptor + extras
  │              Flattens WHERE → filters/filterGroups
  │              Extracts HAVING, ORDER BY, computed exprs
  │              Decomposes OR into filterGroups[][]
  │
  ▼
executor.ts ──► QueryResult    Wraps pipeline execution
  │              SqlWrappingExecutor handles post-pipeline ops:
  │              HAVING, multi-column ORDER BY, CASE/CAST/arithmetic
  │
  ▼
evaluator.ts                   Expression evaluator for computed columns
                               Handles: CASE, CAST, arithmetic, string ops,
                               COALESCE, NULLIF, IN, BETWEEN, LIKE
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| lexer.ts | ~400 | Tokenizer. Keywords, string/number literals, operators |
| parser.ts | ~1200 | Recursive-descent parser → AST nodes |
| ast.ts | ~200 | AST type definitions (SelectStatement, Expression, etc.) |
| compiler.ts | ~800 | AST → QueryDescriptor. Filter pushdown, OR decomposition |
| executor.ts | ~500 | Query execution wrapper. Delegates agg to partial-agg |
| evaluator.ts | ~600 | Row-level expression evaluator (CASE, CAST, math) |
| index.ts | ~30 | Re-exports: parse, compile, sqlToDescriptor, buildSqlDataFrame |

## Key Design Decisions

- **All 14 filter ops push down.** The compiler flattens WHERE predicates into descriptor filters (eq, neq, gt, gte, lt, lte, in, not_in, between, not_between, like, not_like, is_null, is_not_null). These reach the scan operator for page-level pruning.
- **OR → filterGroups.** `WHERE a = 1 OR b = 2` becomes `filterGroups: [[{a, eq, 1}], [{b, eq, 2}]]`. Each group is AND-connected internally, groups are OR-connected.
- **compileFull()** returns both the descriptor and extras (whereExpr, havingExpr, computedExprs, allOrderBy) for the wrapping executor.
- **NEAR operator** for vector search in WHERE clause, following LanceDB syntax.
- **CTE inlining** — WITH clauses inline into the main query (no materialized CTEs).
- **Parsed but not compiled**: EXISTS/NOT EXISTS, SHOW VERSIONS, DIFF — these are in the parser but not wired to execution.

## Supported SQL

```sql
-- Basics
SELECT col1, col2 FROM table WHERE col1 > 5 ORDER BY col2 LIMIT 10 OFFSET 5

-- Aggregation
SELECT category, COUNT(*), AVG(amount) FROM t GROUP BY category HAVING COUNT(*) > 3

-- Joins
SELECT a.id, b.name FROM t1 a JOIN t2 b ON a.id = b.fk

-- Window functions
SELECT id, SUM(amount) OVER (PARTITION BY cat ORDER BY dt ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)

-- Set operations
SELECT id FROM t1 UNION ALL SELECT id FROM t2

-- Vector search
SELECT * FROM embeddings WHERE NEAR(vector, [0.1, 0.2, ...], 10, 'cosine')

-- Expressions
SELECT CASE WHEN status = 'A' THEN 'active' ELSE 'inactive' END,
       CAST(amount AS FLOAT), price * qty AS total
```
