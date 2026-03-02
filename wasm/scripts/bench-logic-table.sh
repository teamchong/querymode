#!/bin/bash
# bench-logic-table.sh - @logic_table ML Workflow Benchmark (LanceQL vs DuckDB vs Polars)
#
# Benchmarks:
#   - Feature Engineering (1B rows): normalize, z-score, log transform
#   - Vector Search (10M docs x 384-dim): cosine similarity, euclidean distance
#   - Fraud Detection (500M transactions): multi-factor risk scoring
#   - Recommendations (5M items x 256-dim): collaborative filtering
#   - SQL Clauses (200M rows): SELECT, WHERE, GROUP BY, ORDER BY
#
# Each benchmark runs 30+ seconds.
#
# Usage: ./scripts/bench-logic-table.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "@logic_table ML Workflow Benchmark (LanceQL vs DuckDB vs Polars)"
echo "================================================================================"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Platform: $(uname -s) $(uname -m)"
echo ""

# Check engines
echo "Engines:"
echo "  - LanceQL: native Zig SIMD"

if python3 -c "import duckdb" 2>/dev/null; then
    echo "  - DuckDB: $(python3 -c 'import duckdb; print(duckdb.__version__)')"
else
    echo "  - DuckDB: not installed (pip install duckdb)"
fi

if python3 -c "import polars" 2>/dev/null; then
    echo "  - Polars: $(python3 -c 'import polars; print(polars.__version__)')"
else
    echo "  - Polars: not installed (pip install polars)"
fi
echo ""

# Build and run
echo "Building bench-logic-table..."
if ! zig build bench-logic-table 2>&1; then
    echo ""
    echo "================================================================================"
    echo "SKIPPED: bench-logic-table requires metal0 module dependencies not yet configured"
    echo "================================================================================"
    echo ""
    echo "The benchmark uses lanceql.codegen which requires metal0 with full module wiring."
    echo "To enable this benchmark, configure all metal0 internal modules in build.zig:"
    echo "  - c_interop"
    echo "  - analysis/native_types.zig"
    echo "  - lexer.zig, parser.zig, compiler.zig"
    echo "  - codegen/native/main.zig"
    echo "  - analysis/types.zig, analysis/lifetime.zig"
    echo ""
    echo "For now, use bench-sql-clauses.sh for SQL benchmarks instead."
    exit 0
fi
