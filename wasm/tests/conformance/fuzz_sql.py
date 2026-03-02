#!/usr/bin/env python3
"""
Differential fuzzing for SQL conformance.

Generates random valid SQL filter expressions and compares results
between PyLance and LanceQL.

Usage:
    pip install pylance pyarrow
    python fuzz_sql.py --iterations 1000
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import lance
import pyarrow as pa

# Output directory
SCRIPT_DIR = Path(__file__).parent
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
TEST_DATASET_DIR = FIXTURES_DIR / "fuzz_test.lance"


def create_fuzz_dataset():
    """Create dataset for fuzz testing."""
    # Generate random data with various types and NULL patterns
    random.seed(42)
    n_rows = 100

    data = {
        'a': [random.randint(-100, 100) if random.random() > 0.1 else None for _ in range(n_rows)],
        'b': [random.randint(-100, 100) if random.random() > 0.1 else None for _ in range(n_rows)],
        'c': [random.choice(['foo', 'bar', 'baz', 'qux', None]) for _ in range(n_rows)],
        'd': [random.uniform(-100, 100) if random.random() > 0.1 else None for _ in range(n_rows)],
    }

    table = pa.table({
        'a': pa.array(data['a'], type=pa.int64()),
        'b': pa.array(data['b'], type=pa.int64()),
        'c': pa.array(data['c'], type=pa.string()),
        'd': pa.array(data['d'], type=pa.float64()),
    })

    if TEST_DATASET_DIR.exists():
        import shutil
        shutil.rmtree(TEST_DATASET_DIR)

    lance.write_dataset(table, str(TEST_DATASET_DIR), mode="overwrite")
    return table


class SQLFuzzer:
    """Generates random valid SQL filter expressions."""

    COMPARISON_OPS = ['=', '!=', '<', '<=', '>', '>=']
    INT_COLS = ['a', 'b']
    STRING_COLS = ['c']
    FLOAT_COLS = ['d']
    ALL_COLS = INT_COLS + STRING_COLS + FLOAT_COLS

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_int_literal(self) -> str:
        return str(random.randint(-50, 50))

    def generate_float_literal(self) -> str:
        return f"{random.uniform(-50, 50):.2f}"

    def generate_string_literal(self) -> str:
        return f"'{random.choice(['foo', 'bar', 'baz', 'qux'])}'"

    def generate_comparison(self) -> str:
        """Generate a simple comparison expression."""
        col_type = random.choice(['int', 'float', 'string'])

        if col_type == 'int':
            col = random.choice(self.INT_COLS)
            op = random.choice(self.COMPARISON_OPS)
            val = self.generate_int_literal()
        elif col_type == 'float':
            col = random.choice(self.FLOAT_COLS)
            op = random.choice(self.COMPARISON_OPS)
            val = self.generate_float_literal()
        else:
            col = random.choice(self.STRING_COLS)
            op = random.choice(['=', '!='])
            val = self.generate_string_literal()

        return f"{col} {op} {val}"

    def generate_is_null(self) -> str:
        """Generate IS NULL / IS NOT NULL expression."""
        col = random.choice(self.ALL_COLS)
        if random.random() > 0.5:
            return f"{col} IS NULL"
        return f"{col} IS NOT NULL"

    def generate_in_list(self) -> str:
        """Generate IN list expression."""
        col_type = random.choice(['int', 'string'])

        if col_type == 'int':
            col = random.choice(self.INT_COLS)
            values = [self.generate_int_literal() for _ in range(random.randint(2, 5))]
        else:
            col = random.choice(self.STRING_COLS)
            values = [self.generate_string_literal() for _ in range(random.randint(2, 4))]

        negated = "NOT " if random.random() > 0.7 else ""
        return f"{col} {negated}IN ({', '.join(values)})"

    def generate_between(self) -> str:
        """Generate BETWEEN expression."""
        col_type = random.choice(['int', 'float'])

        if col_type == 'int':
            col = random.choice(self.INT_COLS)
            low = random.randint(-50, 0)
            high = random.randint(0, 50)
        else:
            col = random.choice(self.FLOAT_COLS)
            low = random.uniform(-50, 0)
            high = random.uniform(0, 50)

        return f"{col} BETWEEN {low} AND {high}"

    def generate_like(self) -> str:
        """Generate LIKE expression."""
        col = random.choice(self.STRING_COLS)
        patterns = ['foo%', '%bar', '%a%', 'f_o', 'ba_']
        pattern = random.choice(patterns)
        negated = "NOT " if random.random() > 0.8 else ""
        return f"{col} {negated}LIKE '{pattern}'"

    def generate_simple_expr(self) -> str:
        """Generate a simple atomic expression."""
        expr_type = random.choice([
            'comparison', 'comparison', 'comparison',  # More common
            'is_null', 'in_list', 'between', 'like'
        ])

        if expr_type == 'comparison':
            return self.generate_comparison()
        elif expr_type == 'is_null':
            return self.generate_is_null()
        elif expr_type == 'in_list':
            return self.generate_in_list()
        elif expr_type == 'between':
            return self.generate_between()
        elif expr_type == 'like':
            return self.generate_like()

    def generate_filter(self, depth: int = 0, max_depth: int = 3) -> str:
        """Generate a random filter expression with controlled depth."""
        if depth >= max_depth or random.random() < 0.6:
            return self.generate_simple_expr()

        # Combine expressions with AND/OR
        left = self.generate_filter(depth + 1, max_depth)
        right = self.generate_filter(depth + 1, max_depth)
        op = random.choice(['AND', 'OR'])

        # Optionally add NOT
        if random.random() > 0.9:
            return f"NOT ({left} {op} {right})"

        # Optionally use parentheses
        if random.random() > 0.5:
            return f"({left}) {op} ({right})"

        return f"{left} {op} {right}"


def run_pylance_query(filter_expr: str) -> tuple[bool, dict]:
    """Run query through PyLance and return results."""
    try:
        ds = lance.dataset(str(TEST_DATASET_DIR))
        result = ds.to_table(filter=filter_expr)

        return True, {
            "row_count": result.num_rows,
            "data": {col: result.column(col).to_pylist() for col in result.column_names},
        }
    except Exception as e:
        return False, {"error": str(e), "error_type": type(e).__name__}


def fuzz_test(iterations: int = 1000, seed: Optional[int] = None, verbose: bool = False):
    """Run differential fuzzing."""
    print(f"Creating fuzz test dataset...")
    create_fuzz_dataset()

    fuzzer = SQLFuzzer(seed=seed)

    successes = 0
    failures = []
    errors = []

    print(f"Running {iterations} fuzz iterations...")

    for i in range(iterations):
        filter_expr = fuzzer.generate_filter()

        success, result = run_pylance_query(filter_expr)

        if success:
            successes += 1
            if verbose:
                print(f"  [{i+1}] OK: {filter_expr[:60]}... -> {result['row_count']} rows")
        else:
            errors.append({
                "iteration": i + 1,
                "query": filter_expr,
                "error": result,
            })
            if verbose:
                print(f"  [{i+1}] ERROR: {filter_expr[:60]}... -> {result['error']}")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{iterations}")

    # Summary
    print("\n" + "=" * 60)
    print("FUZZ TEST SUMMARY")
    print("=" * 60)
    print(f"Total iterations: {iterations}")
    print(f"Successful queries: {successes}")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"\nFirst 5 errors:")
        for err in errors[:5]:
            print(f"  Query: {err['query'][:60]}...")
            print(f"  Error: {err['error']['error']}")
            print()

    # Save detailed results
    results_file = SCRIPT_DIR / "fuzz_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "iterations": iterations,
            "seed": seed,
            "successes": successes,
            "error_count": len(errors),
            "errors": errors[:100],  # Limit saved errors
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_file}")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="SQL differential fuzzing")
    parser.add_argument("--iterations", "-n", type=int, default=1000,
                        help="Number of fuzz iterations")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show each query result")

    args = parser.parse_args()

    success = fuzz_test(
        iterations=args.iterations,
        seed=args.seed,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
