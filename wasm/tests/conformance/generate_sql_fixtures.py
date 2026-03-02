#!/usr/bin/env python3
"""
Generate SQL conformance test fixtures using PyLance as the reference implementation.

This script creates test data with edge cases, runs queries through PyLance,
and saves expected results as JSON fixtures for validation by the Zig test runner.

Usage:
    pip install pylance pyarrow numpy
    python generate_sql_fixtures.py
"""

import json
import os
import shutil
from pathlib import Path

import lance
import pyarrow as pa
import numpy as np

# Output directories
SCRIPT_DIR = Path(__file__).parent
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
TEST_DATASET_DIR = FIXTURES_DIR / "conformance_test.lance"


def create_test_dataset():
    """Create test dataset with edge cases for SQL conformance testing."""

    # Test data with NULLs, various types, edge values
    test_table = pa.table({
        # Integer column with NULLs
        'a': pa.array([1, 2, 3, None, 5, None, 7, 0, -1, 10], type=pa.int64()),

        # Integer column with different NULL positions
        'b': pa.array([10, None, 30, 40, None, 60, 70, 0, -10, 100], type=pa.int64()),

        # String column with NULLs and various patterns
        'c': pa.array(['foo', 'bar', None, 'baz', 'qux', None, 'quux', 'foo%bar', 'f_o', ''], type=pa.string()),

        # Float column for type coercion tests
        'd': pa.array([1.5, 2.0, 3.0, None, 5.5, 6.0, 7.7, 0.0, -1.5, 10.0], type=pa.float64()),

        # Boolean column
        'e': pa.array([True, False, None, True, False, None, True, False, True, False]),
    })

    # Remove existing dataset if present
    if TEST_DATASET_DIR.exists():
        shutil.rmtree(TEST_DATASET_DIR)

    # Write dataset using PyLance
    lance.write_dataset(test_table, str(TEST_DATASET_DIR), mode="overwrite")

    print(f"Created test dataset at {TEST_DATASET_DIR}")
    print(f"  Rows: {test_table.num_rows}")
    print(f"  Columns: {test_table.column_names}")

    return test_table


def run_query(filter_expr: str) -> dict:
    """Run a filter query through PyLance and return results as dict."""
    try:
        ds = lance.dataset(str(TEST_DATASET_DIR))
        result = ds.to_table(filter=filter_expr)

        # Convert to dict format
        return {
            "success": True,
            "row_count": result.num_rows,
            "columns": result.column_names,
            "data": {col: result.column(col).to_pylist() for col in result.column_names},
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def save_fixture(name: str, query: str, result: dict):
    """Save query result as JSON fixture."""
    fixture = {
        "name": name,
        "query": query,
        "result": result,
    }

    fixture_path = FIXTURES_DIR / f"{name}.expected.json"
    with open(fixture_path, "w") as f:
        json.dump(fixture, f, indent=2, default=str)

    print(f"  Saved: {name}.expected.json (rows: {result.get('row_count', 'error')})")


def generate_and_or_precedence_tests():
    """Test AND/OR operator precedence."""
    print("\n=== AND/OR Precedence Tests ===")

    queries = {
        # AND has higher precedence than OR
        "and_or_precedence_1": "a > 2 AND b < 50 OR a = 1",  # (a > 2 AND b < 50) OR (a = 1)
        "and_or_precedence_2": "a > 2 AND (b < 50 OR a = 1)",  # a > 2 AND (b < 50 OR a = 1)
        "and_or_precedence_3": "a = 1 OR b > 50 AND a < 10",  # a = 1 OR (b > 50 AND a < 10)
        "and_or_precedence_4": "(a = 1 OR b > 50) AND a < 10",

        # NOT precedence
        "not_precedence_1": "NOT a > 5 AND b < 50",  # (NOT (a > 5)) AND (b < 50)
        "not_precedence_2": "NOT (a > 5 AND b < 50)",  # NOT ((a > 5) AND (b < 50))
    }

    for name, query in queries.items():
        result = run_query(query)
        save_fixture(name, query, result)


def generate_null_handling_tests():
    """Test NULL handling in comparisons."""
    print("\n=== NULL Handling Tests ===")

    queries = {
        # Direct NULL comparisons (SQL standard: always false)
        "null_equals_null": "a = NULL",
        "null_not_equals_null": "a != NULL",
        "null_less_than": "a < NULL",
        "null_greater_than": "a > NULL",

        # IS NULL / IS NOT NULL
        "is_null": "a IS NULL",
        "is_not_null": "a IS NOT NULL",
        "is_null_string": "c IS NULL",

        # NULL with AND/OR (three-valued logic)
        "null_and_true": "a IS NULL AND b > 0",
        "null_and_false": "a IS NULL AND b < -100",
        "null_or_true": "a IS NULL OR b > 0",
        "null_or_false": "a IS NULL OR b < -100",

        # Comparison with NULL column values
        "compare_with_null_column": "a > b",  # Some rows have NULL b
        "null_in_both_columns": "a = b",  # Both can be NULL
    }

    for name, query in queries.items():
        result = run_query(query)
        save_fixture(name, query, result)


def generate_in_operator_tests():
    """Test IN operator with various edge cases."""
    print("\n=== IN Operator Tests ===")

    queries = {
        # Basic IN
        "in_basic": "a IN (1, 2, 3)",
        "in_single": "a IN (1)",
        "in_many": "a IN (1, 2, 3, 5, 7)",

        # NOT IN
        "not_in_basic": "a NOT IN (1, 2, 3)",

        # IN with NULL in the list
        "in_with_null_in_list": "a IN (1, NULL, 3)",
        "not_in_with_null_in_list": "a NOT IN (1, NULL, 3)",

        # IN where the column has NULL
        "in_null_column": "a IN (1, 2, 3)",  # a has NULLs

        # IN with floats in int column (type coercion)
        "in_float_for_int": "a IN (1.0, 2.0, 3.0)",

        # String IN
        "in_string": "c IN ('foo', 'bar')",
        "in_string_with_null": "c IN ('foo', NULL)",
    }

    for name, query in queries.items():
        result = run_query(query)
        save_fixture(name, query, result)


def generate_between_tests():
    """Test BETWEEN operator with edge cases."""
    print("\n=== BETWEEN Tests ===")

    queries = {
        # Basic BETWEEN (inclusive)
        "between_basic": "a BETWEEN 2 AND 5",
        "between_edge_equal": "a BETWEEN 5 AND 5",  # a = 5

        # BETWEEN with reversed bounds (key edge case!)
        "between_reversed": "a BETWEEN 5 AND 2",

        # BETWEEN with negative numbers
        "between_negative": "a BETWEEN -1 AND 2",

        # BETWEEN with NULL bounds
        "between_null_low": "a BETWEEN NULL AND 5",
        "between_null_high": "a BETWEEN 2 AND NULL",
        "between_both_null": "a BETWEEN NULL AND NULL",

        # NULL column with BETWEEN
        "null_column_between": "a BETWEEN 1 AND 10",  # a has NULLs

        # NOT BETWEEN
        "not_between": "a NOT BETWEEN 2 AND 5",

        # Float BETWEEN
        "between_float": "d BETWEEN 1.0 AND 5.0",
    }

    for name, query in queries.items():
        result = run_query(query)
        save_fixture(name, query, result)


def generate_like_tests():
    """Test LIKE pattern matching."""
    print("\n=== LIKE Tests ===")

    queries = {
        # Basic patterns
        "like_exact": "c LIKE 'foo'",
        "like_prefix": "c LIKE 'foo%'",
        "like_suffix": "c LIKE '%ux'",
        "like_contains": "c LIKE '%a%'",

        # Single character wildcard
        "like_single_char": "c LIKE 'f_o'",
        "like_multiple_single": "c LIKE 'b__'",

        # Mixed patterns
        "like_mixed": "c LIKE 'f%o'",
        "like_start_end": "c LIKE 'f%x'",

        # Escape sequences (testing PyLance's convention)
        "like_literal_percent": "c LIKE '%\\%%'",  # Match literal %
        "like_literal_underscore": "c LIKE '%\\_%'",  # Match literal _

        # NOT LIKE
        "not_like": "c NOT LIKE 'foo%'",

        # LIKE with NULL
        "like_null_pattern": "c LIKE NULL",
        "null_column_like": "c LIKE 'foo'",  # c has NULLs

        # Empty string
        "like_empty": "c LIKE ''",
        "like_match_empty": "c LIKE '%'",  # Should match everything including empty
    }

    for name, query in queries.items():
        result = run_query(query)
        save_fixture(name, query, result)


def generate_type_coercion_tests():
    """Test type coercion in comparisons."""
    print("\n=== Type Coercion Tests ===")

    queries = {
        # Int/Float comparison
        "int_equals_float": "a = 3.0",
        "float_equals_int": "d = 2",
        "int_compare_float": "a > 2.5",
        "float_compare_int": "d < 5",

        # Float with integer value
        "float_int_value": "d = 2.0",  # d is float, 2.0 is int-like float
    }

    for name, query in queries.items():
        result = run_query(query)
        save_fixture(name, query, result)


def generate_order_by_null_tests():
    """Test ORDER BY with NULL values."""
    print("\n=== ORDER BY with NULLs Tests ===")

    # Note: PyLance filter doesn't support ORDER BY directly in filter expression
    # We'll test the data ordering separately
    # For now, document what we find about NULL ordering

    queries = {
        # These test that NULLs are properly filtered (prerequisite for ORDER BY)
        "nulls_filtered_out": "a IS NOT NULL",
        "nulls_filtered_in": "a IS NULL",
    }

    for name, query in queries.items():
        result = run_query(query)
        save_fixture(name, query, result)

    # Document expected ORDER BY NULL behavior
    doc = {
        "note": "ORDER BY NULL handling documentation",
        "sql_standard": {
            "NULLS FIRST": "NULLs appear before non-NULL values",
            "NULLS LAST": "NULLs appear after non-NULL values",
            "default_asc": "Implementation-defined (typically NULLS LAST)",
            "default_desc": "Implementation-defined (typically NULLS FIRST)",
        },
        "recommendation": "Test ORDER BY separately in executor tests",
    }

    with open(FIXTURES_DIR / "order_by_nulls.doc.json", "w") as f:
        json.dump(doc, f, indent=2)


def generate_complex_expression_tests():
    """Test complex nested expressions."""
    print("\n=== Complex Expression Tests ===")

    queries = {
        # Deeply nested
        "nested_and_or": "((a > 1 AND b < 50) OR (a < 0 AND b > 0)) AND c IS NOT NULL",

        # Multiple NOT
        "multiple_not": "NOT (NOT (a > 5))",

        # Combination of operators
        "combo_in_and_compare": "a IN (1, 2, 3) AND b > 20",
        "combo_between_and_like": "a BETWEEN 1 AND 5 AND c LIKE 'f%'",
        "combo_null_and_in": "a IS NOT NULL AND a IN (1, 2, 3, 5, 7)",
    }

    for name, query in queries.items():
        result = run_query(query)
        save_fixture(name, query, result)


def generate_edge_value_tests():
    """Test edge values (zero, negative, empty string)."""
    print("\n=== Edge Value Tests ===")

    queries = {
        # Zero
        "equals_zero": "a = 0",
        "greater_than_zero": "a > 0",
        "less_than_zero": "a < 0",

        # Negative
        "equals_negative": "a = -1",
        "between_negative": "a BETWEEN -5 AND 0",

        # Empty string
        "equals_empty_string": "c = ''",
        "not_equals_empty_string": "c != ''",
    }

    for name, query in queries.items():
        result = run_query(query)
        save_fixture(name, query, result)


def create_summary():
    """Create summary of all generated fixtures."""
    print("\n=== Creating Summary ===")

    fixtures = list(FIXTURES_DIR.glob("*.expected.json"))
    summary = {
        "total_fixtures": len(fixtures),
        "categories": {},
        "fixtures": [],
    }

    for fixture_path in sorted(fixtures):
        with open(fixture_path) as f:
            fixture = json.load(f)

        name = fixture["name"]
        category = name.split("_")[0]

        if category not in summary["categories"]:
            summary["categories"][category] = 0
        summary["categories"][category] += 1

        summary["fixtures"].append({
            "name": name,
            "query": fixture["query"],
            "success": fixture["result"].get("success", True),
            "row_count": fixture["result"].get("row_count", "error"),
        })

    summary_path = FIXTURES_DIR / "SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGenerated {len(fixtures)} fixtures in {len(summary['categories'])} categories:")
    for cat, count in sorted(summary["categories"].items()):
        print(f"  {cat}: {count} tests")


def main():
    """Generate all conformance test fixtures."""
    print("=" * 60)
    print("PyLance SQL Conformance Test Fixture Generator")
    print("=" * 60)

    # Ensure fixtures directory exists
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Create test dataset
    create_test_dataset()

    # Generate test fixtures by category
    generate_and_or_precedence_tests()
    generate_null_handling_tests()
    generate_in_operator_tests()
    generate_between_tests()
    generate_like_tests()
    generate_type_coercion_tests()
    generate_order_by_null_tests()
    generate_complex_expression_tests()
    generate_edge_value_tests()

    # Create summary
    create_summary()

    print("\n" + "=" * 60)
    print("Fixture generation complete!")
    print(f"Fixtures saved to: {FIXTURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
