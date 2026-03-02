//! SQL Executor Integration Tests
//!
//! Tests the SQL executor against real Lance files

const std = @import("std");
const Table = @import("lanceql.table").Table;
const ast = @import("lanceql.sql.ast");
const parser = @import("lanceql.sql.parser");
const executor_mod = @import("lanceql.sql.executor");
const Executor = executor_mod.Executor;
const Result = executor_mod.Result;
const Value = ast.Value;

// ============================================================================
// Test Fixtures
// ============================================================================

const int64_fixture = @embedFile("fixtures/simple_int64.lance/data/001110111111001100101010e79bea45fd969c9e1da619850b.lance");
const float64_fixture = @embedFile("fixtures/simple_float64.lance/data/111100011001011101011011e9643e40abbeac7897d4a95cc3.lance");
const mixed_fixture = @embedFile("fixtures/mixed_types.lance/data/110101110101011000001010a45f35499b9af0396c9ed741b0.lance");
const sqlite_fixture = @embedFile("fixtures/mixed_types.lance/data/110101110101011000001010a45f35499b9af0396c9ed741b0.lance");
const with_nulls_fixture = @embedFile("fixtures/with_nulls.lance/data/10010011001000000000001166409f4102b3c1e02271ce9ab7.lance");

// ============================================================================
// Test Context Helper
// ============================================================================

/// Test context that manages table, executor, and result lifecycle.
/// Use init() to set up, exec() to run queries, and deinit() to clean up.
const TestContext = struct {
    table: Table,
    executor: Executor,
    allocator: std.mem.Allocator,
    result: ?Result,
    stmt: ?ast.Statement,

    /// Initialize in-place to avoid pointer invalidation
    pub fn init(self: *TestContext, allocator: std.mem.Allocator, data: []const u8) !void {
        self.allocator = allocator;
        self.result = null;
        self.stmt = null;
        self.table = try Table.init(allocator, data);
        self.executor = Executor.init(&self.table, allocator);
    }

    /// Execute a SQL query and return the result
    pub fn exec(self: *TestContext, sql: []const u8) !*Result {
        // Clean up previous result/stmt if any
        if (self.result) |*r| r.deinit();
        if (self.stmt) |*s| ast.deinitSelectStmt(&s.select, self.allocator);

        self.stmt = try parser.parseSQL(sql, self.allocator);
        self.result = try self.executor.execute(&self.stmt.?.select, &[_]Value{});
        return &self.result.?;
    }

    /// Clean up all resources
    pub fn deinit(self: *TestContext) void {
        if (self.result) |*r| r.deinit();
        if (self.stmt) |*s| ast.deinitSelectStmt(&s.select, self.allocator);
        self.executor.deinit();
        self.table.deinit();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "execute simple SELECT *" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT * FROM t");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 2), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
    try std.testing.expectEqual(@as(i64, 4), values[3]);
    try std.testing.expectEqual(@as(i64, 5), values[4]);
}

test "execute SELECT with WHERE clause" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT * FROM t WHERE id > 2");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 5), values[2]);
}

test "execute SELECT with ORDER BY DESC" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT * FROM t ORDER BY id DESC");

    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 5), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
    try std.testing.expectEqual(@as(i64, 2), values[3]);
    try std.testing.expectEqual(@as(i64, 1), values[4]);
}

test "execute SELECT with LIMIT" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT * FROM t LIMIT 3");

    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 2), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
}

test "execute SELECT with OFFSET" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT * FROM t LIMIT 2 OFFSET 2");

    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
}

test "execute SELECT with float64 column" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, float64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT * FROM t WHERE value > 3.0");

    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.float64;
    try std.testing.expectEqual(@as(f64, 3.5), values[0]);
    try std.testing.expectEqual(@as(f64, 4.5), values[1]);
    try std.testing.expectEqual(@as(f64, 5.5), values[2]);
}

test "execute SELECT with mixed types" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, mixed_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT * FROM t");

    try std.testing.expectEqual(@as(usize, 3), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    try std.testing.expect(result.columns[0].data == .int64);
    try std.testing.expect(result.columns[1].data == .float64);
    try std.testing.expect(result.columns[2].data == .string);
}

test "execute SELECT * on mixed_types fixture" {
    const allocator = std.testing.allocator;

    // Read the mixed_types fixture (has 'id' int64, 'value' float64, 'name' string)
    const data = @embedFile("fixtures/mixed_types.lance/data/110101110101011000001010a45f35499b9af0396c9ed741b0.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Verify schema
    const schema = table.schema orelse return error.NoSchema;
    try std.testing.expectEqual(@as(usize, 3), schema.fields.len);

    // Parse SQL
    const sql = "SELECT * FROM t";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Verify results - should have 3 columns
    try std.testing.expectEqual(@as(usize, 3), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    // Check column types (id is int64, value is float64, name is string)
    try std.testing.expect(result.columns[0].data == .int64);
    try std.testing.expect(result.columns[1].data == .float64);
    try std.testing.expect(result.columns[2].data == .string);

    // Check int values
    const ints = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), ints[0]);
    try std.testing.expectEqual(@as(i64, 2), ints[1]);
    try std.testing.expectEqual(@as(i64, 3), ints[2]);

    // Check float values
    const floats = result.columns[1].data.float64;
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), floats[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), floats[1], 0.001);

    // Check string values
    const strings = result.columns[2].data.string;
    try std.testing.expectEqualStrings("alice", strings[0]);
    try std.testing.expectEqualStrings("bob", strings[1]);
}

// ============================================================================
// GROUP BY / Aggregate Tests
// ============================================================================

test "execute SELECT COUNT(*)" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT COUNT(*) FROM t");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    try std.testing.expect(result.columns[0].data == .int64);
    try std.testing.expectEqual(@as(i64, 5), result.columns[0].data.int64[0]);
}

test "execute SELECT SUM(id)" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT SUM(id) FROM t");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    try std.testing.expect(result.columns[0].data == .int64);
    try std.testing.expectEqual(@as(i64, 15), result.columns[0].data.int64[0]);
}

test "execute SELECT AVG(id)" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT AVG(id) FROM t");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    try std.testing.expect(result.columns[0].data == .float64);
    try std.testing.expectEqual(@as(f64, 3.0), result.columns[0].data.float64[0]);
}

test "execute SELECT MIN/MAX(id)" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // Test MIN
    var result = try ctx.exec("SELECT MIN(id) FROM t");
    try std.testing.expectEqual(@as(i64, 1), result.columns[0].data.int64[0]);

    // Test MAX
    result = try ctx.exec("SELECT MAX(id) FROM t");
    try std.testing.expectEqual(@as(i64, 5), result.columns[0].data.int64[0]);
}

test "execute SELECT STDDEV and VARIANCE" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // For values 1,2,3,4,5: Mean=3, PopVar=2.0, SampleVar=2.5

    // Test VARIANCE (sample)
    var result = try ctx.exec("SELECT VARIANCE(id) FROM t");
    try std.testing.expect(result.columns[0].data == .float64);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), result.columns[0].data.float64[0], 0.0001);

    // Test VAR_POP (population)
    result = try ctx.exec("SELECT VAR_POP(id) FROM t");
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result.columns[0].data.float64[0], 0.0001);

    // Test STDDEV (sample)
    result = try ctx.exec("SELECT STDDEV(id) FROM t");
    try std.testing.expectApproxEqAbs(@as(f64, 1.5811388300841898), result.columns[0].data.float64[0], 0.0001);

    // Test STDDEV_POP (population)
    result = try ctx.exec("SELECT STDDEV_POP(id) FROM t");
    try std.testing.expectApproxEqAbs(@as(f64, 1.4142135623730951), result.columns[0].data.float64[0], 0.0001);
}

test "execute SELECT MEDIAN and PERCENTILE" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // For sorted values [1,2,3,4,5]: MEDIAN=3.0

    // Test MEDIAN
    var result = try ctx.exec("SELECT MEDIAN(id) FROM t");
    try std.testing.expect(result.columns[0].data == .float64);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result.columns[0].data.float64[0], 0.0001);

    // Test PERCENTILE 0th (min)
    result = try ctx.exec("SELECT PERCENTILE(id, 0.0) FROM t");
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.columns[0].data.float64[0], 0.0001);

    // Test PERCENTILE 25th
    result = try ctx.exec("SELECT PERCENTILE(id, 0.25) FROM t");
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result.columns[0].data.float64[0], 0.0001);

    // Test PERCENTILE 75th
    result = try ctx.exec("SELECT PERCENTILE(id, 0.75) FROM t");
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result.columns[0].data.float64[0], 0.0001);

    // Test PERCENTILE 100th (max)
    result = try ctx.exec("SELECT PERCENTILE(id, 1.0) FROM t");
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result.columns[0].data.float64[0], 0.0001);
}

test "execute SELECT with GROUP BY" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, sqlite_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT name, COUNT(*) FROM t GROUP BY name");

    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);
    try std.testing.expect(result.columns[0].data == .string);
    try std.testing.expect(result.columns[1].data == .int64);

    // Each group should have exactly 1 row (all unique strings)
    for (result.columns[1].data.int64) |c| {
        try std.testing.expectEqual(@as(i64, 1), c);
    }
}

test "execute SELECT with GROUP BY and SUM" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, sqlite_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT name, SUM(id) FROM t GROUP BY name");

    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);
    try std.testing.expect(result.columns[0].data == .string);
    try std.testing.expect(result.columns[1].data == .int64);

    // Verify total sum of all SUM(id) values equals 1+2+3 = 6
    var total: i64 = 0;
    for (result.columns[1].data.int64) |s| {
        total += s;
    }
    try std.testing.expectEqual(@as(i64, 6), total);
}

// ============================================================================
// Expression Evaluation Tests (Phase 2)
// ============================================================================

test "execute SELECT with arithmetic multiplication" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT id * 2 AS doubled FROM t");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);
    try std.testing.expectEqualStrings("doubled", result.columns[0].name);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 2), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 6), values[2]);
    try std.testing.expectEqual(@as(i64, 8), values[3]);
    try std.testing.expectEqual(@as(i64, 10), values[4]);
}

test "execute SELECT with arithmetic addition" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT id + 10 AS plus_ten FROM t");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 11), values[0]);
    try std.testing.expectEqual(@as(i64, 12), values[1]);
    try std.testing.expectEqual(@as(i64, 13), values[2]);
    try std.testing.expectEqual(@as(i64, 14), values[3]);
    try std.testing.expectEqual(@as(i64, 15), values[4]);
}

test "execute SELECT with division" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT id / 2 AS halved FROM t");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.float64;
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), values[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), values[4], 0.001);
}

test "execute SELECT with complex expression" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT id * 2 + 1 AS computed FROM t");

    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 5), values[1]);
    try std.testing.expectEqual(@as(i64, 7), values[2]);
    try std.testing.expectEqual(@as(i64, 9), values[3]);
    try std.testing.expectEqual(@as(i64, 11), values[4]);
}

test "execute SELECT with mixed column and expression" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT id, id * 2 AS doubled FROM t");

    // Verify results - 2 columns
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    // First column is 'id'
    try std.testing.expectEqualStrings("id", result.columns[0].name);
    const ids = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), ids[0]);
    try std.testing.expectEqual(@as(i64, 5), ids[4]);

    // Second column is 'doubled'
    try std.testing.expectEqualStrings("doubled", result.columns[1].name);
    const doubled = result.columns[1].data.int64;
    try std.testing.expectEqual(@as(i64, 2), doubled[0]);
    try std.testing.expectEqual(@as(i64, 10), doubled[4]);
}

test "execute SELECT with UPPER function" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, sqlite_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT UPPER(name) AS upper_name FROM t LIMIT 3");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    try std.testing.expect(result.columns[0].data == .string);
    const values = result.columns[0].data.string;
    try std.testing.expectEqualStrings("ALICE", values[0]);
    try std.testing.expectEqualStrings("BOB", values[1]);
    try std.testing.expectEqualStrings("CHARLIE", values[2]);
}

test "execute SELECT with LENGTH function" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, sqlite_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT name, LENGTH(name) AS len FROM t LIMIT 3");

    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    // First column is 'name' (string)
    try std.testing.expect(result.columns[0].data == .string);

    // Second column is 'len' (int64 from LENGTH)
    try std.testing.expect(result.columns[1].data == .int64);
    const lengths = result.columns[1].data.int64;
    try std.testing.expectEqual(@as(i64, 5), lengths[0]); // "alice" = 5
    try std.testing.expectEqual(@as(i64, 3), lengths[1]); // "bob" = 3
    try std.testing.expectEqual(@as(i64, 7), lengths[2]); // "charlie" = 7
}

test "execute SELECT with ABS function" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // ABS(id - 3) gives: |1-3|=2, |2-3|=1, |3-3|=0, |4-3|=1, |5-3|=2
    const result = try ctx.exec("SELECT ABS(id - 3) AS abs_val FROM t");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    // ABS returns float64
    try std.testing.expect(result.columns[0].data == .float64);
    const values = result.columns[0].data.float64;
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), values[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), values[4], 0.001);
}

test "execute SELECT with string concatenation" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, sqlite_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT name || '_suffix' AS with_suffix FROM t LIMIT 3");

    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    try std.testing.expect(result.columns[0].data == .string);
    const values = result.columns[0].data.string;
    try std.testing.expectEqualStrings("alice_suffix", values[0]);
    try std.testing.expectEqualStrings("bob_suffix", values[1]);
    try std.testing.expectEqualStrings("charlie_suffix", values[2]);
}

// ============================================================================
// Parameter Binding Tests (Phase 3)
// ============================================================================

test "execute SELECT with single integer parameter" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/001110111111001100101010e79bea45fd969c9e1da619850b.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM t WHERE id = ?
    const sql = "SELECT * FROM t WHERE id = ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with parameter [3]
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{Value{ .integer = 3 }};
    var result = try executor.execute(&stmt.select, &params);
    defer result.deinit();

    // Verify results - should have 1 row with id=3
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
}

test "execute SELECT with multiple integer parameters" {
    const allocator = std.testing.allocator;

    // Open test Lance file (id column: 1, 2, 3, 4, 5)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/001110111111001100101010e79bea45fd969c9e1da619850b.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM t WHERE id > ? AND id < ?
    const sql = "SELECT * FROM t WHERE id > ? AND id < ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with parameters [2, 5] - should match id=3, id=4
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{ Value{ .integer = 2 }, Value{ .integer = 5 } };
    var result = try executor.execute(&stmt.select, &params);
    defer result.deinit();

    // Verify results - should have 2 rows with id=3 and id=4
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
}

test "execute SELECT with float parameter" {
    const allocator = std.testing.allocator;

    // Open test Lance file with float64 (value: 1.5, 2.5, 3.5, 4.5, 5.5)
    const lance_data = @embedFile("fixtures/simple_float64.lance/data/111100011001011101011011e9643e40abbeac7897d4a95cc3.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL: SELECT * FROM t WHERE value > ?
    const sql = "SELECT * FROM t WHERE value > ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with parameter [3.0] - should match 3.5, 4.5, 5.5
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{Value{ .float = 3.0 }};
    var result = try executor.execute(&stmt.select, &params);
    defer result.deinit();

    // Verify results - should have 3 rows
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.float64;
    try std.testing.expectEqual(@as(f64, 3.5), values[0]);
    try std.testing.expectEqual(@as(f64, 4.5), values[1]);
    try std.testing.expectEqual(@as(f64, 5.5), values[2]);
}

test "execute SELECT with string parameter" {
    const allocator = std.testing.allocator;

    // Use mixed_types fixture: id=[1,2,3], value=[1.5,2.5,3.5], name=["alice","bob","charlie"]
    const data = @embedFile("fixtures/mixed_types.lance/data/110101110101011000001010a45f35499b9af0396c9ed741b0.lance");

    var table = try Table.init(allocator, data);
    defer table.deinit();

    // Parse SQL: SELECT id FROM t WHERE name = ?
    const sql = "SELECT id FROM t WHERE name = ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with parameter ['alice'] - should match the first row where name='alice'
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{Value{ .string = "alice" }};
    var result = try executor.execute(&stmt.select, &params);
    defer result.deinit();

    // Verify results - should have 1 row where name='alice' (first row, id=1)
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
}

test "parameter out of bounds returns error" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/001110111111001100101010e79bea45fd969c9e1da619850b.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    // Parse SQL with 2 parameters: SELECT * FROM t WHERE id > ? AND id < ?
    const sql = "SELECT * FROM t WHERE id > ? AND id < ?";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute with only 1 parameter - should fail
    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    const params = [_]Value{Value{ .integer = 2 }}; // Only 1 param, need 2
    const result = executor.execute(&stmt.select, &params);

    // Should return ParameterOutOfBounds error
    try std.testing.expectError(error.ParameterOutOfBounds, result);
}

// ============================================================================
// DISTINCT Tests (Phase 4)
// ============================================================================

test "execute SELECT DISTINCT on unique values" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT DISTINCT id FROM t");

    // All values are unique, so DISTINCT should return all 5 rows
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 2), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
    try std.testing.expectEqual(@as(i64, 4), values[3]);
    try std.testing.expectEqual(@as(i64, 5), values[4]);
}

test "execute SELECT DISTINCT with WHERE clause" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT DISTINCT id FROM t WHERE id > 2");

    // Should return 3 unique rows (3, 4, 5)
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 3), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 5), values[2]);
}

test "execute SELECT DISTINCT with ORDER BY" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT DISTINCT id FROM t ORDER BY id DESC");

    // Should return all 5 rows in descending order
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 5), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 5), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
    try std.testing.expectEqual(@as(i64, 2), values[3]);
    try std.testing.expectEqual(@as(i64, 1), values[4]);
}

test "execute SELECT DISTINCT with LIMIT" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT DISTINCT id FROM t LIMIT 3");

    // Should return only 3 rows
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);
}

test "execute SELECT DISTINCT on strings" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, sqlite_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT DISTINCT name FROM t");

    // Verify results - should have unique string values
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expect(result.columns[0].data == .string);

    // The mixed_types fixture has 3 unique names: alice, bob, charlie
    try std.testing.expectEqual(@as(usize, 3), result.row_count);
}

// =============================================================================
// @logic_table Integration Tests
// =============================================================================

test "executor registerLogicTableAlias" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/001110111111001100101010e79bea45fd969c9e1da619850b.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    // Register alias
    try executor.registerLogicTableAlias("t", "FraudDetector");

    // Check it's stored
    const class_name = executor.logic_table_aliases.get("t");
    try std.testing.expect(class_name != null);
    try std.testing.expectEqualStrings("FraudDetector", class_name.?);
}

test "executor registerLogicTableAlias rejects duplicates" {
    const allocator = std.testing.allocator;

    // Open test Lance file
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/001110111111001100101010e79bea45fd969c9e1da619850b.lance");
    var table = try Table.init(allocator, lance_data);
    defer table.deinit();

    var executor = Executor.init(&table, allocator);
    defer executor.deinit();

    // Register alias once
    try executor.registerLogicTableAlias("t", "FraudDetector");

    // Attempt to register again - should fail
    const result = executor.registerLogicTableAlias("t", "OtherClass");
    try std.testing.expectError(error.DuplicateAlias, result);
}

test "LogicTableExecutor Python parsing" {
    const allocator = std.testing.allocator;

    const LogicTableExecutor = @import("lanceql.logic_table").LogicTableExecutor;

    // Test in-memory parsing (without actual file)
    var executor = try LogicTableExecutor.init(allocator, "test_fraud.py");
    defer executor.deinit();

    // Manually parse test content
    const content =
        \\from lanceql import logic_table, Table
        \\
        \\@logic_table
        \\class FraudDetector:
        \\    orders = Table("orders.lance")
        \\    customers = Table("customers.lance")
        \\
        \\    def risk_score(self):
        \\        return self.orders.amount * 0.01
        \\
        \\    def velocity(self, days=30):
        \\        return 0.5
    ;

    // Extract class name
    const class_name = try executor.extractClassName(content);
    defer allocator.free(class_name);
    try std.testing.expectEqualStrings("FraudDetector", class_name);

    // Extract table declarations
    try executor.extractTableDecls(content);
    try std.testing.expectEqual(@as(usize, 2), executor.table_decls.items.len);
    try std.testing.expectEqualStrings("orders", executor.table_decls.items[0].name);
    try std.testing.expectEqualStrings("orders.lance", executor.table_decls.items[0].path);
    try std.testing.expectEqualStrings("customers", executor.table_decls.items[1].name);

    // Extract methods
    try executor.extractMethods(content);
    try std.testing.expectEqual(@as(usize, 2), executor.methods.items.len);
    try std.testing.expectEqualStrings("risk_score", executor.methods.items[0].name);
    try std.testing.expectEqualStrings("velocity", executor.methods.items[1].name);
}

test "LogicTableExecutor batch dispatch types" {
    const allocator = std.testing.allocator;

    const dispatch = @import("lanceql.sql.executor").logic_table_dispatch;
    const ColumnBinding = dispatch.ColumnBinding;
    const ColumnBuffer = dispatch.ColumnBuffer;

    // Test ColumnBinding
    const values = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const binding = ColumnBinding{
        .name = "amount",
        .table_alias = "orders",
        .data = .{ .f64 = &values },
    };
    try std.testing.expectEqual(@as(usize, 5), binding.len());

    // Test ColumnBuffer
    var buffer = try ColumnBuffer.initFloat64(allocator, 10);
    defer buffer.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 10), buffer.len());

    // Write to buffer
    const out = buffer.f64.?;
    out[0] = 42.0;
    try std.testing.expectEqual(@as(f64, 42.0), out[0]);
}

test "Dispatcher batch method registration and dispatch" {
    const allocator = std.testing.allocator;

    const dispatch = @import("lanceql.sql.executor").logic_table_dispatch;
    const Dispatcher = dispatch.Dispatcher;
    const ColumnBinding = dispatch.ColumnBinding;
    const ColumnBuffer = dispatch.ColumnBuffer;

    var dispatcher = Dispatcher.init(allocator);
    defer dispatcher.deinit();

    // Create a test batch function
    const TestBatchFn = struct {
        fn computeRiskScore(
            inputs: [*]const ColumnBinding,
            num_inputs: usize,
            selection: ?[*]const u32,
            selection_len: usize,
            output: *ColumnBuffer,
            ctx: ?*dispatch.logic_table.QueryContext,
        ) callconv(.c) void {
            _ = ctx;
            if (num_inputs == 0) return;

            // Get amount column (assume it's f64)
            const amounts = switch (inputs[0].data) {
                .f64 => |d| d,
                else => return,
            };

            const out_buf = output.f64 orelse return;

            // Compute risk score: amount * 0.01
            if (selection) |sel| {
                for (0..selection_len) |i| {
                    const idx = sel[i];
                    out_buf[i] = amounts[idx] * 0.01;
                }
            } else {
                for (amounts, 0..) |amt, i| {
                    out_buf[i] = amt * 0.01;
                }
            }
        }
    };

    // Register the batch method
    const input_cols = [_][]const u8{"amount"};
    try dispatcher.registerBatchMethod(
        "FraudDetector",
        "risk_score",
        &TestBatchFn.computeRiskScore,
        &input_cols,
        .f64,
    );

    // Verify it's registered as batch
    try std.testing.expect(dispatcher.isBatchMethod("FraudDetector", "risk_score"));

    // Create test input
    const amounts = [_]f64{ 100.0, 200.0, 300.0, 400.0, 500.0 };
    const inputs = [_]ColumnBinding{
        .{
            .name = "amount",
            .table_alias = "orders",
            .data = .{ .f64 = &amounts },
        },
    };

    // Test dispatch without selection
    {
        var output = try ColumnBuffer.initFloat64(allocator, 5);
        defer output.deinit(allocator);

        try dispatcher.callMethodBatch("FraudDetector", "risk_score", &inputs, null, &output, null);

        const out_buf = output.f64.?;
        try std.testing.expectEqual(@as(f64, 1.0), out_buf[0]); // 100 * 0.01
        try std.testing.expectEqual(@as(f64, 2.0), out_buf[1]); // 200 * 0.01
        try std.testing.expectEqual(@as(f64, 5.0), out_buf[4]); // 500 * 0.01
    }

    // Test dispatch with selection (only rows 1, 3)
    {
        const selection = [_]u32{ 1, 3 };
        var output = try ColumnBuffer.initFloat64(allocator, 2);
        defer output.deinit(allocator);

        try dispatcher.callMethodBatch("FraudDetector", "risk_score", &inputs, &selection, &output, null);

        const out_buf = output.f64.?;
        try std.testing.expectEqual(@as(f64, 2.0), out_buf[0]); // 200 * 0.01
        try std.testing.expectEqual(@as(f64, 4.0), out_buf[1]); // 400 * 0.01
    }
}

// ============================================================================
// JOIN Tests
// ============================================================================

test "parse INNER JOIN syntax" {
    const allocator = std.testing.allocator;

    // Test parsing of INNER JOIN (just verify it parses without error)
    const sql = "SELECT a.id, b.id FROM orders AS a INNER JOIN items AS b ON a.id = b.order_id";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Verify the FROM clause is a JOIN
    try std.testing.expect(stmt.select.from == .join);
    const join = stmt.select.from.join;

    // Verify left side is a simple table
    try std.testing.expect(join.left.* == .simple);

    // Verify join clause
    try std.testing.expectEqual(ast.JoinType.inner, join.join_clause.join_type);
    try std.testing.expect(join.join_clause.on_condition != null);
}

test "parse LEFT JOIN syntax" {
    const allocator = std.testing.allocator;

    const sql = "SELECT * FROM orders LEFT JOIN items ON orders.id = items.order_id";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    try std.testing.expect(stmt.select.from == .join);
    const join = stmt.select.from.join;
    try std.testing.expectEqual(ast.JoinType.left, join.join_clause.join_type);
}

test "parse RIGHT JOIN syntax" {
    const allocator = std.testing.allocator;

    const sql = "SELECT * FROM orders RIGHT JOIN items ON orders.id = items.order_id";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    try std.testing.expect(stmt.select.from == .join);
    const join = stmt.select.from.join;
    try std.testing.expectEqual(ast.JoinType.right, join.join_clause.join_type);
}

test "parse FULL OUTER JOIN syntax" {
    const allocator = std.testing.allocator;

    const sql = "SELECT * FROM orders FULL OUTER JOIN items ON orders.id = items.order_id";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    try std.testing.expect(stmt.select.from == .join);
    const join = stmt.select.from.join;
    try std.testing.expectEqual(ast.JoinType.full, join.join_clause.join_type);
}

test "execute INNER JOIN (self-join)" {
    const allocator = std.testing.allocator;

    // Use the same Lance file twice (self-join)
    const lance_data = @embedFile("fixtures/simple_int64.lance/data/001110111111001100101010e79bea45fd969c9e1da619850b.lance");
    var table1 = try Table.init(allocator, lance_data);
    defer table1.deinit();
    var table2 = try Table.init(allocator, lance_data);
    defer table2.deinit();

    // Create executor and register both tables
    var executor = Executor.init(null, allocator);
    defer executor.deinit();

    try executor.registerTable("a", &table1);
    try executor.registerTable("b", &table2);

    // For self-join, we need both tables to have the same data
    // Parse SQL: SELECT a.id, b.id FROM a INNER JOIN b ON a.id = b.id
    const sql = "SELECT a.id, b.id FROM a INNER JOIN b ON a.id = b.id";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Execute - this tests the hash join algorithm
    var result = try executor.execute(&stmt.select, &[_]Value{});
    defer result.deinit();

    // Self-join on same data with equality should produce same row count as original
    // Each row matches exactly one row in the other table
    std.debug.print("\nINNER JOIN (self-join) result:\n", .{});
    std.debug.print("  columns: {d}\n", .{result.columns.len});
    std.debug.print("  row_count: {d}\n", .{result.row_count});

    // Should have 5 rows (each row matches itself)
    try std.testing.expectEqual(@as(usize, 5), result.row_count);
}

// ============================================================================
// Window Function Tests
// Note: Window function parsing is tested in src/sql/parser.zig
// Window function execution is implemented in src/sql/executor.zig
// ============================================================================

// ============================================================================
// Set Operation Tests (UNION, INTERSECT, EXCEPT)
// ============================================================================

test "parse UNION" {
    const allocator = std.testing.allocator;

    // Parse UNION SQL
    const sql = "SELECT id FROM t1 UNION SELECT id FROM t2";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Verify set operation was parsed
    try std.testing.expect(stmt.select.set_operation != null);
    try std.testing.expectEqual(ast.SetOperationType.union_distinct, stmt.select.set_operation.?.op_type);
}

test "parse UNION ALL" {
    const allocator = std.testing.allocator;

    // Parse UNION ALL SQL
    const sql = "SELECT id FROM t1 UNION ALL SELECT id FROM t2";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Verify set operation was parsed
    try std.testing.expect(stmt.select.set_operation != null);
    try std.testing.expectEqual(ast.SetOperationType.union_all, stmt.select.set_operation.?.op_type);
}

test "parse INTERSECT" {
    const allocator = std.testing.allocator;

    // Parse INTERSECT SQL
    const sql = "SELECT id FROM t1 INTERSECT SELECT id FROM t2";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Verify set operation was parsed
    try std.testing.expect(stmt.select.set_operation != null);
    try std.testing.expectEqual(ast.SetOperationType.intersect, stmt.select.set_operation.?.op_type);
}

test "parse EXCEPT" {
    const allocator = std.testing.allocator;

    // Parse EXCEPT SQL
    const sql = "SELECT id FROM t1 EXCEPT SELECT id FROM t2";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    // Verify set operation was parsed
    try std.testing.expect(stmt.select.set_operation != null);
    try std.testing.expectEqual(ast.SetOperationType.except, stmt.select.set_operation.?.op_type);
}

test "execute UNION ALL" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // Left: [1, 2], Right: [4, 5], Union All: [1, 2, 4, 5]
    const result = try ctx.exec("SELECT id FROM t WHERE id <= 2 UNION ALL SELECT id FROM t WHERE id >= 4");

    try std.testing.expectEqual(@as(usize, 4), result.row_count);
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 2), values[1]);
    try std.testing.expectEqual(@as(i64, 4), values[2]);
    try std.testing.expectEqual(@as(i64, 5), values[3]);
}

test "execute UNION (distinct)" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // Left: [1, 2, 3], Right: [2, 3, 4, 5], Union (distinct): [1, 2, 3, 4, 5]
    const result = try ctx.exec("SELECT id FROM t WHERE id <= 3 UNION SELECT id FROM t WHERE id >= 2");

    try std.testing.expectEqual(@as(usize, 5), result.row_count);
}

test "execute INTERSECT" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // Left: [1, 2, 3], Right: [2, 3, 4, 5], Intersect: [2, 3]
    const result = try ctx.exec("SELECT id FROM t WHERE id <= 3 INTERSECT SELECT id FROM t WHERE id >= 2");

    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 2), values[0]);
    try std.testing.expectEqual(@as(i64, 3), values[1]);
}

test "execute EXCEPT" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // Left: [1, 2, 3, 4, 5], Right: [3, 4, 5], Except: [1, 2]
    const result = try ctx.exec("SELECT id FROM t EXCEPT SELECT id FROM t WHERE id >= 3");

    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 2), values[1]);
}

// ============================================================================
// Subquery Tests (EXISTS, IN)
// ============================================================================

test "parse IN list" {
    const allocator = std.testing.allocator;

    const sql = "SELECT id FROM t WHERE id IN (1, 2, 3)";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    try std.testing.expect(stmt.select.where != null);
    try std.testing.expect(stmt.select.where.? == .in_list);
}

test "parse NOT IN list" {
    const allocator = std.testing.allocator;

    const sql = "SELECT id FROM t WHERE id NOT IN (4, 5)";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    try std.testing.expect(stmt.select.where != null);
    try std.testing.expect(stmt.select.where.? == .in_list);
    try std.testing.expect(stmt.select.where.?.in_list.negated);
}

test "parse IN subquery" {
    const allocator = std.testing.allocator;

    const sql = "SELECT id FROM t1 WHERE id IN (SELECT id FROM t2)";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    try std.testing.expect(stmt.select.where != null);
    try std.testing.expect(stmt.select.where.? == .in_subquery);
}

test "parse EXISTS" {
    const allocator = std.testing.allocator;

    const sql = "SELECT id FROM t1 WHERE EXISTS (SELECT 1 FROM t2)";
    var stmt = try parser.parseSQL(sql, allocator);
    defer ast.deinitSelectStmt(&stmt.select, allocator);

    try std.testing.expect(stmt.select.where != null);
    try std.testing.expect(stmt.select.where.? == .exists);
    try std.testing.expect(!stmt.select.where.?.exists.negated);
}

test "execute IN list" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT id FROM t WHERE id IN (2, 4)");

    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 2), values[0]);
    try std.testing.expectEqual(@as(i64, 4), values[1]);
}

test "execute NOT IN list" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT id FROM t WHERE id NOT IN (2, 4)");

    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 3), values[1]);
    try std.testing.expectEqual(@as(i64, 5), values[2]);
}

test "execute EXISTS subquery" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // EXISTS subquery returns true (there are rows with id > 3), so all rows are returned
    const result = try ctx.exec("SELECT id FROM t WHERE EXISTS (SELECT id FROM t WHERE id > 3)");

    try std.testing.expectEqual(@as(usize, 5), result.row_count);
}

test "execute NOT EXISTS subquery" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // NOT EXISTS subquery returns true (no rows with id > 10), so all rows are returned
    const result = try ctx.exec("SELECT id FROM t WHERE NOT EXISTS (SELECT id FROM t WHERE id > 10)");

    try std.testing.expectEqual(@as(usize, 5), result.row_count);
}

test "execute IN subquery" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // Subquery returns [4, 5], so we get rows where id IN (4, 5)
    const result = try ctx.exec("SELECT id FROM t WHERE id IN (SELECT id FROM t WHERE id > 3)");

    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    const values = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 4), values[0]);
    try std.testing.expectEqual(@as(i64, 5), values[1]);
}

// ============================================================================
// Date/Time Function Tests
// ============================================================================

// Note: Date/time functions use column values (id) as epoch timestamps
// id values [1, 2, 3, 4, 5] represent epoch seconds from 1970
// For more meaningful date tests, we use computed expressions

test "execute YEAR function on column" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // YEAR(id) where id = [1,2,3,4,5] (all 1970 since they're tiny epoch values)
    const result = try ctx.exec("SELECT YEAR(id) FROM t LIMIT 1");

    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    // id=1 is epoch second 1, which is Jan 1, 1970
    try std.testing.expectEqual(@as(i64, 1970), result.columns[0].data.int64[0]);
}

test "execute MONTH and DAY functions on column" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    const result = try ctx.exec("SELECT MONTH(id), DAY(id) FROM t LIMIT 1");

    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(i64, 1), result.columns[0].data.int64[0]); // Month 1
    try std.testing.expectEqual(@as(i64, 1), result.columns[1].data.int64[0]); // Day 1
}

test "execute HOUR/MINUTE/SECOND functions on column" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // id=1 is epoch second 1 = 00:00:01 on Jan 1, 1970
    const result = try ctx.exec("SELECT HOUR(id), MINUTE(id), SECOND(id) FROM t LIMIT 1");

    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    try std.testing.expectEqual(@as(usize, 3), result.columns.len);
    try std.testing.expectEqual(@as(i64, 0), result.columns[0].data.int64[0]); // Hour 0
    try std.testing.expectEqual(@as(i64, 0), result.columns[1].data.int64[0]); // Minute 0
    try std.testing.expectEqual(@as(i64, 1), result.columns[2].data.int64[0]); // Second 1
}

test "execute DAYOFWEEK function on column" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // id=1 is epoch second 1 = Jan 1, 1970 = Thursday (day 4)
    const result = try ctx.exec("SELECT DAYOFWEEK(id) FROM t LIMIT 1");

    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    // Jan 1, 1970 was a Thursday (day 4, 0-indexed from Sunday)
    try std.testing.expectEqual(@as(i64, 4), result.columns[0].data.int64[0]);
}

test "execute QUARTER function on column" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // id=1 is Jan 1970 = Q1
    const result = try ctx.exec("SELECT QUARTER(id) FROM t LIMIT 1");

    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    try std.testing.expectEqual(@as(i64, 1), result.columns[0].data.int64[0]); // Q1
}

test "execute DATE_TRUNC on column" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // DATE_TRUNC('day', id) should truncate to 0 (start of Jan 1, 1970)
    const result = try ctx.exec("SELECT DATE_TRUNC('day', id) FROM t LIMIT 1");

    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    try std.testing.expectEqual(@as(i64, 0), result.columns[0].data.int64[0]); // Truncated to day start
}

test "execute DATE_ADD on column" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // DATE_ADD(id, 1, 'day') adds 86400 seconds
    const result = try ctx.exec("SELECT DATE_ADD(id, 1, 'day') FROM t LIMIT 1");

    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    // id=1 + 86400 = 86401
    try std.testing.expectEqual(@as(i64, 86401), result.columns[0].data.int64[0]);
}

test "execute EPOCH function on column" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // EPOCH(id) should return the same value
    const result = try ctx.exec("SELECT EPOCH(id) FROM t LIMIT 1");

    try std.testing.expectEqual(@as(usize, 1), result.row_count);
    try std.testing.expectEqual(@as(i64, 1), result.columns[0].data.int64[0]);
}

// ============================================================================
// Compiled Execution Tests
// ============================================================================

test "compiled vs interpreted produce same results" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    // Force compilation with low threshold
    ctx.executor.setCompileThreshold(0);
    const r1 = try ctx.exec("SELECT id FROM t WHERE id > 2");
    const compiled_count = r1.row_count;

    // Disable compilation, run same query
    ctx.executor.enableCompiledExecution(false);
    const r2 = try ctx.exec("SELECT id FROM t WHERE id > 2");

    try std.testing.expectEqual(compiled_count, r2.row_count);
}

test "fallback to interpreted on GROUP BY" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    ctx.executor.setCompileThreshold(0);
    // GROUP BY not compiled - should fallback gracefully
    const result = try ctx.exec("SELECT COUNT(*) FROM t GROUP BY id");
    try std.testing.expect(result.row_count > 0);
}

// Removed: "compilation can be disabled" test - interpreted fallback was removed,
// all queries now use compiled execution

test "compiled filter with AND" {
    var ctx: TestContext = undefined;
    try ctx.init(std.testing.allocator, int64_fixture);
    defer ctx.deinit();

    ctx.executor.setCompileThreshold(0);
    const result = try ctx.exec("SELECT id FROM t WHERE id > 1 AND id < 5");
    try std.testing.expectEqual(@as(usize, 3), result.row_count);
}

// ============================================================================
// ORDER BY with NULLs Tests
// ============================================================================
// JIT NULL Handling Status:
// - [x] Code generation: genSortContext generates NULL-safe comparisons
// - [x] Memory layout: RuntimeColumns supports validity bitmaps
// - [x] Executor: Allocates and passes validity bitmaps to JIT
// - [x] Actual NULL loading: Reads validity bitmaps from Lance file pages
//
// The JIT codegen properly generates NULL-handling code with:
// - isValid() helper to check validity bitmaps
// - NULLS FIRST/LAST ordering in sort comparisons
// - Proper if-else-if chains for NULL cases
//
// Lance file format:
// - Nullable columns have 2 buffers: [validity_bitmap, data]
// - Non-nullable columns have 1 buffer: [data]
// - Validity bitmap uses Arrow format: bit 1 = valid, bit 0 = null
//
// SQL standard NULL ordering behavior:
// - NULLS FIRST: NULLs appear before non-NULL values
// - NULLS LAST: NULLs appear after non-NULL values
// - Default ASC: implementation-defined (we use NULLS LAST)
// - Default DESC: implementation-defined (we use NULLS FIRST)
