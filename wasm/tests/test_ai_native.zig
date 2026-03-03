//! Integration test for native AI module
//!
//! Tests TinyBERT cross-encoder with actual GGUF model file.

const std = @import("std");
const ai = @import("querymode.ai");

test "tinybert: load model and cross-encode" {
    const allocator = std.testing.allocator;

    var tinybert = try ai.TinyBERT.init(allocator);
    defer tinybert.deinit();

    // Load model
    tinybert.loadFromFile("models/ms-marco-tinybert-l2.gguf") catch |err| {
        std.debug.print("Model not found (expected in CI): {}\n", .{err});
        return; // Skip if model not present
    };

    // Test cross-encoding
    const query = "how to fix error 500";
    const doc = "Error 500 occurs when the server encounters an unexpected condition";

    const score = try tinybert.crossEncode(query, doc);

    std.debug.print("\nQuery: {s}\n", .{query});
    std.debug.print("Doc: {s}\n", .{doc});
    std.debug.print("Score: {d:.4}\n", .{score});

    // Score should be between 0 and 1
    try std.testing.expect(score >= 0.0);
    try std.testing.expect(score <= 1.0);

    // This query-doc pair should have a reasonably high relevance score
    try std.testing.expect(score > 0.3);
}

test "tinybert: multiple document scoring" {
    const allocator = std.testing.allocator;

    var tinybert = try ai.TinyBERT.init(allocator);
    defer tinybert.deinit();

    tinybert.loadFromFile("models/ms-marco-tinybert-l2.gguf") catch |err| {
        std.debug.print("Model not found (expected in CI): {}\n", .{err});
        return;
    };

    const query = "what is machine learning";

    // Score multiple documents
    const docs = [_][]const u8{
        "Machine learning is a branch of artificial intelligence",
        "Python is a programming language",
        "Deep learning uses neural networks for AI tasks",
    };

    std.debug.print("\nQuery: {s}\n", .{query});
    for (docs) |doc| {
        const score = try tinybert.crossEncode(query, doc);
        std.debug.print("  {d:.4} - {s}\n", .{ score, doc });

        // All scores should be valid (0-1)
        try std.testing.expect(score >= 0.0);
        try std.testing.expect(score <= 1.0);
    }
}
