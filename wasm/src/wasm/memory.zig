const std = @import("std");

// Purpose-built allocator for WASM linear memory (used by vectorjson + termweb)
pub const wasm_allocator: std.mem.Allocator = .{
    .ptr = undefined,
    .vtable = &std.heap.WasmAllocator.vtable,
};

pub fn wasmAlloc(len: usize) ?[*]u8 {
    if (len == 0 or len > 256 * 1024 * 1024) return null;
    const slice = wasm_allocator.alloc(u8, len) catch return null;
    return slice.ptr;
}

pub fn wasmFree(ptr: [*]u8, len: usize) void {
    wasm_allocator.free(ptr[0..len]);
}

pub fn wasmReset() void {}

pub fn getHeapUsage() usize {
    return 0;
}

pub fn getHeapCapacity() usize {
    return 64 * 1024 * 1024;
}

// ============================================================================
// WASM Exports
// ============================================================================

pub export fn alloc(len: usize) ?[*]u8 {
    return wasmAlloc(len);
}

pub export fn free(ptr: [*]u8, len: usize) void {
    wasmFree(ptr, len);
}

// ============================================================================
// Tests
// ============================================================================

test "memory: basic allocation" {
    const ptr1 = wasmAlloc(100);
    try std.testing.expect(ptr1 != null);

    const ptr2 = wasmAlloc(200);
    try std.testing.expect(ptr2 != null);

    const addr1 = @intFromPtr(ptr1.?);
    const addr2 = @intFromPtr(ptr2.?);
    try std.testing.expect(addr2 > addr1);
}

test "memory: alignment" {
    _ = wasmAlloc(1);
    const ptr = wasmAlloc(8);
    try std.testing.expect(ptr != null);

    const addr = @intFromPtr(ptr.?);
    try std.testing.expectEqual(@as(usize, 0), addr % 8);
}

test "memory: heap capacity" {
    try std.testing.expectEqual(@as(usize, 64 * 1024 * 1024), getHeapCapacity());
}

test "memory: reject zero and oversized" {
    try std.testing.expect(wasmAlloc(0) == null);
    try std.testing.expect(wasmAlloc(257 * 1024 * 1024) == null);
}
