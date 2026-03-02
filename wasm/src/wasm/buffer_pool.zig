//! LRU Buffer Pool for WASM
//!
//! Manages memory usage by evicting cold data when a byte limit is reached.
//! This is a Zig-native implementation to cache column data efficiently.

const std = @import("std");
const memory = @import("memory.zig");

// ============================================================================
// Constants
// ============================================================================

const MAX_ENTRIES = 1024;
/// Default 64MB - safe for Cloudflare Workers (128MB limit).
/// For AWS Lambda or browsers with more memory, call bufferPoolInit() with larger value.
const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024; // 64MB (Worker-safe)

// ============================================================================
// Types
// ============================================================================

pub const CacheEntry = struct {
    key: [128]u8, // Fixed-size key buffer
    key_len: usize,
    data: ?[]u8, // Pointer to cached data
    size: usize,
    prev: ?*CacheEntry,
    next: ?*CacheEntry,
    in_use: bool,
};

// ============================================================================
// Global State
// ============================================================================

var entries: [MAX_ENTRIES]CacheEntry = undefined;
var entry_count: usize = 0;
var head: ?*CacheEntry = null; // Most recently used
var tail: ?*CacheEntry = null; // Least recently used
var current_bytes: usize = 0;
var max_bytes: usize = DEFAULT_MAX_BYTES;
var initialized: bool = false;

// ============================================================================
// Initialization
// ============================================================================

fn initPool() void {
    if (initialized) return;
    
    for (&entries) |*e| {
        e.* = CacheEntry{
            .key = undefined,
            .key_len = 0,
            .data = null,
            .size = 0,
            .prev = null,
            .next = null,
            .in_use = false,
        };
    }
    entry_count = 0;
    head = null;
    tail = null;
    current_bytes = 0;
    initialized = true;
}

// ============================================================================
// Helper Functions
// ============================================================================

fn findEntry(key: []const u8) ?*CacheEntry {
    for (&entries) |*e| {
        if (e.in_use and e.key_len == key.len and std.mem.eql(u8, e.key[0..e.key_len], key)) {
            return e;
        }
    }
    return null;
}

fn allocEntry() ?*CacheEntry {
    for (&entries) |*e| {
        if (!e.in_use) {
            return e;
        }
    }
    return null;
}

fn moveToHead(entry: *CacheEntry) void {
    if (entry == head) return;
    
    removeNode(entry);
    addToHead(entry);
}

fn addToHead(entry: *CacheEntry) void {
    entry.next = head;
    entry.prev = null;
    
    if (head) |h| {
        h.prev = entry;
    }
    head = entry;
    
    if (tail == null) {
        tail = entry;
    }
}

fn removeNode(entry: *CacheEntry) void {
    if (entry.prev) |p| {
        p.next = entry.next;
    } else {
        head = entry.next;
    }
    
    if (entry.next) |n| {
        n.prev = entry.prev;
    } else {
        tail = entry.prev;
    }
    
    entry.prev = null;
    entry.next = null;
}

fn evictIfNeeded() void {
    while (current_bytes > max_bytes and tail != null) {
        const node = tail.?;
        removeNode(node);
        current_bytes -= node.size;
        node.in_use = false;
        node.data = null;
        entry_count -= 1;
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Initialize or reset the buffer pool
pub export fn bufferPoolInit(max_size: usize) void {
    initPool();
    max_bytes = if (max_size > 0) max_size else DEFAULT_MAX_BYTES;
}

/// Get cached data by key
/// Returns pointer and size via out parameters, returns 1 if found, 0 otherwise
pub export fn bufferPoolGet(
    key_ptr: [*]const u8,
    key_len: usize,
    out_data_ptr: *?[*]const u8,
    out_size: *usize,
) u32 {
    if (!initialized) initPool();
    
    const key = key_ptr[0..key_len];
    const entry = findEntry(key) orelse return 0;
    
    moveToHead(entry);
    
    if (entry.data) |data| {
        out_data_ptr.* = data.ptr;
        out_size.* = entry.size;
        return 1;
    }
    return 0;
}

/// Store data in the cache
/// Returns 1 on success, 0 on failure
pub export fn bufferPoolSet(
    key_ptr: [*]const u8,
    key_len: usize,
    data_ptr: [*]const u8,
    data_len: usize,
) u32 {
    if (!initialized) initPool();
    if (key_len > 128) return 0;
    
    const key = key_ptr[0..key_len];
    
    // Check if entry already exists
    if (findEntry(key)) |existing| {
        // Update existing entry
        current_bytes -= existing.size;
        current_bytes += data_len;
        
        // Allocate new buffer and copy data
        const buf = memory.wasmAlloc(data_len) orelse return 0;
        @memcpy(buf[0..data_len], data_ptr[0..data_len]);
        existing.data = buf[0..data_len];
        existing.size = data_len;
        
        moveToHead(existing);
        evictIfNeeded();
        return 1;
    }
    
    // Allocate new entry
    const entry = allocEntry() orelse {
        // Pool is full, evict tail and try again
        if (tail) |t| {
            removeNode(t);
            current_bytes -= t.size;
            t.in_use = false;
            t.data = null;
            entry_count -= 1;
        }
        return bufferPoolSet(key_ptr, key_len, data_ptr, data_len);
    };
    
    // Allocate buffer and copy data
    const buf = memory.wasmAlloc(data_len) orelse return 0;
    @memcpy(buf[0..data_len], data_ptr[0..data_len]);
    
    // Initialize entry
    @memcpy(entry.key[0..key_len], key);
    entry.key_len = key_len;
    entry.data = buf[0..data_len];
    entry.size = data_len;
    entry.in_use = true;
    
    addToHead(entry);
    entry_count += 1;
    current_bytes += data_len;
    
    evictIfNeeded();
    return 1;
}

/// Check if a key exists in the cache
pub export fn bufferPoolHas(key_ptr: [*]const u8, key_len: usize) u32 {
    if (!initialized) initPool();
    const key = key_ptr[0..key_len];
    return if (findEntry(key) != null) 1 else 0;
}

/// Remove an entry from the cache
pub export fn bufferPoolDelete(key_ptr: [*]const u8, key_len: usize) u32 {
    if (!initialized) initPool();
    const key = key_ptr[0..key_len];
    
    const entry = findEntry(key) orelse return 0;
    
    removeNode(entry);
    current_bytes -= entry.size;
    entry.in_use = false;
    entry.data = null;
    entry_count -= 1;
    
    return 1;
}

/// Clear the entire cache
pub export fn bufferPoolClear() void {
    if (!initialized) return;
    
    for (&entries) |*e| {
        e.in_use = false;
        e.data = null;
        e.prev = null;
        e.next = null;
    }
    
    head = null;
    tail = null;
    current_bytes = 0;
    entry_count = 0;
}

/// Get current cache statistics
pub export fn bufferPoolGetStats(out_count: *usize, out_bytes: *usize, out_max: *usize) void {
    if (!initialized) initPool();
    out_count.* = entry_count;
    out_bytes.* = current_bytes;
    out_max.* = max_bytes;
}
