//! I/O abstraction layer for QueryMode.
//!
//! This module provides a VFS (Virtual File System) abstraction that allows
//! the same Lance parsing code to work with:
//! - Native file system (FileReader)
//! - Memory-mapped files (MmapReader) - zero-copy, cached
//! - In-memory buffers (MemoryReader)
//! - HTTP Range requests (HttpReader)

const std = @import("std");

pub const reader = @import("reader.zig");
pub const file_reader = @import("file_reader.zig");
pub const mmap_reader = @import("mmap_reader.zig");
pub const memory_reader = @import("memory_reader.zig");
pub const http_reader = @import("http_reader.zig");
pub const batch_reader = @import("batch_reader.zig");
pub const s3_client = @import("s3_client.zig");

// Re-export main types
pub const Reader = reader.Reader;
pub const ReadError = reader.ReadError;
pub const ByteRange = reader.ByteRange;
pub const BatchReadResult = reader.BatchReadResult;
pub const FileReader = file_reader.FileReader;
pub const MmapReader = mmap_reader.MmapReader;
pub const FileCache = mmap_reader.FileCache;
pub const MemoryReader = memory_reader.MemoryReader;
pub const HttpReader = http_reader.HttpReader;
pub const BatchReader = batch_reader.BatchReader;
pub const S3Client = s3_client.S3Client;
pub const S3Error = s3_client.S3Error;
pub const GetResult = s3_client.GetResult;
pub const PutResult = s3_client.PutResult;
pub const parseVersionFromLatest = s3_client.parseVersionFromLatest;

// Global cache functions
pub const initGlobalCache = mmap_reader.initGlobalCache;
pub const deinitGlobalCache = mmap_reader.deinitGlobalCache;

test {
    std.testing.refAllDecls(@This());
}
