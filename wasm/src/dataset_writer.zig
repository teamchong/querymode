//! Lance Dataset Writer with ETag-based distributed coordination.
//!
//! High-level API for writing to Lance datasets stored on S3-compatible storage.
//! Uses ETag-based Compare-And-Swap (CAS) for distributed coordination without
//! requiring external lock services.
//!
//! ## Dataset Structure
//! ```
//! dataset.lance/
//! ├── _versions/
//! │   ├── 1.manifest           # Version 1 manifest (protobuf)
//! │   ├── 2.manifest           # Version 2 manifest
//! │   └── _latest              # Version pointer for CAS (just "2")
//! ├── _transactions/
//! │   └── {version}-{uuid}.txn # Transaction markers
//! └── data/
//!     └── {uuid}.lance         # Column data files
//! ```
//!
//! ## CAS Protocol
//! ```
//! 1. Writer creates data file: data/{uuid}.lance (immediate, no lock)
//! 2. Writer reads _versions/_latest + ETag
//! 3. Writer reads current manifest, adds new fragment
//! 4. Writer writes new manifest: _versions/{N+1}.manifest
//! 5. Writer PUTs _versions/_latest with If-Match: {old-etag}
//!    - Success: Done!
//!    - 412 Precondition Failed: Retry from step 2
//! ```

const std = @import("std");
const io_mod = @import("querymode.io");
const format_mod = @import("querymode.format");
const writer_mod = @import("querymode.writer");
const proto_mod = @import("querymode.proto");

const S3Client = io_mod.S3Client;
const PutResult = io_mod.PutResult;
const parseVersionFromLatest = io_mod.parseVersionFromLatest;

const ManifestWriter = format_mod.ManifestWriter;
const Manifest = format_mod.Manifest;
const Fragment = format_mod.Fragment;
const generateUUID = format_mod.generateUUID;

const LanceWriter = writer_mod.LanceWriter;
const ColumnType = writer_mod.ColumnType;

/// Maximum retries for CAS operations before giving up
const MAX_CAS_RETRIES = 10;

/// Result of an append operation
pub const AppendResult = struct {
    /// New version number
    version: u64,
    /// Path to the data file that was written
    data_file_path: []const u8,
    /// Number of retries needed (0 = first try succeeded)
    retries: u32,
    /// Allocator for cleanup
    allocator: std.mem.Allocator,

    pub fn deinit(self: *AppendResult) void {
        self.allocator.free(self.data_file_path);
    }
};

/// Schema field for creating datasets
pub const SchemaField = struct {
    name: []const u8,
    col_type: ColumnType,
    nullable: bool = true,
    /// For fixed_size_list: dimension
    dimension: ?u32 = null,
};

/// Error types for dataset operations
pub const DatasetWriteError = error{
    OutOfMemory,
    NetworkError,
    CASFailed, // Max retries exceeded
    InvalidSchema,
    S3Error,
    NoLatestVersion,
    ManifestParseError,
    DataWriteError,
};

/// High-level dataset writer with CAS-based distributed coordination.
pub const DatasetWriter = struct {
    allocator: std.mem.Allocator,
    s3: S3Client,
    base_url: []const u8,

    const Self = @This();

    /// Create a new dataset writer.
    ///
    /// base_url: Full URL to the dataset (e.g., "https://data.example.com/my-dataset.lance")
    pub fn init(allocator: std.mem.Allocator, base_url: []const u8) Self {
        return Self{
            .allocator = allocator,
            .s3 = S3Client.init(allocator, base_url),
            .base_url = base_url,
        };
    }

    /// Create a new empty dataset with the given schema.
    ///
    /// This creates the initial manifest (version 1) with no fragments.
    pub fn create(self: *Self, schema: []const SchemaField) DatasetWriteError!void {
        // Check if dataset already exists
        const latest_exists = self.s3.exists("_versions/_latest") catch false;
        if (latest_exists) {
            // Dataset already exists - could be an error or we could ignore
            return;
        }

        // Create version 1 manifest with schema but no fragments
        var manifest_writer = ManifestWriter.init(self.allocator);
        defer manifest_writer.deinit();

        // Add schema fields
        for (schema, 0..) |field, i| {
            manifest_writer.addField(.{
                .name = field.name,
                .id = @intCast(i),
                .logical_type = columnTypeToLogicalType(field.col_type),
                .nullable = field.nullable,
                .dimension = field.dimension,
            }) catch return DatasetWriteError.OutOfMemory;
        }

        manifest_writer.setVersion(1);

        // Generate transaction ID
        const txn_id = generateUUID(self.allocator) catch return DatasetWriteError.OutOfMemory;
        defer self.allocator.free(txn_id);
        manifest_writer.setTransactionId(txn_id);

        // Encode manifest
        const manifest_bytes = manifest_writer.encode() catch return DatasetWriteError.OutOfMemory;
        defer self.allocator.free(manifest_bytes);

        // Write manifest
        self.s3.put("_versions/1.manifest", manifest_bytes) catch return DatasetWriteError.S3Error;

        // Write _latest pointer
        self.s3.put("_versions/_latest", "1") catch return DatasetWriteError.S3Error;
    }

    /// Append data to the dataset using CAS retry loop.
    ///
    /// This is the main write path:
    /// 1. Write data file with unique UUID name
    /// 2. Read current version and ETag
    /// 3. Create new manifest with new fragment
    /// 4. Atomically update _latest using If-Match
    /// 5. Retry on conflict
    pub fn append(self: *Self, lance_data: []const u8, row_count: u64) DatasetWriteError!AppendResult {
        // 1. Generate UUID and write data file (no conflict possible)
        const uuid = generateUUID(self.allocator) catch return DatasetWriteError.OutOfMemory;
        defer self.allocator.free(uuid);

        const data_file_path = std.fmt.allocPrint(self.allocator, "data/{s}.lance", .{uuid}) catch return DatasetWriteError.OutOfMemory;
        errdefer self.allocator.free(data_file_path);

        self.s3.put(data_file_path, lance_data) catch return DatasetWriteError.S3Error;

        // 2. CAS retry loop
        var retries: u32 = 0;
        while (retries < MAX_CAS_RETRIES) : (retries += 1) {
            // Read current _latest + ETag
            var latest_result = self.s3.getWithETag("_versions/_latest") catch {
                // If _latest doesn't exist, we might need to create the dataset
                return DatasetWriteError.NoLatestVersion;
            };
            defer latest_result.deinit();

            const current_version = parseVersionFromLatest(latest_result.data) orelse return DatasetWriteError.ManifestParseError;

            // Read current manifest (for schema)
            const manifest_path = std.fmt.allocPrint(self.allocator, "_versions/{d}.manifest", .{current_version}) catch return DatasetWriteError.OutOfMemory;
            defer self.allocator.free(manifest_path);

            const manifest_data = self.s3.get(manifest_path) catch return DatasetWriteError.S3Error;
            defer self.allocator.free(manifest_data);

            // Parse existing manifest to get schema
            var old_manifest = Manifest.parse(self.allocator, manifest_data) catch return DatasetWriteError.ManifestParseError;
            defer old_manifest.deinit();

            // Create new manifest with new fragment
            const new_version = current_version + 1;

            var manifest_writer = ManifestWriter.init(self.allocator);
            defer manifest_writer.deinit();

            // Copy schema fields (need to read from old manifest - simplified for now)
            // In a full implementation, we'd copy the schema from old_manifest
            // For now, we just track the new fragment

            // Add existing fragments (from old manifest)
            for (old_manifest.fragments) |frag| {
                manifest_writer.addFragment(.{
                    .id = frag.id,
                    .files = &[_]format_mod.manifest_writer.DataFile{.{
                        .path = frag.file_path,
                    }},
                    .physical_rows = frag.physical_rows,
                }) catch return DatasetWriteError.OutOfMemory;
            }

            // Add new fragment
            manifest_writer.addFragment(.{
                .id = new_version - 1, // Fragment IDs are 0-based
                .files = &[_]format_mod.manifest_writer.DataFile{.{
                    .path = data_file_path,
                }},
                .physical_rows = row_count,
            }) catch return DatasetWriteError.OutOfMemory;

            manifest_writer.setVersion(new_version);

            const txn_id = generateUUID(self.allocator) catch return DatasetWriteError.OutOfMemory;
            defer self.allocator.free(txn_id);
            manifest_writer.setTransactionId(txn_id);

            // Encode new manifest
            const new_manifest_bytes = manifest_writer.encode() catch return DatasetWriteError.OutOfMemory;
            defer self.allocator.free(new_manifest_bytes);

            // Write new manifest (unique filename, no conflict)
            const new_manifest_path = std.fmt.allocPrint(self.allocator, "_versions/{d}.manifest", .{new_version}) catch return DatasetWriteError.OutOfMemory;
            defer self.allocator.free(new_manifest_path);

            self.s3.put(new_manifest_path, new_manifest_bytes) catch return DatasetWriteError.S3Error;

            // 5. CAS update _latest
            const new_version_str = std.fmt.allocPrint(self.allocator, "{d}", .{new_version}) catch return DatasetWriteError.OutOfMemory;
            defer self.allocator.free(new_version_str);

            switch (self.s3.putIfMatch("_versions/_latest", new_version_str, latest_result.etag)) {
                .success => {
                    // Success! Return result
                    return AppendResult{
                        .version = new_version,
                        .data_file_path = data_file_path,
                        .retries = retries,
                        .allocator = self.allocator,
                    };
                },
                .precondition_failed => {
                    // Another writer updated _latest - retry
                    continue;
                },
                .http_error, .network_error => {
                    return DatasetWriteError.S3Error;
                },
            }
        }

        // Max retries exceeded
        return DatasetWriteError.CASFailed;
    }

    /// Append data using a LanceWriter.
    ///
    /// Convenience method that finalizes the writer and appends.
    pub fn appendFromWriter(self: *Self, lance_writer: *LanceWriter, row_count: u64) DatasetWriteError!AppendResult {
        const data = lance_writer.finalize() catch return DatasetWriteError.DataWriteError;
        return self.append(data, row_count);
    }

    /// Get the current version number.
    pub fn currentVersion(self: *Self) DatasetWriteError!u64 {
        const latest_data = self.s3.get("_versions/_latest") catch return DatasetWriteError.S3Error;
        defer self.allocator.free(latest_data);

        return parseVersionFromLatest(latest_data) orelse DatasetWriteError.ManifestParseError;
    }
};

/// Convert ColumnType to logical type string for manifest
fn columnTypeToLogicalType(col_type: ColumnType) []const u8 {
    return switch (col_type) {
        .int32 => "int32",
        .int64 => "int64",
        .float32 => "float",
        .float64 => "double",
        .string => "string",
        .bool => "bool",
        .fixed_size_list_f32 => "fixed_size_list:float32",
    };
}

// ============================================================================
// Tests
// ============================================================================

test "DatasetWriter interface compiles" {
    // Just verify the type compiles correctly
    const T = DatasetWriter;
    _ = @TypeOf(T.init);
    _ = @TypeOf(T.create);
    _ = @TypeOf(T.append);
}

test "column type to logical type" {
    try std.testing.expectEqualSlices(u8, "int64", columnTypeToLogicalType(.int64));
    try std.testing.expectEqualSlices(u8, "double", columnTypeToLogicalType(.float64));
    try std.testing.expectEqualSlices(u8, "string", columnTypeToLogicalType(.string));
}
