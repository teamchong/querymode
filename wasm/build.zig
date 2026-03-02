const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // === Platform Detection (comptime) ===
    // Accelerate framework for SIMD on macOS
    const use_accelerate = target.result.os.tag == .macos;

    // === GPU Module ===
    // Cross-platform GPU via wgpu-native
    // Uses same WGSL shaders as browser for code sharing
    const wgpu_dep = b.dependency("wgpu_native_zig", .{
        .target = target,
        .optimize = optimize,
    });

    const gpu_mod = b.addModule("lanceql.gpu", .{
        .root_source_file = b.path("src/gpu/gpu.zig"),
        .imports = &.{
            .{ .name = "wgpu", .module = wgpu_dep.module("wgpu") },
        },
    });

    // === Optional ONNX Runtime Support ===
    // Enable with: zig build -Donnx=/path/to/onnxruntime
    const onnx_path = b.option([]const u8, "onnx", "Path to ONNX Runtime installation (enables embedding support)");

    // === WASM AI Features ===
    // Enable MiniLM + TinyBERT for Cloudflare Workers (not browser)
    // Browser build: zig build wasm (default, no AI, ~500KB)
    // Worker build:  zig build wasm -Denable_ai=true (~1.5MB + models from R2)
    const enable_ai = b.option(bool, "enable_ai", "Enable AI/ML features in WASM (MiniLM + TinyBERT for Workers)") orelse false;

    // === Core Modules ===
    const proto_mod = b.addModule("lanceql.proto", .{
        .root_source_file = b.path("src/proto/proto.zig"),
    });

    const io_mod = b.addModule("lanceql.io", .{
        .root_source_file = b.path("src/io/io.zig"),
    });

    const snappy_mod = b.addModule("lanceql.encoding.snappy", .{
        .root_source_file = b.path("src/encoding/snappy.zig"),
    });

    const table_utils_mod = b.addModule("lanceql.table_utils", .{
        .root_source_file = b.path("src/table_utils.zig"),
    });

    const encoding_mod = b.addModule("lanceql.encoding", .{
        .root_source_file = b.path("src/encoding/encoding.zig"),
        .imports = &.{
            .{ .name = "lanceql.encoding.snappy", .module = snappy_mod },
        },
    });

    const writer_mod = b.addModule("lanceql.writer", .{
        .root_source_file = b.path("src/writer/writer.zig"),
        .imports = &.{
            .{ .name = "lanceql.proto", .module = proto_mod },
        },
    });

    const value_mod = b.addModule("lanceql.value", .{
        .root_source_file = b.path("src/value.zig"),
    });

    // Query expression module - for predicate fusion in codegen
    // Defined here before query_mod to avoid circular dependency
    const query_expr_mod = b.addModule("lanceql.query.expr", .{
        .root_source_file = b.path("src/query/expr.zig"),
        .imports = &.{
            .{ .name = "lanceql.value", .module = value_mod },
        },
    });

    const query_mod = b.addModule("lanceql.query", .{
        .root_source_file = b.path("src/query/query.zig"),
        .imports = &.{
            .{ .name = "lanceql.value", .module = value_mod },
            .{ .name = "lanceql.gpu", .module = gpu_mod },
            .{ .name = "lanceql.query.expr", .module = query_expr_mod },
        },
    });

    const format_mod = b.addModule("lanceql.format", .{
        .root_source_file = b.path("src/format/format.zig"),
        .imports = &.{
            .{ .name = "lanceql.proto", .module = proto_mod },
            .{ .name = "lanceql.io", .module = io_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
        },
    });

    const parquet_encoding_mod = b.addModule("lanceql.encoding.parquet", .{
        .root_source_file = b.path("src/encoding/parquet/parquet_encoding.zig"),
        .imports = &.{
            .{ .name = "lanceql.proto", .module = proto_mod },
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.encoding.snappy", .module = snappy_mod },
        },
    });

    // SQL modules
    const sql_ast_mod = b.addModule("lanceql.sql.ast", .{
        .root_source_file = b.path("src/sql/ast.zig"),
    });

    const sql_lexer_mod = b.addModule("lanceql.sql.lexer", .{
        .root_source_file = b.path("src/sql/lexer.zig"),
    });

    const sql_parser_mod = b.addModule("lanceql.sql.parser", .{
        .root_source_file = b.path("src/sql/parser.zig"),
        .imports = &.{
            .{ .name = "ast", .module = sql_ast_mod },
            .{ .name = "lexer", .module = sql_lexer_mod },
        },
    });

    // Column dependency extraction for @logic_table
    const sql_column_deps_mod = b.addModule("lanceql.sql.column_deps", .{
        .root_source_file = b.path("src/sql/column_deps.zig"),
        .imports = &.{
            .{ .name = "ast", .module = sql_ast_mod },
        },
    });

    // Batch code generator for @logic_table GPU/SIMD dispatch
    _ = b.addModule("lanceql.sql.batch_codegen", .{
        .root_source_file = b.path("src/sql/batch_codegen.zig"),
        .imports = &.{
            .{ .name = "ast", .module = sql_ast_mod },
            .{ .name = "column_deps", .module = sql_column_deps_mod },
        },
    });

    // SIMD and parallel compute primitives
    const simd_mod = b.addModule("lanceql.simd", .{
        .root_source_file = b.path("src/simd.zig"),
    });

    // Hash functions for GROUP BY and JOIN
    const hash_mod = b.addModule("lanceql.hash", .{
        .root_source_file = b.path("src/hash.zig"),
    });

    // SIMD columnar operations for SQL executor
    const columnar_ops_mod = b.addModule("lanceql.columnar_ops", .{
        .root_source_file = b.path("src/columnar_ops.zig"),
    });

    // DuckDB-style vectorized query engine primitives
    // Shared between native and WASM executors for consistent performance
    const vector_engine_mod = b.addModule("lanceql.query.vector_engine", .{
        .root_source_file = b.path("src/query/vector_engine.zig"),
    });

    const table_mod = b.addModule("lanceql.table", .{
        .root_source_file = b.path("src/table.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.proto", .module = proto_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
            .{ .name = "lanceql.io", .module = io_mod },
            .{ .name = "simd", .module = simd_mod },
        },
    });

    const parquet_table_mod = b.addModule("lanceql.parquet_table", .{
        .root_source_file = b.path("src/parquet_table.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.encoding.parquet", .module = parquet_encoding_mod },
        },
    });

    const delta_table_mod = b.addModule("lanceql.delta_table", .{
        .root_source_file = b.path("src/delta_table.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
            .{ .name = "lanceql.parquet_table", .module = parquet_table_mod },
        },
    });

    const iceberg_table_mod = b.addModule("lanceql.iceberg_table", .{
        .root_source_file = b.path("src/iceberg_table.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
            .{ .name = "lanceql.parquet_table", .module = parquet_table_mod },
        },
    });

    const arrow_table_mod = b.addModule("lanceql.arrow_table", .{
        .root_source_file = b.path("src/arrow_table.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
            .{ .name = "lanceql.table_utils", .module = table_utils_mod },
        },
    });

    const avro_table_mod = b.addModule("lanceql.avro_table", .{
        .root_source_file = b.path("src/avro_table.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
            .{ .name = "lanceql.table_utils", .module = table_utils_mod },
        },
    });

    const orc_table_mod = b.addModule("lanceql.orc_table", .{
        .root_source_file = b.path("src/orc_table.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
            .{ .name = "lanceql.table_utils", .module = table_utils_mod },
        },
    });

    const xlsx_table_mod = b.addModule("lanceql.xlsx_table", .{
        .root_source_file = b.path("src/xlsx_table.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
            .{ .name = "lanceql.table_utils", .module = table_utils_mod },
        },
    });

    const any_table_mod = b.addModule("lanceql.any_table", .{
        .root_source_file = b.path("src/any_table.zig"),
        .imports = &.{
            .{ .name = "lanceql.table", .module = table_mod },
            .{ .name = "lanceql.parquet_table", .module = parquet_table_mod },
            .{ .name = "lanceql.delta_table", .module = delta_table_mod },
            .{ .name = "lanceql.iceberg_table", .module = iceberg_table_mod },
            .{ .name = "lanceql.arrow_table", .module = arrow_table_mod },
            .{ .name = "lanceql.avro_table", .module = avro_table_mod },
            .{ .name = "lanceql.orc_table", .module = orc_table_mod },
            .{ .name = "lanceql.xlsx_table", .module = xlsx_table_mod },
        },
    });

    // Logic table static library - compiled @logic_table functions from metal0
    // Generated by: metal0 build --emit-logic-table <python_file> -o lib/logic_table.a
    // The .a file contains all @logic_table structs with metal0 runtime bundled
    //
    // Build option to specify the path (defaults to lib/logic_table.a)
    const logic_table_lib_opt = b.option([]const u8, "logic-table-lib", "Path to logic_table.a static library");

    // Logic table module - provides LogicTableContext, LogicTableRegistry, etc.
    const logic_table_mod = b.addModule("lanceql.logic_table", .{
        .root_source_file = b.path("src/logic_table/logic_table.zig"),
        .imports = &.{
            .{ .name = "lanceql.table", .module = table_mod },
        },
    });

    // ============================================================================
    // metal0 module - Python to native compiler (as submodule)
    // ============================================================================
    // metal0 has many internal modules that need to be wired up for full functionality.
    // This includes c_interop (for Python extension interop), runtime, collections, etc.

    // Utility modules
    const metal0_hashmap_helper = b.addModule("utils.hashmap_helper", .{
        .root_source_file = b.path("deps/metal0/src/utils/hashmap_helper.zig"),
    });
    const metal0_allocator_helper = b.addModule("utils.allocator_helper", .{
        .root_source_file = b.path("deps/metal0/src/utils/allocator_helper.zig"),
    });
    const metal0_platform = b.addModule("utils.platform", .{
        .root_source_file = b.path("deps/metal0/src/utils/platform.zig"),
    });
    const metal0_fnv_hash = b.addModule("utils.fnv_hash", .{
        .root_source_file = b.path("deps/metal0/src/utils/fnv_hash.zig"),
    });
    // Collections module
    const metal0_collections = b.addModule("collections", .{
        .root_source_file = b.path("deps/metal0/packages/collections/collections.zig"),
    });

    // JSON SIMD dispatch
    const metal0_json_simd = b.addModule("json_simd", .{
        .root_source_file = b.path("deps/metal0/packages/shared/json/simd/dispatch.zig"),
    });

    // JSON module
    const metal0_json = b.addModule("json", .{
        .root_source_file = b.path("deps/metal0/packages/shared/json/json.zig"),
    });
    metal0_json.addImport("json_simd", metal0_json_simd);
    metal0_json.addImport("utils.hashmap_helper", metal0_hashmap_helper);

    // Gzip module
    const metal0_gzip = b.addModule("gzip", .{
        .root_source_file = b.path("deps/metal0/packages/runtime/src/Modules/gzip/gzip.zig"),
    });
    metal0_gzip.addIncludePath(b.path("deps/metal0/vendor/libdeflate"));

    // HTTP/2 module
    const metal0_h2 = b.addModule("h2", .{
        .root_source_file = b.path("deps/metal0/packages/shared/http/h2/Client.zig"),
    });
    metal0_h2.addImport("gzip", metal0_gzip);
    metal0_h2.addImport("utils.hashmap_helper", metal0_hashmap_helper);

    // Regex module
    const metal0_regex = b.addModule("regex", .{
        .root_source_file = b.path("deps/metal0/packages/regex/src/pyregex/regex.zig"),
    });

    // BigInt module
    const metal0_bigint = b.addModule("bigint", .{
        .root_source_file = b.path("deps/metal0/packages/bigint/src/bigint.zig"),
    });

    // Data structures module
    const metal0_ds = b.addModule("ds", .{
        .root_source_file = b.path("deps/metal0/packages/ds/src/ds.zig"),
    });

    // Glob module
    const metal0_glob = b.addModule("glob", .{
        .root_source_file = b.path("deps/metal0/packages/glob/src/glob.zig"),
    });
    metal0_glob.addImport("ds", metal0_ds);

    // Tokenizer module
    const metal0_tokenizer = b.addModule("tokenizer", .{
        .root_source_file = b.path("deps/metal0/packages/tokenizer/src/tokenizer.zig"),
    });
    metal0_tokenizer.addImport("json", metal0_json);
    metal0_tokenizer.addImport("utils.hashmap_helper", metal0_hashmap_helper);
    metal0_tokenizer.addImport("utils.fnv_hash", metal0_fnv_hash);

    // Package manager module
    const metal0_pkg = b.addModule("pkg", .{
        .root_source_file = b.path("deps/metal0/packages/pkg/src/pkg.zig"),
    });
    metal0_pkg.addImport("json", metal0_json);
    metal0_pkg.addImport("h2", metal0_h2);
    metal0_pkg.addImport("utils.hashmap_helper", metal0_hashmap_helper);

    // Runtime module (core Python runtime)
    const metal0_runtime = b.createModule(.{
        .root_source_file = b.path("deps/metal0/packages/runtime/src/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    metal0_runtime.addIncludePath(b.path("deps/metal0/vendor/libdeflate"));
    // Add libdeflate source files
    metal0_runtime.addCSourceFiles(.{
        .files = &.{
            "deps/metal0/vendor/libdeflate/lib/deflate_compress.c",
            "deps/metal0/vendor/libdeflate/lib/deflate_decompress.c",
            "deps/metal0/vendor/libdeflate/lib/utils.c",
            "deps/metal0/vendor/libdeflate/lib/gzip_compress.c",
            "deps/metal0/vendor/libdeflate/lib/gzip_decompress.c",
            "deps/metal0/vendor/libdeflate/lib/zlib_compress.c",
            "deps/metal0/vendor/libdeflate/lib/zlib_decompress.c",
            "deps/metal0/vendor/libdeflate/lib/adler32.c",
            "deps/metal0/vendor/libdeflate/lib/crc32.c",
            "deps/metal0/vendor/libdeflate/lib/arm/cpu_features.c",
            "deps/metal0/vendor/libdeflate/lib/x86/cpu_features.c",
        },
        // Disable AVX-512 to avoid compilation errors on CI runners without evex512 support
        .flags = &.{
            "-std=c99",
            "-O3",
            "-DLIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_AVX512VNNI",
            "-DLIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_VPCLMULQDQ",
        },
    });
    // Link libc for malloc/free used by libdeflate
    metal0_runtime.link_libc = true;
    // Link compression libraries
    metal0_runtime.linkSystemLibrary("bz2", .{});
    metal0_runtime.linkSystemLibrary("lzma", .{});
    metal0_runtime.addImport("utils.hashmap_helper", metal0_hashmap_helper);
    metal0_runtime.addImport("utils.allocator_helper", metal0_allocator_helper);
    metal0_runtime.addImport("utils.platform", metal0_platform);
    metal0_runtime.addImport("json_simd", metal0_json_simd);
    metal0_runtime.addImport("json", metal0_json);
    metal0_runtime.addImport("regex", metal0_regex);
    metal0_runtime.addImport("bigint", metal0_bigint);
    metal0_runtime.addImport("gzip", metal0_gzip);
    metal0_runtime.addImport("h2", metal0_h2);
    metal0_runtime.addImport("tokenizer", metal0_tokenizer);

    // C interop module (Python extension interop)
    const metal0_c_interop = b.createModule(.{
        .root_source_file = b.path("deps/metal0/packages/c_interop/src/registry.zig"),
        .target = target,
        .optimize = optimize,
    });
    metal0_c_interop.addImport("runtime", metal0_runtime);
    metal0_c_interop.addImport("collections", metal0_collections);
    metal0_c_interop.addImport("utils.hashmap_helper", metal0_hashmap_helper);

    // AST module
    const metal0_ast_mod = b.addModule("analysis.ast", .{
        .root_source_file = b.path("deps/metal0/src/ast.zig"),
    });

    // Main metal0 module
    const metal0_mod = b.addModule("metal0", .{
        .root_source_file = b.path("deps/metal0/src/main.zig"),
    });
    metal0_mod.addImport("analysis.ast", metal0_ast_mod);
    metal0_mod.addImport("c_interop", metal0_c_interop);
    metal0_mod.addImport("json", metal0_json);
    metal0_mod.addImport("h2", metal0_h2);
    metal0_mod.addImport("regex", metal0_regex);
    metal0_mod.addImport("bigint", metal0_bigint);
    metal0_mod.addImport("tokenizer", metal0_tokenizer);
    metal0_mod.addImport("pkg", metal0_pkg);
    metal0_mod.addImport("ds", metal0_ds);
    metal0_mod.addImport("glob", metal0_glob);
    metal0_mod.addImport("utils.hashmap_helper", metal0_hashmap_helper);
    metal0_mod.addImport("utils.allocator_helper", metal0_allocator_helper);
    metal0_mod.addImport("utils.fnv_hash", metal0_fnv_hash);
    metal0_mod.addImport("collections", metal0_collections);
    metal0_mod.addImport("runtime", metal0_runtime);

    // Analysis modules needed by metal0
    const metal0_function_traits = b.addModule("analysis.function_traits", .{
        .root_source_file = b.path("deps/metal0/src/analysis/traits/function_traits.zig"),
    });
    metal0_function_traits.addImport("analysis.ast", metal0_ast_mod);
    metal0_function_traits.addImport("utils.hashmap_helper", metal0_hashmap_helper);

    const metal0_module_traits = b.addModule("analysis.module_traits", .{
        .root_source_file = b.path("deps/metal0/src/analysis/traits/module_traits.zig"),
    });
    metal0_module_traits.addImport("analysis.ast", metal0_ast_mod);
    metal0_module_traits.addImport("utils.hashmap_helper", metal0_hashmap_helper);
    metal0_module_traits.addImport("analysis.function_traits", metal0_function_traits);

    metal0_mod.addImport("analysis.function_traits", metal0_function_traits);
    metal0_mod.addImport("analysis.module_traits", metal0_module_traits);

    // Debug info module
    const metal0_debug_info = b.addModule("debug.debug_info", .{
        .root_source_file = b.path("deps/metal0/src/debug/debug_info.zig"),
    });
    metal0_debug_info.addImport("utils.hashmap_helper", metal0_hashmap_helper);
    metal0_mod.addImport("debug.debug_info", metal0_debug_info);

    // Codegen modules
    const metal0_zig_keywords = b.addModule("utils.zig_keywords", .{
        .root_source_file = b.path("deps/metal0/src/utils/zig_keywords.zig"),
    });
    const metal0_name_gen = b.addModule("codegen.name_gen", .{
        .root_source_file = b.path("deps/metal0/src/codegen/native/name_gen.zig"),
    });
    metal0_name_gen.addImport("utils.zig_keywords", metal0_zig_keywords);
    metal0_mod.addImport("codegen.name_gen", metal0_name_gen);
    metal0_mod.addImport("utils.zig_keywords", metal0_zig_keywords);

    // Builder module for structured Zig code generation
    const metal0_builder = b.addModule("codegen.builder", .{
        .root_source_file = b.path("deps/metal0/src/codegen/builder/builder.zig"),
    });
    metal0_builder.addImport("codegen.name_gen", metal0_name_gen);
    metal0_builder.addImport("utils.zig_keywords", metal0_zig_keywords);
    metal0_mod.addImport("codegen.builder", metal0_builder);

    // String interner
    const metal0_string_interner = b.addModule("utils.string_interner", .{
        .root_source_file = b.path("deps/metal0/src/utils/string_interner.zig"),
    });
    metal0_mod.addImport("utils.string_interner", metal0_string_interner);
    metal0_mod.addImport("utils.platform", metal0_platform);

    // Codegen module - JIT compilation integration with metal0
    const codegen_mod = b.addModule("lanceql.codegen", .{
        .root_source_file = b.path("src/codegen/metal0_jit.zig"),
        .imports = &.{
            .{ .name = "metal0", .module = metal0_mod },
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.proto", .module = proto_mod },
            .{ .name = "lanceql.query.expr", .module = query_expr_mod },
        },
    });

    // Helper to link logic_table.a to a compile step
    const linkLogicTableLib = struct {
        fn link(step: *std.Build.Step.Compile, lib_path: ?[]const u8, builder: *std.Build) void {
            if (lib_path) |path| {
                // User-specified path
                step.addObjectFile(.{ .cwd_relative = path });
            } else {
                // Default path - only link if file exists
                const default_path = builder.path("lib/logic_table.a");
                step.addObjectFile(default_path);
            }
        }
    }.link;

    _ = linkLogicTableLib; // Available for executables that need logic_table
    _ = logic_table_lib_opt; // Used by linkLogicTableLib

    const sql_executor_mod = b.addModule("lanceql.sql.executor", .{
        .root_source_file = b.path("src/sql/executor.zig"),
        .imports = &.{
            .{ .name = "ast", .module = sql_ast_mod },
            .{ .name = "parser", .module = sql_parser_mod },
            .{ .name = "lanceql.table", .module = table_mod },
            .{ .name = "lanceql.parquet_table", .module = parquet_table_mod },
            .{ .name = "lanceql.delta_table", .module = delta_table_mod },
            .{ .name = "lanceql.iceberg_table", .module = iceberg_table_mod },
            .{ .name = "lanceql.arrow_table", .module = arrow_table_mod },
            .{ .name = "lanceql.avro_table", .module = avro_table_mod },
            .{ .name = "lanceql.orc_table", .module = orc_table_mod },
            .{ .name = "lanceql.xlsx_table", .module = xlsx_table_mod },
            .{ .name = "lanceql.any_table", .module = any_table_mod },
            .{ .name = "lanceql.logic_table", .module = logic_table_mod },
            .{ .name = "lanceql.hash", .module = hash_mod },
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.codegen", .module = codegen_mod },
            .{ .name = "lanceql.vector_engine", .module = vector_engine_mod },
            .{ .name = "lanceql.columnar_ops", .module = columnar_ops_mod },
            .{ .name = "lanceql.query", .module = query_mod },
        },
    });

    const sql_dispatch_mod = b.addModule("lanceql.sql.logic_table_dispatch", .{
        .root_source_file = b.path("src/sql/logic_table_dispatch.zig"),
        .imports = &.{
            .{ .name = "ast", .module = sql_ast_mod },
            .{ .name = "lanceql.logic_table", .module = logic_table_mod },
        },
    });
    _ = sql_dispatch_mod; // Used in SQL tests and CLI

    const dataframe_mod = b.addModule("lanceql.dataframe", .{
        .root_source_file = b.path("src/dataframe.zig"),
        .imports = &.{
            .{ .name = "lanceql.value", .module = value_mod },
            .{ .name = "lanceql.query", .module = query_mod },
            .{ .name = "lanceql.table", .module = table_mod },
        },
    });

    // Dataset module - high-level API mirroring browser vault.js
    const dataset_mod = b.addModule("lanceql.dataset", .{
        .root_source_file = b.path("src/dataset.zig"),
        .imports = &.{
            .{ .name = "lanceql.sql.executor", .module = sql_executor_mod },
            .{ .name = "lexer", .module = sql_lexer_mod },
            .{ .name = "parser", .module = sql_parser_mod },
            .{ .name = "ast", .module = sql_ast_mod },
            .{ .name = "lanceql.table", .module = table_mod },
            .{ .name = "lanceql.dataframe", .module = dataframe_mod },
            .{ .name = "lanceql.format", .module = format_mod },
        },
    });

    // Dataset writer module - distributed writes with ETag-based CAS
    const dataset_writer_mod = b.addModule("lanceql.dataset_writer", .{
        .root_source_file = b.path("src/dataset_writer.zig"),
        .imports = &.{
            .{ .name = "lanceql.io", .module = io_mod },
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.writer", .module = writer_mod },
            .{ .name = "lanceql.proto", .module = proto_mod },
        },
    });

    // AI module - native GGUF model inference (TinyBERT, MiniLM)
    // Only built when enable_ai=true
    const ai_mod = b.addModule("lanceql.ai", .{
        .root_source_file = b.path("src/ai/ai.zig"),
    });

    // Root module exports all
    const lanceql_mod = b.addModule("lanceql", .{
        .root_source_file = b.path("src/lanceql.zig"),
        .imports = &.{
            .{ .name = "lanceql.format", .module = format_mod },
            .{ .name = "lanceql.io", .module = io_mod },
            .{ .name = "lanceql.proto", .module = proto_mod },
            .{ .name = "lanceql.encoding", .module = encoding_mod },
            .{ .name = "lanceql.writer", .module = writer_mod },
            .{ .name = "lanceql.table", .module = table_mod },
            .{ .name = "lanceql.query", .module = query_mod },
            .{ .name = "lanceql.value", .module = value_mod },
            .{ .name = "lanceql.dataframe", .module = dataframe_mod },
            .{ .name = "lanceql.dataset", .module = dataset_mod },
            .{ .name = "lanceql.dataset_writer", .module = dataset_writer_mod },
            .{ .name = "lanceql.gpu", .module = gpu_mod },
            .{ .name = "lanceql.logic_table", .module = logic_table_mod },
            .{ .name = "lanceql.codegen", .module = codegen_mod },
        },
    });

    // Pass build options to modules
    const build_options = b.addOptions();
    build_options.addOption(bool, "use_gpu", true); // GPU via wgpu-native (cross-platform)
    build_options.addOption(bool, "use_accelerate", use_accelerate);
    build_options.addOption(bool, "use_onnx", onnx_path != null);
    build_options.addOption(bool, "enable_ai", enable_ai); // Native GGUF inference (TinyBERT, MiniLM)
    gpu_mod.addOptions("build_options", build_options);
    lanceql_mod.addOptions("build_options", build_options);

    // === Tests ===
    const test_footer = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_footer.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.format", .module = format_mod },
            },
        }),
    });

    const test_proto = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_proto.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.proto", .module = proto_mod },
            },
        }),
    });

    const test_integration = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_integration.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.io", .module = io_mod },
                .{ .name = "lanceql.proto", .module = proto_mod },
                .{ .name = "lanceql.encoding", .module = encoding_mod },
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });

    // Run tests
    const run_test_footer = b.addRunArtifact(test_footer);
    const run_test_proto = b.addRunArtifact(test_proto);
    const run_test_integration = b.addRunArtifact(test_integration);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_test_footer.step);
    test_step.dependOn(&run_test_proto.step);
    test_step.dependOn(&run_test_integration.step);

    const test_footer_step = b.step("test-footer", "Run footer tests");
    test_footer_step.dependOn(&run_test_footer.step);

    const test_proto_step = b.step("test-proto", "Run protobuf tests");
    test_proto_step.dependOn(&run_test_proto.step);

    const test_integration_step = b.step("test-integration", "Run integration tests with real .lance files");
    test_integration_step.dependOn(&run_test_integration.step);

    // Parquet parser tests - use ReleaseFast for benchmark
    const test_parquet = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_parquet.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.encoding.parquet", .module = parquet_encoding_mod },
            },
        }),
    });

    const run_test_parquet = b.addRunArtifact(test_parquet);
    test_step.dependOn(&run_test_parquet.step);

    const test_parquet_step = b.step("test-parquet", "Run Parquet parser tests");
    test_parquet_step.dependOn(&run_test_parquet.step);

    // Query module tests (with Metal/Accelerate for logic_table)
    const test_query = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/query/query.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql.value", .module = value_mod },
                .{ .name = "lanceql.gpu", .module = gpu_mod },
                .{ .name = "lanceql.query.expr", .module = query_expr_mod },
            },
        }),
    });

    // Link Accelerate framework on macOS for SIMD acceleration
    if (use_accelerate) {
        test_query.root_module.linkFramework("Accelerate", .{});
    }
    // wgpu-native requires platform-specific libraries
    switch (target.result.os.tag) {
        .macos => {
            test_query.root_module.linkFramework("Metal", .{});
            test_query.root_module.linkFramework("QuartzCore", .{});
            test_query.root_module.linkFramework("Foundation", .{});
            test_query.root_module.linkFramework("CoreFoundation", .{});
            test_query.linkLibC();
        },
        .linux => {
            test_query.root_module.linkSystemLibrary("vulkan", .{});
            test_query.linkLibC();
        },
        .windows => {
            test_query.root_module.linkSystemLibrary("d3d12", .{});
            test_query.root_module.linkSystemLibrary("dxgi", .{});
            test_query.root_module.linkSystemLibrary("dxguid", .{});
            test_query.linkLibC();
        },
        else => {},
    }

    const run_test_query = b.addRunArtifact(test_query);
    test_step.dependOn(&run_test_query.step);

    const test_query_step = b.step("test-query", "Run query module tests");
    test_query_step.dependOn(&run_test_query.step);

    // Value module tests
    const test_value = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/value.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_value = b.addRunArtifact(test_value);
    test_step.dependOn(&run_test_value.step);

    // DataFrame module tests
    const test_dataframe = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/dataframe.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql.value", .module = value_mod },
                .{ .name = "lanceql.query", .module = query_mod },
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });

    const run_test_dataframe = b.addRunArtifact(test_dataframe);
    test_step.dependOn(&run_test_dataframe.step);

    // LogicTable end-to-end tests (with Metal/Accelerate for GPU acceleration)
    const test_logic_table_e2e = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_logic_table_e2e.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.table", .module = table_mod },
                .{ .name = "lanceql.query", .module = query_mod },
            },
        }),
    });

    // Link Accelerate framework on macOS for SIMD acceleration
    if (use_accelerate) {
        test_logic_table_e2e.root_module.linkFramework("Accelerate", .{});
    }
    // wgpu-native requires platform-specific libraries
    switch (target.result.os.tag) {
        .macos => {
            test_logic_table_e2e.root_module.linkFramework("Metal", .{});
            test_logic_table_e2e.root_module.linkFramework("QuartzCore", .{});
            test_logic_table_e2e.root_module.linkFramework("Foundation", .{});
            test_logic_table_e2e.root_module.linkFramework("CoreFoundation", .{});
            test_logic_table_e2e.linkLibC();
        },
        .linux => {
            test_logic_table_e2e.root_module.linkSystemLibrary("vulkan", .{});
            test_logic_table_e2e.linkLibC();
        },
        .windows => {
            test_logic_table_e2e.root_module.linkSystemLibrary("d3d12", .{});
            test_logic_table_e2e.root_module.linkSystemLibrary("dxgi", .{});
            test_logic_table_e2e.root_module.linkSystemLibrary("dxguid", .{});
            test_logic_table_e2e.linkLibC();
        },
        else => {},
    }

    const run_test_logic_table_e2e = b.addRunArtifact(test_logic_table_e2e);
    test_step.dependOn(&run_test_logic_table_e2e.step);

    const test_logic_table_step = b.step("test-logic-table", "Run @logic_table end-to-end tests");
    test_logic_table_step.dependOn(&run_test_logic_table_e2e.step);

    const test_dataframe_step = b.step("test-dataframe", "Run DataFrame module tests");
    test_dataframe_step.dependOn(&run_test_dataframe.step);

    // Ingest command tests (CSV, TSV, JSON, JSONL, Parquet → Lance)
    const test_ingest = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_ingest.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.encoding", .module = encoding_mod },
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.encoding.parquet", .module = parquet_encoding_mod },
            },
        }),
    });

    const run_test_ingest = b.addRunArtifact(test_ingest);
    test_step.dependOn(&run_test_ingest.step);

    const test_ingest_step = b.step("test-ingest", "Run ingest command tests (CSV, TSV, JSON, JSONL, Parquet)");
    test_ingest_step.dependOn(&run_test_ingest.step);

    // GPU module tests (cross-platform via wgpu-native, currently CPU fallback)
    const test_gpu = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/gpu.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_gpu = b.addRunArtifact(test_gpu);
    test_step.dependOn(&run_test_gpu.step);

    const test_gpu_step = b.step("test-gpu", "Run GPU module tests (wgpu-native)");
    test_gpu_step.dependOn(&run_test_gpu.step);

    // Vector benchmark (Column-first I/O)
    const bench_vector = b.addExecutable(.{
        .name = "bench_vector",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_vector_ops.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.io", .module = io_mod },
            },
        }),
    });
    const run_bench_vector = b.addRunArtifact(bench_vector);
    const bench_vector_step = b.step("bench-vector", "FAIR end-to-end: L2 norm (vector ops)");
    bench_vector_step.dependOn(&run_bench_vector.step);

    // SQL clause benchmark - uses real SQL executor for honest benchmarks
    const bench_sql = b.addExecutable(.{
        .name = "bench_sql",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_sql_clauses.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
                .{ .name = "lanceql.simd", .module = simd_mod },
                .{ .name = "lanceql.sql.ast", .module = sql_ast_mod },
                .{ .name = "lanceql.sql.parser", .module = sql_parser_mod },
                .{ .name = "lanceql.sql.executor", .module = sql_executor_mod },
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.io", .module = io_mod },
            },
        }),
    });
    const run_bench_sql = b.addRunArtifact(bench_sql);
    const bench_sql_step = b.step("bench-sql", "FAIR end-to-end: SQL clauses (FILTER, AGGREGATE, GROUP BY, JOIN)");
    bench_sql_step.dependOn(&run_bench_sql.step);

    // Column-first I/O benchmark - compares full-file vs column-first reading
    const bench_column_io = b.addExecutable(.{
        .name = "bench_column_io",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_column_io.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.io", .module = io_mod },
                .{ .name = "lanceql.simd", .module = simd_mod },
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    const run_bench_column_io = b.addRunArtifact(bench_column_io);
    const bench_column_io_step = b.step("bench-column-io", "Column-first I/O benchmark");
    bench_column_io_step.dependOn(&run_bench_column_io.step);

    // @logic_table benchmark - uses built-in SIMD functions and JIT
    const bench_logic_table = b.addExecutable(.{
        .name = "bench_logic_table",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_logic_table.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
                .{ .name = "lanceql.simd", .module = simd_mod },
                .{ .name = "lanceql.codegen", .module = codegen_mod },
            },
        }),
    });
    const run_bench_logic_table = b.addRunArtifact(bench_logic_table);
    const bench_logic_table_step = b.step("bench-logic-table", "Benchmark @logic_table (SIMD)");
    bench_logic_table_step.dependOn(&run_bench_logic_table.step);

    // @logic_table End-to-End benchmark - full pipeline with comparisons
    const bench_logic_table_e2e = b.addExecutable(.{
        .name = "bench_logic_table_e2e",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_logic_table_e2e.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    const run_bench_logic_table_e2e = b.addRunArtifact(bench_logic_table_e2e);
    const bench_logic_table_e2e_step = b.step("bench-logic-table-e2e", "End-to-end @logic_table benchmark with DuckDB/Polars comparison");
    bench_logic_table_e2e_step.dependOn(&run_bench_logic_table_e2e.step);

    // @logic_table Pushdown benchmark - demonstrates filtered_indices optimization
    const bench_pushdown = b.addExecutable(.{
        .name = "bench_pushdown",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_pushdown.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    const run_bench_pushdown = b.addRunArtifact(bench_pushdown);
    const bench_pushdown_step = b.step("bench-pushdown", "Benchmark pushdown (filtered_indices optimization)");
    bench_pushdown_step.dependOn(&run_bench_pushdown.step);

    // Window functions benchmark - @logic_table vs Python UDFs
    const bench_window = b.addExecutable(.{
        .name = "bench_window",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_window_functions.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    const run_bench_window = b.addRunArtifact(bench_window);
    const bench_window_step = b.step("bench-window", "Window functions: @logic_table (compiled) vs Python UDFs");
    bench_window_step.dependOn(&run_bench_window.step);

    // Tiered dispatch benchmark - SIMD vs GPU for batch vector operations
    const bench_tiered = b.addExecutable(.{
        .name = "bench_tiered_dispatch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_tiered_dispatch.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    const run_bench_tiered = b.addRunArtifact(bench_tiered);
    const bench_tiered_step = b.step("bench-tiered", "FAIR end-to-end: SIMD dot product");
    bench_tiered_step.dependOn(&run_bench_tiered.step);

    // Parquet benchmark - LanceQL vs DuckDB vs Polars
    const bench_parquet = b.addExecutable(.{
        .name = "bench_parquet",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_parquet.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.encoding.parquet", .module = parquet_encoding_mod },
            },
        }),
    });
    b.installArtifact(bench_parquet);
    const bench_parquet_step = b.step("bench-parquet", "Benchmark Parquet reading: LanceQL vs DuckDB vs Polars");
    bench_parquet_step.dependOn(&b.addInstallArtifact(bench_parquet, .{}).step);

    // In-process benchmark: LanceQL vs DuckDB C API (FAIR comparison)
    const bench_inprocess = b.addExecutable(.{
        .name = "bench_inprocess",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_inprocess.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    const run_bench_inprocess = b.addRunArtifact(bench_inprocess);
    const bench_inprocess_step = b.step("bench-inprocess", "FAIR end-to-end: LanceQL vs DuckDB vs Polars");
    bench_inprocess_step.dependOn(&run_bench_inprocess.step);

    // RAG Pipeline benchmark - end-to-end document retrieval
    const bench_rag = b.addExecutable(.{
        .name = "bench_rag_pipeline",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_rag_pipeline.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    const run_bench_rag = b.addRunArtifact(bench_rag);
    const bench_rag_step = b.step("bench-rag", "FAIR end-to-end: similarity search (RAG pipeline)");
    bench_rag_step.dependOn(&run_bench_rag.step);

    // Hybrid Search benchmark - vector + SQL filters
    const bench_hybrid = b.addExecutable(.{
        .name = "bench_hybrid_search",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_hybrid_search.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.gpu", .module = gpu_mod },
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    if (use_accelerate) {
        bench_hybrid.root_module.linkFramework("Accelerate", .{});
    }
    const run_bench_hybrid = b.addRunArtifact(bench_hybrid);
    const bench_hybrid_step = b.step("bench-hybrid", "Hybrid search: vector similarity + SQL filters");
    bench_hybrid_step.dependOn(&run_bench_hybrid.step);

    // Feature Engineering benchmark - ML transformations
    const bench_feature = b.addExecutable(.{
        .name = "bench_feature_engineering",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_feature_engineering.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    const run_bench_feature = b.addRunArtifact(bench_feature);
    const bench_feature_step = b.step("bench-feature", "Feature engineering: normalization, binning, transforms");
    bench_feature_step.dependOn(&run_bench_feature.step);

    // Analytics benchmark - aggregations, window functions
    const bench_analytics = b.addExecutable(.{
        .name = "bench_analytics",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_analytics.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.io", .module = io_mod },
            },
        }),
    });
    const run_bench_analytics = b.addRunArtifact(bench_analytics);
    const bench_analytics_step = b.step("bench-analytics", "Analytics: aggregations, GROUP BY, window functions");
    bench_analytics_step.dependOn(&run_bench_analytics.step);

    // Embedding Pipeline benchmark - text to vector
    const bench_embed = b.addExecutable(.{
        .name = "bench_embedding_pipeline",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_embedding_pipeline.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });
    const run_bench_embed = b.addRunArtifact(bench_embed);
    const bench_embed_step = b.step("bench-embed", "Embedding pipeline: chunking, tokenization, embedding");
    bench_embed_step.dependOn(&run_bench_embed.step);

    // LanceQL CLI
    const cli = b.addExecutable(.{
        .name = "lanceql",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.gpu", .module = gpu_mod },
                .{ .name = "lanceql.query", .module = query_mod },
                .{ .name = "lanceql.table", .module = table_mod },
                .{ .name = "lanceql.parquet_table", .module = parquet_table_mod },
                .{ .name = "lanceql.delta_table", .module = delta_table_mod },
                .{ .name = "lanceql.iceberg_table", .module = iceberg_table_mod },
                .{ .name = "lanceql.arrow_table", .module = arrow_table_mod },
                .{ .name = "lanceql.avro_table", .module = avro_table_mod },
                .{ .name = "lanceql.orc_table", .module = orc_table_mod },
                .{ .name = "lanceql.xlsx_table", .module = xlsx_table_mod },
                .{ .name = "lanceql.any_table", .module = any_table_mod },
                .{ .name = "lanceql.sql.ast", .module = sql_ast_mod },
                .{ .name = "lanceql.sql.lexer", .module = sql_lexer_mod },
                .{ .name = "lanceql.sql.parser", .module = sql_parser_mod },
                .{ .name = "lanceql.sql.executor", .module = sql_executor_mod },
                .{ .name = "lanceql.ai", .module = ai_mod },
            },
        }),
    });
    if (use_accelerate) {
        cli.root_module.linkFramework("Accelerate", .{});
    }
    // wgpu-native requires platform-specific libraries
    switch (target.result.os.tag) {
        .macos => {
            cli.root_module.linkFramework("Metal", .{});
            cli.root_module.linkFramework("QuartzCore", .{});
            cli.root_module.linkFramework("Foundation", .{});
            cli.root_module.linkFramework("CoreFoundation", .{});
            cli.linkLibC();
        },
        .linux => {
            // Vulkan backend on Linux
            cli.root_module.linkSystemLibrary("vulkan", .{});
            cli.linkLibC();
        },
        .windows => {
            // DirectX 12 backend on Windows (d3d12, dxgi)
            cli.root_module.linkSystemLibrary("d3d12", .{});
            cli.root_module.linkSystemLibrary("dxgi", .{});
            cli.root_module.linkSystemLibrary("dxguid", .{});
            cli.linkLibC();
        },
        else => {},
    }
    // ONNX Runtime for embedding support
    if (onnx_path) |path| {
        const lib_path = std.fmt.allocPrint(b.allocator, "{s}/lib", .{path}) catch @panic("OOM");
        const include_path = std.fmt.allocPrint(b.allocator, "{s}/include", .{path}) catch @panic("OOM");
        cli.root_module.addLibraryPath(.{ .cwd_relative = lib_path });
        cli.root_module.addIncludePath(.{ .cwd_relative = include_path });
        cli.root_module.linkSystemLibrary("onnxruntime", .{});
        cli.linkLibC();
    }
    // Pass build options to CLI (for ONNX availability check)
    cli.root_module.addOptions("build_options", build_options);
    b.installArtifact(cli);
    const run_cli = b.addRunArtifact(cli);
    run_cli.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cli.addArgs(args);
    }
    const cli_step = b.step("cli", "Build and run LanceQL CLI");
    cli_step.dependOn(&run_cli.step);

    // SQL executor tests
    const test_sql_executor = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_sql_executor.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql.table", .module = table_mod },
                .{ .name = "lanceql.sql.ast", .module = sql_ast_mod },
                .{ .name = "lanceql.sql.parser", .module = sql_parser_mod },
                .{ .name = "lanceql.sql.executor", .module = sql_executor_mod },
                .{ .name = "lanceql.sql.column_deps", .module = sql_column_deps_mod },
                .{ .name = "lanceql.logic_table", .module = logic_table_mod },
            },
        }),
    });

    const run_test_sql_executor = b.addRunArtifact(test_sql_executor);
    test_step.dependOn(&run_test_sql_executor.step);

    // Column dependency extraction tests
    const test_column_deps = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/sql/column_deps.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ast", .module = sql_ast_mod },
            },
        }),
    });
    const run_test_column_deps = b.addRunArtifact(test_column_deps);
    test_step.dependOn(&run_test_column_deps.step);

    // Batch codegen tests
    const test_batch_codegen = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/sql/batch_codegen.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ast", .module = sql_ast_mod },
                .{ .name = "column_deps", .module = sql_column_deps_mod },
            },
        }),
    });
    const run_test_batch_codegen = b.addRunArtifact(test_batch_codegen);
    test_step.dependOn(&run_test_batch_codegen.step);

    const test_sql_step = b.step("test-sql", "Run SQL executor tests");
    test_sql_step.dependOn(&run_test_sql_executor.step);
    test_sql_step.dependOn(&run_test_column_deps.step);
    test_sql_step.dependOn(&run_test_batch_codegen.step);

    // @logic_table FFI tests - verify compiled Python functions work via C FFI
    const test_logic_table_ffi = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests/test_logic_table_ffi.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    // Link against the compiled @logic_table static library
    test_logic_table_ffi.addObjectFile(b.path("lib/simple_logic_table.a"));
    // Link system libraries that the .a depends on
    test_logic_table_ffi.linkLibC();

    const run_test_logic_table_ffi = b.addRunArtifact(test_logic_table_ffi);
    const test_logic_table_ffi_step = b.step("test-logic-table-ffi", "Run @logic_table FFI tests (compiled Python functions)");
    test_logic_table_ffi_step.dependOn(&run_test_logic_table_ffi.step);

    // JIT codegen tests - metal0 integration and JIT compilation
    const test_codegen = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/codegen/metal0_jit.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "metal0", .module = metal0_mod },
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.proto", .module = proto_mod },
                .{ .name = "lanceql.query.expr", .module = query_expr_mod },
            },
        }),
    });
    const run_test_codegen = b.addRunArtifact(test_codegen);
    const test_codegen_step = b.step("test-codegen", "Run JIT codegen tests (metal0 integration)");
    test_codegen_step.dependOn(&run_test_codegen.step);
    test_step.dependOn(&run_test_codegen.step);

    // AI module tests (TinyBERT native inference)
    const test_ai = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_ai_native.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql.ai", .module = ai_mod },
            },
        }),
    });
    const run_test_ai = b.addRunArtifact(test_ai);
    const test_ai_step = b.step("test-ai", "Run AI module tests (TinyBERT native inference)");
    test_ai_step.dependOn(&run_test_ai.step);
    // Don't add to main test step - requires model file

    // Stress tests - large datasets, memory pressure, edge cases
    const test_stress = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/stress/stress_tests.zig"),
            .target = target,
            .optimize = .ReleaseFast, // Use ReleaseFast for performance tests
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.io", .module = io_mod },
                .{ .name = "lanceql.table", .module = table_mod },
            },
        }),
    });

    const run_test_stress = b.addRunArtifact(test_stress);
    const test_stress_step = b.step("test-stress", "Run stress tests (large datasets, memory, edge cases)");
    test_stress_step.dependOn(&run_test_stress.step);

    // === Conformance Tests (PyLance reference) ===
    // Python-based conformance tests that validate against PyLance
    // Requires: pip install pylance pyarrow numpy pytest
    const conformance_generate = b.addSystemCommand(&.{ "python3", "tests/conformance/generate_sql_fixtures.py" });
    const conformance_generate_step = b.step("conformance-generate", "Generate SQL conformance fixtures from PyLance");
    conformance_generate_step.dependOn(&conformance_generate.step);

    const conformance_vector = b.addSystemCommand(&.{ "pytest", "tests/conformance/test_vector_conformance.py", "-v" });
    const conformance_vector_step = b.step("conformance-vector", "Run vector algorithm conformance tests");
    conformance_vector_step.dependOn(&conformance_vector.step);

    const conformance_fuzz = b.addSystemCommand(&.{ "python3", "tests/conformance/fuzz_sql.py", "--iterations", "100" });
    const conformance_fuzz_step = b.step("conformance-fuzz", "Run SQL differential fuzzing (quick mode)");
    conformance_fuzz_step.dependOn(&conformance_fuzz.step);

    // Combined conformance test step
    const conformance_step = b.step("test-conformance", "Run all PyLance conformance tests");
    conformance_step.dependOn(&conformance_generate.step);
    conformance_step.dependOn(&conformance_vector.step);

    // === WASM Build ===
    // Browser build (default): zig build wasm
    // Worker build (with AI):  zig build wasm -Denable_ai=true
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .cpu_features_add = std.Target.wasm.featureSet(&.{.simd128}),
    });

    // Build options module for conditional compilation
    const wasm_options = b.addOptions();
    wasm_options.addOption(bool, "enable_ai", enable_ai);

    const wasm = b.addExecutable(.{
        .name = "lanceql",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasm.zig"),
            .target = wasm_target,
            .optimize = .ReleaseSmall,
            .imports = &.{
                // DuckDB-style vectorized query engine (shared with native)
                .{ .name = "vector_engine", .module = vector_engine_mod },
                // Build options for conditional AI compilation
                .{ .name = "build_options", .module = wasm_options.createModule() },
            },
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;

    const wasm_step = b.step("wasm", "Build WASM module (browser, no AI)");
    const install_wasm = b.addInstallArtifact(wasm, .{});
    wasm_step.dependOn(&install_wasm.step);

    // === WASM Tests (Node.js) ===
    // Requires: node tests/wasm_test.mjs
    // This step builds WASM first, then runs the Node.js test suite
    const wasm_test = b.addSystemCommand(&.{ "node", "tests/wasm_test.mjs" });
    wasm_test.step.dependOn(&install_wasm.step);

    const wasm_test_step = b.step("test-wasm", "Run WASM parser tests (requires Node.js)");
    wasm_test_step.dependOn(&wasm_test.step);

    // Add to main test step
    test_step.dependOn(&wasm_test.step);

    // Arrow C Data Interface module for zero-copy Python interop
    const arrow_c_mod = b.addModule("arrow_c", .{
        .root_source_file = b.path("src/arrow_c.zig"),
    });

    // === Native Shared Library for Python ===
    const lib = b.addLibrary(.{
        .name = "lanceql",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/python.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql", .module = lanceql_mod },
                .{ .name = "lanceql.table", .module = table_mod },
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.proto", .module = proto_mod },
                .{ .name = "lanceql.io", .module = io_mod },
                .{ .name = "lanceql.encoding", .module = encoding_mod },
                .{ .name = "lanceql.value", .module = value_mod },
                .{ .name = "lanceql.gpu", .module = gpu_mod },
                .{ .name = "arrow_c", .module = arrow_c_mod },
            },
        }),
    });

    // Link Accelerate framework on macOS for SIMD acceleration
    if (use_accelerate) {
        lib.root_module.linkFramework("Accelerate", .{});
    }

    const lib_step = b.step("lib", "Build native shared library for Python");
    const install_lib = b.addInstallArtifact(lib, .{});
    lib_step.dependOn(&install_lib.step);

    // === Node.js Shared Library ===
    const nodejs_lib = b.addLibrary(.{
        .name = "lanceql",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/nodejs.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lanceql.format", .module = format_mod },
                .{ .name = "lanceql.io", .module = io_mod },
                .{ .name = "lanceql.proto", .module = proto_mod },
                .{ .name = "lanceql.encoding", .module = encoding_mod },
                .{ .name = "lanceql.table", .module = table_mod },
                .{ .name = "lanceql.value", .module = value_mod },
                .{ .name = "lanceql.sql.ast", .module = sql_ast_mod },
                .{ .name = "lanceql.sql.lexer", .module = sql_lexer_mod },
                .{ .name = "lanceql.sql.parser", .module = sql_parser_mod },
                .{ .name = "lanceql.sql.executor", .module = sql_executor_mod },
            },
        }),
    });

    const nodejs_lib_step = b.step("lib-nodejs", "Build native shared library for Node.js");
    const install_nodejs_lib = b.addInstallArtifact(nodejs_lib, .{});
    nodejs_lib_step.dependOn(&install_nodejs_lib.step);

    // Default build includes lib-nodejs
    b.default_step.dependOn(&install_nodejs_lib.step);

    // === Metal Shaders (macOS only) ===
    // Compiles .metal shader files to .metallib for GPU acceleration
    const metal_shaders_step = b.step("metal-shaders", "Build Metal shaders (macOS only)");
    if (target.result.os.tag == .macos) {
        // Ensure zig-out directory exists
        const mkdir = b.addSystemCommand(&.{ "mkdir", "-p", "zig-out" });

        // Compile metal0 kernel shaders
        const metal_kernels = b.addSystemCommand(&.{
            "xcrun", "-sdk", "macosx", "metal",
            "-c", "deps/metal0/packages/runtime/src/metal/kernels/metal_kernels.metal",
            "-o", "zig-out/metal_kernels.air",
        });
        metal_kernels.step.dependOn(&mkdir.step);

        const matmul_kernels = b.addSystemCommand(&.{
            "xcrun", "-sdk", "macosx", "metal",
            "-c", "deps/metal0/packages/metal/kernels/matmul.metal",
            "-o", "zig-out/matmul.air",
        });
        matmul_kernels.step.dependOn(&mkdir.step);

        // Link into metallib
        const metallib = b.addSystemCommand(&.{
            "xcrun", "-sdk", "macosx", "metallib",
            "zig-out/metal_kernels.air",
            "zig-out/matmul.air",
            "-o", "zig-out/lanceql.metallib",
        });
        metallib.step.dependOn(&metal_kernels.step);
        metallib.step.dependOn(&matmul_kernels.step);

        metal_shaders_step.dependOn(&metallib.step);
    }
    // On non-macOS, step exists but does nothing

    // === Cross-compilation targets for NPM prebuilds ===
    const prebuild_step = b.step("prebuild", "Build prebuilt binaries for all platforms");

    // Cross-compilation targets
    const cross_targets = [_]struct {
        query: std.Target.Query,
        name: []const u8,
    }{
        .{ .query = .{ .cpu_arch = .x86_64, .os_tag = .macos }, .name = "darwin-x64" },
        .{ .query = .{ .cpu_arch = .aarch64, .os_tag = .macos }, .name = "darwin-arm64" },
        .{ .query = .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu }, .name = "linux-x64" },
        .{ .query = .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu }, .name = "linux-arm64" },
        .{ .query = .{ .cpu_arch = .x86_64, .os_tag = .windows }, .name = "win32-x64" },
    };

    for (cross_targets) |cross_target| {
        const resolved_target = b.resolveTargetQuery(cross_target.query);

        const cross_lib = b.addLibrary(.{
            .name = "lanceql",
            .linkage = .dynamic,
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/nodejs.zig"),
                .target = resolved_target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "lanceql.format", .module = format_mod },
                    .{ .name = "lanceql.io", .module = io_mod },
                    .{ .name = "lanceql.proto", .module = proto_mod },
                    .{ .name = "lanceql.encoding", .module = encoding_mod },
                    .{ .name = "lanceql.table", .module = table_mod },
                    .{ .name = "lanceql.value", .module = value_mod },
                    .{ .name = "lanceql.sql.ast", .module = sql_ast_mod },
                    .{ .name = "lanceql.sql.lexer", .module = sql_lexer_mod },
                    .{ .name = "lanceql.sql.parser", .module = sql_parser_mod },
                    .{ .name = "lanceql.sql.executor", .module = sql_executor_mod },
                },
            }),
        });

        // Install to prebuilds/{platform}/
        const install_cross = b.addInstallArtifact(cross_lib, .{
            .dest_dir = .{ .override = .{ .custom = b.fmt("prebuilds/{s}", .{cross_target.name}) } },
        });
        prebuild_step.dependOn(&install_cross.step);

        // Also create individual platform steps
        const platform_step = b.step(b.fmt("prebuild-{s}", .{cross_target.name}), b.fmt("Build for {s}", .{cross_target.name}));
        platform_step.dependOn(&install_cross.step);
    }
}
