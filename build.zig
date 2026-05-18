const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gpu_enabled = b.option(bool, "gpu", "Enable GPU/CUDA acceleration via Futhark CUDA backend") orelse false;

    const build_options = b.addOptions();
    build_options.addOption(bool, "gpu_acceleration", gpu_enabled);

    const futhark_c = b.path("src/hw/accel/futhark_kernels.c");
    const futhark_include = b.path("src/hw/accel");

    const main_exe = b.addExecutable(.{
        .name = "jaide",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_exe.linkLibC();
    main_exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
    main_exe.addIncludePath(futhark_include);
    main_exe.root_module.addOptions("build_options", build_options);
    b.installArtifact(main_exe);

    const inference_server_exe = b.addExecutable(.{
        .name = "jaide-inference-server",
        .root_source_file = b.path("src/inference_server_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    inference_server_exe.linkLibC();
    inference_server_exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
    inference_server_exe.addIncludePath(futhark_include);
    inference_server_exe.root_module.addOptions("build_options", build_options);
    b.installArtifact(inference_server_exe);

    if (gpu_enabled) {
        const distributed_exe = b.addExecutable(.{
            .name = "jaide-distributed",
            .root_source_file = b.path("src/main_distributed.zig"),
            .target = target,
            .optimize = optimize,
        });
        distributed_exe.linkLibC();
        distributed_exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
        distributed_exe.addIncludePath(futhark_include);
        distributed_exe.root_module.addOptions("build_options", build_options);
        b.installArtifact(distributed_exe);

        const run_distributed_cmd = b.addRunArtifact(distributed_exe);
        run_distributed_cmd.step.dependOn(&distributed_exe.step);
        const run_distributed_step = b.step("run-distributed", "Run the distributed trainer");
        run_distributed_step.dependOn(&run_distributed_cmd.step);

        const distributed_futhark_exe = b.addExecutable(.{
            .name = "jaide-distributed-futhark",
            .root_source_file = b.path("src/main_distributed_futhark.zig"),
            .target = target,
            .optimize = optimize,
        });
        distributed_futhark_exe.linkLibC();
        distributed_futhark_exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
        distributed_futhark_exe.addIncludePath(futhark_include);
        distributed_futhark_exe.root_module.addOptions("build_options", build_options);
        b.installArtifact(distributed_futhark_exe);

        const gpu_exe = b.addExecutable(.{
            .name = "jaide-gpu",
            .root_source_file = b.path("src/main_gpu.zig"),
            .target = target,
            .optimize = optimize,
        });
        gpu_exe.linkLibC();
        gpu_exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
        gpu_exe.addIncludePath(futhark_include);
        gpu_exe.root_module.addOptions("build_options", build_options);
        b.installArtifact(gpu_exe);
    }

    const run_cmd = b.addRunArtifact(main_exe);
    run_cmd.step.dependOn(&main_exe.step);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the JAIDE main executable");
    run_step.dependOn(&run_cmd.step);

    const main_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_tests.linkLibC();
    main_tests.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
    main_tests.addIncludePath(futhark_include);
    main_tests.root_module.addOptions("build_options", build_options);
    const run_main_tests = b.addRunArtifact(main_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_main_tests.step);

    const tensor_tests = b.addTest(.{
        .root_source_file = b.path("src/core/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_tensor_tests = b.addRunArtifact(tensor_tests);
    const tensor_test_step = b.step("test-tensor", "Run tensor tests");
    tensor_test_step.dependOn(&run_tensor_tests.step);

    const memory_tests = b.addTest(.{
        .root_source_file = b.path("src/core/memory.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_memory_tests = b.addRunArtifact(memory_tests);
    const memory_test_step = b.step("test-memory", "Run memory tests");
    memory_test_step.dependOn(&run_memory_tests.step);
}
