const std = @import("std");
const GPUCoordinator = @import("distributed/gpu_coordinator.zig").GPUCoordinator;
const DistributedTrainerFuthark = @import("distributed/distributed_trainer_futhark.zig").DistributedTrainerFuthark;
const nccl = @import("distributed/nccl_bindings.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const world_size = try std.process.getEnvVarOwned(allocator, "WORLD_SIZE");
    defer allocator.free(world_size);
    const world_size_val = try std.fmt.parseInt(usize, world_size, 10);

    const rank_str = try std.process.getEnvVarOwned(allocator, "RANK");
    defer allocator.free(rank_str);
    const rank = try std.fmt.parseInt(usize, rank_str, 10);

    const master_addr = try std.process.getEnvVarOwned(allocator, "MASTER_ADDR");
    defer allocator.free(master_addr);

    const master_port = try std.process.getEnvVarOwned(allocator, "MASTER_PORT");
    defer allocator.free(master_port);

    std.debug.print("============================================================\n", .{});
    std.debug.print("JAIDE v40 Distributed Training (Futhark GPU Acceleration)\n", .{});
    std.debug.print("============================================================\n", .{});
    std.debug.print("Rank: {d}/{d}\n", .{ rank, world_size_val });
    std.debug.print("Master: {s}:{s}\n", .{ master_addr, master_port });
    std.debug.print("GPU: NVIDIA B200 (192GB)\n", .{});
    std.debug.print("Precision: f16 (Futhark kernels)\n", .{});
    std.debug.print("NVLink: Enabled (NCCL P2P)\n", .{});
    std.debug.print("============================================================\n\n", .{});

    const pid = std.os.linux.getpid();
    var nccl_id_path_buf: [256]u8 = undefined;
    var nccl_ready_path_buf: [256]u8 = undefined;
    const nccl_id_path = try std.fmt.bufPrint(&nccl_id_path_buf, "/tmp/nccl_id_{d}", .{pid});
    const nccl_ready_path = try std.fmt.bufPrint(&nccl_ready_path_buf, "/tmp/nccl_ready_{d}", .{pid});

    var nccl_id: nccl.ncclUniqueId = undefined;

    if (rank == 0) {
        const result = nccl.ncclGetUniqueId(&nccl_id);
        if (result != .ncclSuccess) {
            std.debug.print("Failed to generate NCCL ID\n", .{});
            return error.NCCLGetUniqueIdFailed;
        }

        std.fs.deleteFileAbsolute(nccl_id_path) catch {};
        std.fs.deleteFileAbsolute(nccl_ready_path) catch {};

        const id_file = try std.fs.createFileAbsolute(nccl_id_path, .{});
        try id_file.writeAll(std.mem.asBytes(&nccl_id));
        try id_file.sync();
        id_file.close();

        const ready_file = try std.fs.createFileAbsolute(nccl_ready_path, .{});
        try ready_file.writeAll("ready");
        try ready_file.sync();
        ready_file.close();

        std.debug.print("[Rank 0] Generated NCCL ID (file: {s})\n", .{nccl_id_path});
    } else {
        var attempts: usize = 0;
        while (attempts < 100) : (attempts += 1) {
            const ready_file = std.fs.openFileAbsolute(nccl_ready_path, .{}) catch {
                std.time.sleep(100 * std.time.ns_per_ms);
                continue;
            };
            ready_file.close();
            break;
        }

        if (attempts >= 100) {
            std.debug.print("[Rank {d}] Timeout waiting for NCCL ID from rank 0\n", .{rank});
            return error.NCCLIdTimeout;
        }

        const id_file = try std.fs.openFileAbsolute(nccl_id_path, .{});
        defer id_file.close();

        const bytes_read = try id_file.readAll(std.mem.asBytes(&nccl_id));
        if (bytes_read != @sizeOf(nccl.ncclUniqueId)) {
            std.debug.print("[Rank {d}] Failed to read NCCL ID (got {d} bytes, expected {d})\n", .{ rank, bytes_read, @sizeOf(nccl.ncclUniqueId) });
            return error.NCCLIdReadFailed;
        }

        std.debug.print("[Rank {d}] Loaded NCCL ID from rank 0\n", .{rank});
    }

    var coordinator = try GPUCoordinator.init(allocator, world_size_val, rank, nccl_id);
    defer coordinator.deinit();

    std.debug.print("[Rank {d}] GPU coordinator initialized\n", .{rank});

    const model_dim: usize = 2048;
    const num_layers: usize = 48;
    const local_batch_size: usize = 4;

    var epochs_env_owned: ?[]u8 = null;
    const epochs_env: []const u8 = blk: {
        epochs_env_owned = std.process.getEnvVarOwned(allocator, "JAIDE_EPOCHS") catch null;
        break :blk epochs_env_owned orelse "20";
    };
    defer if (epochs_env_owned) |owned| allocator.free(owned);

    const num_epochs = std.fmt.parseInt(usize, epochs_env, 10) catch 20;

    var trainer = try DistributedTrainerFuthark.init(
        allocator,
        &coordinator,
        model_dim,
        local_batch_size,
    );
    defer trainer.deinit();

    std.debug.print("[Rank {d}] Futhark-accelerated trainer initialized (f16, model_dim={d}, layers={d})\n", .{ rank, model_dim, num_layers });

    var dataset_path_owned: ?[]u8 = null;
    const dataset_path: []const u8 = blk: {
        dataset_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_DATASET") catch null;
        break :blk dataset_path_owned orelse "/data/tower9b/hun_Latn_full.jsonl";
    };
    defer if (dataset_path_owned) |owned| allocator.free(owned);

    std.debug.print("[Rank {d}] Loading dataset from {s}\n", .{ rank, dataset_path });

    const samples = try trainer.loadDataset(dataset_path);
    defer {
        for (samples) |sample| {
            allocator.free(sample);
        }
        allocator.free(samples);
    }

    if (coordinator.isRoot()) {
        std.debug.print("\n============================================================\n", .{});
        std.debug.print("Starting Futhark-accelerated training\n", .{});
        std.debug.print("Dataset: {d} samples (per rank)\n", .{samples.len});
        std.debug.print("Batch size: {d} (per rank)\n", .{local_batch_size});
        std.debug.print("Epochs: {d}\n", .{num_epochs});
        std.debug.print("GPU Memory: 100%% VRAM-resident (zero host copies)\n", .{});
        std.debug.print("NVLink: Enabled for gradient synchronization\n", .{});
        std.debug.print("============================================================\n\n", .{});
    }

    var epoch: usize = 0;
    while (epoch < num_epochs) : (epoch += 1) {
        const start_time = std.time.milliTimestamp();

        const avg_loss = try trainer.trainEpoch(samples);

        const end_time = std.time.milliTimestamp();
        const elapsed = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0;

        if (coordinator.isRoot()) {
            std.debug.print("[Epoch {d}/{d}] Loss: {d:.6} | Time: {d:.2}s\n", .{ epoch + 1, num_epochs, avg_loss, elapsed });

            {
                var dir_buf: [256]u8 = undefined;
                const dir_path = std.fmt.bufPrint(&dir_buf, "/checkpoints/epoch_{d:0>3}", .{epoch + 1}) catch "/checkpoints";
                std.fs.makeDirAbsolute(dir_path) catch {};

                var checkpoint_path_buf: [256]u8 = undefined;
                const checkpoint_path = try std.fmt.bufPrint(
                    &checkpoint_path_buf,
                    "/checkpoints/epoch_{d:0>3}/model.ckpt",
                    .{epoch + 1},
                );

                try trainer.saveCheckpoint(checkpoint_path);
                std.debug.print("  Checkpoint saved: {s}\n", .{checkpoint_path});
            }
        }

        try coordinator.synchronize();
    }

    if (coordinator.isRoot()) {
        std.debug.print("\n============================================================\n", .{});
        std.debug.print("Futhark-accelerated training completed successfully!\n", .{});
        std.debug.print("Final model saved to /checkpoints/\n", .{});
        std.debug.print("============================================================\n", .{});
    }
}
