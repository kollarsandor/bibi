const std = @import("std");
const GPUCoordinator = @import("distributed/gpu_coordinator.zig").GPUCoordinator;
const DistributedTrainerFuthark = @import("distributed/distributed_trainer_futhark.zig").DistributedTrainerFuthark;
const nccl = @import("distributed/nccl_bindings.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();

    try stdout.writeAll("============================================================\n");
    try stdout.writeAll("JAIDE v40 GPU Training (Single H100, Futhark Acceleration)\n");
    try stdout.writeAll("============================================================\n");
    try stdout.writeAll("GPU: NVIDIA H100 (80GB)\n");
    try stdout.writeAll("Precision: f16 (Futhark CUDA kernels)\n");
    try stdout.writeAll("Mode: Single-GPU (WORLD_SIZE=1)\n");
    try stdout.writeAll("============================================================\n\n");

    var nccl_id: nccl.ncclUniqueId = undefined;
    const result = nccl.ncclGetUniqueId(&nccl_id);
    if (result != .ncclSuccess) {
        try stdout.writeAll("❌ Failed to generate NCCL ID\n");
        return error.NCCLGetUniqueIdFailed;
    }

    var coordinator = try GPUCoordinator.init(allocator, 1, 0, nccl_id);
    defer coordinator.deinit();

    try stdout.writeAll("[✅] GPU coordinator initialized\n");

    const model_dim: usize = 2048;
    const num_layers: usize = 48;
    const local_batch_size: usize = 32;
    const num_epochs: usize = 10;

    var trainer = try DistributedTrainerFuthark.init(
        allocator,
        &coordinator,
        model_dim,
        local_batch_size,
    );
    defer trainer.deinit();

    try stdout.print("[✅] Futhark GPU trainer initialized (model_dim={d}, layers={d}, batch={d})\n",
        .{model_dim, num_layers, local_batch_size});

    const dataset_path = "/app/arxiv_hungarian_dataset.jsonl";
    try stdout.print("\n[DATA] Loading dataset from {s}\n", .{dataset_path});

    const samples = try trainer.loadDataset(dataset_path);
    defer {
        for (samples) |sample| {
            allocator.free(sample);
        }
        allocator.free(samples);
    }

    try stdout.print("[✅] Loaded {d} training samples\n\n", .{samples.len});

    try stdout.writeAll("============================================================\n");
    try stdout.print("Starting {d}-epoch GPU training loop\n", .{num_epochs});
    try stdout.writeAll("============================================================\n\n");

    var epoch: usize = 0;
    while (epoch < num_epochs) : (epoch += 1) {
        const epoch_start = std.time.milliTimestamp();

        try stdout.print("Epoch {d}/{d}: GPU forward/backward with Futhark kernels...\n",
            .{epoch + 1, num_epochs});

        const loss = try trainer.trainEpoch(samples);

        const epoch_duration = std.time.milliTimestamp() - epoch_start;
        try stdout.print("  Loss: {d:.6} | Time: {d}ms\n", .{loss, epoch_duration});
    }

    try stdout.writeAll("\n============================================================\n");
    try stdout.writeAll("Saving checkpoints to /mnt/checkpoints/\n");
    try stdout.writeAll("============================================================\n");

    try trainer.saveCheckpoint("/mnt/checkpoints");

    try stdout.writeAll("\n[✅] GPU training completed successfully!\n");
    try stdout.writeAll("============================================================\n");
}
