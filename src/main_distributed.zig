const std = @import("std");
const GPUCoordinator = @import("distributed/gpu_coordinator.zig").GPUCoordinator;
const DistributedTrainer = @import("distributed/distributed_trainer.zig").DistributedTrainer;
const QuantumTrainingConfig = @import("distributed/distributed_trainer.zig").QuantumTrainingConfig;
const HybridStepResult = @import("distributed/distributed_trainer.zig").HybridStepResult;
const nccl = @import("distributed/nccl_bindings.zig");
const core_relational = @import("core_relational/mod.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const world_size_str = std.posix.getenv("WORLD_SIZE") orelse "8";
    const rank_str = std.posix.getenv("RANK") orelse "0";
    const master_addr = std.posix.getenv("MASTER_ADDR") orelse "127.0.0.1";
    const master_port = std.posix.getenv("MASTER_PORT") orelse "29500";

    const ibm_quantum_crn = std.posix.getenv("IBM_QUANTUM_CRN");
    const ibm_quantum_api_key = std.posix.getenv("IBM_QUANTUM_API_KEY");

    const quantum_enabled = ibm_quantum_crn != null and ibm_quantum_api_key != null;

    const world_size = try std.fmt.parseInt(usize, world_size_str, 10);
    const rank = try std.fmt.parseInt(usize, rank_str, 10);

    std.debug.print("============================================================\n", .{});
    std.debug.print("JAIDE v40 Distributed Training - Quantum-GPU Hybrid\n", .{});
    std.debug.print("============================================================\n", .{});
    std.debug.print("Rank: {d}/{d}\n", .{rank, world_size});
    std.debug.print("Master: {s}:{s}\n", .{master_addr, master_port});
    std.debug.print("GPU: NVIDIA B200 (192GB)\n", .{});
    std.debug.print("Quantum Backend: {s}\n", .{if (quantum_enabled) "IBM Quantum (enabled)" else "Disabled"});
    std.debug.print("============================================================\n\n", .{});

    var nccl_id: nccl.ncclUniqueId = undefined;

    if (rank == 0) {
        const result = nccl.ncclGetUniqueId(&nccl_id);
        if (result != .ncclSuccess) {
            std.debug.print("Failed to generate NCCL ID\n", .{});
            return error.NCCLGetUniqueIdFailed;
        }

        try DistributedTrainer.writeNcclId(allocator, &nccl_id);
        std.debug.print("[Rank 0] Generated and shared NCCL ID via NCCL_ID_FILE\n", .{});
    } else {
        DistributedTrainer.readNcclId(allocator, &nccl_id, 10000) catch |err| {
            std.debug.print("[Rank {d}] Failed to read NCCL ID: {}\n", .{ rank, err });
            return err;
        };
        std.debug.print("[Rank {d}] Loaded NCCL ID from rank 0\n", .{rank});
    }

    var coordinator = try GPUCoordinator.init(allocator, world_size, rank, nccl_id);
    defer coordinator.deinit();

    std.debug.print("[Rank {d}] GPU coordinator initialized\n", .{rank});

    try coordinator.barrier();
    std.debug.print("[Rank {d}] All ranks synchronized\n", .{rank});

    const model_dim: usize = 2048;
    const num_layers: usize = 48;
    const vocab_size: usize = 65536;
    const local_batch_size: usize = 4;
    const num_epochs: usize = 10;

    var trainer: DistributedTrainer = undefined;

    if (quantum_enabled) {
        const quantum_config = QuantumTrainingConfig{
            .ibm_crn = ibm_quantum_crn.?,
            .ibm_api_key = ibm_quantum_api_key.?,
            .num_qubits = 8,
            .vqe_layers = 2,
            .quantum_shots = 1024,
            .enable_hybrid = true,
            .enable_verification = true,
            .quantum_learning_rate = 0.01,
            .max_quantum_iterations = 100,
            .verification_frequency = 10,
        };

        trainer = try DistributedTrainer.initWithQuantum(
            allocator,
            &coordinator,
            model_dim,
            num_layers,
            vocab_size,
            local_batch_size,
            quantum_config
        );

        if (coordinator.isRoot()) {
            std.debug.print("[Rank 0] Quantum-GPU hybrid trainer initialized\n", .{});
            std.debug.print("  - IBM Quantum CRN: {s}\n", .{ibm_quantum_crn.?});
            std.debug.print("  - Qubits: {d}, VQE Layers: {d}\n", .{quantum_config.num_qubits, quantum_config.vqe_layers});
            std.debug.print("  - Quantum Shots: {d}\n", .{quantum_config.quantum_shots});
            std.debug.print("  - Hybrid Training: {s}\n", .{if (quantum_config.enable_hybrid) "Enabled" else "Disabled"});
            std.debug.print("  - Formal Verification: {s}\n", .{if (quantum_config.enable_verification) "Enabled" else "Disabled"});
        }
    } else {
        trainer = try DistributedTrainer.init(
            allocator,
            &coordinator,
            model_dim,
            num_layers,
            vocab_size,
            local_batch_size
        );

        if (coordinator.isRoot()) {
            std.debug.print("[Rank 0] Classical distributed trainer initialized\n", .{});
            std.debug.print("  - Quantum backend not configured (set IBM_QUANTUM_CRN and IBM_QUANTUM_API_KEY to enable)\n", .{});
        }
    }
    defer trainer.deinit();

    std.debug.print("[Rank {d}] Trainer initialized\n", .{rank});
    std.debug.print("[Rank {d}] Model: {d} layers, {d} dim\n", .{rank, num_layers, model_dim});

    try DistributedTrainer.ensureCheckpointDirExists(allocator);
    const checkpoint_path = try DistributedTrainer.getCheckpointPath(allocator, "latest.ckpt");
    defer allocator.free(checkpoint_path);

    trainer.loadCheckpoint(checkpoint_path) catch |err| {
        if (coordinator.isRoot()) {
            std.debug.print("No checkpoint found or error loading: {}\n", .{err});
            std.debug.print("Starting from scratch\n", .{});
        }
    };

    const dataset_path = "/mnt/datasets/arxiv_hungarian_dataset.jsonl";
    const samples = try trainer.loadDataset(dataset_path);
    defer {
        for (samples) |sample| {
            allocator.free(sample);
        }
        allocator.free(samples);
    }

    if (coordinator.isRoot()) {
        std.debug.print("Loaded {d} training samples\n", .{samples.len});
    }

    try coordinator.barrier();

    if (coordinator.isRoot()) {
        std.debug.print("\n============================================================\n", .{});
        std.debug.print("Starting {s} Training\n", .{if (quantum_enabled) "Quantum-GPU Hybrid" else "Classical Distributed"});
        std.debug.print("============================================================\n\n", .{});
    }

    var epoch: usize = 0;
    while (epoch < num_epochs) : (epoch += 1) {
        const epoch_start_time = std.time.milliTimestamp();

        if (coordinator.isRoot()) {
            std.debug.print("\n[Epoch {d}/{d}]\n", .{epoch + 1, num_epochs});
        }

        if (quantum_enabled) {
            const hybrid_result: HybridStepResult = try trainer.trainEpochHybrid(samples);

            const epoch_end_time = std.time.milliTimestamp();
            const epoch_duration = epoch_end_time - epoch_start_time;

            if (coordinator.isRoot()) {
                std.debug.print("[Epoch {d}] Classical Loss: {d:.4}\n", .{epoch + 1, hybrid_result.classical_loss});
                std.debug.print("[Epoch {d}] Quantum Loss: {d:.6}\n", .{epoch + 1, hybrid_result.quantum_loss});
                std.debug.print("[Epoch {d}] Combined Loss: {d:.6}\n", .{epoch + 1, hybrid_result.combined_loss});
                std.debug.print("[Epoch {d}] Quantum Contribution: {d:.6}\n", .{epoch + 1, hybrid_result.quantum_contribution});
                std.debug.print("[Epoch {d}] Gradient Norm: {d:.6}\n", .{epoch + 1, hybrid_result.gradient_norm});
                std.debug.print("[Epoch {d}] Verification: {s}\n", .{epoch + 1, if (hybrid_result.verification_passed) "PASSED" else "FAILED"});
                std.debug.print("[Epoch {d}] Duration: {d}ms\n", .{epoch + 1, epoch_duration});

                if (trainer.getQuantumStatistics()) |qstats| {
                    std.debug.print("\n[Quantum Statistics]\n", .{});
                    std.debug.print("  - Total Quantum Shots: {d}\n", .{qstats.total_shots});
                    std.debug.print("  - Successful Verifications: {d}\n", .{qstats.successful_verifications});
                    std.debug.print("  - Z-Runtime Memory Used: {d} bytes\n", .{qstats.z_runtime_memory_used});
                    std.debug.print("  - Z-Runtime Variables: {d}\n", .{qstats.z_runtime_variables});
                    std.debug.print("  - Verification Engine:\n", .{});
                    std.debug.print("      Total Verifications: {d}\n", .{qstats.ve_total_verifications});
                    std.debug.print("      Successful: {d}\n", .{qstats.ve_successful_verifications});
                    std.debug.print("      Invariant Count: {d}\n", .{qstats.ve_invariant_count});
                }

                try trainer.saveCheckpoint(checkpoint_path);
            }
        } else {
            const avg_loss = try trainer.trainEpoch(samples);

            const epoch_end_time = std.time.milliTimestamp();
            const epoch_duration = epoch_end_time - epoch_start_time;

            if (coordinator.isRoot()) {
                std.debug.print("[Epoch {d}] Average Loss: {d:.4}\n", .{epoch + 1, avg_loss});
                std.debug.print("[Epoch {d}] Duration: {d}ms\n", .{epoch + 1, epoch_duration});

                try trainer.saveCheckpoint(checkpoint_path);
            }
        }

        try coordinator.barrier();
    }

    if (coordinator.isRoot()) {
        std.debug.print("\n============================================================\n", .{});
        std.debug.print("Training Complete\n", .{});
        std.debug.print("============================================================\n", .{});

        if (quantum_enabled) {
            if (trainer.getQuantumStatistics()) |final_stats| {
                std.debug.print("\n[Final Quantum Training Statistics]\n", .{});
                std.debug.print("  - Total Quantum Shots Executed: {d}\n", .{final_stats.total_shots});
                std.debug.print("  - Total Successful Verifications: {d}\n", .{final_stats.successful_verifications});
                std.debug.print("  - Quantum Enabled: {s}\n", .{if (final_stats.quantum_enabled) "Yes" else "No"});
                std.debug.print("  - Hybrid Enabled: {s}\n", .{if (final_stats.hybrid_enabled) "Yes" else "No"});
                std.debug.print("  - Verification Enabled: {s}\n", .{if (final_stats.verification_enabled) "Yes" else "No"});
                std.debug.print("  - Final Z-Runtime Memory: {d} bytes\n", .{final_stats.z_runtime_memory_used});
                std.debug.print("  - Final Z-Runtime Variables: {d}\n", .{final_stats.z_runtime_variables});

                const success_rate = if (final_stats.ve_total_verifications > 0)
                    @as(f64, @floatFromInt(final_stats.ve_successful_verifications)) / @as(f64, @floatFromInt(final_stats.ve_total_verifications)) * 100.0
                else
                    0.0;
                std.debug.print("  - Verification Success Rate: {d:.2}%\n", .{success_rate});
            }
        }

        const final_model_path = DistributedTrainer.getCheckpointPath(allocator, "final_model.ckpt") catch |err| {
            std.debug.print("Failed to get final model path: {}\n", .{err});
            return err;
        };
        defer allocator.free(final_model_path);
        try trainer.saveCheckpoint(final_model_path);
        std.debug.print("Final model saved to: {s}\n", .{final_model_path});
    }

    try coordinator.barrier();

    std.debug.print("[Rank {d}] Exiting\n", .{rank});
}
