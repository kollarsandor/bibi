const std = @import("std");
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const accel = @import("../hw/accel/accel_interface.zig");
const RSFAccelerator = accel.RSFAccelerator;
const FutharkArray2DF16 = accel.FutharkArray2DF16;
const FutharkArray1DF16 = accel.FutharkArray1DF16;
const PinnedMemory = accel.PinnedMemory;

pub const TrainerConfig = struct {
    learning_rate: f32 = 0.001,
    momentum: f32 = 0.0,
    max_line_size: usize = 10 * 1024 * 1024,
    checkpoint_version: u32 = 4,
};

pub const DistributedTrainerFuthark = struct {
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    tokenizer: MGT,
    accelerator: RSFAccelerator,
    model_dim: usize,
    vocab_size: usize,
    local_batch_size: usize,
    global_step: u64,
    learning_rate: f32,
    momentum: f32,
    config: TrainerConfig,

    pub fn init(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
    ) !DistributedTrainerFuthark {
        return initWithConfig(allocator, coordinator, model_dim, local_batch_size, .{});
    }

    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
        config: TrainerConfig,
    ) !DistributedTrainerFuthark {
        if (model_dim == 0) return error.InvalidModelDim;
        if (model_dim % 2 != 0) return error.InvalidModelDim;
        if (local_batch_size == 0) return error.InvalidBatchSize;
        if (coordinator.world_size == 0) return error.InvalidWorldSize;
        if (coordinator.rank >= coordinator.world_size) return error.InvalidRank;
        if (config.max_line_size == 0) return error.InvalidMaxLineSize;
        if (config.checkpoint_version == 0) return error.InvalidCheckpointVersion;
        try validateHyperparameters(config.learning_rate, config.momentum);
        if (coordinator.world_size > 1 and config.momentum != 0.0) return error.UnsupportedDistributedMomentum;

        const vocab = &[_][]const u8{
            "a",     "about",   "all",   "also",  "and",   "as",    "at",
            "be",    "because", "but",   "by",    "can",   "come",  "could",
            "day",   "do",      "even",  "find",  "first", "for",   "from",
            "get",   "give",    "go",    "have",  "he",    "her",   "here",
            "him",   "his",     "how",   "i",     "if",    "in",    "into",
            "it",    "its",     "just",  "know",  "like",  "look",  "make",
            "man",   "many",    "me",    "more",  "my",    "new",   "no",
            "not",   "now",     "of",    "on",    "one",   "only",  "or",
            "other", "our",     "out",   "people", "say",  "see",   "she",
            "so",    "some",    "take",  "tell",  "than",  "that",  "the",
            "their", "them",    "then",  "there", "these", "they",  "thing",
            "think", "this",    "those", "time",  "to",    "two",   "up",
            "use",   "very",    "want",  "way",   "we",    "well",  "what",
            "when",  "which",   "who",   "will",  "with",  "would", "year",
            "you",   "your",
        };
        if (model_dim < vocab.len) return error.ModelDimTooSmallForVocabulary;
        const empty_anchors: []const []const u8 = &.{};

        var tokenizer = try MGT.init(allocator, vocab, empty_anchors);
        errdefer tokenizer.deinit();

        var accelerator = try RSFAccelerator.init(model_dim);
        errdefer accelerator.deinit();

        return DistributedTrainerFuthark{
            .allocator = allocator,
            .coordinator = coordinator,
            .tokenizer = tokenizer,
            .accelerator = accelerator,
            .model_dim = model_dim,
            .vocab_size = vocab.len,
            .local_batch_size = local_batch_size,
            .global_step = 0,
            .learning_rate = config.learning_rate,
            .momentum = config.momentum,
            .config = config,
        };
    }

    pub fn deinit(self: *DistributedTrainerFuthark) void {
        self.accelerator.sync() catch {};
        self.accelerator.deinit();
        self.tokenizer.deinit();
    }

    fn validateHyperparameters(learning_rate: f32, momentum: f32) !void {
        if (!std.math.isFinite(learning_rate)) return error.InvalidLearningRate;
        if (!std.math.isFinite(momentum)) return error.InvalidMomentum;
        if (learning_rate < 0.0 or learning_rate > 65504.0) return error.InvalidLearningRate;
        if (momentum < 0.0 or momentum >= 1.0) return error.InvalidMomentum;
    }

    fn openReadFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) {
            return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
        }
        return std.fs.cwd().openFile(path, .{ .mode = .read_only });
    }

    fn createWriteFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) {
            return std.fs.createFileAbsolute(path, .{ .mode = 0o600 });
        }
        return std.fs.cwd().createFile(path, .{ .mode = 0o600 });
    }

    fn writeF32(writer: anytype, value: f32) !void {
        try writer.writeInt(u32, @as(u32, @bitCast(value)), .little);
    }

    fn readF32(reader: anytype) !f32 {
        const bits = try reader.readInt(u32, .little);
        return @as(f32, @bitCast(bits));
    }

    fn isTokenizableText(self: *DistributedTrainerFuthark, text: []const u8) !bool {
        var token_list = std.ArrayList(u32).init(self.allocator);
        defer token_list.deinit();
        try self.tokenizer.encode(text, &token_list);
        return token_list.items.len > 0;
    }

    fn extractDatasetText(self: *DistributedTrainerFuthark, line: []const u8) !?[]const u8 {
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            line,
            .{ .allocate = .alloc_always },
        ) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => return null,
        };
        defer parsed.deinit();

        return switch (parsed.value) {
            .object => |obj| blk: {
                const text_value = obj.get("text") orelse break :blk null;
                break :blk switch (text_value) {
                    .string => |text| if (text.len > 0)
                        try self.allocator.dupe(u8, text)
                    else
                        null,
                    else => null,
                };
            },
            else => null,
        };
    }

    fn isUsableDatasetLine(self: *DistributedTrainerFuthark, line: []const u8) !bool {
        const maybe_text = try self.extractDatasetText(line);
        if (maybe_text) |text| {
            defer self.allocator.free(text);
            return try self.isTokenizableText(text);
        }
        return false;
    }

    fn appendDatasetRange(
        self: *DistributedTrainerFuthark,
        dataset_path: []const u8,
        start_valid_index: usize,
        count: usize,
        samples: *std.ArrayList([]const u8),
    ) !void {
        if (count == 0) {
            return;
        }

        const end_valid_index = try std.math.add(usize, start_valid_index, count);
        var appended: usize = 0;
        var valid_index: usize = 0;

        const load_file = openReadFile(dataset_path) catch |err| {
            std.debug.print("[Rank {d}] ERROR: Cannot open dataset: {}\n", .{ self.coordinator.rank, err });
            return err;
        };
        defer load_file.close();

        var load_buf_reader = std.io.bufferedReader(load_file.reader());
        var load_stream = load_buf_reader.reader();

        while (try load_stream.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
            defer self.allocator.free(line);

            const maybe_text = try self.extractDatasetText(line);
            if (maybe_text) |text_copy| {
                var keep = false;
                defer if (!keep) self.allocator.free(text_copy);

                if (!try self.isTokenizableText(text_copy)) {
                    continue;
                }

                if (valid_index >= start_valid_index and valid_index < end_valid_index) {
                    try samples.append(text_copy);
                    keep = true;
                    appended += 1;
                }
                valid_index += 1;

                if (appended == count) {
                    return;
                }
            }
        }

        return error.UnexpectedEndOfFile;
    }

    fn readWeightsFlat(self: *DistributedTrainerFuthark, matrix: anytype) ![]f16 {
        const rows = try matrix.values(&self.accelerator.ctx, self.allocator);
        defer {
            for (rows) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(rows);
        }

        if (rows.len != self.model_dim) {
            return error.InvalidWeightsShape;
        }

        const weight_count = try std.math.mul(usize, self.model_dim, self.model_dim);
        var flat = try self.allocator.alloc(f16, weight_count);
        errdefer self.allocator.free(flat);

        var idx: usize = 0;
        for (rows) |row| {
            if (row.len != self.model_dim) {
                return error.InvalidWeightsShape;
            }
            for (row) |value| {
                flat[idx] = value;
                idx += 1;
            }
        }

        if (idx != weight_count) {
            return error.InvalidWeightsShape;
        }

        return flat;
    }

    fn readBiasFlat(self: *DistributedTrainerFuthark, bias: anytype) ![]f16 {
        const values = try bias.values1D(&self.accelerator.ctx, self.allocator);
        errdefer self.allocator.free(values);
        if (values.len != self.model_dim) {
            self.allocator.free(values);
            return error.InvalidBiasShape;
        }
        return values;
    }

    fn allReduceFloat32Values(self: *DistributedTrainerFuthark, values: []f32) !void {
        if (values.len == 0 or self.coordinator.world_size <= 1) {
            return;
        }

        const byte_count = try std.math.mul(usize, values.len, @sizeOf(f32));
        const values_dev = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(values_dev);

        try self.coordinator.copyHostToDevice(values_dev, std.mem.sliceAsBytes(values), byte_count);
        try self.coordinator.allReduceFloat32(values_dev, values_dev, values.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(values), values_dev, byte_count);
        try self.coordinator.synchronize();
    }

    fn allReduceScalarF32(self: *DistributedTrainerFuthark, value: f32) !f32 {
        var values = [1]f32{value};
        try self.allReduceFloat32Values(values[0..]);
        return values[0];
    }

    fn averageDeltaInPlace(self: *DistributedTrainerFuthark, delta: []f16) !void {
        if (delta.len == 0 or self.coordinator.world_size <= 1) {
            return;
        }

        const byte_count = try std.math.mul(usize, delta.len, @sizeOf(f16));
        const delta_dev = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(delta_dev);

        try self.coordinator.copyHostToDevice(delta_dev, std.mem.sliceAsBytes(delta), byte_count);
        try self.coordinator.allReduceFloat16(delta_dev, delta_dev, delta.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(delta), delta_dev, byte_count);
        try self.coordinator.synchronize();

        const inv_world: f32 = 1.0 / @as(f32, @floatFromInt(self.coordinator.world_size));
        for (delta) |*value| {
            const scaled = @as(f32, @floatCast(value.*)) * inv_world;
            value.* = @floatCast(scaled);
        }
    }

    fn applyDelta(self: *DistributedTrainerFuthark, base: []const f16, delta: []const f16, which: enum { s, t }) !void {
        if (base.len != delta.len) {
            return error.InvalidWeightsShape;
        }
        const expected_len = try std.math.mul(usize, self.model_dim, self.model_dim);
        if (base.len != expected_len) {
            return error.InvalidWeightsShape;
        }

        var merged = try self.allocator.alloc(f16, base.len);
        defer self.allocator.free(merged);

        for (base, delta, 0..) |base_value, delta_value, idx| {
            const merged_value = @as(f32, @floatCast(base_value)) + @as(f32, @floatCast(delta_value));
            merged[idx] = @floatCast(merged_value);
        }

        switch (which) {
            .s => try self.accelerator.setWeightsS(merged, self.model_dim, self.model_dim),
            .t => try self.accelerator.setWeightsT(merged, self.model_dim, self.model_dim),
        }
    }

    pub fn loadDataset(self: *DistributedTrainerFuthark, dataset_path: []const u8) ![][]const u8 {
        if (self.coordinator.world_size == 0) return error.InvalidWorldSize;
        if (self.coordinator.rank >= self.coordinator.world_size) return error.InvalidRank;

        var total_line_count: usize = 0;
        var valid_sample_count: usize = 0;

        {
            const count_file = openReadFile(dataset_path) catch |err| {
                std.debug.print("[Rank {d}] ERROR: Cannot open dataset: {}\n", .{ self.coordinator.rank, err });
                return err;
            };
            defer count_file.close();

            var count_buf_reader = std.io.bufferedReader(count_file.reader());
            var count_stream = count_buf_reader.reader();

            while (try count_stream.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
                defer self.allocator.free(line);
                total_line_count = try std.math.add(usize, total_line_count, 1);
                if (try self.isUsableDatasetLine(line)) {
                    valid_sample_count = try std.math.add(usize, valid_sample_count, 1);
                }
            }
        }

        if (valid_sample_count == 0) {
            std.debug.print("[Rank {d}] ERROR: Dataset does not contain any usable samples\n", .{self.coordinator.rank});
            return error.EmptyDataset;
        }

        const base_per_rank = valid_sample_count / self.coordinator.world_size;
        const remainder = valid_sample_count % self.coordinator.world_size;
        const samples_per_rank = if (self.coordinator.rank < remainder) base_per_rank + 1 else base_per_rank;
        const start_valid_index = if (self.coordinator.rank < remainder)
            self.coordinator.rank * (base_per_rank + 1)
        else
            remainder * (base_per_rank + 1) + (self.coordinator.rank - remainder) * base_per_rank;

        var samples = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (samples.items) |sample| {
                self.allocator.free(sample);
            }
            samples.deinit();
        }

        if (samples_per_rank > 0) {
            try self.appendDatasetRange(dataset_path, start_valid_index, samples_per_rank, &samples);
        }

        if (samples.items.len != samples_per_rank) {
            return error.InvalidDatasetPartition;
        }

        if (self.coordinator.isRoot()) {
            std.debug.print("[Rank {d}] Loaded {d} samples on this rank from {d} usable samples across {d} lines\n", .{
                self.coordinator.rank,
                samples.items.len,
                valid_sample_count,
                total_line_count,
            });
        }

        return samples.toOwnedSlice();
    }

    pub fn trainEpoch(self: *DistributedTrainerFuthark, samples: [][]const u8) !f32 {
        if (self.local_batch_size == 0) return error.InvalidBatchSize;

        var total_loss: f32 = 0.0;
        var num_batches: usize = 0;

        const local_total: f32 = @floatFromInt(samples.len);
        const global_total = try self.allReduceScalarF32(local_total);
        _ = global_total;

        const local_batches_count: usize = (samples.len + self.local_batch_size - 1) / self.local_batch_size;
        var max_batches_local: f32 = @floatFromInt(local_batches_count);
        if (self.coordinator.world_size > 1) {
            var arr = [1]f32{max_batches_local};
            const byte_count = arr.len * @sizeOf(f32);
            const dev = try self.coordinator.allocDeviceMemory(byte_count);
            defer self.coordinator.freeDeviceMemory(dev);
            try self.coordinator.copyHostToDevice(dev, std.mem.sliceAsBytes(arr[0..]), byte_count);
            try self.coordinator.allReduceFloat32Max(dev, dev, arr.len);
            try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(arr[0..]), dev, byte_count);
            try self.coordinator.synchronize();
            max_batches_local = arr[0];
        }

        const target_batches: usize = @intFromFloat(max_batches_local);
        var batch_idx: usize = 0;
        var batch_start: usize = 0;
        while (batch_idx < target_batches) : (batch_idx += 1) {
            var batch: [][]const u8 = &.{};
            if (batch_start < samples.len) {
                const remaining = samples.len - batch_start;
                const batch_len = @min(self.local_batch_size, remaining);
                const batch_end = batch_start + batch_len;
                batch = samples[batch_start..batch_end];
                batch_start = batch_end;
            }

            const loss = try self.trainStepFuthark(batch);
            if (!std.math.isFinite(loss)) return error.InvalidLoss;
            if (batch.len > 0) {
                total_loss += loss;
                num_batches = try std.math.add(usize, num_batches, 1);
            }

            if (self.coordinator.isRoot() and self.global_step % 10 == 0) {
                std.debug.print("[Step {d}] Loss: {d:.4}\n", .{ self.global_step, loss });
            }

            self.global_step = try std.math.add(u64, self.global_step, 1);
        }

        var loss_and_count = [2]f32{ total_loss, @floatFromInt(num_batches) };
        try self.allReduceFloat32Values(loss_and_count[0..]);

        const global_loss_sum = loss_and_count[0];
        const global_batch_count = loss_and_count[1];

        if (global_batch_count < 1.0) {
            std.debug.print("[WARNING] No batches processed across all ranks\n", .{});
            return 0.0;
        }

        return global_loss_sum / global_batch_count;
    }

    pub fn trainStepFuthark(self: *DistributedTrainerFuthark, batch: [][]const u8) !f32 {
        var local_active_f: f32 = if (batch.len > 0) 1.0 else 0.0;
        if (self.coordinator.world_size > 1) {
            const active_count = try self.allReduceScalarF32(local_active_f);
            if (active_count == 0.0) return 0.0;
        } else {
            if (batch.len == 0) return 0.0;
        }

        var token_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        defer {
            for (token_lists.items) |*list| {
                list.deinit();
            }
            token_lists.deinit();
        }

        for (batch) |text| {
            var token_list = std.ArrayList(u32).init(self.allocator);
            errdefer token_list.deinit();
            try self.tokenizer.encode(text, &token_list);
            try token_lists.append(token_list);
        }

        var max_seq_len: usize = 0;
        for (token_lists.items) |list| {
            max_seq_len = @max(max_seq_len, list.items.len);
        }

        if (self.coordinator.world_size > 1) {
            var msl_arr = [1]f32{@floatFromInt(max_seq_len)};
            const byte_count = msl_arr.len * @sizeOf(f32);
            const dev = try self.coordinator.allocDeviceMemory(byte_count);
            defer self.coordinator.freeDeviceMemory(dev);
            try self.coordinator.copyHostToDevice(dev, std.mem.sliceAsBytes(msl_arr[0..]), byte_count);
            try self.coordinator.allReduceFloat32Max(dev, dev, msl_arr.len);
            try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(msl_arr[0..]), dev, byte_count);
            try self.coordinator.synchronize();
            const global_msl: usize = @intFromFloat(msl_arr[0]);
            if (global_msl == 0) return 0.0;
            max_seq_len = global_msl;
        } else {
            if (max_seq_len == 0) return 0.0;
        }

        const effective_batch_size: usize = if (batch.len == 0) 1 else batch.len;
        const batch_rows = try std.math.mul(usize, effective_batch_size, max_seq_len);
        const data_elements = try std.math.mul(usize, batch_rows, self.model_dim);
        const data_size = try std.math.mul(usize, data_elements, @sizeOf(f16));

        var pinned_input = try PinnedMemory.alloc(data_size);
        defer pinned_input.free();
        var pinned_target = try PinnedMemory.alloc(data_size);
        defer pinned_target.free();

        const input_f16_data = pinned_input.asSlice(f16) orelse return error.AllocationFailed;
        const target_f16_data = pinned_target.asSlice(f16) orelse return error.AllocationFailed;
        if (input_f16_data.len != data_elements or target_f16_data.len != data_elements) {
            return error.InvalidPinnedMemorySize;
        }
        @memset(input_f16_data, @as(f16, 0.0));
        @memset(target_f16_data, @as(f16, 0.0));

        var b_idx: usize = 0;
        while (b_idx < token_lists.items.len) : (b_idx += 1) {
            const list = token_lists.items[b_idx].items;
            if (list.len == 0) continue;
            var seq_idx: usize = 0;
            while (seq_idx < list.len) : (seq_idx += 1) {
                const token_index: usize = @intCast(list[seq_idx]);
                if (token_index >= self.vocab_size) return error.InvalidToken;
                if (token_index >= self.model_dim) return error.InvalidToken;

                const row_offset = try std.math.mul(usize, b_idx, max_seq_len);
                const row_index = try std.math.add(usize, row_offset, seq_idx);
                const base_idx = try std.math.mul(usize, row_index, self.model_dim);
                const final_idx = try std.math.add(usize, base_idx, token_index);
                if (final_idx >= input_f16_data.len) return error.IndexOutOfBounds;
                input_f16_data[final_idx] = @as(f16, 1.0);

                if (seq_idx + 1 < list.len) {
                    const next_token: usize = @intCast(list[seq_idx + 1]);
                    if (next_token >= self.vocab_size) return error.InvalidToken;
                    if (next_token >= self.model_dim) return error.InvalidToken;
                    const tgt_final = try std.math.add(usize, base_idx, next_token);
                    if (tgt_final >= target_f16_data.len) return error.IndexOutOfBounds;
                    target_f16_data[tgt_final] = @as(f16, 1.0);
                }
            }
        }

        var inputs = try FutharkArray2DF16.newFromFlat(&self.accelerator.ctx, input_f16_data, batch_rows, self.model_dim);
        defer inputs.free(&self.accelerator.ctx);

        var targets = try FutharkArray2DF16.newFromFlat(&self.accelerator.ctx, target_f16_data, batch_rows, self.model_dim);
        defer targets.free(&self.accelerator.ctx);

        const lr_f16: f16 = @floatCast(self.learning_rate);
        const mom_f16: f16 = @floatCast(self.momentum);

        if (self.coordinator.world_size <= 1) {
            const loss_f16 = try self.accelerator.trainingStep(&inputs, &targets, lr_f16, mom_f16);
            try self.accelerator.sync();
            const loss_f32: f32 = @floatCast(loss_f16);
            if (!std.math.isFinite(loss_f32)) return error.InvalidLoss;
            return loss_f32;
        }

        const weights_s_before = try self.readWeightsFlat(self.accelerator.weights_s);
        defer self.allocator.free(weights_s_before);
        const weights_t_before = try self.readWeightsFlat(self.accelerator.weights_t);
        defer self.allocator.free(weights_t_before);

        const loss_f16 = try self.accelerator.trainingStep(&inputs, &targets, lr_f16, mom_f16);
        try self.accelerator.sync();

        const weights_s_after = try self.readWeightsFlat(self.accelerator.weights_s);
        defer self.allocator.free(weights_s_after);
        const weights_t_after = try self.readWeightsFlat(self.accelerator.weights_t);
        defer self.allocator.free(weights_t_after);

        if (weights_s_before.len != weights_s_after.len or weights_t_before.len != weights_t_after.len) {
            return error.InvalidWeightsShape;
        }

        for (weights_s_after, weights_s_before) |*after_value, before_value| {
            const delta = @as(f32, @floatCast(after_value.*)) - @as(f32, @floatCast(before_value));
            after_value.* = @floatCast(delta);
        }
        for (weights_t_after, weights_t_before) |*after_value, before_value| {
            const delta = @as(f32, @floatCast(after_value.*)) - @as(f32, @floatCast(before_value));
            after_value.* = @floatCast(delta);
        }

        try self.averageDeltaInPlace(weights_s_after);
        try self.averageDeltaInPlace(weights_t_after);

        try self.applyDelta(weights_s_before, weights_s_after, .s);
        try self.applyDelta(weights_t_before, weights_t_after, .t);
        try self.accelerator.sync();

        var loss_arr = [1]f32{@as(f32, @floatCast(loss_f16))};
        try self.allReduceFloat32Values(loss_arr[0..]);
        const inv_w: f32 = 1.0 / @as(f32, @floatFromInt(self.coordinator.world_size));
        const final_loss = loss_arr[0] * inv_w;
        if (!std.math.isFinite(final_loss)) return error.InvalidLoss;
        return final_loss;
    }

    pub fn saveCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        if (!self.coordinator.isRoot()) {
            try self.coordinator.synchronize();
            return;
        }

        try self.coordinator.synchronize();
        try self.accelerator.sync();

        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{path});
        defer self.allocator.free(tmp_path);

        const file = createWriteFile(tmp_path) catch |err| {
            std.debug.print("Failed to create checkpoint file: {}\n", .{err});
            return err;
        };
        var file_closed = false;
        defer if (!file_closed) file.close();

        var buffered_writer = std.io.bufferedWriter(file.writer());
        var writer = buffered_writer.writer();

        try writer.writeInt(u32, self.config.checkpoint_version, .little);
        try writer.writeInt(u64, self.global_step, .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.model_dim)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.vocab_size)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.local_batch_size)), .little);
        try writeF32(writer, self.learning_rate);
        try writeF32(writer, self.momentum);

        const weights_s_vals = try self.readWeightsFlat(self.accelerator.weights_s);
        defer self.allocator.free(weights_s_vals);
        for (weights_s_vals) |w| try writeF32(writer, @floatCast(w));

        const weights_t_vals = try self.readWeightsFlat(self.accelerator.weights_t);
        defer self.allocator.free(weights_t_vals);
        for (weights_t_vals) |w| try writeF32(writer, @floatCast(w));

        const s_bias_vals = try self.readBiasFlat(self.accelerator.s_bias);
        defer self.allocator.free(s_bias_vals);
        for (s_bias_vals) |b| try writeF32(writer, @floatCast(b));

        const t_bias_vals = try self.readBiasFlat(self.accelerator.t_bias);
        defer self.allocator.free(t_bias_vals);
        for (t_bias_vals) |b| try writeF32(writer, @floatCast(b));

        const vel_s_vals = try self.readWeightsFlat(self.accelerator.velocity_s);
        defer self.allocator.free(vel_s_vals);
        for (vel_s_vals) |v| try writeF32(writer, @floatCast(v));

        const vel_t_vals = try self.readWeightsFlat(self.accelerator.velocity_t);
        defer self.allocator.free(vel_t_vals);
        for (vel_t_vals) |v| try writeF32(writer, @floatCast(v));

        const vel_sb_vals = try self.readBiasFlat(self.accelerator.velocity_sb);
        defer self.allocator.free(vel_sb_vals);
        for (vel_sb_vals) |v| try writeF32(writer, @floatCast(v));

        const vel_tb_vals = try self.readBiasFlat(self.accelerator.velocity_tb);
        defer self.allocator.free(vel_tb_vals);
        for (vel_tb_vals) |v| try writeF32(writer, @floatCast(v));

        try writeF32(writer, @as(f32, @floatCast(self.accelerator.clip_min)));
        try writeF32(writer, @as(f32, @floatCast(self.accelerator.clip_max)));

        try buffered_writer.flush();
        try file.sync();
        file.close();
        file_closed = true;

        if (std.fs.path.isAbsolute(path)) {
            try std.fs.renameAbsolute(tmp_path, path);
        } else {
            try std.fs.cwd().rename(tmp_path, path);
        }

        std.debug.print("Checkpoint saved to {s} at step {d}\n", .{ path, self.global_step });
    }

    pub fn loadCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        const file = openReadFile(path) catch |err| {
            std.debug.print("Failed to open checkpoint file: {}\n", .{err});
            return err;
        };
        defer file.close();

        var buf_reader = std.io.bufferedReader(file.reader());
        var reader = buf_reader.reader();

        const version = try reader.readInt(u32, .little);
        if (version != self.config.checkpoint_version) {
            return error.CheckpointVersionMismatch;
        }

        const saved_global_step = try reader.readInt(u64, .little);
        const saved_model_dim_u64 = try reader.readInt(u64, .little);
        const saved_vocab_size_u64 = try reader.readInt(u64, .little);
        const saved_local_batch_size_u64 = try reader.readInt(u64, .little);
        const saved_learning_rate = try readF32(reader);
        const saved_momentum = try readF32(reader);

        const saved_model_dim: usize = std.math.cast(usize, saved_model_dim_u64) orelse return error.ModelDimMismatch;
        const saved_vocab_size: usize = std.math.cast(usize, saved_vocab_size_u64) orelse return error.VocabSizeMismatch;
        const saved_local_batch_size: usize = std.math.cast(usize, saved_local_batch_size_u64) orelse return error.InvalidBatchSize;

        if (saved_model_dim != self.model_dim) return error.ModelDimMismatch;
        if (saved_vocab_size != self.vocab_size) return error.VocabSizeMismatch;
        _ = saved_local_batch_size;

        try validateHyperparameters(saved_learning_rate, saved_momentum);
        if (self.coordinator.world_size > 1 and saved_momentum != 0.0) {
            return error.UnsupportedDistributedMomentum;
        }
        self.learning_rate = saved_learning_rate;
        self.momentum = saved_momentum;
        self.global_step = saved_global_step;

        const weight_count = try std.math.mul(usize, self.model_dim, self.model_dim);

        const s_weights = try self.allocator.alloc(f16, weight_count);
        defer self.allocator.free(s_weights);
        for (s_weights) |*w| {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return error.InvalidWeightValue;
            w.* = @floatCast(v);
        }

        const t_weights = try self.allocator.alloc(f16, weight_count);
        defer self.allocator.free(t_weights);
        for (t_weights) |*w| {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return error.InvalidWeightValue;
            w.* = @floatCast(v);
        }

        try self.accelerator.setWeightsS(s_weights, self.model_dim, self.model_dim);
        try self.accelerator.setWeightsT(t_weights, self.model_dim, self.model_dim);

        const s_bias_data = try self.allocator.alloc(f16, self.model_dim);
        defer self.allocator.free(s_bias_data);
        for (s_bias_data) |*b| {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return error.InvalidWeightValue;
            b.* = @floatCast(v);
        }

        const t_bias_data = try self.allocator.alloc(f16, self.model_dim);
        defer self.allocator.free(t_bias_data);
        for (t_bias_data) |*b| {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return error.InvalidWeightValue;
            b.* = @floatCast(v);
        }

        try self.accelerator.setSBias(s_bias_data, self.model_dim);
        try self.accelerator.setTBias(t_bias_data, self.model_dim);

        const vel_s = try self.allocator.alloc(f16, weight_count);
        defer self.allocator.free(vel_s);
        for (vel_s) |*w| {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return error.InvalidWeightValue;
            w.* = @floatCast(v);
        }

        const vel_t = try self.allocator.alloc(f16, weight_count);
        defer self.allocator.free(vel_t);
        for (vel_t) |*w| {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return error.InvalidWeightValue;
            w.* = @floatCast(v);
        }

        const vel_sb = try self.allocator.alloc(f16, self.model_dim);
        defer self.allocator.free(vel_sb);
        for (vel_sb) |*w| {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return error.InvalidWeightValue;
            w.* = @floatCast(v);
        }

        const vel_tb = try self.allocator.alloc(f16, self.model_dim);
        defer self.allocator.free(vel_tb);
        for (vel_tb) |*w| {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return error.InvalidWeightValue;
            w.* = @floatCast(v);
        }

        try self.accelerator.setVelocityS(vel_s, self.model_dim, self.model_dim);
        try self.accelerator.setVelocityT(vel_t, self.model_dim, self.model_dim);
        try self.accelerator.setVelocitySB(vel_sb, self.model_dim);
        try self.accelerator.setVelocityTB(vel_tb, self.model_dim);

        const clip_min_f32 = try readF32(reader);
        const clip_max_f32 = try readF32(reader);
        if (!std.math.isFinite(clip_min_f32) or !std.math.isFinite(clip_max_f32) or !(clip_min_f32 < clip_max_f32)) {
            return error.InvalidClipRange;
        }
        try self.accelerator.setClipRange(@floatCast(clip_min_f32), @floatCast(clip_max_f32));

        try self.accelerator.sync();

        std.debug.print("Checkpoint loaded from {s} at step {d}\n", .{ path, self.global_step });
    }
};
