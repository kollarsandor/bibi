const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");
const core_tensor = @import("../../core/tensor.zig");
const core_memory = @import("../../core/memory.zig");

pub const gpu_enabled: bool = @import("build_options").gpu_acceleration;

pub const AccelError = error{
    FutharkConfigFailed,
    FutharkContextFailed,
    FutharkSyncFailed,
    FutharkArrayNewFailed,
    FutharkValuesFailed,
    FutharkForwardFailed,
    FutharkTrainingStepFailed,
    FutharkScaleWeightsFailed,
    FutharkShapeFailed,
    CudaHostAllocFailed,
    CudaFreeFailed,
    NullPointer,
    InvalidDimensions,
    AllocationFailed,
    PartialRowCleanup,
};

pub const FutharkContext = struct {
    ctx: ?*futhark.struct_futhark_context,

    const Self = @This();

    pub fn init() AccelError!Self {
        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return AccelError.FutharkConfigFailed;

        futhark.futhark_context_config_set_device(cfg, 0);
        futhark.futhark_context_config_set_default_group_size(cfg, 256);
        futhark.futhark_context_config_set_default_num_groups(cfg, 128);
        futhark.futhark_context_config_set_default_tile_size(cfg, 32);

        const ctx = futhark.futhark_context_new(cfg);
        futhark.futhark_context_config_free(cfg);

        if (ctx == null) return AccelError.FutharkContextFailed;

        if (futhark.futhark_context_sync(ctx) != 0) {
            futhark.futhark_context_free(ctx);
            return AccelError.FutharkSyncFailed;
        }

        return Self{ .ctx = ctx };
    }

    pub fn deinit(self: *Self) void {
        if (self.ctx) |ctx| {
            futhark.futhark_context_free(ctx);
            self.ctx = null;
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (self.ctx == null) return AccelError.NullPointer;
        if (futhark.futhark_context_sync(self.ctx) != 0) {
            return AccelError.FutharkSyncFailed;
        }
    }

    pub fn getDataPointer(self: *Self, array: *FutharkArray2DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_2d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }
};

pub const PinnedMemory = struct {
    ptr: ?*anyopaque,
    size: usize,

    const Self = @This();

    pub fn alloc(size: usize) AccelError!Self {
        if (size == 0) {
            return Self{ .ptr = null, .size = 0 };
        }

        var ptr: ?*anyopaque = null;
        const err = cuda.cudaHostAlloc(&ptr, size, cuda.cudaHostAllocDefault);
        if (err != cuda.cudaSuccess) {
            return AccelError.CudaHostAllocFailed;
        }

        return Self{
            .ptr = ptr,
            .size = size,
        };
    }

    pub fn free(self: *Self) void {
        if (self.ptr) |p| {
            _ = cuda.cudaFreeHost(p);
            self.ptr = null;
            self.size = 0;
        }
    }

    pub fn asSlice(self: *Self, comptime T: type) ?[]T {
        if (self.ptr == null) return null;
        if (self.size == 0) return &[_]T{};
        const count = self.size / @sizeOf(T);
        if (count == 0) return &[_]T{};
        const aligned: [*]T = @ptrCast(@alignCast(self.ptr.?));
        return aligned[0..count];
    }
};

pub const FutharkArray1DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_1d,
    len: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, length: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != length) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn newZeros(ctx: *FutharkContext, length: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;

        const zeros = std.heap.page_allocator.alloc(f16, length) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn values1D(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;

        const buf = allocator.alloc(f16, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);

        const result = futhark.futhark_values_f16_1d(ctx.ctx, self.arr, @ptrCast(buf.ptr));
        if (result != 0) {
            allocator.free(buf);
            return AccelError.FutharkValuesFailed;
        }

        const sync_result = futhark.futhark_context_sync(ctx.ctx);
        if (sync_result != 0) {
            allocator.free(buf);
            return AccelError.FutharkSyncFailed;
        }

        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const FutharkArray2DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn new(ctx: *FutharkContext, data: []const []const f16) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;

        const rows = data.len;
        const cols = data[0].len;
        if (cols == 0) return AccelError.InvalidDimensions;

        for (data) |row| {
            if (row.len != cols) return AccelError.InvalidDimensions;
        }

        const total = rows * cols;
        var flat_data = std.ArrayList(f16).init(std.heap.page_allocator);
        defer flat_data.deinit();

        flat_data.ensureTotalCapacity(total) catch return AccelError.AllocationFailed;

        for (data) |row| {
            flat_data.appendSlice(row) catch return AccelError.AllocationFailed;
        }

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.items.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != rows * cols) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;

        const total = rows * cols;
        const zeros = std.heap.page_allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn values(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![][]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;

        var dims: [2]i64 = undefined;
        if (futhark.futhark_shape_f16_2d(ctx.ctx, self.arr, &dims) != 0) {
            return AccelError.FutharkShapeFailed;
        }
        const rows = @as(usize, @intCast(dims[0]));
        const cols = @as(usize, @intCast(dims[1]));

        if (rows == 0 or cols == 0) {
            return allocator.alloc([]f16, 0) catch return AccelError.AllocationFailed;
        }

        const flat = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(flat);

        if (futhark.futhark_values_f16_2d(ctx.ctx, self.arr, @ptrCast(flat.ptr)) != 0) {
            return AccelError.FutharkValuesFailed;
        }

        const result = allocator.alloc([]f16, rows) catch return AccelError.AllocationFailed;
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            result[i] = allocator.alloc(f16, cols) catch {
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    allocator.free(result[j]);
                }
                allocator.free(result);
                return AccelError.PartialRowCleanup;
            };
            @memcpy(result[i], flat[i * cols .. (i + 1) * cols]);
        }

        return result;
    }
};

pub const FutharkArray2DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = tensor.shape.dims[0];
        const cols = tensor.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, tensor.data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const zeros = allocator.alloc(f32, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, zeros.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_f32_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{ self.rows, self.cols };
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_2d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }
};

pub const FutharkArray1DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_1d,
    len: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 1) return AccelError.InvalidDimensions;
        const n = tensor.shape.dims[0];
        if (n == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, tensor.data.ptr, @intCast(n));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = n };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_f32_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{self.len};
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }
};

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    weights_s: FutharkArray2DF16,
    weights_t: FutharkArray2DF16,
    s_bias: FutharkArray1DF16,
    t_bias: FutharkArray1DF16,
    velocity_s: FutharkArray2DF16,
    velocity_t: FutharkArray2DF16,
    velocity_sb: FutharkArray1DF16,
    velocity_tb: FutharkArray1DF16,
    model_dim: usize,
    clip_min: f16,
    clip_max: f16,
    initialized: bool,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;

        var ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        var weights_s = try FutharkArray2DF16.newZeros(&ctx, model_dim, model_dim);
        errdefer weights_s.free(&ctx);

        var weights_t = try FutharkArray2DF16.newZeros(&ctx, model_dim, model_dim);
        errdefer weights_t.free(&ctx);

        var s_bias = try FutharkArray1DF16.newZeros(&ctx, model_dim);
        errdefer s_bias.free(&ctx);

        var t_bias = try FutharkArray1DF16.newZeros(&ctx, model_dim);
        errdefer t_bias.free(&ctx);

        var velocity_s = try FutharkArray2DF16.newZeros(&ctx, model_dim, model_dim);
        errdefer velocity_s.free(&ctx);

        var velocity_t = try FutharkArray2DF16.newZeros(&ctx, model_dim, model_dim);
        errdefer velocity_t.free(&ctx);

        var velocity_sb = try FutharkArray1DF16.newZeros(&ctx, model_dim);
        errdefer velocity_sb.free(&ctx);

        var velocity_tb = try FutharkArray1DF16.newZeros(&ctx, model_dim);
        errdefer velocity_tb.free(&ctx);

        return Self{
            .ctx = ctx,
            .weights_s = weights_s,
            .weights_t = weights_t,
            .s_bias = s_bias,
            .t_bias = t_bias,
            .velocity_s = velocity_s,
            .velocity_t = velocity_t,
            .velocity_sb = velocity_sb,
            .velocity_tb = velocity_tb,
            .model_dim = model_dim,
            .clip_min = @as(f16, -5.0),
            .clip_max = @as(f16, 5.0),
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;

        self.velocity_tb.free(&self.ctx);
        self.velocity_sb.free(&self.ctx);
        self.velocity_t.free(&self.ctx);
        self.velocity_s.free(&self.ctx);
        self.t_bias.free(&self.ctx);
        self.s_bias.free(&self.ctx);
        self.weights_t.free(&self.ctx);
        self.weights_s.free(&self.ctx);
        self.ctx.deinit();
        self.initialized = false;
    }

    pub fn forward(self: *Self, input: *FutharkArray2DF16) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (input.arr == null) return AccelError.NullPointer;
        if (self.weights_s.arr == null) return AccelError.NullPointer;
        if (self.weights_t.arr == null) return AccelError.NullPointer;
        if (self.s_bias.arr == null) return AccelError.NullPointer;
        if (self.t_bias.arr == null) return AccelError.NullPointer;

        var output: ?*futhark.struct_futhark_f16_2d = null;
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        const result = futhark.futhark_entry_rsf_forward(
            self.ctx.ctx,
            &output,
            input.arr,
            self.weights_s.arr,
            self.weights_t.arr,
            self.s_bias.arr,
            self.t_bias.arr,
            clip_min_bits,
            clip_max_bits,
        );

        if (result != 0) {
            return AccelError.FutharkForwardFailed;
        }

        if (output == null) {
            return AccelError.NullPointer;
        }

        return FutharkArray2DF16{
            .arr = output,
            .rows = input.rows,
            .cols = input.cols,
        };
    }

    pub fn trainingStep(
        self: *Self,
        inputs: *FutharkArray2DF16,
        targets: *FutharkArray2DF16,
        learning_rate: f16,
        momentum: f16,
    ) AccelError!f16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null or targets.arr == null) return AccelError.NullPointer;
        if (self.weights_s.arr == null or self.weights_t.arr == null) return AccelError.NullPointer;
        if (self.s_bias.arr == null or self.t_bias.arr == null) return AccelError.NullPointer;
        if (self.velocity_s.arr == null or self.velocity_t.arr == null) return AccelError.NullPointer;
        if (self.velocity_sb.arr == null or self.velocity_tb.arr == null) return AccelError.NullPointer;

        var new_ws: ?*futhark.struct_futhark_f16_2d = null;
        var new_wt: ?*futhark.struct_futhark_f16_2d = null;
        var new_sb: ?*futhark.struct_futhark_f16_1d = null;
        var new_tb: ?*futhark.struct_futhark_f16_1d = null;
        var new_vs: ?*futhark.struct_futhark_f16_2d = null;
        var new_vt: ?*futhark.struct_futhark_f16_2d = null;
        var new_vsb: ?*futhark.struct_futhark_f16_1d = null;
        var new_vtb: ?*futhark.struct_futhark_f16_1d = null;
        var loss: u16 = 0;

        const lr_bits: u16 = @bitCast(learning_rate);
        const momentum_bits: u16 = @bitCast(momentum);
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        const result = futhark.futhark_entry_training_step(
            self.ctx.ctx,
            &new_ws,
            &new_wt,
            &new_sb,
            &new_tb,
            &new_vs,
            &new_vt,
            &new_vsb,
            &new_vtb,
            &loss,
            inputs.arr,
            targets.arr,
            self.weights_s.arr,
            self.weights_t.arr,
            self.s_bias.arr,
            self.t_bias.arr,
            self.velocity_s.arr,
            self.velocity_t.arr,
            self.velocity_sb.arr,
            self.velocity_tb.arr,
            lr_bits,
            momentum_bits,
            clip_min_bits,
            clip_max_bits,
        );

        if (result != 0) {
            return AccelError.FutharkTrainingStepFailed;
        }

        if (new_ws == null or new_wt == null or new_sb == null or new_tb == null or
            new_vs == null or new_vt == null or new_vsb == null or new_vtb == null)
        {
            return AccelError.NullPointer;
        }

        const old_ws = self.weights_s.arr;
        const old_wt = self.weights_t.arr;
        const old_sb = self.s_bias.arr;
        const old_tb = self.t_bias.arr;
        const old_vs = self.velocity_s.arr;
        const old_vt = self.velocity_t.arr;
        const old_vsb = self.velocity_sb.arr;
        const old_vtb = self.velocity_tb.arr;

        self.weights_s.arr = new_ws;
        self.weights_t.arr = new_wt;
        self.s_bias.arr = new_sb;
        self.t_bias.arr = new_tb;
        self.velocity_s.arr = new_vs;
        self.velocity_t.arr = new_vt;
        self.velocity_sb.arr = new_vsb;
        self.velocity_tb.arr = new_vtb;

        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_ws);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_wt);
        _ = futhark.futhark_free_f16_1d(self.ctx.ctx, old_sb);
        _ = futhark.futhark_free_f16_1d(self.ctx.ctx, old_tb);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_vs);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_vt);
        _ = futhark.futhark_free_f16_1d(self.ctx.ctx, old_vsb);
        _ = futhark.futhark_free_f16_1d(self.ctx.ctx, old_vtb);

        const loss_f16: f16 = @bitCast(loss);
        return loss_f16;
    }

    pub fn scaleWeights(self: *Self, scale_factor: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (self.weights_s.arr == null or self.weights_t.arr == null) return AccelError.NullPointer;

        if (scale_factor == @as(f16, 0.0)) return AccelError.InvalidDimensions;

        const scale_bits: u16 = @bitCast(scale_factor);

        var new_ws: ?*futhark.struct_futhark_f16_2d = null;
        const result_s = futhark.futhark_entry_scale_weights_inplace(
            self.ctx.ctx,
            &new_ws,
            self.weights_s.arr,
            scale_bits,
        );

        if (result_s != 0) {
            return AccelError.FutharkScaleWeightsFailed;
        }

        if (new_ws != null) {
            const old = self.weights_s.arr;
            self.weights_s.arr = new_ws;
            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
        }

        var new_wt: ?*futhark.struct_futhark_f16_2d = null;
        const result_t = futhark.futhark_entry_scale_weights_inplace(
            self.ctx.ctx,
            &new_wt,
            self.weights_t.arr,
            scale_bits,
        );

        if (result_t != 0) {
            return AccelError.FutharkScaleWeightsFailed;
        }

        if (new_wt != null) {
            const old = self.weights_t.arr;
            self.weights_t.arr = new_wt;
            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
        }
    }

    pub fn getWeightsSDataPointer(self: *Self) AccelError!*anyopaque {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.getDataPointer(&self.weights_s);
    }

    pub fn getWeightsTDataPointer(self: *Self) AccelError!*anyopaque {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.getDataPointer(&self.weights_t);
    }

    pub fn sync(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.sync();
    }

    pub fn setWeightsS(self: *Self, data: []const f16, rows: usize, cols: usize) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;

        self.weights_s.free(&self.ctx);
        self.weights_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setWeightsT(self: *Self, data: []const f16, rows: usize, cols: usize) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;

        self.weights_t.free(&self.ctx);
        self.weights_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setSBias(self: *Self, data: []const f16, length: usize) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;
        if (data.len != length) return AccelError.InvalidDimensions;

        self.s_bias.free(&self.ctx);
        self.s_bias = try FutharkArray1DF16.newFromFlat(&self.ctx, data, length);
    }

    pub fn setTBias(self: *Self, data: []const f16, length: usize) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;
        if (data.len != length) return AccelError.InvalidDimensions;

        self.t_bias.free(&self.ctx);
        self.t_bias = try FutharkArray1DF16.newFromFlat(&self.ctx, data, length);
    }

    pub fn setClipRange(self: *Self, clip_min_val: f16, clip_max_val: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (clip_min_val >= clip_max_val) return AccelError.InvalidDimensions;
        self.clip_min = clip_min_val;
        self.clip_max = clip_max_val;
    }

    pub fn forwardFromTensor(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (!self.initialized) return AccelError.NullPointer;
        if (input.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = input.shape.dims[0];
        const cols = input.shape.dims[1];
        const f16_data = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(f16_data);
        {
            var i: usize = 0;
            while (i < input.data.len) : (i += 1) {
                const v = input.data[i];
                f16_data[i] = @floatCast(v);
            }
        }
        var f16_input = try FutharkArray2DF16.newFromFlat(&self.ctx, f16_data, rows, cols);
        defer f16_input.free(&self.ctx);
        var output = try self.forward(&f16_input);
        defer output.free(&self.ctx);
        const shape = [_]usize{ output.rows, output.cols };
        var result = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        const out_f16 = allocator.alloc(f16, output.rows * output.cols) catch {
            result.deinit();
            return AccelError.AllocationFailed;
        };
        defer allocator.free(out_f16);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, output.arr, @ptrCast(out_f16.ptr)) != 0) {
            result.deinit();
            return AccelError.FutharkValuesFailed;
        }
        {
            var i: usize = 0;
            while (i < out_f16.len) : (i += 1) {
                const v = out_f16[i];
                result.data[i] = @floatCast(v);
            }
        }
        return result;
    }
};

pub const GPUOps = struct {
    ctx: FutharkContext,

    const Self = @This();

    pub fn init() AccelError!Self {
        return Self{ .ctx = try FutharkContext.init() };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
    }

    pub fn matmul(self: *Self, a: *const core_tensor.Tensor, b: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fa = try FutharkArray2DF32.fromTensor(&self.ctx, a);
        defer fa.free(&self.ctx);
        var fb = try FutharkArray2DF32.fromTensor(&self.ctx, b);
        defer fb.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_matmul(self.ctx.ctx, &out_arr, fa.arr, fb.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray2DF32{ .arr = out_arr, .rows = a.shape.dims[0], .cols = b.shape.dims[1] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }
};
