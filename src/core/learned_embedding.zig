const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

pub const LearnedEmbedding = struct {
    weight: Tensor,
    grad: Tensor,
    vocab_size: usize,
    dim: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, v_size: usize, d: usize, seed: u64) !LearnedEmbedding {
        var w = try Tensor.init(allocator, &.{ v_size, d });
        errdefer w.deinit();
        var g = try Tensor.init(allocator, &.{ v_size, d });
        errdefer g.deinit();
        @memset(g.data, 0.0);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        var i: usize = 0;
        while (i < w.data.len) : (i += 1) {
            w.data[i] = (random.float(f32) - 0.5) * 0.02;
        }
        return LearnedEmbedding{
            .weight = w,
            .grad = g,
            .vocab_size = v_size,
            .dim = d,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LearnedEmbedding) void {
        self.weight.deinit();
        self.grad.deinit();
    }

    pub fn forward(self: *LearnedEmbedding, allocator: Allocator, tokens: []const u32, output_dim: usize) !Tensor {
        var out = try Tensor.init(allocator, &.{ 1, output_dim });
        @memset(out.data, 0.0);
        const max_tokens = @min(tokens.len, self.dim);
        var r: usize = 0;
        while (r < max_tokens) : (r += 1) {
            const t = @min(@as(usize, tokens[r]), self.vocab_size - 1);
            var c: usize = 0;
            while (c < self.dim) : (c += 1) {
                const w_idx = t * self.dim + c;
                if (w_idx < self.weight.data.len) {
                    if (r * self.dim + c < output_dim) {
                        const out_idx = r * self.dim + c;
                        if (out_idx < out.data.len) {
                            out.data[out_idx] = self.weight.data[w_idx];
                        }
                    }
                }
            }
        }
        return out;
    }

    pub fn backward(self: *LearnedEmbedding, tokens: []const u32, out_grad: []const f32) void {
        const max_tokens = @min(tokens.len, self.dim);
        var r: usize = 0;
        while (r < max_tokens) : (r += 1) {
            const t = @min(@as(usize, tokens[r]), self.vocab_size - 1);
            var c: usize = 0;
            while (c < self.dim) : (c += 1) {
                const g_idx = t * self.dim + c;
                const o_idx = r * self.dim + c;
                if (g_idx < self.grad.data.len and o_idx < out_grad.len) {
                    self.grad.data[g_idx] += out_grad[o_idx];
                }
            }
        }
    }

    pub fn zeroGrad(self: *LearnedEmbedding) void {
        @memset(self.grad.data, 0.0);
    }

    pub fn applyGradients(self: *LearnedEmbedding, lr: f32, momentum: f32) void {
        var i: usize = 0;
        while (i < self.weight.data.len) : (i += 1) {
            self.weight.data[i] -= lr * self.grad.data[i];
            self.grad.data[i] *= momentum;
        }
    }

    pub fn paramCount(self: *const LearnedEmbedding) usize {
        return self.vocab_size * self.dim;
    }

    pub fn flattenParams(self: *const LearnedEmbedding, dst: []f32) void {
        const count = @min(dst.len, self.weight.data.len);
        @memcpy(dst[0..count], self.weight.data[0..count]);
    }

    pub fn flattenGrads(self: *const LearnedEmbedding, dst: []f32) void {
        const count = @min(dst.len, self.grad.data.len);
        @memcpy(dst[0..count], self.grad.data[0..count]);
    }

    pub fn scatterParams(self: *LearnedEmbedding, src: []const f32) void {
        const count = @min(src.len, self.weight.data.len);
        @memcpy(self.weight.data[0..count], src[0..count]);
    }

    pub fn save(self: *const LearnedEmbedding, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        var buf_writer = std.io.bufferedWriter(file.writer());
        const writer = buf_writer.writer();
        try writer.writeInt(u32, 0x4A454D42, .little);
        try writer.writeInt(u32, 1, .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.vocab_size)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.dim)), .little);
        for (self.weight.data) |w| {
            try writer.writeInt(u32, @as(u32, @bitCast(w)), .little);
        }
        try buf_writer.flush();
    }

    pub fn load(allocator: Allocator, path: []const u8) !LearnedEmbedding {
        const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
        defer file.close();
        var buf_reader = std.io.bufferedReader(file.reader());
        const reader = buf_reader.reader();
        const magic = try reader.readInt(u32, .little);
        if (magic != 0x4A454D42) return error.InvalidFormat;
        _ = try reader.readInt(u32, .little);
        const v_size = @as(usize, @intCast(try reader.readInt(u64, .little)));
        const d = @as(usize, @intCast(try reader.readInt(u64, .little)));
        var w = try Tensor.init(allocator, &.{ v_size, d });
        errdefer w.deinit();
        var g = try Tensor.init(allocator, &.{ v_size, d });
        errdefer g.deinit();
        @memset(g.data, 0.0);
        var i: usize = 0;
        while (i < w.data.len) : (i += 1) {
            w.data[i] = @bitCast(try reader.readInt(u32, .little));
        }
        return LearnedEmbedding{
            .weight = w,
            .grad = g,
            .vocab_size = v_size,
            .dim = d,
            .allocator = allocator,
        };
    }
};
