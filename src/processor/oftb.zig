Const std = @import("std");  
const Allocator = std.mem.Allocator;  
const Tensor = @import("../core/tensor.zig").Tensor;  
  
pub const OFTB = struct {  
    pub const FRACTAL_SCALE: f32 = 0.7071067811865476;  
  
    dim: usize,  
  
    pub fn init(d: usize) OFTB {  
        std.debug.assert(d != 0);  
        std.debug.assert(d <= std.math.maxInt(usize) / 2);  
        return OFTB{  
            .dim = d,  
        };  
    }  
  
    pub fn deinit(self: *OFTB) void {  
        self.* = undefined;  
    }  
  
    pub fn forwardInPlace(self: OFTB, x: *Tensor) !void {  
        if (self.dim == 0) return error.InvalidDimension;  
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;  
        const total = self.dim * 2;  
        if (x.data.len < total) return error.TensorTooSmall;  
        const half = self.dim;  
        const x1 = x.data[0..half];  
        const x2 = x.data[half..][0..half];  
        const scale: f32 = FRACTAL_SCALE;  
        const VLEN: usize = 8;  
        var i: usize = 0;  
        while (i + VLEN <= half) : (i += VLEN) {  
            const va: @Vector(VLEN, f32) = x1[i..][0..VLEN].*;  
            const vb: @Vector(VLEN, f32) = x2[i..][0..VLEN].*;  
            const vscale: @Vector(VLEN, f32) = @splat(scale);  
            x1[i..][0..VLEN].* = (va - vb) * vscale;  
            x2[i..][0..VLEN].* = (va + vb) * vscale;  
        }  
        while (i < half) : (i += 1) {  
            const a = x1[i];  
            const b = x2[i];  
            x1[i] = (a - b) * scale;  
            x2[i] = (a + b) * scale;  
        }  
    }  
  
    pub fn backwardInPlace(self: OFTB, grad: []f32) !void {  
        if (self.dim == 0) return error.InvalidDimension;  
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;  
        const total = self.dim * 2;  
        if (grad.len < total) return error.GradientTooSmall;  
        const half = self.dim;  
        const g1 = grad[0..half];  
        const g2 = grad[half..][0..half];  
        const scale: f32 = FRACTAL_SCALE;  
        const VLEN: usize = 8;  
        var i: usize = 0;  
        while (i + VLEN <= half) : (i += VLEN) {  
            const va: @Vector(VLEN, f32) = g1[i..][0..VLEN].*;  
            const vb: @Vector(VLEN, f32) = g2[i..][0..VLEN].*;  
            const vscale: @Vector(VLEN, f32) = @splat(scale);  
            g1[i..][0..VLEN].* = (va + vb) * vscale;  
            g2[i..][0..VLEN].* = (vb - va) * vscale;  
        }  
        while (i < half) : (i += 1) {  
            const a = g1[i];  
            const b = g2[i];  
            g1[i] = (a + b) * scale;  
            g2[i] = (b - a) * scale;  
        }  
    }  
  
    pub fn backwardInPlaceSlice(self: OFTB, grad: []f32) !void {  
        return self.backwardInPlace(grad);  
    }  
  
    comptime {  
        _ = Allocator;  
    }  
};  
  
comptime {  
    _ = OFTB;  
}
