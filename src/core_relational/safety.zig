const std = @import("std");
const Allocator = std.mem.Allocator;

pub const SafetyError = error{
    IntegerOverflow,
    IntegerUnderflow,
    InvalidPointer,
    NullPointer,
    MisalignedPointer,
    InvalidProvenance,
    BufferTooSmall,
    InvalidLength,
};

pub fn safeIntCast(comptime T: type, value: anytype) SafetyError!T {
    const SourceT = @TypeOf(value);
    const source_info = @typeInfo(SourceT);
    const target_info = @typeInfo(T);

    if (source_info == .Int and target_info == .Int) {
        const source_int = source_info.Int;
        const target_int = target_info.Int;

        if (target_int.signedness == .unsigned) {
            if (source_int.signedness == .signed) {
                if (value < 0) {
                    return SafetyError.IntegerUnderflow;
                }
            }
            const max_val = std.math.maxInt(T);
            if (source_int.bits > target_int.bits or source_int.signedness == .signed) {
                if (@as(u128, @intCast(if (value < 0) 0 else @as(u128, @intCast(value)))) > max_val) {
                    return SafetyError.IntegerOverflow;
                }
            }
        } else {
            const max_val = std.math.maxInt(T);
            const min_val = std.math.minInt(T);
            const v_i128: i128 = @intCast(value);
            if (v_i128 > max_val or v_i128 < min_val) {
                if (v_i128 > max_val) {
                    return SafetyError.IntegerOverflow;
                }
                return SafetyError.IntegerUnderflow;
            }
        }
    }

    return @intCast(value);
}

pub fn safeUsizeToInt(comptime T: type, value: usize) ?T {
    const max_val = std.math.maxInt(T);
    if (value > max_val) {
        return null;
    }
    return @intCast(value);
}

pub fn safeI64ToU64(value: i64) ?u64 {
    if (value < 0) {
        return null;
    }
    return @intCast(value);
}

pub fn safePtrCast(comptime T: type, ptr: anytype) SafetyError!T {
    if (@TypeOf(ptr) == @TypeOf(null)) {
        return SafetyError.NullPointer;
    }

    const ptr_info = @typeInfo(@TypeOf(ptr));
    if (ptr_info != .Pointer and ptr_info != .Optional) {
        return SafetyError.InvalidPointer;
    }

    if (ptr_info == .Optional) {
        if (ptr == null) {
            return SafetyError.NullPointer;
        }
    }

    const target_info = @typeInfo(T);
    if (target_info != .Pointer) {
        return SafetyError.InvalidPointer;
    }

    const target_alignment = target_info.Pointer.alignment;
    const addr = @intFromPtr(if (ptr_info == .Optional) ptr.? else ptr);

    if (addr == 0) {
        return SafetyError.NullPointer;
    }

    if (addr % target_alignment != 0) {
        return SafetyError.MisalignedPointer;
    }

    return @ptrCast(@alignCast(if (ptr_info == .Optional) ptr.? else ptr));
}

pub fn validatePointer(ptr: anytype) bool {
    const ptr_info = @typeInfo(@TypeOf(ptr));

    if (ptr_info == .Optional) {
        if (ptr == null) {
            return false;
        }
    }

    if (ptr_info != .Pointer and ptr_info != .Optional) {
        return false;
    }

    const addr = @intFromPtr(if (ptr_info == .Optional) ptr.? else ptr);
    return addr != 0;
}

pub const SecureRng = struct {
    fallback_state: u64,

    const Self = @This();

    pub fn init() Self {
        var seed: [8]u8 = undefined;
        std.crypto.random.bytes(&seed);
        return Self{
            .fallback_state = std.mem.readInt(u64, &seed, .little),
        };
    }

    pub fn bytes(self: *Self, buffer: []u8) void {
        _ = self;
        std.crypto.random.bytes(buffer);
    }

    pub fn int(self: *Self, comptime T: type) T {
        _ = self;
        return std.crypto.random.int(T);
    }

    pub fn intRange(self: *Self, comptime T: type, min: T, max: T) T {
        _ = self;
        if (min >= max) {
            return min;
        }
        const range = max - min;
        const random_val = std.crypto.random.int(T);
        return min + @mod(random_val, range);
    }

    pub fn uintLessThan(self: *Self, comptime T: type, upper: T) T {
        _ = self;
        return std.crypto.random.uintLessThan(T, upper);
    }

    pub fn float(self: *Self, comptime T: type) T {
        _ = self;
        if (T == f32) {
            const bits = std.crypto.random.int(u32);
            const mantissa = bits & 0x7FFFFF;
            return @as(f32, @floatFromInt(mantissa)) / @as(f32, @floatFromInt(0x800000));
        } else {
            const bits = std.crypto.random.int(u64);
            const mantissa = bits & 0xFFFFFFFFFFFFF;
            return @as(f64, @floatFromInt(mantissa)) / @as(f64, @floatFromInt(0x10000000000000));
        }
    }
};

pub const MonotonicClock = struct {
    base_nano: i128,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .base_nano = std.time.nanoTimestamp(),
        };
    }

    pub fn now(self: *Self) i128 {
        return std.time.nanoTimestamp() - self.base_nano;
    }

    pub fn nowMillis(self: *Self) i64 {
        const nanos = self.now();
        const millis = @divFloor(nanos, 1_000_000);
        if (millis > std.math.maxInt(i64)) {
            return std.math.maxInt(i64);
        }
        if (millis < std.math.minInt(i64)) {
            return std.math.minInt(i64);
        }
        return @intCast(millis);
    }

    pub fn elapsed(self: *Self, start: i128) i128 {
        return self.now() - start;
    }
};

pub fn secureZeroBytes(buffer: []u8) void {
    for (buffer) |*b| {
        @as(*volatile u8, b).* = 0;
    }
}

pub fn secureZeroSlice(comptime T: type, buffer: []T) void {
    const bytes: []u8 = @as([*]u8, @ptrCast(buffer.ptr))[0 .. buffer.len * @sizeOf(T)];
    secureZeroBytes(bytes);
}

pub fn secureCompare(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) {
        return false;
    }
    var result: u8 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        result |= a[i] ^ b[i];
    }
    return result == 0;
}

pub fn hashWithSeed(data: []const u8, seed: u64) u64 {
    var hasher = std.hash.Wyhash.init(seed);
    hasher.update(data);
    return hasher.final();
}

pub fn cryptoHashSeed() u64 {
    var seed_bytes: [8]u8 = undefined;
    std.crypto.random.bytes(&seed_bytes);
    return std.mem.readInt(u64, &seed_bytes, .little);
}

pub fn safeSlice(comptime T: type, slice: []T, start: usize, end: usize) SafetyError![]T {
    if (start > end) {
        return SafetyError.InvalidLength;
    }
    if (end > slice.len) {
        return SafetyError.BufferTooSmall;
    }
    return slice[start..end];
}

pub fn safeCopy(dest: []u8, src: []const u8) SafetyError!usize {
    if (dest.len < src.len) {
        return SafetyError.BufferTooSmall;
    }
    @memcpy(dest[0..src.len], src);
    return src.len;
}

pub fn safeCopyWithMax(dest: []u8, src: []const u8, max_len: usize) usize {
    if (max_len == 0) {
        return 0;
    }
    const copy_len = @min(src.len, @min(dest.len, max_len));
    if (copy_len > 0) {
        @memcpy(dest[0..copy_len], src[0..copy_len]);
    }
    return copy_len;
}

pub const BigInt512 = struct {
    low: u256,
    high: u256,

    const Self = @This();

    pub fn init(val: u256) Self {
        return Self{ .low = val, .high = 0 };
    }

    pub fn fromU512(val: u512) Self {
        return Self{
            .low = @truncate(val),
            .high = @truncate(val >> 256),
        };
    }

    pub fn toU512(self: Self) u512 {
        return (@as(u512, self.high) << 256) | @as(u512, self.low);
    }

    pub fn add(self: Self, other: Self) Self {
        var result: Self = undefined;
        var carry: u256 = 0;

        const low_sum = @addWithOverflow(self.low, other.low);
        result.low = low_sum[0];
        carry = low_sum[1];

        result.high = self.high +% other.high +% carry;
        return result;
    }

    pub fn mul(a: u256, b: u256) Self {
        const result: u512 = @as(u512, a) * @as(u512, b);
        return Self.fromU512(result);
    }

    pub fn mod(self: Self, modulus: u256) u256 {
        if (modulus == 0) return 0;
        const val = self.toU512();
        return @truncate(val % @as(u512, modulus));
    }

    pub fn isZero(self: Self) bool {
        return self.low == 0 and self.high == 0;
    }

    pub fn compare(self: Self, other: Self) std.math.Order {
        if (self.high != other.high) {
            return std.math.order(self.high, other.high);
        }
        return std.math.order(self.low, other.low);
    }
};

test "safeIntCast" {
    const testing = std.testing;

    const result1 = try safeIntCast(u8, @as(u32, 255));
    try testing.expectEqual(@as(u8, 255), result1);

    const result2 = safeIntCast(u8, @as(u32, 256));
    try testing.expectError(SafetyError.IntegerOverflow, result2);

    const result3 = safeIntCast(u32, @as(i32, -1));
    try testing.expectError(SafetyError.IntegerUnderflow, result3);
}

test "SecureRng" {
    const testing = std.testing;

    var rng = SecureRng.init();
    var buf: [32]u8 = undefined;
    rng.bytes(&buf);

    var all_zero = true;
    for (buf) |b| {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    try testing.expect(!all_zero);
}

test "MonotonicClock" {
    const testing = std.testing;

    var clock = MonotonicClock.init();
    const t1 = clock.now();
    std.time.sleep(1_000_000);
    const t2 = clock.now();

    try testing.expect(t2 > t1);
}

test "secureCompare" {
    const testing = std.testing;

    const a = [_]u8{ 1, 2, 3, 4 };
    const b = [_]u8{ 1, 2, 3, 4 };
    const c = [_]u8{ 1, 2, 3, 5 };

    try testing.expect(secureCompare(&a, &b));
    try testing.expect(!secureCompare(&a, &c));
}
