const std = @import("std");
const net = std.net;
const mem = std.mem;
const fs = std.fs;
const Thread = std.Thread;
const Allocator = mem.Allocator;
const RSFLayer = @import("../processor/rsf.zig").RSFLayer;
const Ranker = @import("../ranker/ranker.zig").Ranker;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const SSI = @import("../index/ssi.zig").SSI;
const Tensor = @import("../core/tensor.zig").Tensor;
const ModelFormat = @import("../core/model_io.zig").ModelFormat;
const importModel = @import("../core/model_io.zig").importModel;
const core_memory = @import("../core/memory.zig");
const nsir = @import("../core_relational/nsir_core.zig");
const sfd = @import("../optimizer/sfd.zig");
const accel = @import("../hw/accel/accel_interface.zig");

pub const ServerConfig = struct {
    port: u16 = 8080,
    host: []const u8 = "127.0.0.1",
    max_connections: u32 = 100,
    request_timeout_ms: u64 = 30000,
    batch_size: usize = 32,
    model_path: ?[]const u8 = null,
    rate_limit_per_minute: u32 = 10,
    max_request_size_bytes: usize = 1024 * 1024,
    require_api_key: bool = true,
    dataset_path: ?[]const u8 = null,
    sample_limit: ?usize = null,
    num_validation_samples: usize = 100,
};

const RateLimiter = struct {
    const RequestLog = struct {
        timestamps: std.ArrayList(i64),
        mutex: Thread.Mutex,
    };

    logs: std.StringHashMap(RequestLog),
    key_storage: std.ArrayList([]u8),
    allocator: Allocator,
    mutex: Thread.Mutex,
    window_seconds: u64,
    max_requests: u32,

    pub fn init(allocator: Allocator, max_requests_per_minute: u32) RateLimiter {
        return RateLimiter{
            .logs = std.StringHashMap(RequestLog).init(allocator),
            .key_storage = std.ArrayList([]u8).init(allocator),
            .allocator = allocator,
            .mutex = Thread.Mutex{},
            .window_seconds = 60,
            .max_requests = max_requests_per_minute,
        };
    }

    pub fn deinit(self: *RateLimiter) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var iter = self.logs.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.timestamps.deinit();
        }
        self.logs.deinit();

        for (self.key_storage.items) |key| {
            self.allocator.free(key);
        }
        self.key_storage.deinit();
    }

    pub fn checkAndRecord(self: *RateLimiter, ip_address: []const u8) !bool {
        const now = std.time.timestamp();
        const cutoff = now - @as(i64, @intCast(self.window_seconds));

        self.mutex.lock();
        defer self.mutex.unlock();

        const result = self.logs.getOrPut(ip_address) catch return error.OutOfMemory;
        if (!result.found_existing) {
            const owned_key = self.allocator.dupe(u8, ip_address) catch return error.OutOfMemory;
            self.key_storage.append(owned_key) catch {
                self.allocator.free(owned_key);
                return error.OutOfMemory;
            };
            result.key_ptr.* = owned_key;
            result.value_ptr.* = RequestLog{
                .timestamps = std.ArrayList(i64).init(self.allocator),
                .mutex = Thread.Mutex{},
            };
        }

        var log = result.value_ptr;
        log.mutex.lock();
        defer log.mutex.unlock();

        var i: usize = 0;
        while (i < log.timestamps.items.len) {
            if (log.timestamps.items[i] < cutoff) {
                _ = log.timestamps.orderedRemove(i);
            } else {
                i += 1;
            }
        }

        if (log.timestamps.items.len >= self.max_requests) {
            return false;
        }

        log.timestamps.append(now) catch return error.OutOfMemory;
        return true;
    }
};

pub const InferenceRequest = struct {
    text: []const u8,
    max_tokens: ?usize = null,
    return_embeddings: bool = false,

    pub fn fromJson(allocator: Allocator, json: []const u8) !InferenceRequest {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidJson;
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidJson;

        const text_val = root.object.get("text") orelse return error.MissingTextField;
        if (text_val != .string) return error.InvalidTextField;

        var max_tokens: ?usize = null;
        if (root.object.get("max_tokens")) |mt| {
            if (mt == .integer) {
                if (mt.integer < 0) return error.InvalidMaxTokens;
                if (mt.integer > 1000000) return error.MaxTokensTooLarge;
                max_tokens = @intCast(mt.integer);
            }
        }

        var return_embeddings = false;
        if (root.object.get("return_embeddings")) |re| {
            if (re == .bool) {
                return_embeddings = re.bool;
            }
        }

        return InferenceRequest{
            .text = try allocator.dupe(u8, text_val.string),
            .max_tokens = max_tokens,
            .return_embeddings = return_embeddings,
        };
    }

    pub fn deinit(self: *InferenceRequest, allocator: Allocator) void {
        allocator.free(self.text);
    }
};

pub const InferenceResponse = struct {
    tokens: []u32,
    embeddings: ?[]f32 = null,
    processing_time_ms: f64,

    pub fn toJson(self: *const InferenceResponse, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{\"tokens\":[");
        var i: usize = 0;
        while (i < self.tokens.len) : (i += 1) {
            if (i > 0) try writer.writeAll(",");
            try writer.print("{d}", .{self.tokens[i]});
        }
        try writer.writeAll("]");

        if (self.embeddings) |emb| {
            try writer.writeAll(",\"embeddings\":[");
            var j: usize = 0;
            while (j < emb.len) : (j += 1) {
                if (j > 0) try writer.writeAll(",");
                try writer.print("{d:.6}", .{emb[j]});
            }
            try writer.writeAll("]");
        }

        try writer.print(",\"processing_time_ms\":{d:.2}", .{self.processing_time_ms});
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }

    pub fn deinit(self: *InferenceResponse, allocator: Allocator) void {
        allocator.free(self.tokens);
        if (self.embeddings) |emb| {
            allocator.free(emb);
        }
    }
};

pub const HealthResponse = struct {
    status: []const u8 = "healthy",
    uptime_seconds: u64,
    model_loaded: bool,
    version: []const u8 = "1.0.0",

    pub fn toJson(self: *const HealthResponse, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{");
        try writer.print("\"status\":\"{s}\",", .{self.status});
        try writer.print("\"uptime_seconds\":{d},", .{self.uptime_seconds});
        try writer.print("\"model_loaded\":{},", .{self.model_loaded});
        try writer.print("\"version\":\"{s}\"", .{self.version});
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }
};

fn nsirModulateForInference(data: []f32) void {
    if (data.len == 0) return;
    var mean: f32 = 0.0;
    for (data) |v| {
        mean += v;
    }
    mean /= @as(f32, @floatFromInt(data.len));
    var i: usize = 0;
    while (i < data.len) : (i += 1) {
        if (data[i] > mean) {
            data[i] *= 1.05;
        }
    }
}

pub const InferenceServer = struct {
    allocator: Allocator,
    config: ServerConfig,
    model: ?ModelFormat = null,
    ssi: ?SSI = null,
    ranker: ?Ranker = null,
    request_count: u64,
    inference_mutex: Thread.Mutex,
    start_time: i64,
    running: std.atomic.Value(bool),
    rate_limiter: RateLimiter,
    api_key: ?[]const u8,

    pub fn init(allocator: Allocator, config: ServerConfig) !InferenceServer {
        var api_key: ?[]const u8 = null;
        if (config.require_api_key) {
            if (std.posix.getenv("JAIDE_API_KEY")) |env_key| {
                api_key = try allocator.dupe(u8, env_key);
                std.debug.print("API key loaded from environment\n", .{});
            }
        }

        return InferenceServer{
            .allocator = allocator,
            .config = config,
            .request_count = 0,
            .inference_mutex = Thread.Mutex{},
            .start_time = std.time.timestamp(),
            .running = std.atomic.Value(bool).init(false),
            .rate_limiter = RateLimiter.init(allocator, config.rate_limit_per_minute),
            .api_key = api_key,
        };
    }

    pub fn deinit(self: *InferenceServer) void {
        if (self.model) |*model| {
            model.deinit();
        }
        if (self.ssi) |*ssi| {
            ssi.deinit();
        }
        if (self.ranker) |*r| {
            r.deinit();
        }
        if (self.api_key) |key| {
            self.allocator.free(key);
        }
        self.rate_limiter.deinit();
    }

    pub fn loadModel(self: *InferenceServer, path: []const u8) !void {
        self.model = try importModel(path, self.allocator);
        self.ssi = SSI.init(self.allocator);
        self.ranker = try Ranker.init(self.allocator, 3, 8, 42);
    }

    pub fn start(self: *InferenceServer) !void {
        const address = try net.Address.parseIp(self.config.host, self.config.port);
        var server = address.listen(.{
            .reuse_address = true,
        }) catch |err| {
            std.debug.print("Failed to listen: {}\n", .{err});
            return err;
        };
        defer server.deinit();

        self.running.store(true, .seq_cst);

        std.debug.print("Security configuration:\n", .{});
        std.debug.print("   - API key auth: {s}\n", .{if (self.api_key != null) "ENABLED" else "DISABLED"});
        std.debug.print("   - Rate limiting: {d} requests/min per IP\n", .{self.config.rate_limit_per_minute});
        std.debug.print("   - Max request size: {d} bytes\n", .{self.config.max_request_size_bytes});
        std.debug.print("\n", .{});
        std.debug.print("Inference server listening on {s}:{d}\n", .{self.config.host, self.config.port});

        while (self.running.load(.seq_cst)) {
            const connection = server.accept() catch |err| {
                std.debug.print("Failed to accept connection: {}\n", .{err});
                continue;
            };

            self.handleStreamConnection(connection.stream, connection.address) catch |err| {
                std.debug.print("Error handling connection: {}\n", .{err});
            };
        }
    }

    pub fn stop(self: *InferenceServer) void {
        self.running.store(false, .seq_cst);
    }

    fn handleStreamConnection(self: *InferenceServer, stream: net.Stream, client_addr: net.Address) !void {
        defer stream.close();

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const temp_allocator = arena.allocator();

        var ip_buf: [64]u8 = undefined;
        const ip_len = std.fmt.bufPrint(&ip_buf, "{}", .{client_addr}) catch return;
        const ip_str = try temp_allocator.dupe(u8, ip_len);

        var total_read: usize = 0;
        var buf: [65536]u8 = undefined;
        const max_read = @min(buf.len, self.config.max_request_size_bytes);

        while (total_read < max_read) {
            const bytes_read = stream.read(buf[total_read..max_read]) catch break;
            if (bytes_read == 0) break;
            total_read += bytes_read;

            if (mem.indexOf(u8, buf[0..total_read], "\r\n\r\n")) |_| {
                break;
            }
        }

        if (total_read == 0) return;

        if (total_read >= self.config.max_request_size_bytes) {
            try self.sendError(stream, "Request too large", 413);
            return;
        }

        const request_data = buf[0..total_read];

        const method_end = mem.indexOf(u8, request_data, " ") orelse return error.InvalidRequest;
        const method = request_data[0..method_end];

        const path_start = method_end + 1;
        const path_end = mem.indexOfPos(u8, request_data, path_start, " ") orelse return error.InvalidRequest;
        const path = request_data[path_start..path_end];

        const headers_end = mem.indexOf(u8, request_data, "\r\n\r\n") orelse return error.InvalidRequest;
        const headers = request_data[0..headers_end];
        const body = if (headers_end + 4 < request_data.len) request_data[headers_end + 4 ..] else "";

        if (mem.eql(u8, method, "GET") and mem.eql(u8, path, "/v1/health")) {
            try self.handleHealth(stream, temp_allocator);
        } else if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/v1/inference")) {
            const rate_allowed = self.rate_limiter.checkAndRecord(ip_str) catch false;
            if (!rate_allowed) {
                try self.sendError(stream, "Rate limit exceeded", 429);
                std.debug.print("Rate limit exceeded for IP: {s}\n", .{ip_str});
                return;
            }

            if (self.api_key) |expected_key| {
                const auth_valid = self.checkAuthorization(headers, expected_key);
                if (!auth_valid) {
                    try self.sendError(stream, "Unauthorized - Invalid or missing API key", 401);
                    std.debug.print("Unauthorized access attempt from IP: {s}\n", .{ip_str});
                    return;
                }
            }

            try self.handleInference(stream, body, temp_allocator);
        } else {
            try self.sendNotFound(stream);
        }
    }

    fn checkAuthorization(self: *InferenceServer, headers: []const u8, expected_key: []const u8) bool {
        _ = self;

        var lines = mem.splitSequence(u8, headers, "\r\n");
        while (lines.next()) |line| {
            const lower_check = if (line.len >= 14)
                mem.startsWith(u8, line, "Authorization:") or mem.startsWith(u8, line, "authorization:")
            else
                false;

            if (lower_check) {
                const value_start = mem.indexOf(u8, line, ":") orelse continue;
                const value = mem.trim(u8, line[value_start + 1 ..], " \t");

                if (value.len > 7) {
                    const prefix_check = mem.startsWith(u8, value, "Bearer ") or mem.startsWith(u8, value, "bearer ");
                    if (prefix_check) {
                        const token = mem.trim(u8, value[7..], " \t");
                        return mem.eql(u8, token, expected_key);
                    }
                }
            }
        }

        return false;
    }

    fn handleHealth(self: *InferenceServer, stream: net.Stream, allocator: Allocator) !void {
        const uptime = @as(u64, @intCast(std.time.timestamp() - self.start_time));

        const response = HealthResponse{
            .uptime_seconds = uptime,
            .model_loaded = self.model != null,
        };

        const json = try response.toJson(allocator);
        defer allocator.free(json);

        var response_buf = std.ArrayList(u8).init(allocator);
        defer response_buf.deinit();
        var writer = response_buf.writer();

        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: application/json\r\n");
        try writer.writeAll("Cache-Control: no-cache\r\n");
        try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
        try writer.print("Content-Length: {d}\r\n", .{json.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(json);

        _ = stream.write(response_buf.items) catch {};
    }

    fn handleInference(self: *InferenceServer, stream: net.Stream, body: []const u8, allocator: Allocator) !void {
        if (self.model == null or self.model.?.mgt == null) {
            try self.sendError(stream, "Model not loaded", 503);
            return;
        }

        const start_time = std.time.milliTimestamp();

        var request = InferenceRequest.fromJson(allocator, body) catch {
            try self.sendError(stream, "Invalid JSON request", 400);
            return;
        };
        defer request.deinit(allocator);

        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        self.model.?.mgt.?.encode(request.text, &tokens) catch {
            try self.sendError(stream, "Encoding failed", 500);
            return;
        };

        const max_tokens = request.max_tokens orelse tokens.items.len;
        const final_tokens = if (tokens.items.len > max_tokens)
            tokens.items[0..max_tokens]
        else
            tokens.items;

        var embeddings: ?[]f32 = null;
        if (request.return_embeddings and self.model.?.rsf != null) {
            const dim = (self.model.?.rsf.?.ctrl orelse {
                try self.sendError(stream, "RSF not initialized", 500);
                return;
            }).dim;
            const batch_size = 1;

            var input_tensor = Tensor.init(allocator, &.{batch_size, dim * 2}) catch {
                try self.sendError(stream, "Failed to create embeddings", 500);
                return;
            };
            defer input_tensor.deinit();

            var k: usize = 0;
            while (k < input_tensor.data.len) : (k += 1) {
                input_tensor.data[k] = if (k < final_tokens.len)
                    @as(f32, @floatFromInt(final_tokens[k])) / 1000.0
                else
                    0.0;
            }

            self.model.?.rsf.?.forward(&input_tensor) catch {
                try self.sendError(stream, "Embedding generation failed", 500);
                return;
            };

            nsirModulateForInference(input_tensor.data);

            embeddings = try allocator.alloc(f32, @min(dim, 128));
            var m: usize = 0;
            while (m < embeddings.?.len) : (m += 1) {
                embeddings.?[m] = if (m < input_tensor.data.len) input_tensor.data[m] else 0.0;
            }
        }

        self.inference_mutex.lock();
        defer self.inference_mutex.unlock();

        if (self.ssi) |*ssi_idx| {
            const is_anchor = (self.request_count % 10 == 0);
            ssi_idx.addSequence(final_tokens, self.request_count, is_anchor) catch {};

            if (self.ranker) |*rnk| {
                const candidates = ssi_idx.retrieveTopK(final_tokens, 3, allocator) catch null;
                if (candidates) |cands| {
                    defer allocator.free(cands);
                    rnk.rankCandidatesWithQuery(cands, final_tokens, ssi_idx, allocator) catch {};
                }
            }
        }
        self.request_count += 1;

        const end_time = std.time.milliTimestamp();
        const processing_time = @as(f64, @floatFromInt(end_time - start_time));

        const tokens_copy = try allocator.dupe(u32, final_tokens);

        var response = InferenceResponse{
            .tokens = tokens_copy,
            .embeddings = embeddings,
            .processing_time_ms = processing_time,
        };
        defer {
            allocator.free(response.tokens);
            if (response.embeddings) |emb| allocator.free(emb);
        }

        const json = try response.toJson(allocator);
        defer allocator.free(json);

        var response_buf = std.ArrayList(u8).init(allocator);
        defer response_buf.deinit();
        var writer = response_buf.writer();

        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: application/json\r\n");
        try writer.writeAll("Cache-Control: no-cache\r\n");
        try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
        try writer.print("Content-Length: {d}\r\n", .{json.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(json);

        _ = stream.write(response_buf.items) catch {};
    }

    fn sendError(self: *InferenceServer, stream: net.Stream, message: []const u8, status_code: u16) !void {
        _ = self;
        var buf: [1024]u8 = undefined;
        const json = std.fmt.bufPrint(&buf, "{{\"error\":\"{s}\"}}", .{message}) catch return;

        const status_text = switch (status_code) {
            400 => "Bad Request",
            401 => "Unauthorized",
            404 => "Not Found",
            413 => "Payload Too Large",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            else => "Error",
        };

        var response_buf: [2048]u8 = undefined;
        const response = std.fmt.bufPrint(
            &response_buf,
            "HTTP/1.1 {d} {s}\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Cache-Control: no-cache\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "Content-Length: {d}\r\n" ++
                "\r\n" ++
                "{s}",
            .{status_code, status_text, json.len, json},
        ) catch return;

        _ = stream.write(response) catch {};
    }

    fn sendNotFound(self: *InferenceServer, stream: net.Stream) !void {
        try self.sendError(stream, "Endpoint not found", 404);
    }
};

pub const BatchInferenceRequest = struct {
    texts: [][]const u8,
    max_tokens: ?usize = null,
    return_embeddings: bool = false,

    pub fn fromJson(allocator: Allocator, json: []const u8) !BatchInferenceRequest {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidJson;
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidJson;

        const texts_array = root.object.get("texts") orelse return error.MissingTextsField;
        if (texts_array != .array) return error.InvalidTextsField;

        var texts = try allocator.alloc([]const u8, texts_array.array.items.len);
        var n: usize = 0;
        while (n < texts_array.array.items.len) : (n += 1) {
            if (texts_array.array.items[n] != .string) {
                var cleanup_idx: usize = 0;
                while (cleanup_idx < n) : (cleanup_idx += 1) {
                    allocator.free(texts[cleanup_idx]);
                }
                allocator.free(texts);
                return error.InvalidTextsField;
            }
            texts[n] = try allocator.dupe(u8, texts_array.array.items[n].string);
        }

        var max_tokens: ?usize = null;
        if (root.object.get("max_tokens")) |mt| {
            if (mt == .integer) {
                if (mt.integer < 0) return error.InvalidMaxTokens;
                if (mt.integer > 1000000) return error.MaxTokensTooLarge;
                max_tokens = @intCast(mt.integer);
            }
        }

        var return_embeddings = false;
        if (root.object.get("return_embeddings")) |re| {
            if (re == .bool) {
                return_embeddings = re.bool;
            }
        }

        return BatchInferenceRequest{
            .texts = texts,
            .max_tokens = max_tokens,
            .return_embeddings = return_embeddings,
        };
    }

    pub fn deinit(self: *BatchInferenceRequest, allocator: Allocator) void {
        for (self.texts) |text| {
            allocator.free(text);
        }
        allocator.free(self.texts);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var config = ServerConfig{};

    if (std.posix.getenv("JAIDE_PORT")) |port_str| {
        config.port = std.fmt.parseInt(u16, port_str, 10) catch 8080;
    }

    if (std.posix.getenv("JAIDE_HOST")) |host| {
        config.host = host;
    }

    var server = try InferenceServer.init(allocator, config);
    defer server.deinit();

    if (config.model_path) |path| {
        server.loadModel(path) catch |err| {
            std.debug.print("Failed to load model: {}\n", .{err});
        };
    }

    try server.start();
}
