const std = @import("std");
const http = std.http;

pub const ModalGPUClient = struct {
    allocator: std.mem.Allocator,
    api_token: []const u8,
    http_client: http.Client,
    gpu_count: usize,
    gpu_preferences: [2][]const u8,

    pub fn init(allocator: std.mem.Allocator, api_token: []const u8) !ModalGPUClient {
        return .{
            .allocator = allocator,
            .api_token = try allocator.dupe(u8, api_token),
            .http_client = http.Client{ .allocator = allocator },
            .gpu_count = 8,
            .gpu_preferences = .{ "B300", "B200" },
        };
    }

    pub fn deinit(self: *ModalGPUClient) void {
        self.allocator.free(self.api_token);
        self.http_client.deinit();
    }

    pub fn deployTrainingJob(self: *ModalGPUClient, model_path: []const u8, dataset_path: []const u8) ![]const u8 {
        const uri = try std.Uri.parse("https://api.modal.com/v1/functions/deploy");
        const payload = try std.json.stringifyAlloc(self.allocator, .{
            .gpu = self.gpu_preferences,
            .gpu_count = self.gpu_count,
            .image = "jaide-v40-training",
            .model_path = model_path,
            .dataset_path = dataset_path,
            .batch_size = 32,
            .epochs = 10,
        }, .{});
        defer self.allocator.free(payload);

        return try self.sendRequest(.POST, uri, payload);
    }

    pub fn getJobStatus(self: *ModalGPUClient, job_id: []const u8) ![]const u8 {
        const uri_str = try std.fmt.allocPrint(self.allocator, "https://api.modal.com/v1/functions/{s}/status", .{job_id});
        defer self.allocator.free(uri_str);

        const uri = try std.Uri.parse(uri_str);
        return try self.sendRequest(.GET, uri, null);
    }

    fn sendRequest(self: *ModalGPUClient, method: http.Method, uri: std.Uri, body: ?[]const u8) ![]const u8 {
        const authorization_value = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_token});
        defer self.allocator.free(authorization_value);

        if (@hasDecl(http.Client, "open")) {
            const open_params_len = @typeInfo(@TypeOf(http.Client.open)).Fn.params.len;
            if (open_params_len == 5) {
                var headers = http.Headers{ .allocator = self.allocator };
                defer headers.deinit();

                try headers.append("Authorization", authorization_value);
                if (body != null) {
                    try headers.append("Content-Type", "application/json");
                }

                var req = try self.http_client.open(method, uri, headers, .{});
                defer req.deinit();

                return try self.sendAndReadResponse(&req, body);
            }

            if (open_params_len == 4) {
                const server_header_buffer = try self.allocator.alloc(u8, 16 * 1024);
                defer self.allocator.free(server_header_buffer);

                var req = try self.http_client.open(method, uri, .{
                    .server_header_buffer = server_header_buffer,
                });
                defer req.deinit();

                try req.headers.append("Authorization", authorization_value);
                if (body != null) {
                    try req.headers.append("Content-Type", "application/json");
                }

                return try self.sendAndReadResponse(&req, body);
            }

            @compileError("Unsupported std.http.Client.open signature");
        }

        if (@hasDecl(http.Client, "request")) {
            var headers = http.Headers{ .allocator = self.allocator };
            defer headers.deinit();

            try headers.append("Authorization", authorization_value);
            if (body != null) {
                try headers.append("Content-Type", "application/json");
            }

            var req = try self.http_client.request(method, uri, headers, .{});
            defer req.deinit();

            return try self.sendAndReadResponse(&req, body);
        }

        @compileError("Unsupported std.http.Client API");
    }

    fn sendAndReadResponse(self: *ModalGPUClient, req: *http.Client.Request, body: ?[]const u8) ![]const u8 {
        if (body) |request_body| {
            req.transfer_encoding = .{ .content_length = request_body.len };
            try sendCompat(req);
            try req.writer().writeAll(request_body);
            try req.finish();
            try req.wait();
        } else {
            try sendCompat(req);
            try req.finish();
            try req.wait();
        }

        const response_body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024);
        return response_body;
    }

    fn sendCompat(req: *http.Client.Request) !void {
        const send_params_len = @typeInfo(@TypeOf(http.Client.Request.send)).Fn.params.len;
        if (send_params_len == 1) {
            try req.send();
            return;
        }
        if (send_params_len == 2) {
            try req.send(.{});
            return;
        }
        @compileError("Unsupported std.http.Client.Request.send signature");
    }
};