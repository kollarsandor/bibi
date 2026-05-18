
const std = @import("std");
const http = std.http;

pub const IBMQuantumClient = struct {
    allocator: std.mem.Allocator,
    api_token: []const u8,
    crn: []const u8,
    http_client: http.Client,
    owns_crn: bool,

    pub fn init(allocator: std.mem.Allocator, api_token: []const u8) !IBMQuantumClient {
        return initWithCrn(allocator, api_token, null);
    }

    pub fn initWithCrn(allocator: std.mem.Allocator, api_token: []const u8, crn_override: ?[]const u8) !IBMQuantumClient {
        const crn = if (crn_override) |c|
            try allocator.dupe(u8, c)
        else if (std.posix.getenv("IBM_QUANTUM_CRN")) |env_crn|
            try allocator.dupe(u8, env_crn)
        else
            return error.MissingIBMQuantumCRN;

        return .{
            .allocator = allocator,
            .api_token = try allocator.dupe(u8, api_token),
            .crn = crn,
            .http_client = http.Client{ .allocator = allocator },
            .owns_crn = true,
        };
    }

    pub fn deinit(self: *IBMQuantumClient) void {
        self.allocator.free(self.api_token);
        if (self.owns_crn) {
            self.allocator.free(self.crn);
        }
        self.http_client.deinit();
    }

    pub fn submitJob(self: *IBMQuantumClient, qasm: []const u8) ![]const u8 {
        const uri = try std.Uri.parse("https://cloud.ibm.com/quantum/api/v1/jobs");

        var headers = http.Headers{ .allocator = self.allocator };
        defer headers.deinit();

        try headers.append("Authorization", try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_token}));
        try headers.append("Content-Type", "application/json");

        const payload = try std.fmt.allocPrint(self.allocator,
            \\{{"qasm": "{s}", "backend": "ibm_brisbane", "shots": 1024}}
        , .{qasm});
        defer self.allocator.free(payload);

        var req = try self.http_client.open(.POST, uri, headers, .{});
        defer req.deinit();

        req.transfer_encoding = .chunked;
        try req.send(.{});
        try req.writeAll(payload);
        try req.finish();
        try req.wait();

        const body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024);
        return body;
    }

    pub fn getJobResult(self: *IBMQuantumClient, job_id: []const u8) ![]const u8 {
        const uri_str = try std.fmt.allocPrint(self.allocator, "https://cloud.ibm.com/quantum/api/v1/jobs/{s}", .{job_id});
        defer self.allocator.free(uri_str);

        const uri = try std.Uri.parse(uri_str);

        var headers = http.Headers{ .allocator = self.allocator };
        defer headers.deinit();

        try headers.append("Authorization", try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_token}));

        var req = try self.http_client.open(.GET, uri, headers, .{});
        defer req.deinit();

        try req.send(.{});
        try req.finish();
        try req.wait();

        const body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024);
        return body;
    }
};
