const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Mutex = std.Thread.Mutex;

const chaos = @import("chaos_core.zig");
const MemoryBlock = chaos.MemoryBlock;
const MemoryBlockState = chaos.MemoryBlockState;
const ContentAddressableStorage = chaos.ContentAddressableStorage;
const DataFlowAnalyzer = chaos.DataFlowAnalyzer;

const RETENTION_AGE_WEIGHT: f64 = 0.3;
const RETENTION_FREQUENCY_WEIGHT: f64 = 0.2;
const RETENTION_BASE_WEIGHT: f64 = 0.5;
const NANOSECONDS_TO_MILLISECONDS: f64 = 1_000_000.0;
const HASH_SIZE: usize = 16;
const HASH_BITS: usize = HASH_SIZE * 8;
const MAX_INPUT_SIZE: usize = 100 * 1024 * 1024;
const JACCARD_SAMPLE_SIZE: usize = 1000;
const MAX_ENTANGLEMENT_PAIRS: usize = 100;
const DEFAULT_SURPRISE_THRESHOLD: f64 = 0.3;
const TEMPORAL_NOVELTY_WINDOW_NS: i128 = 86_400_000_000_000;
const BIGRAM_SPACE: usize = 1 << 16;
const BIGRAM_WORDS: usize = BIGRAM_SPACE / 64;
const FREQUENCY_SATURATION: f64 = 8.0;
const FLOAT_EPSILON: f64 = 1e-12;

fn sanitizeUnit(value: f64, fallback: f64) f64 {
    if (!std.math.isFinite(value)) return fallback;
    return @max(0.0, @min(1.0, value));
}

fn sanitizeThreshold(value: f64) f64 {
    return sanitizeUnit(value, DEFAULT_SURPRISE_THRESHOLD);
}

fn stableNow(previous: ?i128) i128 {
    const observed = std.time.nanoTimestamp();
    if (previous) |prior| {
        if (observed < prior) return prior;
    }
    return observed;
}

fn lexLessThan(a: [HASH_SIZE]u8, b: [HASH_SIZE]u8) bool {
    return std.mem.order(u8, a[0..], b[0..]) == .lt;
}

pub const SurpriseMetrics = struct {
    jaccard_dissimilarity: f64,
    content_hash_distance: f64,
    temporal_novelty: f64,
    combined_surprise: f64,

    pub fn init(jaccard: f64, hash_dist: f64, temporal: f64) SurpriseMetrics {
        const clamped_jaccard = sanitizeUnit(jaccard, 0.0);
        const clamped_hash = sanitizeUnit(hash_dist, 0.0);
        const clamped_temporal = sanitizeUnit(temporal, 0.0);
        const combined = (clamped_jaccard + clamped_hash + clamped_temporal) / 3.0;
        return SurpriseMetrics{
            .jaccard_dissimilarity = clamped_jaccard,
            .content_hash_distance = clamped_hash,
            .temporal_novelty = clamped_temporal,
            .combined_surprise = sanitizeUnit(combined, 0.0),
        };
    }

    pub fn exceedsThreshold(self: *const SurpriseMetrics, threshold: f64) bool {
        return self.combined_surprise > sanitizeThreshold(threshold);
    }
};

pub const SurpriseRecord = struct {
    block_id: [HASH_SIZE]u8,
    surprise_score: f64,
    creation_time: i128,
    last_access_time: i128,
    retention_priority: f64,
    access_frequency: usize,

    fn recomputeRetention(self: *SurpriseRecord, now: i128) void {
        const effective_now = if (now < self.last_access_time) self.last_access_time else now;
        const age_ns = effective_now - self.last_access_time;
        const clamped_age: i64 = @intCast(@min(age_ns, @as(i128, std.math.maxInt(i64))));
        const age_ms: f64 = @as(f64, @floatFromInt(clamped_age)) / NANOSECONDS_TO_MILLISECONDS;
        const recency_factor = 1.0 / (1.0 + age_ms);
        const freq_f: f64 = @floatFromInt(self.access_frequency);
        const frequency_factor = freq_f / (freq_f + FREQUENCY_SATURATION);
        const weight = RETENTION_BASE_WEIGHT + RETENTION_AGE_WEIGHT * recency_factor + RETENTION_FREQUENCY_WEIGHT * frequency_factor;
        self.retention_priority = sanitizeUnit(self.surprise_score, 0.0) * sanitizeUnit(weight, 1.0);
    }

    pub fn init(block_id: [HASH_SIZE]u8, score: f64) SurpriseRecord {
        const now = stableNow(null);
        var record = SurpriseRecord{
            .block_id = block_id,
            .surprise_score = sanitizeUnit(score, 0.0),
            .creation_time = now,
            .last_access_time = now,
            .retention_priority = 0.0,
            .access_frequency = 1,
        };
        record.recomputeRetention(now);
        return record;
    }

    pub fn updateRetention(self: *SurpriseRecord) void {
        self.recomputeRetention(stableNow(self.last_access_time));
    }

    pub fn recordAccess(self: *SurpriseRecord) void {
        const now = stableNow(self.last_access_time);
        self.access_frequency += 1;
        self.last_access_time = now;
        self.recomputeRetention(now);
    }

    pub fn getRetentionPriority(self: *const SurpriseRecord) f64 {
        return self.retention_priority;
    }

    pub fn getAccessFrequency(self: *const SurpriseRecord) usize {
        return self.access_frequency;
    }
};

pub const SurpriseMemoryStatistics = struct {
    total_blocks: usize,
    high_surprise_blocks: usize,
    low_surprise_blocks: usize,
    average_surprise: f64,
    surprise_threshold: f64,
    evictions_due_to_low_surprise: usize,
    novel_block_allocations: usize,
    total_surprise_sum: f64,

    pub fn init(threshold: f64) SurpriseMemoryStatistics {
        return SurpriseMemoryStatistics{
            .total_blocks = 0,
            .high_surprise_blocks = 0,
            .low_surprise_blocks = 0,
            .average_surprise = 0.0,
            .surprise_threshold = sanitizeThreshold(threshold),
            .evictions_due_to_low_surprise = 0,
            .novel_block_allocations = 0,
            .total_surprise_sum = 0.0,
        };
    }

    pub fn addBlock(self: *SurpriseMemoryStatistics, surprise_score: f64, threshold: f64) void {
        const score = sanitizeUnit(surprise_score, 0.0);
        const clean_threshold = sanitizeThreshold(threshold);
        self.total_blocks += 1;
        self.total_surprise_sum += score;
        if (score > clean_threshold) {
            self.high_surprise_blocks += 1;
            self.novel_block_allocations += 1;
        } else {
            self.low_surprise_blocks += 1;
        }
        self.recalculateAverage();
    }

    pub fn removeBlock(self: *SurpriseMemoryStatistics, surprise_score: f64, threshold: f64) void {
        if (self.total_blocks == 0) return;
        const score = sanitizeUnit(surprise_score, 0.0);
        const clean_threshold = sanitizeThreshold(threshold);
        self.total_blocks -= 1;
        if (self.total_surprise_sum >= score) {
            self.total_surprise_sum -= score;
            if (self.total_surprise_sum < FLOAT_EPSILON) self.total_surprise_sum = 0.0;
        } else {
            self.total_surprise_sum = 0.0;
        }
        if (score > clean_threshold) {
            if (self.high_surprise_blocks > 0) self.high_surprise_blocks -= 1;
            if (self.novel_block_allocations > 0) self.novel_block_allocations -= 1;
        } else {
            if (self.low_surprise_blocks > 0) self.low_surprise_blocks -= 1;
        }
        self.recalculateAverage();
    }

    fn recalculateAverage(self: *SurpriseMemoryStatistics) void {
        if (self.total_blocks > 0) {
            self.average_surprise = self.total_surprise_sum / @as(f64, @floatFromInt(self.total_blocks));
        } else {
            self.average_surprise = 0.0;
            self.total_surprise_sum = 0.0;
        }
    }
};

const CandidateItem = struct {
    block_id: [HASH_SIZE]u8,
    priority: f64,
};

pub const SurpriseMemoryManager = struct {
    storage: *ContentAddressableStorage,
    flow_analyzer: *DataFlowAnalyzer,
    surprise_records: std.HashMap([HASH_SIZE]u8, SurpriseRecord, chaos.BlockIdContext, std.hash_map.default_max_load_percentage),
    surprise_threshold: f64,
    statistics: SurpriseMemoryStatistics,
    allocator: Allocator,
    mutex: Mutex,
    owns_storage: bool,
    owns_analyzer: bool,

    const Self = @This();

    fn computeContentHash(data: []const u8) [HASH_SIZE]u8 {
        var hash_out: [32]u8 = undefined;
        Sha256.hash(data, &hash_out, .{});
        var result: [HASH_SIZE]u8 = undefined;
        var i: usize = 0;
        while (i < HASH_SIZE) : (i += 1) {
            result[i] = hash_out[i] ^ hash_out[i + HASH_SIZE];
        }
        return result;
    }

    fn clearPresence(presence: *[BIGRAM_WORDS]u64) void {
        @memset(presence[0..], 0);
    }

    fn setPresenceBit(presence: *[BIGRAM_WORDS]u64, key: usize) bool {
        const word_index = key >> 6;
        const bit_index: u6 = @intCast(key & 63);
        const mask = @as(u64, 1) << bit_index;
        const was_set = (presence[word_index] & mask) != 0;
        if (!was_set) {
            presence[word_index] |= mask;
        }
        return !was_set;
    }

    fn buildBigramPresence(data: []const u8, presence: *[BIGRAM_WORDS]u64) usize {
        clearPresence(presence);
        if (data.len < 2) {
            if (data.len == 1) {
                const key = @as(usize, data[0]) << 8;
                _ = setPresenceBit(presence, key);
                return 1;
            }
            return 0;
        }

        const total_windows = data.len - 1;
        const max_windows = @min(total_windows, JACCARD_SAMPLE_SIZE);
        const stride = if (total_windows <= max_windows) @as(usize, 1) else (total_windows + max_windows - 1) / max_windows;

        var distinct: usize = 0;
        var sampled: usize = 0;
        var position: usize = 0;
        while (position < total_windows and sampled < max_windows) : (sampled += 1) {
            const idx = position;
            const key = (@as(usize, data[idx]) << 8) | @as(usize, data[idx + 1]);
            if (setPresenceBit(presence, key)) {
                distinct += 1;
            }
            position += stride;
        }

        const tail_key = (@as(usize, data[data.len - 2]) << 8) | @as(usize, data[data.len - 1]);
        if (setPresenceBit(presence, tail_key)) {
            distinct += 1;
        }
        return distinct;
    }

    fn computeJaccardDistance(data_a: []const u8, data_b: []const u8) f64 {
        var set_a: [BIGRAM_WORDS]u64 = undefined;
        var set_b: [BIGRAM_WORDS]u64 = undefined;
        const count_a = buildBigramPresence(data_a, &set_a);
        const count_b = buildBigramPresence(data_b, &set_b);

        if (count_a == 0 and count_b == 0) return 0.0;

        var intersection_count: usize = 0;
        var union_count: usize = 0;
        var idx: usize = 0;
        while (idx < BIGRAM_WORDS) : (idx += 1) {
            intersection_count += @popCount(set_a[idx] & set_b[idx]);
            union_count += @popCount(set_a[idx] | set_b[idx]);
        }
        if (union_count == 0) return 0.0;
        const similarity = @as(f64, @floatFromInt(intersection_count)) / @as(f64, @floatFromInt(union_count));
        return sanitizeUnit(1.0 - similarity, 0.0);
    }

    fn computeHashDistance(hash_a: [HASH_SIZE]u8, hash_b: [HASH_SIZE]u8) f64 {
        var hamming_dist: usize = 0;
        var hash_idx: usize = 0;
        while (hash_idx < HASH_SIZE) : (hash_idx += 1) {
            hamming_dist += @popCount(hash_a[hash_idx] ^ hash_b[hash_idx]);
        }
        return @as(f64, @floatFromInt(hamming_dist)) / @as(f64, @floatFromInt(HASH_BITS));
    }

    fn recomputeStatisticsLocked(self: *Self) void {
        var stats = SurpriseMemoryStatistics.init(self.surprise_threshold);
        stats.evictions_due_to_low_surprise = self.statistics.evictions_due_to_low_surprise;
        var iter = self.surprise_records.iterator();
        while (iter.next()) |entry| {
            stats.addBlock(entry.value_ptr.surprise_score, self.surprise_threshold);
        }
        self.statistics = stats;
    }

    fn computeTemporalNoveltyLocked(self: *Self, now: i128) f64 {
        if (self.surprise_records.count() == 0) return 1.0;

        var total_age: f64 = 0.0;
        var samples: usize = 0;
        var iter = self.surprise_records.iterator();
        while (iter.next()) |entry| {
            const ts = entry.value_ptr.last_access_time;
            const age_ns = if (now > ts) now - ts else @as(i128, 0);
            const bounded_age = @min(age_ns, TEMPORAL_NOVELTY_WINDOW_NS);
            total_age += @as(f64, @floatFromInt(bounded_age));
            samples += 1;
        }
        if (samples == 0) return 1.0;
        const window = @as(f64, @floatFromInt(TEMPORAL_NOVELTY_WINDOW_NS));
        const average_age = total_age / @as(f64, @floatFromInt(samples));
        return sanitizeUnit(average_age / window, 0.0);
    }

    fn sampleExistingBlocksLocked(self: *Self, new_data: []const u8, exclude_block_id: ?[HASH_SIZE]u8) !struct { min_jaccard: f64, min_hash: f64, compared: usize } {
        const block_count = self.storage.storage.count();
        const max_samples = @min(block_count, JACCARD_SAMPLE_SIZE);
        const new_hash = computeContentHash(new_data);
        var min_jaccard_dist: f64 = 1.0;
        var min_hash_dist: f64 = 1.0;
        var compared: usize = 0;
        var seen: usize = 0;
        const stride = if (block_count <= max_samples or max_samples == 0) @as(usize, 1) else (block_count + max_samples - 1) / max_samples;

        var iter = self.storage.storage.iterator();
        while (iter.next()) |entry| {
            const block_id = entry.key_ptr.*;
            if (exclude_block_id) |excluded| {
                if (std.mem.eql(u8, excluded[0..], block_id[0..])) {
                    continue;
                }
            }
            if (max_samples != 0 and (seen % stride) != 0 and compared + 1 < max_samples) {
                seen += 1;
                continue;
            }
            const existing_block = entry.value_ptr;
            const jaccard = computeJaccardDistance(new_data, existing_block.data);
            if (jaccard < min_jaccard_dist) min_jaccard_dist = jaccard;
            const hash_dist = computeHashDistance(new_hash, existing_block.content_hash);
            if (hash_dist < min_hash_dist) min_hash_dist = hash_dist;
            compared += 1;
            seen += 1;
            if (compared >= max_samples) break;
        }

        return .{ .min_jaccard = min_jaccard_dist, .min_hash = min_hash_dist, .compared = compared };
    }

    fn computeSurpriseLocked(self: *Self, new_data: []const u8, exclude_block_id: ?[HASH_SIZE]u8) !SurpriseMetrics {
        if (new_data.len > MAX_INPUT_SIZE) {
            return error.InputTooLarge;
        }

        if (self.storage.storage.count() == 0) {
            return SurpriseMetrics.init(1.0, 1.0, 1.0);
        }

        const now = stableNow(null);
        const sample = try self.sampleExistingBlocksLocked(new_data, exclude_block_id);
        if (sample.compared == 0) {
            return SurpriseMetrics.init(1.0, 1.0, 1.0);
        }
        const temporal_novelty = computeTemporalNoveltyLocked(self, now);
        return SurpriseMetrics.init(sample.min_jaccard, sample.min_hash, temporal_novelty);
    }

    fn refreshRecordPrioritiesLocked(self: *Self) void {
        var iter = self.surprise_records.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.updateRetention();
        }
    }

    fn appendStorageCandidatesLocked(self: *Self, candidates: *ArrayList(CandidateItem)) !void {
        try candidates.ensureTotalCapacity(self.storage.storage.count());
        var iter = self.storage.storage.iterator();
        while (iter.next()) |entry| {
            const block_id = entry.key_ptr.*;
            const priority = if (self.surprise_records.get(block_id)) |record| record.retention_priority else 0.0;
            candidates.appendAssumeCapacity(.{ .block_id = block_id, .priority = priority });
        }
    }

    fn candidateLess(_: void, a: CandidateItem, b: CandidateItem) bool {
        if (a.priority < b.priority) return true;
        if (a.priority > b.priority) return false;
        return lexLessThan(a.block_id, b.block_id);
    }

    pub fn init(allocator: Allocator, storage: *ContentAddressableStorage, analyzer: *DataFlowAnalyzer) Self {
        _ = MemoryBlock;
        _ = MemoryBlockState;
        _ = AutoHashMap;
        return Self{
            .storage = storage,
            .flow_analyzer = analyzer,
            .surprise_records = std.HashMap([HASH_SIZE]u8, SurpriseRecord, chaos.BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .surprise_threshold = DEFAULT_SURPRISE_THRESHOLD,
            .statistics = SurpriseMemoryStatistics.init(DEFAULT_SURPRISE_THRESHOLD),
            .allocator = allocator,
            .mutex = Mutex{},
            .owns_storage = false,
            .owns_analyzer = false,
        };
    }

    pub fn initWithOwnership(allocator: Allocator, storage: *ContentAddressableStorage, analyzer: *DataFlowAnalyzer, owns_storage: bool, owns_analyzer: bool) Self {
        var self = init(allocator, storage, analyzer);
        self.owns_storage = owns_storage;
        self.owns_analyzer = owns_analyzer;
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.surprise_records.deinit();
        if (self.owns_storage) {
            self.storage.deinit();
        }
        if (self.owns_analyzer) {
            self.flow_analyzer.deinit();
        }
    }

    pub fn setSurpriseThreshold(self: *Self, threshold: f64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.surprise_threshold = sanitizeThreshold(threshold);
        self.recomputeStatisticsLocked();
    }

    pub fn getSurpriseThreshold(self: *Self) f64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.surprise_threshold;
    }

    pub fn computeSurprise(self: *Self, new_data: []const u8) !SurpriseMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.computeSurpriseLocked(new_data, null);
    }

    fn storeBlockInternal(self: *Self, data: []const u8, preferred_core: ?usize, surprise: SurpriseMetrics) ![HASH_SIZE]u8 {
        const block_id = try self.storage.store(data, preferred_core);
        errdefer _ = self.storage.removeBlock(block_id);
        const record = SurpriseRecord.init(block_id, surprise.combined_surprise);
        try self.surprise_records.put(block_id, record);
        self.statistics.addBlock(surprise.combined_surprise, self.surprise_threshold);
        return block_id;
    }

    pub fn storeWithSurprise(self: *Self, data: []const u8, preferred_core: ?usize) ![HASH_SIZE]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (data.len > MAX_INPUT_SIZE) {
            return error.InputTooLarge;
        }

        if (self.storage.retrieveByContent(data)) |block_id| {
            if (self.surprise_records.getPtr(block_id)) |record| {
                record.recordAccess();
                return block_id;
            }

            const recovered_surprise = try self.computeSurpriseLocked(data, block_id);
            try self.surprise_records.put(block_id, SurpriseRecord.init(block_id, recovered_surprise.combined_surprise));
            self.statistics.addBlock(recovered_surprise.combined_surprise, self.surprise_threshold);
            return block_id;
        }

        const surprise = try self.computeSurpriseLocked(data, null);
        return try self.storeBlockInternal(data, preferred_core, surprise);
    }

    fn partialSort(items: []CandidateItem, k: usize) void {
        if (items.len <= 1 or k == 0) return;
        std.mem.sort(CandidateItem, items, {}, candidateLess);
    }

    pub fn evictLowSurpriseBlocks(self: *Self, target_capacity: usize) !usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        const current_size = self.storage.storage.count();
        if (current_size <= target_capacity) return 0;

        self.refreshRecordPrioritiesLocked();

        const to_evict = current_size - target_capacity;
        var candidates = ArrayList(CandidateItem).init(self.allocator);
        defer candidates.deinit();
        try self.appendStorageCandidatesLocked(&candidates);

        if (candidates.items.len == 0) return 0;

        const k = @min(to_evict, candidates.items.len);
        partialSort(candidates.items, k);

        var evicted_count: usize = 0;
        var idx: usize = 0;
        while (idx < k) : (idx += 1) {
            const candidate = candidates.items[idx];
            if (!self.storage.containsBlock(candidate.block_id)) continue;
            if (self.surprise_records.get(candidate.block_id)) |record| {
                self.statistics.removeBlock(record.surprise_score, self.surprise_threshold);
            }
            if (self.storage.removeBlock(candidate.block_id)) |_| {} else |_| {
                return error.StorageRemoveFailed;
            }
            if (self.storage.containsBlock(candidate.block_id)) {
                return error.StorageRemoveFailed;
            }
            _ = self.surprise_records.remove(candidate.block_id);
            evicted_count += 1;
        }

        self.statistics.evictions_due_to_low_surprise += evicted_count;
        return evicted_count;
    }

    pub fn organizeByEntanglement(self: *Self) !usize {
        var high_surprise_ids = ArrayList([HASH_SIZE]u8).init(self.allocator);
        errdefer high_surprise_ids.deinit();

        self.mutex.lock();
        var iter = self.surprise_records.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.surprise_score > self.surprise_threshold and high_surprise_ids.items.len < MAX_ENTANGLEMENT_PAIRS) {
                try high_surprise_ids.append(entry.key_ptr.*);
            }
        }
        self.mutex.unlock();
        defer high_surprise_ids.deinit();

        var entangled_pairs: usize = 0;
        var i: usize = 0;
        while (i < high_surprise_ids.items.len) : (i += 1) {
            var j: usize = i + 1;
            while (j < high_surprise_ids.items.len) : (j += 1) {
                if (self.storage.entangleBlocks(high_surprise_ids.items[i], high_surprise_ids.items[j])) |_| {
                    entangled_pairs += 1;
                } else |_| {}
            }
        }

        return entangled_pairs;
    }

    pub fn getStatistics(self: *Self) SurpriseMemoryStatistics {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.statistics;
    }

    pub fn getStatisticsConst(self: *Self) SurpriseMemoryStatistics {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.statistics;
    }

    pub fn getSurpriseRecord(self: *Self, block_id: [HASH_SIZE]u8) ?SurpriseRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.surprise_records.get(block_id);
    }

    pub fn getSurpriseRecordConst(self: *Self, block_id: [HASH_SIZE]u8) ?SurpriseRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.surprise_records.get(block_id);
    }

    pub fn containsRecord(self: *Self, block_id: [HASH_SIZE]u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.surprise_records.contains(block_id);
    }

    pub fn getRecordCount(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.surprise_records.count();
    }
};

test "surprise_memory_basic" {
    const allocator = std.testing.allocator;

    var storage = ContentAddressableStorage.init(allocator, 1024);
    defer storage.deinit();

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var manager = SurpriseMemoryManager.init(allocator, &storage, &analyzer);
    defer manager.deinit();

    const data1 = "unique_data_content_1";
    const data2 = "unique_data_content_2";

    const block1 = try manager.storeWithSurprise(data1, null);
    const block2 = try manager.storeWithSurprise(data2, null);

    try std.testing.expect(!std.mem.eql(u8, block1[0..], block2[0..]));

    const stats = manager.getStatistics();
    try std.testing.expectEqual(@as(usize, 2), stats.total_blocks);
}

test "surprise_metrics_validation" {
    const metrics = SurpriseMetrics.init(1.5, -0.5, 0.5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.jaccard_dissimilarity, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), metrics.content_hash_distance, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), metrics.temporal_novelty, 0.001);
}

test "surprise_record_retention" {
    var record = SurpriseRecord.init([_]u8{0} ** HASH_SIZE, 0.8);
    const initial_priority = record.getRetentionPriority();
    record.recordAccess();
    try std.testing.expectEqual(@as(usize, 2), record.getAccessFrequency());
    try std.testing.expect(record.getRetentionPriority() >= initial_priority);
}

test "statistics_incremental_update" {
    var stats = SurpriseMemoryStatistics.init(0.5);
    stats.addBlock(0.8, 0.5);
    try std.testing.expectEqual(@as(usize, 1), stats.total_blocks);
    try std.testing.expectEqual(@as(usize, 1), stats.high_surprise_blocks);
    stats.addBlock(0.3, 0.5);
    try std.testing.expectEqual(@as(usize, 2), stats.total_blocks);
    try std.testing.expectEqual(@as(usize, 1), stats.low_surprise_blocks);
    stats.removeBlock(0.8, 0.5);
    try std.testing.expectEqual(@as(usize, 1), stats.total_blocks);
    try std.testing.expectEqual(@as(usize, 0), stats.high_surprise_blocks);
}

test "hash_distance_calculation" {
    const hash1 = [_]u8{0xFF} ** HASH_SIZE;
    const hash2 = [_]u8{0x00} ** HASH_SIZE;
    const distance = SurpriseMemoryManager.computeHashDistance(hash1, hash2);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), distance, 0.001);
    const hash3 = [_]u8{0xFF} ** HASH_SIZE;
    const same_distance = SurpriseMemoryManager.computeHashDistance(hash1, hash3);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), same_distance, 0.001);
}

test "surprise_metrics_threshold" {
    const low_metrics = SurpriseMetrics.init(0.1, 0.1, 0.1);
    try std.testing.expect(!low_metrics.exceedsThreshold(0.3));
    const high_metrics = SurpriseMetrics.init(0.9, 0.9, 0.9);
    try std.testing.expect(high_metrics.exceedsThreshold(0.3));
}

test "partial_sort_correctness" {
    var items = [_]CandidateItem{
        .{ .block_id = [_]u8{5} ** HASH_SIZE, .priority = 5.0 },
        .{ .block_id = [_]u8{1} ** HASH_SIZE, .priority = 1.0 },
        .{ .block_id = [_]u8{3} ** HASH_SIZE, .priority = 3.0 },
        .{ .block_id = [_]u8{2} ** HASH_SIZE, .priority = 2.0 },
        .{ .block_id = [_]u8{4} ** HASH_SIZE, .priority = 4.0 },
    };
    SurpriseMemoryManager.partialSort(&items, 3);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), items[0].priority, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), items[1].priority, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), items[2].priority, 0.001);
}

test "statistics_edge_cases" {
    var stats = SurpriseMemoryStatistics.init(0.5);
    stats.removeBlock(0.5, 0.5);
    try std.testing.expectEqual(@as(usize, 0), stats.total_blocks);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), stats.average_surprise, 0.0);
    stats.addBlock(0.0, 0.5);
    try std.testing.expectEqual(@as(usize, 1), stats.total_blocks);
    try std.testing.expectEqual(@as(usize, 1), stats.low_surprise_blocks);
}

test "content_hash_consistency" {
    const data = "test_data_for_hashing";
    const hash1 = SurpriseMemoryManager.computeContentHash(data);
    const hash2 = SurpriseMemoryManager.computeContentHash(data);
    try std.testing.expect(std.mem.eql(u8, hash1[0..], hash2[0..]));
    const different_data = "different_test_data";
    const hash3 = SurpriseMemoryManager.computeContentHash(different_data);
    try std.testing.expect(!std.mem.eql(u8, hash1[0..], hash3[0..]));
}
