const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;
const core_tensor = @import("../core/tensor.zig");
const core_memory = @import("../core/memory.zig");

pub const MGT = struct {
    token_to_id: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    prefixes: std.StringHashMap(u32),
    suffixes: std.StringHashMap(u32),
    roots: std.StringHashMap(u32),
    bpe_pairs: std.StringHashMap(BPEMerge),
    anchors: std.StringHashMap(u64),
    allocated_strings: std.ArrayList([]u8),
    allocator: Allocator,
    next_token_id: u32,

    const BPEMerge = struct {
        token_id: u32,
        priority: u32,
    };

    const TokenPairKey = struct {
        first: u32,
        second: u32,
    };

    const PairFreq = struct {
        key: TokenPairKey,
        freq: u32,
    };

    const SPECIAL_TOKENS = struct {
        const PAD: u32 = 0;
        const UNK: u32 = 1;
        const BOS: u32 = 2;
        const EOS: u32 = 3;
    };

    pub fn init(allocator: Allocator, vocab: []const []const u8, anchors: []const []const u8) !MGT {
        var mgt = initEmpty(allocator);
        errdefer mgt.deinit();

        try mgt.addToken("[PAD]");
        try mgt.addToken("[UNK]");
        try mgt.addToken("[BOS]");
        try mgt.addToken("[EOS]");

        for (vocab) |word| {
            _ = try mgt.addToken(word);
        }

        try mgt.initMorphemes();

        for (anchors) |anch| {
            const tid = mgt.token_to_id.get(anch) orelse try mgt.addToken(anch);
            const key = mgt.id_to_token.get(tid).?;
            try mgt.anchors.put(key, @as(u64, @intCast(tid)));
        }

        return mgt;
    }

    pub fn initWithArena(arena: *core_memory.ArenaAllocator, vocab: []const []const u8, anchors_list: []const []const u8) !MGT {
        return init(arena.allocator(), vocab, anchors_list);
    }

    pub fn initWithPool(pool: *core_memory.PoolAllocator, vocab: []const []const u8, anchors_list: []const []const u8) !MGT {
        return init(pool.allocator(), vocab, anchors_list);
    }

    pub fn initWithBuddy(buddy: *core_memory.BuddyAllocator, vocab: []const []const u8, anchors_list: []const []const u8) !MGT {
        return init(buddy.allocator(), vocab, anchors_list);
    }

    fn initEmpty(allocator: Allocator) MGT {
        return .{
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
            .prefixes = std.StringHashMap(u32).init(allocator),
            .suffixes = std.StringHashMap(u32).init(allocator),
            .roots = std.StringHashMap(u32).init(allocator),
            .bpe_pairs = std.StringHashMap(BPEMerge).init(allocator),
            .anchors = std.StringHashMap(u64).init(allocator),
            .allocated_strings = std.ArrayList([]u8).init(allocator),
            .allocator = allocator,
            .next_token_id = 0,
        };
    }

    fn reset(self: *MGT) void {
        const allocator = self.allocator;
        self.deinit();
        self.* = initEmpty(allocator);
    }

    fn initMorphemes(self: *MGT) !void {
        const prefix_list = [_][]const u8{
            "un",  "re",   "pre",  "dis",  "mis",  "over", "under", "out",
            "sub", "inter", "fore", "de",   "trans", "super", "semi", "anti",
            "mid", "non",  "ex",   "post", "pro",  "co",   "en",   "em",
            "meg", "el", "fel", "le", "be", "ki", "rá", "át", "szét", "vissza",
            "ide", "oda", "alá", "fölé", "közé", "egy", "össze", "tul", "hozzá", "körül",
            "alig", "éppen", "majd", "csak", "is", "leg", "legesleg",
        };

        for (prefix_list) |prefix| {
            const id = self.token_to_id.get(prefix) orelse try self.addToken(prefix);
            const key = self.id_to_token.get(id).?;
            try self.prefixes.put(key, id);
        }

        const suffix_list = [_][]const u8{
            "ing", "ed",  "er",   "est",  "ly",   "tion", "sion", "ness",
            "ment", "ful", "less", "ous",  "ive",  "able", "ible", "al",
            "ial", "y",   "s",    "es",   "en",   "ize",  "ise",  "ate",
            "ság", "ség", "ságú", "ségű", "é", "je", "ja", "ban", "ben",
            "ba", "be", "ból", "ből", "hoz", "hez", "höz", "tól", "től",
            "nak", "nek", "val", "vel", "ért", "ul", "ül", "ként", "án",
            "én", "ig", "at", "et", "tat", "tet", "ott", "ett", "atlan",
            "etlen", "talan", "telen", "ál", "él", "oz", "ez", "öd", "ed",
            "gyet", "get", "j", "unk", "jatok", "játok", "i", "ni", "nként",
            "kor", "ra", "re",
        };

        for (suffix_list) |suffix| {
            const id = self.token_to_id.get(suffix) orelse try self.addToken(suffix);
            const key = self.id_to_token.get(id).?;
            try self.suffixes.put(key, id);
        }
    }

    pub fn deinit(self: *MGT) void {
        self.token_to_id.deinit();
        self.id_to_token.deinit();
        self.prefixes.deinit();
        self.suffixes.deinit();
        self.roots.deinit();
        self.bpe_pairs.deinit();
        self.anchors.deinit();
        for (self.allocated_strings.items) |str| {
            self.allocator.free(str);
        }
        self.allocated_strings.deinit();
    }

    fn isWhitespace(c: u8) bool {
        return c == ' ' or c == '\n' or c == '\t' or c == '\r';
    }

    fn isPunctuation(c: u8) bool {
        return c == '.' or c == ',' or c == '!' or c == '?' or c == ';' or
            c == ':' or c == '"' or c == '\'' or c == '(' or c == ')' or
            c == '{' or c == '}';
    }

    fn isKnownSpecialTokenStart(self: *const MGT, text: []const u8, pos: usize) bool {
        if (pos >= text.len or text[pos] != '[') return false;
        const specials = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };
        for (specials) |special| {
            if (pos + special.len <= text.len and mem.eql(u8, text[pos .. pos + special.len], special) and self.token_to_id.contains(special)) {
                return true;
            }
        }
        return false;
    }

    fn getKnownSpecialTokenLen(self: *const MGT, text: []const u8, pos: usize) ?usize {
        if (pos >= text.len or text[pos] != '[') return null;
        const specials = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };
        for (specials) |special| {
            if (pos + special.len <= text.len and mem.eql(u8, text[pos .. pos + special.len], special) and self.token_to_id.contains(special)) {
                return special.len;
            }
        }
        return null;
    }

    fn utf8CharLen(first_byte: u8) u8 {
        if (first_byte & 0x80 == 0) return 1;
        if (first_byte & 0xE0 == 0xC0) return 2;
        if (first_byte & 0xF0 == 0xE0) return 3;
        if (first_byte & 0xF8 == 0xF0) return 4;
        return 1;
    }

    fn safeUtf8SequenceLenAt(text: []const u8, pos: usize) usize {
        if (pos >= text.len) return 0;
        const len = utf8CharLen(text[pos]);
        if (pos + len > text.len) return 1;
        if (len == 1) return 1;
        var i: usize = 1;
        while (i < len) : (i += 1) {
            if ((text[pos + i] & 0xC0) != 0x80) return 1;
        }
        return len;
    }

    fn emitToken(self: *const MGT, tid: u32, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        _ = self;
        try out_tokens.append(tid);
        if (out_anchors) |anchors_out| {
            if (self.id_to_token.get(tid)) |token_str| {
                if (self.anchors.contains(token_str)) {
                    try anchors_out.append(byte_pos);
                }
            }
        }
    }

    fn appendUnknownForSlice(self: *const MGT, slice: []const u8, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        const tid = self.unknownReplacement(slice);
        try self.emitToken(tid, byte_pos, out_tokens, out_anchors);
    }

    fn appendBPEOrUnknown(self: *MGT, slice: []const u8, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        const tokens = try self.encodeBPE(slice);
        defer self.allocator.free(tokens);
        if (tokens.len == 0) {
            try self.appendUnknownForSlice(slice, byte_pos, out_tokens, out_anchors);
            return;
        }
        for (tokens) |tid| {
            try self.emitToken(tid, byte_pos, out_tokens, out_anchors);
        }
    }

    fn encodeInternal(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        var i: usize = 0;
        while (i < text.len) {
            if (self.getKnownSpecialTokenLen(text, i)) |special_len| {
                const special_token = text[i .. i + special_len];
                const tid = self.token_to_id.get(special_token).?;
                try self.emitToken(tid, i, out_tokens, out_anchors);
                i += special_len;
                continue;
            }

            if (isWhitespace(text[i])) {
                const ws = text[i .. i + 1];
                if (self.token_to_id.get(ws)) |tid| {
                    try self.emitToken(tid, i, out_tokens, out_anchors);
                } else if (text[i] == ' ') {
                    if (self.token_to_id.get(" ")) |space_tid| {
                        try self.emitToken(space_tid, i, out_tokens, out_anchors);
                    } else {
                        try self.appendUnknownForSlice(ws, i, out_tokens, out_anchors);
                    }
                } else {
                    try self.appendUnknownForSlice(ws, i, out_tokens, out_anchors);
                }
                i += 1;
                continue;
            }

            if (isPunctuation(text[i])) {
                const punct = text[i .. i + 1];
                if (self.token_to_id.get(punct)) |tid| {
                    try self.emitToken(tid, i, out_tokens, out_anchors);
                } else {
                    try self.appendBPEOrUnknown(punct, i, out_tokens, out_anchors);
                }
                i += 1;
                continue;
            }

            var word_end = i;
            while (word_end < text.len) {
                if (self.isKnownSpecialTokenStart(text, word_end)) break;
                const c = text[word_end];
                if (isWhitespace(c) or isPunctuation(c)) break;
                const char_len = safeUtf8SequenceLenAt(text, word_end);
                if (char_len == 0) break;
                word_end += char_len;
            }

            if (word_end == i) {
                const char_len = safeUtf8SequenceLenAt(text, i);
                try self.appendBPEOrUnknown(text[i .. i + char_len], i, out_tokens, out_anchors);
                i += char_len;
                continue;
            }

            const word = text[i..word_end];
            if (self.token_to_id.get(word)) |tid| {
                try self.emitToken(tid, i, out_tokens, out_anchors);
            } else if (try self.morphDecompose(word, i, out_tokens, out_anchors)) {
            } else {
                try self.subwordSplitInto(word, i, out_tokens, out_anchors);
            }
            i = word_end;
        }
    }

    pub fn encode(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32)) !void {
        try self.encodeInternal(text, out_tokens, null);
    }

    fn findLongestPrefix(self: *MGT, word: []const u8) ?struct { prefix: []const u8, len: usize } {
        var max_len: usize = 0;
        var best: ?[]const u8 = null;

        var prefix_it = self.prefixes.iterator();
        while (prefix_it.next()) |entry| {
            const prefix = entry.key_ptr.*;
            if (word.len > prefix.len and mem.startsWith(u8, word, prefix)) {
                if (prefix.len > max_len) {
                    max_len = prefix.len;
                    best = prefix;
                }
            }
        }

        if (best) |p| {
            return .{ .prefix = p, .len = max_len };
        }
        return null;
    }

    fn findLongestSuffix(self: *MGT, word: []const u8) ?struct { suffix: []const u8, len: usize } {
        var max_len: usize = 0;
        var best: ?[]const u8 = null;

        var suffix_it = self.suffixes.iterator();
        while (suffix_it.next()) |entry| {
            const suffix = entry.key_ptr.*;
            if (word.len > suffix.len and mem.endsWith(u8, word, suffix)) {
                if (suffix.len > max_len) {
                    max_len = suffix.len;
                    best = suffix;
                }
            }
        }

        if (best) |s| {
            return .{ .suffix = s, .len = max_len };
        }
        return null;
    }

    fn morphDecompose(self: *MGT, word: []const u8, word_start: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !bool {
        if (word.len < 4) return false;

        const prefix_result = self.findLongestPrefix(word);
        const suffix_result = self.findLongestSuffix(word);

        const prefix_len = if (prefix_result) |p| p.len else 0;
        const suffix_len = if (suffix_result) |s| s.len else 0;

        if (prefix_len == 0 and suffix_len == 0) return false;

        const root_start = prefix_len;
        const root_end = word.len - suffix_len;
        if (root_end <= root_start or root_end - root_start < 2) return false;

        const root = word[root_start..root_end];
        const root_tid = self.token_to_id.get(root) orelse return false;

        if (prefix_result) |p| {
            const tid = self.token_to_id.get(p.prefix) orelse return false;
            try self.emitToken(tid, word_start, out_tokens, out_anchors);
        }

        try self.emitToken(root_tid, word_start + root_start, out_tokens, out_anchors);

        if (suffix_result) |s| {
            const tid = self.token_to_id.get(s.suffix) orelse return false;
            try self.emitToken(tid, word_start + word.len - s.len, out_tokens, out_anchors);
        }

        return true;
    }

    fn addToken(self: *MGT, token: []const u8) !u32 {
        if (self.token_to_id.get(token)) |existing| {
            return existing;
        }

        const id = self.next_token_id;
        const token_copy = try self.allocator.dupe(u8, token);
        errdefer self.allocator.free(token_copy);

        try self.token_to_id.put(token_copy, id);
        errdefer _ = self.token_to_id.remove(token_copy);

        try self.id_to_token.put(id, token_copy);
        errdefer _ = self.id_to_token.remove(id);

        try self.allocated_strings.append(token_copy);

        self.next_token_id = id + 1;
        return id;
    }

    fn adoptTokenWithId(self: *MGT, token: []u8, id: u32) !void {
        errdefer self.allocator.free(token);
        try self.token_to_id.put(token, id);
        errdefer _ = self.token_to_id.remove(token);
        try self.id_to_token.put(id, token);
        errdefer _ = self.id_to_token.remove(id);
        try self.allocated_strings.append(token);
        if (id >= self.next_token_id) {
            self.next_token_id = id + 1;
        }
    }

    fn getCanonicalTokenForLoad(self: *MGT, raw: []u8, id: u32) ![]const u8 {
        if (self.id_to_token.get(id)) |canonical| {
            if (!mem.eql(u8, canonical, raw)) return error.InvalidData;
            self.allocator.free(raw);
            return canonical;
        }
        if (self.token_to_id.get(raw)) |existing_id| {
            if (existing_id != id) return error.InvalidData;
            const canonical = self.id_to_token.get(existing_id).?;
            self.allocator.free(raw);
            return canonical;
        }
        try self.adoptTokenWithId(raw, id);
        return self.id_to_token.get(id).?;
    }

    fn encodeBPE(self: *MGT, text: []const u8) ![]u32 {
        if (text.len == 0) return self.allocator.alloc(u32, 0);

        var current = std.ArrayList(u32).init(self.allocator);
        defer current.deinit();

        for (text) |byte| {
            var buf: [16]u8 = undefined;
            const byte_str = try std.fmt.bufPrint(&buf, "<{x:0>2}>", .{byte});
            const tid = self.token_to_id.get(byte_str) orelse SPECIAL_TOKENS.UNK;
            try current.append(tid);
        }

        while (current.items.len > 1) {
            var best_priority: ?u32 = null;
            var best_idx: ?usize = null;
            var i: usize = 0;
            while (i + 1 < current.items.len) : (i += 1) {
                const left = self.id_to_token.get(current.items[i]) orelse return error.InvalidData;
                const right = self.id_to_token.get(current.items[i + 1]) orelse return error.InvalidData;
                const pair = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ left, right });
                defer self.allocator.free(pair);
                if (self.bpe_pairs.get(pair)) |merge| {
                    if (best_priority == null or merge.priority < best_priority.?) {
                        best_priority = merge.priority;
                        best_idx = i;
                    }
                }
            }

            if (best_idx == null) break;
            const idx = best_idx.?;
            const pair_left = current.items[idx];
            const pair_right = current.items[idx + 1];
            const left = self.id_to_token.get(pair_left) orelse return error.InvalidData;
            const right = self.id_to_token.get(pair_right) orelse return error.InvalidData;
            const pair = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ left, right });
            defer self.allocator.free(pair);
            const merge = self.bpe_pairs.get(pair) orelse break;
            current.items[idx] = merge.token_id;
            _ = current.orderedRemove(idx + 1);
        }

        return try current.toOwnedSlice();
    }

    const LessThanContext = struct {
        fn lessThan(_: @This(), a: PairFreq, b: PairFreq) bool {
            if (a.freq != b.freq) return a.freq > b.freq;
            if (a.key.first != b.key.first) return a.key.first < b.key.first;
            return a.key.second < b.key.second;
        }
    };

    pub fn trainBPE(self: *MGT, corpus: []const []const u8, num_merges: u32) !void {
        var sequences = try self.allocator.alloc([]u32, corpus.len);
        errdefer self.allocator.free(sequences);
        var seq_count: usize = 0;
        errdefer {
            for (sequences[0..seq_count]) |seq| {
                self.allocator.free(seq);
            }
        }
        defer {
            for (sequences[0..seq_count]) |seq| {
                self.allocator.free(seq);
            }
            self.allocator.free(sequences);
        }

        for (corpus) |text| {
            const seq = try self.allocator.alloc(u32, text.len);
            errdefer self.allocator.free(seq);
            var i: usize = 0;
            while (i < text.len) : (i += 1) {
                var buf: [16]u8 = undefined;
                const byte_str = try std.fmt.bufPrint(&buf, "<{x:0>2}>", .{text[i]});
                seq[i] = try self.addToken(byte_str);
            }
            sequences[seq_count] = seq;
            seq_count += 1;
        }

        var merge_count: u32 = 0;
        while (merge_count < num_merges) {
            var pair_freqs = std.AutoHashMap(TokenPairKey, u32).init(self.allocator);
            defer pair_freqs.deinit();

            for (sequences[0..seq_count]) |seq| {
                if (seq.len < 2) continue;
                var i: usize = 0;
                while (i + 1 < seq.len) : (i += 1) {
                    const key = TokenPairKey{ .first = seq[i], .second = seq[i + 1] };
                    const entry = try pair_freqs.getOrPut(key);
                    if (entry.found_existing) {
                        entry.value_ptr.* += 1;
                    } else {
                        entry.value_ptr.* = 1;
                    }
                }
            }

            var pair_list = std.ArrayList(PairFreq).init(self.allocator);
            defer pair_list.deinit();
            var it = pair_freqs.iterator();
            while (it.next()) |entry| {
                try pair_list.append(.{ .key = entry.key_ptr.*, .freq = entry.value_ptr.* });
            }
            if (pair_list.items.len == 0) break;

            std.mem.sort(PairFreq, pair_list.items, LessThanContext{}, LessThanContext.lessThan);
            const best = pair_list.items[0];
            if (best.freq < 2) break;

            const first_str = self.id_to_token.get(best.key.first) orelse return error.InvalidData;
            const second_str = self.id_to_token.get(best.key.second) orelse return error.InvalidData;
            const merged_text = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ first_str, second_str });
            defer self.allocator.free(merged_text);
            const merge_token_id = try self.addToken(merged_text);
            const canonical = self.id_to_token.get(merge_token_id).?;
            try self.bpe_pairs.put(canonical, .{ .token_id = merge_token_id, .priority = merge_count });

            for (sequences[0..seq_count]) |*seq_ptr| {
                const old_seq = seq_ptr.*;
                var rebuilt = std.ArrayList(u32).init(self.allocator);
                defer rebuilt.deinit();
                var i: usize = 0;
                while (i < old_seq.len) {
                    if (i + 1 < old_seq.len and old_seq[i] == best.key.first and old_seq[i + 1] == best.key.second) {
                        try rebuilt.append(merge_token_id);
                        i += 2;
                    } else {
                        try rebuilt.append(old_seq[i]);
                        i += 1;
                    }
                }
                const new_seq = try rebuilt.toOwnedSlice();
                self.allocator.free(old_seq);
                seq_ptr.* = new_seq;
            }

            merge_count += 1;
        }
    }

    pub fn decode(self: *MGT, tokens: []const u32, out_text: *std.ArrayList(u8)) !void {
        for (tokens) |tok| {
            if (self.id_to_token.get(tok)) |token_str| {
                if (mem.startsWith(u8, token_str, "<") and mem.endsWith(u8, token_str, ">") and token_str.len == 4) {
                    const hex = token_str[1 .. token_str.len - 1];
                    if (std.fmt.parseInt(u8, hex, 16)) |byte| {
                        try out_text.append(byte);
                    } else |_| {
                        try out_text.appendSlice(token_str);
                    }
                } else {
                    try out_text.appendSlice(token_str);
                }
            } else {
                const unk = self.id_to_token.get(SPECIAL_TOKENS.UNK) orelse "[UNK]";
                try out_text.appendSlice(unk);
            }
        }
    }

    pub fn longestMatch(self: *MGT, text: []const u8, start: usize) usize {
        if (start >= text.len) return 0;
        var max_len: usize = 0;
        var end = start;
        while (end < text.len) {
            const step = safeUtf8SequenceLenAt(text, end);
            if (step == 0) break;
            end += step;
            const substr = text[start..end];
            if (self.token_to_id.contains(substr)) {
                max_len = end - start;
            }
        }
        return max_len;
    }

    pub fn vocabSize(self: *const MGT) usize {
        return self.token_to_id.count();
    }

    pub fn addVocabWord(self: *MGT, word: []const u8, is_anchor: bool) !void {
        const id = try self.addToken(word);
        if (is_anchor) {
            const key = self.id_to_token.get(id).?;
            try self.anchors.put(key, @as(u64, id));
        }
    }

    pub fn removeVocabWord(self: *MGT, word: []const u8) void {
        if (mem.eql(u8, word, "[PAD]") or mem.eql(u8, word, "[UNK]") or mem.eql(u8, word, "[BOS]") or mem.eql(u8, word, "[EOS]")) {
            return;
        }
        if (self.token_to_id.get(word)) |id| {
            if (self.id_to_token.get(id)) |allocated_ptr| {
                _ = self.token_to_id.remove(word);
                _ = self.id_to_token.remove(id);
                _ = self.anchors.remove(word);
                _ = self.prefixes.remove(word);
                _ = self.suffixes.remove(word);
                _ = self.roots.remove(word);

                var bpe_remove = std.ArrayList([]const u8).init(self.allocator);
                defer bpe_remove.deinit();
                var bpe_it = self.bpe_pairs.iterator();
                while (bpe_it.next()) |entry| {
                    const key = entry.key_ptr.*;
                    const merge = entry.value_ptr.*;
                    if (key.ptr == allocated_ptr.ptr or merge.token_id == id) {
                        bpe_remove.append(key) catch {};
                    }
                }
                for (bpe_remove.items) |key| {
                    _ = self.bpe_pairs.remove(key);
                }

                var idx: usize = 0;
                while (idx < self.allocated_strings.items.len) : (idx += 1) {
                    const str = self.allocated_strings.items[idx];
                    if (str.ptr == allocated_ptr.ptr) {
                        self.allocator.free(str);
                        _ = self.allocated_strings.orderedRemove(idx);
                        break;
                    }
                }
            }
        }
    }

    pub fn tokenizeWithAnchors(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: *std.ArrayList(usize)) !void {
        try self.encodeInternal(text, out_tokens, out_anchors);
    }

    pub fn detokenize(self: *MGT, tokens: []const u32) ![]u8 {
        return self.detokenizeAlloc(tokens, self.allocator);
    }

    fn detokenizeAlloc(self: *MGT, tokens: []const u32, allocator: Allocator) ![]u8 {
        var text = std.ArrayList(u8).init(allocator);
        defer text.deinit();
        try self.decode(tokens, &text);
        return try text.toOwnedSlice();
    }

    pub fn encodeBatch(self: *MGT, texts: []const []const u8, allocator: Allocator) ![][]u32 {
        const results = try allocator.alloc([]u32, texts.len);
        errdefer allocator.free(results);
        var i: usize = 0;
        errdefer {
            for (results[0..i]) |r| {
                allocator.free(r);
            }
        }
        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();
            try self.encode(text, &tokens);
            results[i] = try tokens.toOwnedSlice();
            i += 1;
        }
        return results;
    }

    pub fn batchDetokenize(self: *MGT, token_lists: []const []const u32, allocator: Allocator) ![][]u8 {
        const results = try allocator.alloc([]u8, token_lists.len);
        errdefer allocator.free(results);
        var i: usize = 0;
        errdefer {
            for (results[0..i]) |r| {
                allocator.free(r);
            }
        }
        for (token_lists) |tokens| {
            results[i] = try self.detokenizeAlloc(tokens, allocator);
            i += 1;
        }
        return results;
    }

    fn writeStringMapSorted(map: std.StringHashMap(u32), writer: anytype, allocator: Allocator) !void {
        const Item = struct {
            key: []const u8,
            value: u32,
        };
        const Ctx = struct {
            fn lessThan(_: @This(), a: Item, b: Item) bool {
                if (a.value != b.value) return a.value < b.value;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var items = std.ArrayList(Item).init(allocator);
        defer items.deinit();

        var it = map.iterator();
        while (it.next()) |entry| {
            try items.append(.{ .key = entry.key_ptr.*, .value = entry.value_ptr.* });
        }

        std.mem.sort(Item, items.items, Ctx{}, Ctx.lessThan);
        try writer.writeInt(u32, @as(u32, @intCast(items.items.len)), .little);
        for (items.items) |item| {
            try writer.writeInt(u32, @as(u32, @intCast(item.key.len)), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u32, item.value, .little);
        }
    }

    pub fn saveVocab(self: *MGT, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();
        var writer = file.writer();

        const TokenItem = struct {
            id: u32,
            token: []const u8,
        };
        const TokenCtx = struct {
            fn lessThan(_: @This(), a: TokenItem, b: TokenItem) bool {
                if (a.id != b.id) return a.id < b.id;
                return std.mem.lessThan(u8, a.token, b.token);
            }
        };

        var token_items = std.ArrayList(TokenItem).init(self.allocator);
        defer token_items.deinit();
        var token_it = self.id_to_token.iterator();
        while (token_it.next()) |entry| {
            try token_items.append(.{ .id = entry.key_ptr.*, .token = entry.value_ptr.* });
        }
        std.mem.sort(TokenItem, token_items.items, TokenCtx{}, TokenCtx.lessThan);

        try writer.writeInt(u32, @as(u32, @intCast(token_items.items.len)), .little);
        for (token_items.items) |item| {
            try writer.writeInt(u32, @as(u32, @intCast(item.token.len)), .little);
            try writer.writeAll(item.token);
            try writer.writeInt(u32, item.id, .little);
        }

        const BpeItem = struct {
            key: []const u8,
            merge: BPEMerge,
        };
        const BpeCtx = struct {
            fn lessThan(_: @This(), a: BpeItem, b: BpeItem) bool {
                if (a.merge.priority != b.merge.priority) return a.merge.priority < b.merge.priority;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var bpe_items = std.ArrayList(BpeItem).init(self.allocator);
        defer bpe_items.deinit();
        var bpe_it = self.bpe_pairs.iterator();
        while (bpe_it.next()) |entry| {
            try bpe_items.append(.{ .key = entry.key_ptr.*, .merge = entry.value_ptr.* });
        }
        std.mem.sort(BpeItem, bpe_items.items, BpeCtx{}, BpeCtx.lessThan);

        try writer.writeInt(u32, @as(u32, @intCast(bpe_items.items.len)), .little);
        for (bpe_items.items) |item| {
            try writer.writeInt(u32, @as(u32, @intCast(item.key.len)), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u32, item.merge.token_id, .little);
            try writer.writeInt(u32, item.merge.priority, .little);
        }

        try writeStringMapSorted(self.prefixes, writer, self.allocator);
        try writeStringMapSorted(self.suffixes, writer, self.allocator);
        try writeStringMapSorted(self.roots, writer, self.allocator);

        const AnchorItem = struct {
            key: []const u8,
            value: u64,
        };
        const AnchorCtx = struct {
            fn lessThan(_: @This(), a: AnchorItem, b: AnchorItem) bool {
                if (a.value != b.value) return a.value < b.value;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var anchor_items = std.ArrayList(AnchorItem).init(self.allocator);
        defer anchor_items.deinit();
        var anchor_it = self.anchors.iterator();
        while (anchor_it.next()) |entry| {
            try anchor_items.append(.{ .key = entry.key_ptr.*, .value = entry.value_ptr.* });
        }
        std.mem.sort(AnchorItem, anchor_items.items, AnchorCtx{}, AnchorCtx.lessThan);

        try writer.writeInt(u32, @as(u32, @intCast(anchor_items.items.len)), .little);
        for (anchor_items.items) |item| {
            try writer.writeInt(u32, @as(u32, @intCast(item.key.len)), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u64, item.value, .little);
        }
    }

    pub fn loadVocab(self: *MGT, path: []const u8) !void {
        self.reset();

        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        var reader = file.reader();

        const size = try reader.readInt(u32, .little);
        var i: usize = 0;
        while (i < size) : (i += 1) {
            const word_len = try reader.readInt(u32, .little);
            const word_buf = try self.allocator.alloc(u8, word_len);
            errdefer self.allocator.free(word_buf);
            try reader.readNoEof(word_buf);
            const id = try reader.readInt(u32, .little);
            if (self.token_to_id.contains(word_buf) or self.id_to_token.contains(id)) return error.InvalidData;
            try self.adoptTokenWithId(word_buf, id);
        }

        const bpe_count = try reader.readInt(u32, .little);
        var j: usize = 0;
        while (j < bpe_count) : (j += 1) {
            const key_len = try reader.readInt(u32, .little);
            const key_buf = try self.allocator.alloc(u8, key_len);
            errdefer self.allocator.free(key_buf);
            try reader.readNoEof(key_buf);
            const token_id = try reader.readInt(u32, .little);
            const priority = try reader.readInt(u32, .little);
            const canonical = try self.getCanonicalTokenForLoad(key_buf, token_id);
            try self.bpe_pairs.put(canonical, .{ .token_id = token_id, .priority = priority });
        }

        const ReadStringMap = struct {
            fn read(self_mgt: *MGT, map: *std.StringHashMap(u32), r: anytype) !void {
                const count = try r.readInt(u32, .little);
                var k: usize = 0;
                while (k < count) : (k += 1) {
                    const len = try r.readInt(u32, .little);
                    const buf = try self_mgt.allocator.alloc(u8, len);
                    errdefer self_mgt.allocator.free(buf);
                    try r.readNoEof(buf);
                    const id = try r.readInt(u32, .little);
                    const canonical = try self_mgt.getCanonicalTokenForLoad(buf, id);
                    try map.put(canonical, id);
                }
            }
        };

        try ReadStringMap.read(self, &self.prefixes, reader);
        try ReadStringMap.read(self, &self.suffixes, reader);
        try ReadStringMap.read(self, &self.roots, reader);

        const anchor_count = try reader.readInt(u32, .little);
        var l: usize = 0;
        while (l < anchor_count) : (l += 1) {
            const key_len = try reader.readInt(u32, .little);
            const key_buf = try self.allocator.alloc(u8, key_len);
            errdefer self.allocator.free(key_buf);
            try reader.readNoEof(key_buf);
            const value = try reader.readInt(u64, .little);
            if (value > std.math.maxInt(u32)) return error.InvalidData;
            const canonical = try self.getCanonicalTokenForLoad(key_buf, @as(u32, @intCast(value)));
            try self.anchors.put(canonical, value);
        }
    }

    pub fn unknownReplacement(self: *MGT, context: []const u8) u32 {
        _ = self;
        _ = context;
        return SPECIAL_TOKENS.UNK;
    }

    fn subwordSplitInto(self: *MGT, word: []const u8, word_start: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        var i: usize = 0;
        while (i < word.len) {
            const match_len = self.longestMatch(word, i);
            if (match_len > 0) {
                const found_word = word[i .. i + match_len];
                if (self.token_to_id.get(found_word)) |tid| {
                    try self.emitToken(tid, word_start + i, out_tokens, out_anchors);
                    i += match_len;
                    continue;
                }
            }

            const char_len = safeUtf8SequenceLenAt(word, i);
            const piece = word[i .. i + char_len];
            try self.appendBPEOrUnknown(piece, word_start + i, out_tokens, out_anchors);
            i += char_len;
        }
    }

    pub fn subwordSplit(self: *MGT, word: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        defer tokens.deinit();
        try self.subwordSplitInto(word, 0, &tokens, null);
        return try tokens.toOwnedSlice();
    }

    pub fn mergeSubwords(self: *MGT, subwords: []const []const u32) ![]u32 {
        var merged = std.ArrayList(u32).init(self.allocator);
        defer merged.deinit();
        for (subwords) |sw| {
            try merged.appendSlice(sw);
        }
        return try merged.toOwnedSlice();
    }

    pub fn validateTokens(self: *MGT, tokens: []const u32) bool {
        for (tokens) |tok| {
            if (!self.id_to_token.contains(tok)) return false;
        }
        return true;
    }

    pub fn coverage(self: *MGT, corpus: []const u8) f32 {
        if (corpus.len == 0) return 0.0;
        var covered: usize = 0;
        var i: usize = 0;
        while (i < corpus.len) {
            if (self.getKnownSpecialTokenLen(corpus, i)) |special_len| {
                covered += special_len;
                i += special_len;
                continue;
            }

            if (isWhitespace(corpus[i]) or isPunctuation(corpus[i])) {
                const slice = corpus[i .. i + 1];
                if (self.token_to_id.contains(slice) or (corpus[i] == ' ' and self.token_to_id.contains(" "))) {
                    covered += 1;
                }
                i += 1;
                continue;
            }

            var word_end = i;
            while (word_end < corpus.len) {
                if (self.isKnownSpecialTokenStart(corpus, word_end)) break;
                const c = corpus[word_end];
                if (isWhitespace(c) or isPunctuation(c)) break;
                word_end += safeUtf8SequenceLenAt(corpus, word_end);
            }

            if (word_end == i) {
                const char_len = safeUtf8SequenceLenAt(corpus, i);
                const maybe_bpe = self.encodeBPE(corpus[i .. i + char_len]) catch null;
                if (maybe_bpe) |bpe| {
                    defer self.allocator.free(bpe);
                    var all_unk = true;
                    for (bpe) |tid| {
                        if (tid != SPECIAL_TOKENS.UNK) {
                            all_unk = false;
                            break;
                        }
                    }
                    if (!all_unk) covered += char_len;
                }
                i += char_len;
                continue;
            }

            const word = corpus[i..word_end];
            if (self.token_to_id.contains(word)) {
                covered += word.len;
            } else {
                var temp = std.ArrayList(u32).init(self.allocator);
                defer temp.deinit();
                if (self.morphDecompose(word, i, &temp, null) catch false) {
                    covered += word.len;
                } else {
                    const maybe_sub = self.subwordSplit(word) catch null;
                    if (maybe_sub) |sub| {
                        defer self.allocator.free(sub);
                        var all_unk = true;
                        for (sub) |tid| {
                            if (tid != SPECIAL_TOKENS.UNK) {
                                all_unk = false;
                                break;
                            }
                        }
                        if (!all_unk) covered += word.len;
                    }
                }
            }
            i = word_end;
        }
        return @as(f32, @floatFromInt(covered)) / @as(f32, @floatFromInt(corpus.len));
    }

    pub fn encodeToTensor(self: *MGT, text: []const u8, allocator: Allocator) !core_tensor.Tensor {
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();
        try self.encode(text, &tokens);
        const shape = [_]usize{tokens.items.len};
        var tensor = try core_tensor.Tensor.init(allocator, &shape);
        var i: usize = 0;
        while (i < tokens.items.len) : (i += 1) {
            tensor.data[i] = @floatFromInt(tokens.items[i]);
        }
        return tensor;
    }

    pub fn encodeBatchToTensor(self: *MGT, texts: []const []const u8, allocator: Allocator) !core_tensor.Tensor {
        var max_len: usize = 0;
        var per_row = std.ArrayList([]u32).init(allocator);
        defer {
            for (per_row.items) |row| {
                allocator.free(row);
            }
            per_row.deinit();
        }

        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();
            try self.encode(text, &tokens);
            const owned = try tokens.toOwnedSlice();
            try per_row.append(owned);
            if (owned.len > max_len) max_len = owned.len;
        }

        if (max_len == 0) max_len = 1;
        const shape = [_]usize{ texts.len, max_len };
        var tensor = try core_tensor.Tensor.init(allocator, &shape);
        @memset(tensor.data, @as(@TypeOf(tensor.data[0]), 0));

        var row_index: usize = 0;
        while (row_index < per_row.items.len) : (row_index += 1) {
            const row = per_row.items[row_index];
            var col: usize = 0;
            while (col < row.len) : (col += 1) {
                tensor.data[row_index * max_len + col] = @floatFromInt(row[col]);
            }
        }

        return tensor;
    }

    pub fn decodeFromTensor(self: *MGT, tensor: *const core_tensor.Tensor, allocator: Allocator) ![]u8 {
        var tokens = try allocator.alloc(u32, tensor.data.len);
        defer allocator.free(tokens);
        var i: usize = 0;
        while (i < tensor.data.len) : (i += 1) {
            const val = tensor.data[i];
            if (std.math.isNan(val) or std.math.isInf(val) or val < 0.0 or val > @as(@TypeOf(val), @floatFromInt(std.math.maxInt(u32)))) {
                tokens[i] = SPECIAL_TOKENS.UNK;
            } else {
                tokens[i] = @as(u32, @intFromFloat(val));
                if (!self.id_to_token.contains(tokens[i])) {
                    tokens[i] = SPECIAL_TOKENS.UNK;
                }
            }
        }
        return self.detokenizeAlloc(tokens, allocator);
    }
};

test "MGT encode decode" {
    const gpa = testing.allocator;
    const vocab = &.{ "hello", "world", " " };
    const anchors = &.{"hello"};
    var mgt = try MGT.init(gpa, vocab, anchors);
    defer mgt.deinit();
    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();
    try mgt.encode("hello world", &tokens);
    try testing.expect(tokens.items.len >= 3);
    var text = std.ArrayList(u8).init(gpa);
    defer text.deinit();
    try mgt.decode(tokens.items, &text);
    try testing.expectEqualStrings("hello world", text.items);
}

test "MGT add remove vocab" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    try mgt.addVocabWord("test", true);
    try testing.expect(mgt.anchors.contains("test"));
    mgt.removeVocabWord("test");
    try testing.expect(!mgt.anchors.contains("test"));
    try testing.expect(!mgt.token_to_id.contains("test"));
}

test "MGT longest match" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "hello", "hell" }, &.{});
    defer mgt.deinit();
    const len = mgt.longestMatch("hello", 0);
    try testing.expectEqual(@as(usize, 5), len);
}

test "MGT batch encode" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "a", "b" }, &.{});
    defer mgt.deinit();
    const texts = &.{ "a", "b" };
    const batches = try mgt.encodeBatch(texts, gpa);
    defer {
        for (batches) |batch| {
            gpa.free(batch);
        }
        gpa.free(batches);
    }
    try testing.expectEqual(@as(usize, 2), batches.len);
    try testing.expectEqual(@as(usize, 1), batches[0].len);
    try testing.expectEqual(@as(usize, 1), batches[1].len);
}

test "MGT subword split" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "hel", "lo" }, &.{});
    defer mgt.deinit();
    const sub = try mgt.subwordSplit("hello");
    defer gpa.free(sub);
    try testing.expectEqual(@as(usize, 2), sub.len);
    try testing.expect(mgt.validateTokens(sub));
}

test "MGT coverage" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "hello", "world", " " }, &.{});
    defer mgt.deinit();
    const cov = mgt.coverage("hello world");
    try testing.expect(cov > 0.99);
}

test "MGT validate" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{"a"}, &.{});
    defer mgt.deinit();
    try testing.expect(mgt.validateTokens(&.{0}));
    try testing.expect(!mgt.validateTokens(&.{999999}));
}

test "MGT tokenize with anchors" {
    const gpa = testing.allocator;
    const vocab = &.{ "test", "anchor" };
    const anchors = &.{"anchor"};
    var mgt = try MGT.init(gpa, vocab, anchors);
    defer mgt.deinit();
    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();
    var anchor_positions = std.ArrayList(usize).init(gpa);
    defer anchor_positions.deinit();
    try mgt.tokenizeWithAnchors("testanchor", &tokens, &anchor_positions);
    try testing.expect(tokens.items.len >= 2);
    try testing.expect(anchor_positions.items.len >= 1);
    try testing.expectEqual(@as(usize, 4), anchor_positions.items[0]);
}

test "MGT batch detokenize" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "a", "b" }, &.{});
    defer mgt.deinit();
    const token_lists = &[_][]const u32{
        &.{4},
        &.{5},
    };
    const results = try mgt.batchDetokenize(token_lists, gpa);
    defer {
        for (results) |result| {
            gpa.free(result);
        }
        gpa.free(results);
    }
    try testing.expectEqual(@as(usize, 2), results.len);
    try testing.expectEqualStrings("a", results[0]);
    try testing.expectEqualStrings("b", results[1]);
}

test "MGT vocab size" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "a", "b", "c" }, &.{});
    defer mgt.deinit();
    const size = mgt.vocabSize();
    try testing.expect(size >= 7);
}

test "MGT save and load vocab" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "test", "vocab", " " }, &.{"test"});
    defer mgt.deinit();
    try mgt.trainBPE(&.{ "test", "text" }, 4);
    try mgt.saveVocab("test_vocab.bin");
    defer {
        std.fs.cwd().deleteFile("test_vocab.bin") catch |err| {
            std.log.warn("Failed to delete test file: {}", .{err});
        };
    }
    var mgt2 = try MGT.init(gpa, &.{}, &.{});
    defer mgt2.deinit();
    try mgt2.loadVocab("test_vocab.bin");
    try testing.expectEqual(mgt.vocabSize(), mgt2.vocabSize());
    try testing.expectEqual(mgt.bpe_pairs.count(), mgt2.bpe_pairs.count());
    try testing.expectEqual(mgt.anchors.count(), mgt2.anchors.count());
    try testing.expect(mgt2.token_to_id.contains("test"));
    try testing.expect(mgt2.anchors.contains("test"));
}

test "MGT merge subwords" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    const sub1 = &[_]u32{ 1, 2 };
    const sub2 = &[_]u32{ 3, 4 };
    const subwords = &[_][]const u32{ sub1, sub2 };
    const merged = try mgt.mergeSubwords(subwords);
    defer gpa.free(merged);
    try testing.expectEqual(@as(usize, 4), merged.len);
}

test "MGT unknown replacement" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    const replacement = mgt.unknownReplacement("context");
    try testing.expectEqual(MGT.SPECIAL_TOKENS.UNK, replacement);
}

test "MGT morphological decomposition" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "run", "walk" }, &.{});
    defer mgt.deinit();
    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();
    try mgt.encode("running", &tokens);
    try testing.expect(tokens.items.len >= 2);
    const run_id = mgt.token_to_id.get("run").?;
    const ing_id = mgt.token_to_id.get("ing").?;
    try testing.expectEqual(run_id, tokens.items[0]);
    try testing.expectEqual(ing_id, tokens.items[1]);
}

test "MGT BPE training" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    const corpus = &.{ "hello", "help", "held" };
    try mgt.trainBPE(corpus, 10);
    try testing.expect(mgt.bpe_pairs.count() > 0);
    const encoded = try mgt.subwordSplit("hello");
    defer gpa.free(encoded);
    try testing.expect(encoded.len >= 1);
}

test "MGT deterministic encoding" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "test", "data", " " }, &.{});
    defer mgt.deinit();

    var tokens1 = std.ArrayList(u32).init(gpa);
    defer tokens1.deinit();
    try mgt.encode("test data", &tokens1);

    const before_vocab = mgt.vocabSize();

    var tokens2 = std.ArrayList(u32).init(gpa);
    defer tokens2.deinit();
    try mgt.encode("test data", &tokens2);

    try testing.expectEqualSlices(u32, tokens1.items, tokens2.items);
    try testing.expectEqual(before_vocab, mgt.vocabSize());
}
