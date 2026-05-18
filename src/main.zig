const std = @import("std");

const types = @import("core/types.zig");
const tensor_mod = @import("core/tensor.zig");
const io_mod = @import("core/io.zig");
const memory_mod = @import("core/memory.zig");
const model_io_mod = @import("core/model_io.zig");

const rsf_mod = @import("processor/rsf.zig");
const mgt_mod = @import("tokenizer/mgt.zig");
const sfd_mod = @import("optimizer/sfd.zig");
const ssi_mod = @import("index/ssi.zig");
const ranker_mod = @import("ranker/ranker.zig");
const learned_embedding_mod = @import("core/learned_embedding.zig");
const oftb_mod = @import("processor/oftb.zig");
const LearnedEmbedding = learned_embedding_mod.LearnedEmbedding;
const OFTBMixer = oftb_mod.OFTB;

const accel_interface = @import("hw/accel/accel_interface.zig");
const cuda_bindings = @import("hw/accel/cuda_bindings.zig");
const futhark_bindings = @import("hw/accel/futhark_bindings.zig");
const fractal_lpu = @import("hw/accel/fractal_lpu.zig");

const distributed_trainer = @import("distributed/distributed_trainer.zig");
const distributed_trainer_futhark = @import("distributed/distributed_trainer_futhark.zig");
const gpu_coordinator = @import("distributed/gpu_coordinator.zig");
const modal_gpu = @import("distributed/modal_gpu.zig");
const nccl_bindings = @import("distributed/nccl_bindings.zig");

const core_relational = @import("core_relational/mod.zig");
const chaos_core = @import("core_relational/chaos_core.zig");
const crev_pipeline = @import("core_relational/crev_pipeline.zig");
const dataset_obfuscation = @import("core_relational/dataset_obfuscation.zig");
const esso_optimizer = @import("core_relational/esso_optimizer.zig");
const fnds = @import("core_relational/fnds.zig");
const formal_verification = @import("core_relational/formal_verification.zig");
const ibm_quantum = @import("core_relational/ibm_quantum.zig");
const nsir_core = @import("core_relational/nsir_core.zig");
const quantum_hardware = @import("core_relational/quantum_hardware.zig");
const quantum_logic = @import("core_relational/quantum_logic.zig");
const quantum_task_adapter = @import("core_relational/quantum_task_adapter.zig");
const r_gpu = @import("core_relational/r_gpu.zig");
const reasoning_orchestrator = @import("core_relational/reasoning_orchestrator.zig");
const safety = @import("core_relational/safety.zig");
const security_proofs = @import("core_relational/security_proofs.zig");
const signal_propagation = @import("core_relational/signal_propagation.zig");
const surprise_memory = @import("core_relational/surprise_memory.zig");
const temporal_graph = @import("core_relational/temporal_graph.zig");
const type_theory = @import("core_relational/type_theory.zig");
const verified_inference_engine = @import("core_relational/verified_inference_engine.zig");
const vpu = @import("core_relational/vpu.zig");
const z_runtime = @import("core_relational/z_runtime.zig");
const zk_verification = @import("core_relational/zk_verification.zig");
const c_api = @import("core_relational/c_api.zig");

const inference_server = @import("api/inference_server.zig");

const Tensor = tensor_mod.Tensor;
const RSF = rsf_mod.RSF;
const MGT = mgt_mod.MGT;
const SFD = sfd_mod.SFD;
const SSI = ssi_mod.SSI;
const Ranker = ranker_mod.Ranker;
const PRNG = types.PRNG;

pub const MainConfig = struct {
    pub const DEFAULT_EMBEDDING_DIM: usize = 128;
    pub const MIN_EMBEDDING_DIM: usize = 8;
    pub const MAX_EMBEDDING_DIM: usize = 16384;
    pub const DEFAULT_RSF_LAYERS: usize = 4;
    pub const MIN_RSF_LAYERS: usize = 1;
    pub const MAX_RSF_LAYERS: usize = 256;
    pub const DEFAULT_BATCH_SIZE: usize = 16;
    pub const MIN_BATCH_SIZE: usize = 1;
    pub const MAX_BATCH_SIZE: usize = 4096;
    pub const DEFAULT_NUM_EPOCHS: usize = 10;
    pub const MAX_NUM_EPOCHS: usize = 100000;
    pub const DEFAULT_LEARNING_RATE: f32 = 0.001;
    pub const MIN_LEARNING_RATE: f32 = 1e-10;
    pub const MAX_LEARNING_RATE: f32 = 10.0;
    pub const DEFAULT_TRAINING_SAMPLES: usize = 100;
    pub const DEFAULT_VALIDATION_SAMPLES: usize = 100;
    pub const MIN_SAMPLES: usize = 1;
    pub const MAX_SAMPLES: usize = 1000000;
    pub const DEFAULT_SAMPLE_LIMIT: usize = 1000000;
    pub const DEFAULT_GRADIENT_CLIP_NORM: f32 = 5.0;
    pub const DEFAULT_SEQUENCE_LENGTH: usize = 64;
    pub const DEFAULT_TOP_K: usize = 5;
    pub const RANKER_NGRAM_SIZE: usize = 10;
    pub const RANKER_LSH_TABLES: usize = 16;
    pub const RANKER_SEED: u64 = 42;
    pub const TEST_DIM: usize = 128;
    pub const TEST_LAYERS: usize = 4;
    pub const TEST_PARAM_SIZE: usize = 128;
    pub const TEST_TOKEN_COUNT: usize = 8;
    pub const REPL_LINE_BUFFER_SIZE: usize = 8192;
    pub const ANCHOR_MODULO: usize = 3;
    pub const TENSOR_INIT_SCALE: f32 = 0.1;
    pub const PARAM_UPDATE_DELTA: f32 = 0.001;
    pub const GRADIENT_SCALE: f32 = 0.01;
    pub const GRADIENT_RANGE_SCALE: f32 = 10.0;
    pub const NORM_TOLERANCE: f32 = 0.1;
    pub const CHANGE_THRESHOLD: f32 = 1e-6;
    pub const GRADIENT_THRESHOLD: f32 = 1e-9;
    pub const R_SQUARED_EPSILON: f64 = 1e-10;
    pub const CONFIDENCE_Z_SCORE: f64 = 1.96;
    pub const PRNG_SEED_FORWARD: u64 = 54321;
    pub const PRNG_SEED_VALIDATION: u64 = 12345;
    pub const PRNG_SEED_GRADIENT: u64 = 99999;
    pub const PRNG_SEED_SYNTHETIC: u64 = 42;
    pub const MAX_VALID_POSITION: u64 = 10000;
    pub const MAX_TOKEN_COUNT: usize = 1000;
    pub const GRADIENT_TENSOR_SIZE: usize = 100;
    pub const PARSE_BASE: u8 = 10;
    pub const DEFAULT_MODELS_DIR: []const u8 = "models";
    pub const FILE_MAGIC_RSF: u32 = 0x4A524653;
    pub const FILE_MAGIC_MGT: u32 = 0x4A4D4754;
    pub const FILE_MAGIC_RANKER: u32 = 0x4A524E4B;
    pub const FILE_MAGIC_PROJ: u32 = 0x4A50524A;
    pub const FILE_MAGIC_EMB: u32 = 0x4A454D42;
    pub const FILE_VERSION: u32 = 1;
    pub const PRNG_SEED_EMBEDDING: u64 = 77777;
    pub const SFD_MOMENTUM: f32 = 0.9;
    pub const NSIR_MODULATION_FACTOR: f32 = 1.05;
    pub const MAX_LINE_LENGTH: usize = 65536;
    pub const MAX_VOCAB_SIZE: u32 = std.math.maxInt(u32);
    pub const MAX_TOKEN_LENGTH: u32 = 65536;
};

const Config = struct {
    embedding_dim: usize,
    rsf_layers: usize,
    batch_size: usize,
    num_epochs: usize,
    learning_rate: f32,
    num_training_samples: usize,
    num_validation_samples: usize,
    models_dir: []const u8,
    vocab_file: ?[]const u8,
    dataset_path: ?[]const u8,
    sample_limit: usize,
    gradient_clip_norm: f32,
    sequence_length: usize,
    top_k: usize,
    allocator: std.mem.Allocator,
    models_dir_allocated: ?[]u8,
    vocab_file_allocated: ?[]u8,
    dataset_path_allocated: ?[]u8,
    mode: []const u8,
    mode_allocated: ?[]const u8,

    pub fn init(allocator: std.mem.Allocator) !Config {
        const models_dir_copy = try allocator.dupe(u8, MainConfig.DEFAULT_MODELS_DIR);
        const mode_str = try allocator.dupe(u8, "interactive");
        return Config{
            .embedding_dim = MainConfig.DEFAULT_EMBEDDING_DIM,
            .rsf_layers = MainConfig.DEFAULT_RSF_LAYERS,
            .batch_size = MainConfig.DEFAULT_BATCH_SIZE,
            .num_epochs = MainConfig.DEFAULT_NUM_EPOCHS,
            .learning_rate = MainConfig.DEFAULT_LEARNING_RATE,
            .num_training_samples = MainConfig.DEFAULT_TRAINING_SAMPLES,
            .num_validation_samples = MainConfig.DEFAULT_VALIDATION_SAMPLES,
            .models_dir = models_dir_copy,
            .vocab_file = null,
            .dataset_path = null,
            .sample_limit = MainConfig.DEFAULT_SAMPLE_LIMIT,
            .gradient_clip_norm = MainConfig.DEFAULT_GRADIENT_CLIP_NORM,
            .sequence_length = MainConfig.DEFAULT_SEQUENCE_LENGTH,
            .top_k = MainConfig.DEFAULT_TOP_K,
            .allocator = allocator,
            .models_dir_allocated = models_dir_copy,
            .vocab_file_allocated = null,
            .dataset_path_allocated = null,
            .mode = mode_str,
            .mode_allocated = mode_str,
        };
    }

    pub fn deinit(self: *Config) void {
        if (self.models_dir_allocated) |dir| self.allocator.free(dir);
        if (self.vocab_file_allocated) |file| self.allocator.free(file);
        if (self.dataset_path_allocated) |path| self.allocator.free(path);
        if (self.mode_allocated) |m| self.allocator.free(m);
    }

    pub fn validate(self: *const Config) error{InvalidConfig}!void {
        if (self.embedding_dim < MainConfig.MIN_EMBEDDING_DIM or self.embedding_dim > MainConfig.MAX_EMBEDDING_DIM) return error.InvalidConfig;
        if (self.rsf_layers < MainConfig.MIN_RSF_LAYERS or self.rsf_layers > MainConfig.MAX_RSF_LAYERS) return error.InvalidConfig;
        if (self.batch_size < MainConfig.MIN_BATCH_SIZE or self.batch_size > MainConfig.MAX_BATCH_SIZE) return error.InvalidConfig;
        if (self.num_epochs > MainConfig.MAX_NUM_EPOCHS) return error.InvalidConfig;
        if (self.learning_rate < MainConfig.MIN_LEARNING_RATE or self.learning_rate > MainConfig.MAX_LEARNING_RATE) return error.InvalidConfig;
        if (std.math.isNan(self.learning_rate) or std.math.isInf(self.learning_rate)) return error.InvalidConfig;
        if (self.num_training_samples < MainConfig.MIN_SAMPLES or self.num_training_samples > MainConfig.MAX_SAMPLES) return error.InvalidConfig;
        if (self.num_validation_samples > MainConfig.MAX_SAMPLES) return error.InvalidConfig;
        if (self.top_k == 0) return error.InvalidConfig;
        if (std.math.isNan(self.gradient_clip_norm) or std.math.isInf(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0) return error.InvalidConfig;
    }

    pub fn parseArgs(allocator: std.mem.Allocator) !Config {
        var config = try Config.init(allocator);
        errdefer config.deinit();

        var args = try std.process.argsWithAllocator(allocator);
        defer args.deinit();

        _ = args.skip();

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--embedding-dim")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (std.mem.startsWith(u8, val, "-")) return error.InvalidArgumentValue;
                config.embedding_dim = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--layers")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (std.mem.startsWith(u8, val, "-")) return error.InvalidArgumentValue;
                config.rsf_layers = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--batch-size")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (std.mem.startsWith(u8, val, "-")) return error.InvalidArgumentValue;
                config.batch_size = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--epochs")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (std.mem.startsWith(u8, val, "-")) return error.InvalidArgumentValue;
                config.num_epochs = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--lr")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                const lr = std.fmt.parseFloat(f32, val) catch return error.InvalidArgumentValue;
                if (std.math.isNan(lr) or std.math.isInf(lr)) return error.InvalidArgumentValue;
                config.learning_rate = lr;
            } else if (std.mem.eql(u8, arg, "--samples")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (std.mem.startsWith(u8, val, "-")) return error.InvalidArgumentValue;
                config.num_training_samples = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--models-dir")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (config.models_dir_allocated) |old| config.allocator.free(old);
                const duped = try allocator.dupe(u8, val);
                config.models_dir_allocated = duped;
                config.models_dir = duped;
            } else if (std.mem.eql(u8, arg, "--vocab-file")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (config.vocab_file_allocated) |old| config.allocator.free(old);
                const duped = try allocator.dupe(u8, val);
                config.vocab_file_allocated = duped;
                config.vocab_file = duped;
            } else if (std.mem.eql(u8, arg, "--dataset-path")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (config.dataset_path_allocated) |old| config.allocator.free(old);
                const duped = try allocator.dupe(u8, val);
                config.dataset_path_allocated = duped;
                config.dataset_path = duped;
            } else if (std.mem.eql(u8, arg, "--sample-limit")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (std.mem.startsWith(u8, val, "-")) return error.InvalidArgumentValue;
                config.sample_limit = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--gradient-clip")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                const clip = std.fmt.parseFloat(f32, val) catch return error.InvalidArgumentValue;
                if (std.math.isNan(clip) or std.math.isInf(clip) or clip <= 0.0) return error.InvalidArgumentValue;
                config.gradient_clip_norm = clip;
            } else if (std.mem.eql(u8, arg, "--sequence-length")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (std.mem.startsWith(u8, val, "-")) return error.InvalidArgumentValue;
                config.sequence_length = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--top-k")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (std.mem.startsWith(u8, val, "-")) return error.InvalidArgumentValue;
                config.top_k = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--help")) {
                try printHelp();
                return error.HelpRequested;
            } else if (std.mem.eql(u8, arg, "--mode")) {
                const mode_val = args.next() orelse return error.MissingArgumentValue;
                if (config.mode_allocated) |m| allocator.free(m);
                const duped = try allocator.dupe(u8, mode_val);
                config.mode_allocated = duped;
                config.mode = duped;
            }
        }
        try config.validate();
        return con