pub const struct_futhark_context_config = opaque {};
pub const struct_futhark_context = opaque {};
pub const struct_futhark_f16_1d = opaque {};
pub const struct_futhark_f16_2d = opaque {};
pub const struct_futhark_f16_3d = opaque {};
pub const struct_futhark_f32_1d = opaque {};
pub const struct_futhark_f32_2d = opaque {};
pub const struct_futhark_f32_3d = opaque {};
pub const struct_futhark_u64_1d = opaque {};
pub const struct_futhark_i64_1d = opaque {};

pub extern "c" fn futhark_context_config_new() ?*struct_futhark_context_config;
pub extern "c" fn futhark_context_config_free(cfg: ?*struct_futhark_context_config) void;
pub extern "c" fn futhark_context_config_set_device(cfg: ?*struct_futhark_context_config, device: c_int) void;
pub extern "c" fn futhark_context_config_set_platform(cfg: ?*struct_futhark_context_config, platform: c_int) void;
pub extern "c" fn futhark_context_config_set_default_group_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_num_groups(cfg: ?*struct_futhark_context_config, num: c_int) void;
pub extern "c" fn futhark_context_config_set_default_tile_size(cfg: ?*struct_futhark_context_config, size: c_int) void;

pub extern "c" fn futhark_context_new(cfg: ?*struct_futhark_context_config) ?*struct_futhark_context;
pub extern "c" fn futhark_context_free(ctx: ?*struct_futhark_context) void;
pub extern "c" fn futhark_context_sync(ctx: ?*struct_futhark_context) c_int;
pub extern "c" fn futhark_context_get_error(ctx: ?*struct_futhark_context) ?[*:0]const u8;

pub extern "c" fn futhark_new_f16_1d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64) ?*struct_futhark_f16_1d;
pub extern "c" fn futhark_new_f16_2d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64, dim1: i64) ?*struct_futhark_f16_2d;
pub extern "c" fn futhark_new_f16_3d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f16_3d;
pub extern "c" fn futhark_new_f16_2d_from_f32(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64) ?*struct_futhark_f16_2d;
pub extern "c" fn futhark_new_f16_3d_from_f32(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f16_3d;

pub extern "c" fn futhark_free_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d) c_int;
pub extern "c" fn futhark_free_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d) c_int;
pub extern "c" fn futhark_free_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d) c_int;

pub extern "c" fn futhark_values_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_2d_to_f32(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f16_3d_to_f32(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d, data: ?[*]f32) c_int;

pub extern "c" fn futhark_values_raw_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d) ?*anyopaque;
pub extern "c" fn futhark_shape_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, dims: ?[*]i64) c_int;

pub extern "c" fn futhark_new_f32_1d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64) ?*struct_futhark_f32_1d;
pub extern "c" fn futhark_new_f32_2d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64) ?*struct_futhark_f32_2d;
pub extern "c" fn futhark_new_f32_3d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f32_3d;
pub extern "c" fn futhark_new_u64_1d(ctx: ?*struct_futhark_context, data: ?[*]const u64, dim0: i64) ?*struct_futhark_u64_1d;
pub extern "c" fn futhark_new_i64_1d(ctx: ?*struct_futhark_context, data: ?[*]const i64, dim0: i64) ?*struct_futhark_i64_1d;

pub extern "c" fn futhark_free_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d) void;
pub extern "c" fn futhark_free_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d) void;
pub extern "c" fn futhark_free_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d) void;
pub extern "c" fn futhark_free_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d) void;
pub extern "c" fn futhark_free_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d) void;

pub extern "c" fn futhark_values_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d, data: ?[*]u64) c_int;
pub extern "c" fn futhark_values_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d, data: ?[*]i64) c_int;

pub extern "c" fn futhark_entry_matmul(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_2d, a: ?*struct_futhark_f32_2d, b: ?*struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_entry_batch_matmul(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_3d, a: ?*struct_futhark_f32_3d, b: ?*struct_futhark_f32_3d) c_int;
pub extern "c" fn futhark_entry_dot(ctx: ?*struct_futhark_context, out: ?*f32, a: ?*struct_futhark_f32_1d, b: ?*struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_clip_fisher(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, fisher: ?*struct_futhark_f32_1d, clip_val: f32) c_int;
pub extern "c" fn futhark_entry_reduce_gradients(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, gradients: ?*struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_entry_rank_segments(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, query_hash: u64, segment_hashes: ?*struct_futhark_u64_1d, base_scores: ?*struct_futhark_f32_1d) c_int;

pub extern "c" fn futhark_entry_rsf_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    input: ?*struct_futhark_f16_2d,
    weights_s: ?*struct_futhark_f16_2d,
    weights_t: ?*struct_futhark_f16_2d,
    s_bias: ?*struct_futhark_f16_1d,
    t_bias: ?*struct_futhark_f16_1d,
    clip_min: u16,
    clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_rsf_backward(
    ctx: ?*struct_futhark_context,
    out_grad_ws: ?*?*struct_futhark_f16_2d,
    out_grad_wt: ?*?*struct_futhark_f16_2d,
    out_grad_sb: ?*?*struct_futhark_f16_1d,
    out_grad_tb: ?*?*struct_futhark_f16_1d,
    input: ?*struct_futhark_f16_2d,
    grad_output: ?*struct_futhark_f16_2d,
    weights_s: ?*struct_futhark_f16_2d,
    weights_t: ?*struct_futhark_f16_2d,
    s_bias: ?*struct_futhark_f16_1d,
    t_bias: ?*struct_futhark_f16_1d,
    clip_min: u16,
    clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_scale_weights_inplace(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f16_2d, weights: ?*struct_futhark_f16_2d, scale: u16) c_int;

pub extern "c" fn futhark_entry_training_step(
    ctx: ?*struct_futhark_context,
    new_weights_s: ?*?*struct_futhark_f16_2d,
    new_weights_t: ?*?*struct_futhark_f16_2d,
    new_s_bias: ?*?*struct_futhark_f16_1d,
    new_t_bias: ?*?*struct_futhark_f16_1d,
    new_velocity_s: ?*?*struct_futhark_f16_2d,
    new_velocity_t: ?*?*struct_futhark_f16_2d,
    new_velocity_sb: ?*?*struct_futhark_f16_1d,
    new_velocity_tb: ?*?*struct_futhark_f16_1d,
    loss: ?*u16,
    inputs: ?*struct_futhark_f16_2d,
    targets: ?*struct_futhark_f16_2d,
    weights_s: ?*struct_futhark_f16_2d,
    weights_t: ?*struct_futhark_f16_2d,
    s_bias: ?*struct_futhark_f16_1d,
    t_bias: ?*struct_futhark_f16_1d,
    velocity_s: ?*struct_futhark_f16_2d,
    velocity_t: ?*struct_futhark_f16_2d,
    velocity_sb: ?*struct_futhark_f16_1d,
    velocity_tb: ?*struct_futhark_f16_1d,
    learning_rate: u16,
    momentum: u16,
    clip_min: u16,
    clip_max: u16,
) c_int;
