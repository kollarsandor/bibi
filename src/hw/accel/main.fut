type f16 = f16

entry rsf_forward [n][half] (input: [n][half*2]f16)
  (weights_s: [half][half]f16) (weights_t: [half][half]f16)
  (s_bias: [half]f16) (t_bias: [half]f16)
  (clip_min: f16) (clip_max: f16) : *[n][half*2]f16 =
  let d = half * 2
  in map (\row ->
    let x1 = row[0:half] :> [half]f16
    let x2 = row[half:d] :> [half]f16
    let scale = map2 (\j bias ->
      let sum = bias f16.+ f16.sum (map2 (\w x -> w f16.* x) weights_s[j] x2)
      let clipped = f16.max clip_min (f16.min clip_max sum)
      in f16.exp clipped
    ) (iota half) s_bias
    let y1 = map2 (\a b -> a f16.* b) x1 scale
    let trans = map2 (\j bias ->
      bias f16.+ f16.sum (map2 (\w x -> w f16.* x) weights_t[j] y1)
    ) (iota half) t_bias
    let y2 = map2 (\a b -> a f16.+ b) x2 trans
    in y1 ++ y2 :> [half*2]f16
  ) input

entry rsf_backward [n][half] (input: [n][half*2]f16) (grad_output: [n][half*2]f16)
  (weights_s: [half][half]f16) (weights_t: [half][half]f16)
  (s_bias: [half]f16) (t_bias: [half]f16)
  (clip_min: f16) (clip_max: f16)
  : (*[half][half]f16, *[half][half]f16, *[half]f16, *[half]f16) =
  let d = half * 2
  let zero_mat = replicate half (replicate half (f16.i32 0))
  let zero_vec = replicate half (f16.i32 0)
  in loop (grad_ws, grad_wt, grad_sb, grad_tb) = (zero_mat, zero_mat, zero_vec, zero_vec) for i < n do
    let row = input[i]
    let g_row = grad_output[i]
    let x1 = row[0:half] :> [half]f16
    let x2 = row[half:d] :> [half]f16
    let pre_scale = map2 (\j bias ->
      bias f16.+ f16.sum (map2 (\w x -> w f16.* x) weights_s[j] x2)
    ) (iota half) s_bias
    let scale = map (\ps ->
      let clipped = f16.max clip_min (f16.min clip_max ps)
      in f16.exp clipped
    ) pre_scale
    let y1 = map2 (\a b -> a f16.* b) x1 scale
    let dy1 = g_row[0:half] :> [half]f16
    let dy2 = g_row[half:d] :> [half]f16
    let grad_wt_batch = map (\j ->
      map (\k -> dy2[j] f16.* y1[k]) (iota half)
    ) (iota half)
    let grad_tb_batch = dy2
    let dy1_total = map2 (\dy1_j j ->
      dy1_j f16.+ f16.sum (map (\k -> weights_t[k][j] f16.* dy2[k]) (iota half))
    ) dy1 (iota half)
    let ds = map2 (\j ps ->
      let in_range = ps f16.>= clip_min && ps f16.<= clip_max
      in if in_range then dy1_total[j] f16.* y1[j] else (f16.i32 0)
    ) (iota half) pre_scale
    let grad_ws_batch = map (\j ->
      map (\k -> ds[j] f16.* x2[k]) (iota half)
    ) (iota half)
    let grad_sb_batch = ds
    let new_grad_ws = map2 (map2 (\a b -> a f16.+ b)) grad_ws grad_ws_batch
    let new_grad_wt = map2 (map2 (\a b -> a f16.+ b)) grad_wt grad_wt_batch
    let new_grad_sb = map2 (\a b -> a f16.+ b) grad_sb grad_sb_batch
    let new_grad_tb = map2 (\a b -> a f16.+ b) grad_tb grad_tb_batch
    in (new_grad_ws, new_grad_wt, new_grad_sb, new_grad_tb)

entry sfd_update_half [d] (weights: *[d][d]f16) (gradients: [d][d]f16) (learning_rate: f16) (momentum: f16) (velocity: *[d][d]f16) : (*[d][d]f16, *[d][d]f16) =
  let new_velocity = map2 (map2 (\v g -> momentum f16.* v f16.+ learning_rate f16.* g)) velocity gradients
  let new_weights = map2 (map2 (\w v -> w f16.- v)) weights new_velocity
  in (new_weights, new_velocity)

entry sfd_update_bias [d] (bias: *[d]f16) (gradients: [d]f16) (learning_rate: f16) (momentum: f16) (velocity: *[d]f16) : (*[d]f16, *[d]f16) =
  let new_velocity = map2 (\v g -> momentum f16.* v f16.+ learning_rate f16.* g) velocity gradients
  let new_bias = map2 (\b v -> b f16.- v) bias new_velocity
  in (new_bias, new_velocity)

entry compute_loss [n][d] (output: [n][d]f16) (target: [n][d]f16) : f16 =
  let squared_diff = map2 (map2 (\o t -> (o f16.- t) f16.* (o f16.- t))) output target
  let total = f16.sum (flatten squared_diff)
  let count = f16.i64 (n * d)
  in total f16./ count

entry batch_forward [batch_size][seq_len][half] (inputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half]f16) (weights_t: [half][half]f16)
  (s_bias: [half]f16) (t_bias: [half]f16)
  (clip_min: f16) (clip_max: f16) : *[batch_size][seq_len][half*2]f16 =
  map (\sample -> rsf_forward sample weights_s weights_t s_bias t_bias clip_min clip_max) inputs

entry batch_compute_loss [batch_size][seq_len][d] (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16) : f16 =
  let squared_diff = map2 (map2 (map2 (\o t -> (o f16.- t) f16.* (o f16.- t)))) outputs targets
  let total = f16.sum (flatten (flatten squared_diff))
  let count = f16.i64 (batch_size * seq_len * d)
  in total f16./ count

entry batch_gradients [batch_size][seq_len][half] (inputs: [batch_size][seq_len][half*2]f16)
  (grad_outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half]f16) (weights_t: [half][half]f16)
  (s_bias: [half]f16) (t_bias: [half]f16)
  (clip_min: f16) (clip_max: f16)
  : (*[half][half]f16, *[half][half]f16, *[half]f16, *[half]f16) =
  let zero_mat = replicate half (replicate half (f16.i32 0))
  let zero_vec = replicate half (f16.i32 0)
  let results = map2 (\inp g_out ->
    rsf_backward inp g_out weights_s weights_t s_bias t_bias clip_min clip_max
  ) inputs grad_outputs
  let (gs_list, gt_list, gsb_list, gtb_list) = unzip4 results
  let gs_total = reduce (map2 (map2 (f16.+))) zero_mat gs_list
  let gt_total = reduce (map2 (map2 (f16.+))) zero_mat gt_list
  let gsb_total = reduce (map2 (f16.+)) zero_vec gsb_list
  let gtb_total = reduce (map2 (f16.+)) zero_vec gtb_list
  in (gs_total, gt_total, gsb_total, gtb_total)

entry xavier_fill_inplace [d] (weights: *[d][d]f16) (seed: i32) : *[d][d]f16 =
  let scale = f16.sqrt (f16.f32 2.0 f16./ f16.i64 d)
  in map (\i ->
    map (\j ->
      let hash = (seed + i32.i64 i * 73856093 + i32.i64 j * 19349663) % 1000000
      let normalized = (f16.i32 hash) f16./ (f16.i32 1000000) f16.- f16.f32 0.5
      in normalized f16.* scale
    ) (iota d)
  ) (iota d)

entry scale_weights_inplace [d] (weights: *[d][d]f16) (scale_factor: f16) : *[d][d]f16 =
  map (map (\w -> w f16./ scale_factor)) weights

entry accumulate_gradients [d] (grad1: *[d][d]f16) (grad2: [d][d]f16) : *[d][d]f16 =
  map2 (map2 (f16.+)) grad1 grad2

entry training_step [batch_size][seq_len][half]
  (inputs: [batch_size][seq_len][half*2]f16)
  (targets: [batch_size][seq_len][half*2]f16)
  (weights_s: *[half][half]f16)
  (weights_t: *[half][half]f16)
  (s_bias: *[half]f16)
  (t_bias: *[half]f16)
  (velocity_s: *[half][half]f16)
  (velocity_t: *[half][half]f16)
  (velocity_sb: *[half]f16)
  (velocity_tb: *[half]f16)
  (learning_rate: f16)
  (momentum: f16)
  (clip_min: f16)
  (clip_max: f16) : (*[half][half]f16, *[half][half]f16, *[half]f16, *[half]f16, *[half][half]f16, *[half][half]f16, *[half]f16, *[half]f16, f16) =

  let outputs = batch_forward inputs weights_s weights_t s_bias t_bias clip_min clip_max
  let loss = batch_compute_loss outputs targets
  let grad_outputs = map2 (map2 (map2 (\o t -> (f16.f32 2.0) f16.* (o f16.- t)))) outputs targets
  let (grad_s, grad_t, grad_sb, grad_tb) = batch_gradients inputs grad_outputs weights_s weights_t s_bias t_bias clip_min clip_max
  let (new_weights_s, new_velocity_s) = sfd_update_half weights_s grad_s learning_rate momentum velocity_s
  let (new_weights_t, new_velocity_t) = sfd_update_half weights_t grad_t learning_rate momentum velocity_t
  let (new_s_bias, new_velocity_sb) = sfd_update_bias s_bias grad_sb learning_rate momentum velocity_sb
  let (new_t_bias, new_velocity_tb) = sfd_update_bias t_bias grad_tb learning_rate momentum velocity_tb

  in (new_weights_s, new_weights_t, new_s_bias, new_t_bias, new_velocity_s, new_velocity_t, new_velocity_sb, new_velocity_tb, loss)
