pragma circom 2.1.8;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/bitify.circom";
include "circomlib/circuits/mux1.circom";

function FIXED_POINT_SCALE() {
    return 1000000;
}

function TAYLOR_COEFF_LINEAR() {
    return 1;
}

function TAYLOR_COEFF_QUADRATIC() {
    return 500000;
}

function TAYLOR_COEFF_CUBIC() {
    return 166667;
}

function REMAINDER_BIT_SIZE() {
    return 21;
}

function SHA256_DIGEST_SIZE() {
    return 32;
}

function DEFAULT_BATCH_SIZE() {
    return 64;
}

template PoseidonChain(n) {
    signal input in[n];
    signal output out;

    var num_chunks = (n + 5) \ 6;
    signal intermediate[num_chunks];

    for (var chunk = 0; chunk < num_chunks; chunk++) {
        var chunk_size = 6;
        if (chunk == num_chunks - 1) {
            var remaining = n - chunk * 6;
            if (remaining < chunk_size) {
                chunk_size = remaining;
            }
        }

        if (chunk_size == 1) {
            component h = Poseidon(2);
            h.inputs[0] <== in[chunk * 6];
            if (chunk > 0) {
                h.inputs[1] <== intermediate[chunk - 1];
            } else {
                h.inputs[1] <== 0;
            }
            intermediate[chunk] <== h.out;
        } else if (chunk_size == 2) {
            component h = Poseidon(3);
            h.inputs[0] <== in[chunk * 6];
            h.inputs[1] <== in[chunk * 6 + 1];
            if (chunk > 0) {
                h.inputs[2] <== intermediate[chunk - 1];
            } else {
                h.inputs[2] <== 0;
            }
            intermediate[chunk] <== h.out;
        } else if (chunk_size == 3) {
            component h = Poseidon(4);
            h.inputs[0] <== in[chunk * 6];
            h.inputs[1] <== in[chunk * 6 + 1];
            h.inputs[2] <== in[chunk * 6 + 2];
            if (chunk > 0) {
                h.inputs[3] <== intermediate[chunk - 1];
            } else {
                h.inputs[3] <== 0;
            }
            intermediate[chunk] <== h.out;
        } else if (chunk_size == 4) {
            component h = Poseidon(5);
            h.inputs[0] <== in[chunk * 6];
            h.inputs[1] <== in[chunk * 6 + 1];
            h.inputs[2] <== in[chunk * 6 + 2];
            h.inputs[3] <== in[chunk * 6 + 3];
            if (chunk > 0) {
                h.inputs[4] <== intermediate[chunk - 1];
            } else {
                h.inputs[4] <== 0;
            }
            intermediate[chunk] <== h.out;
        } else if (chunk_size == 5) {
            component h = Poseidon(6);
            h.inputs[0] <== in[chunk * 6];
            h.inputs[1] <== in[chunk * 6 + 1];
            h.inputs[2] <== in[chunk * 6 + 2];
            h.inputs[3] <== in[chunk * 6 + 3];
            h.inputs[4] <== in[chunk * 6 + 4];
            if (chunk > 0) {
                h.inputs[5] <== intermediate[chunk - 1];
            } else {
                h.inputs[5] <== 0;
            }
            intermediate[chunk] <== h.out;
        } else {
            component h = Poseidon(6);
            h.inputs[0] <== in[chunk * 6];
            h.inputs[1] <== in[chunk * 6 + 1];
            h.inputs[2] <== in[chunk * 6 + 2];
            h.inputs[3] <== in[chunk * 6 + 3];
            h.inputs[4] <== in[chunk * 6 + 4];
            h.inputs[5] <== in[chunk * 6 + 5];

            if (chunk > 0) {
                component h2 = Poseidon(2);
                h2.inputs[0] <== h.out;
                h2.inputs[1] <== intermediate[chunk - 1];
                intermediate[chunk] <== h2.out;
            } else {
                intermediate[chunk] <== h.out;
            }
        }
    }

    out <== intermediate[num_chunks - 1];
}

template SafeIsZero() {
    signal input in;
    signal output out;

    signal inv;
    inv <-- in != 0 ? 1 / in : 0;

    signal prod;
    prod <== in * inv;

    out <== 1 - prod;

    in * out === 0;
}

template SafeIsEqual() {
    signal input a;
    signal input b;
    signal output out;

    signal diff;
    diff <== a - b;

    component isz = SafeIsZero();
    isz.in <== diff;

    out <== isz.out;
}

template PedersenCommit() {
    signal input value;
    signal input blinding;
    signal output commitment;

    component hasher = Poseidon(2);
    hasher.inputs[0] <== value;
    hasher.inputs[1] <== blinding;

    commitment <== hasher.out;
}

template VerifyMerkleProof(depth) {
    signal input leaf;
    signal input path_elements[depth];
    signal input path_indices[depth];
    signal output root;

    signal hashes[depth + 1];
    hashes[0] <== leaf;

    component hashers[depth];
    component muxers_left[depth];
    component muxers_right[depth];

    for (var i = 0; i < depth; i++) {
        path_indices[i] * (path_indices[i] - 1) === 0;

        muxers_left[i] = Mux1();
        muxers_left[i].c[0] <== hashes[i];
        muxers_left[i].c[1] <== path_elements[i];
        muxers_left[i].s <== path_indices[i];

        muxers_right[i] = Mux1();
        muxers_right[i].c[0] <== path_elements[i];
        muxers_right[i].c[1] <== hashes[i];
        muxers_right[i].s <== path_indices[i];

        hashers[i] = Poseidon(2);
        hashers[i].inputs[0] <== muxers_left[i].out;
        hashers[i].inputs[1] <== muxers_right[i].out;

        hashes[i + 1] <== hashers[i].out;
    }

    root <== hashes[depth];
}

template RangeProof(bits) {
    signal input value;
    signal input min_value;
    signal input max_value;
    signal input commitments[bits];
    signal input openings[bits];
    signal output valid;

    signal normalized;
    normalized <== value - min_value;

    component bit_decomp = Num2Bits(bits);
    bit_decomp.in <== normalized;

    component lt_check = LessThan(bits + 1);
    lt_check.in[0] <== normalized;
    lt_check.in[1] <== max_value - min_value + 1;

    component pedersen_commits[bits];
    component eq_checks[bits];
    signal bit_valid[bits];

    for (var i = 0; i < bits; i++) {
        pedersen_commits[i] = PedersenCommit();
        pedersen_commits[i].value <== bit_decomp.out[i];
        pedersen_commits[i].blinding <== openings[i];

        eq_checks[i] = SafeIsEqual();
        eq_checks[i].a <== pedersen_commits[i].commitment;
        eq_checks[i].b <== commitments[i];

        bit_valid[i] <== eq_checks[i].out;
    }

    signal all_valid[bits + 1];
    all_valid[0] <== 1;
    for (var i = 0; i < bits; i++) {
        all_valid[i + 1] <== all_valid[i] * bit_valid[i];
    }

    valid <== all_valid[bits] * lt_check.out;
    valid * (1 - valid) === 0;
}

template RSFLayerComputation(dim) {
    signal input x[dim];
    signal input weights_s[dim / 2][dim / 2];
    signal input weights_t[dim / 2][dim / 2];
    signal input s_bias[dim / 2];
    signal input t_bias[dim / 2];
    signal input expected_commitment;
    signal output y[dim];
    signal output valid_commitment;

    var half = dim / 2;

    signal x1[half];
    signal x2[half];
    for (var i = 0; i < half; i++) {
        x1[i] <== x[i];
        x2[i] <== x[half + i];
    }

    signal s_x2[half];
    for (var i = 0; i < half; i++) {
        signal partial[half + 1];
        partial[0] <== 0;
        for (var j = 0; j < half; j++) {
            signal term;
            term <== weights_s[i][j] * x2[j];
            partial[j + 1] <== partial[j] + term;
        }
        s_x2[i] <== partial[half] + s_bias[i];
    }

    signal y1[half];
    for (var i = 0; i < half; i++) {
        signal s_val;
        s_val <== s_x2[i];

        component s_val_range_low = LessThan(64);
        s_val_range_low.in[0] <== s_val + 1000000;
        s_val_range_low.in[1] <== 2000001;

        component s_val_range_high = LessThan(64);
        s_val_range_high.in[0] <== s_val + 1000000;
        s_val_range_high.in[1] <== 2000001;

        s_val_range_low.out === 1;
        s_val_range_high.out === 1;

        signal s_sq;
        s_sq <== s_val * s_val;

        signal s_cu;
        s_cu <== s_sq * s_val;

        signal taylor_1;
        taylor_1 <== FIXED_POINT_SCALE();

        signal taylor_2;
        taylor_2 <== s_val * TAYLOR_COEFF_LINEAR();

        signal taylor_3;
        taylor_3 <== s_sq * TAYLOR_COEFF_QUADRATIC();

        signal taylor_4;
        taylor_4 <== s_cu * TAYLOR_COEFF_CUBIC();

        signal exp_scaled;
        exp_scaled <== taylor_1 + taylor_2 + taylor_3 + taylor_4;

        signal x1_scaled;
        x1_scaled <== x1[i] * exp_scaled;

        signal quotient;
        signal remainder;
        quotient <-- x1_scaled \ FIXED_POINT_SCALE();
        remainder <-- x1_scaled % FIXED_POINT_SCALE();

        quotient * FIXED_POINT_SCALE() + remainder === x1_scaled;

        component lt_rem = LessThan(REMAINDER_BIT_SIZE());
        lt_rem.in[0] <== remainder;
        lt_rem.in[1] <== FIXED_POINT_SCALE();
        lt_rem.out === 1;

        y1[i] <== quotient;
    }

    signal t_y1[half];
    for (var i = 0; i < half; i++) {
        signal partial[half + 1];
        partial[0] <== 0;
        for (var j = 0; j < half; j++) {
            signal term;
            term <== weights_t[i][j] * y1[j];
            partial[j + 1] <== partial[j] + term;
        }
        t_y1[i] <== partial[half] + t_bias[i];
    }

    signal y2[half];
    for (var i = 0; i < half; i++) {
        y2[i] <== x2[i] + t_y1[i];
    }

    for (var i = 0; i < half; i++) {
        y[i] <== y1[i];
        y[half + i] <== y2[i];
    }

    component output_hash = PoseidonChain(dim);
    for (var i = 0; i < dim; i++) {
        output_hash.in[i] <== y[i];
    }

    component commit_check = SafeIsEqual();
    commit_check.a <== output_hash.out;
    commit_check.b <== expected_commitment;

    valid_commitment <== commit_check.out;
    valid_commitment * (1 - valid_commitment) === 0;
}

template VerifyBatchInference(batch_size, dim) {
    signal input inputs[batch_size][dim];
    signal input outputs[batch_size][dim];
    signal input commitments[batch_size];
    signal input expected_root;
    signal output valid;

    signal computed_commits[batch_size];
    component hashers[batch_size];
    component commit_checks[batch_size];

    for (var b = 0; b < batch_size; b++) {
        hashers[b] = PoseidonChain(dim);
        for (var i = 0; i < dim; i++) {
            hashers[b].in[i] <== outputs[b][i];
        }
        computed_commits[b] <== hashers[b].out;

        commit_checks[b] = SafeIsEqual();
        commit_checks[b].a <== computed_commits[b];
        commit_checks[b].b <== commitments[b];
    }

    var tree_depth = 1;
    var temp_size = batch_size;
    while (temp_size > 1) {
        temp_size = (temp_size + 1) / 2;
        tree_depth = tree_depth + 1;
    }

    signal tree_nodes[tree_depth + 1][batch_size];
    for (var i = 0; i < batch_size; i++) {
        tree_nodes[0][i] <== commitments[i];
    }

    component tree_hashers[tree_depth][batch_size];
    var current_width = batch_size;

    for (var level = 0; level < tree_depth; level++) {
        var next_width = (current_width + 1) / 2;
        for (var i = 0; i < next_width; i++) {
            tree_hashers[level][i] = Poseidon(2);
            tree_hashers[level][i].inputs[0] <== tree_nodes[level][i * 2];

            if (i * 2 + 1 < current_width) {
                tree_hashers[level][i].inputs[1] <== tree_nodes[level][i * 2 + 1];
            } else {
                tree_hashers[level][i].inputs[1] <== tree_nodes[level][i * 2];
            }

            tree_nodes[level + 1][i] <== tree_hashers[level][i].out;
        }
        current_width = next_width;
    }

    component root_check = SafeIsEqual();
    root_check.a <== tree_nodes[tree_depth][0];
    root_check.b <== expected_root;

    signal commit_valid[batch_size + 1];
    commit_valid[0] <== 1;
    for (var b = 0; b < batch_size; b++) {
        commit_valid[b + 1] <== commit_valid[b] * commit_checks[b].out;
    }

    valid <== commit_valid[batch_size] * root_check.out;
    valid * (1 - valid) === 0;
}

template VerifyNoiseBound(dim, precision_bits) {
    signal input original[dim];
    signal input noisy[dim];
    signal input max_noise;
    signal output valid;

    signal noise[dim];
    signal abs_noise[dim];
    component neg_checks[dim];
    component bound_checks[dim];

    for (var i = 0; i < dim; i++) {
        noise[i] <== noisy[i] - original[i];

        component noise_abs = Num2Bits(precision_bits);
        noise_abs.in <== noise[i];

        abs_noise[i] <== noise[i];

        bound_checks[i] = LessThan(precision_bits);
        bound_checks[i].in[0] <== abs_noise[i];
        bound_checks[i].in[1] <== max_noise;
    }

    signal all_valid[dim + 1];
    all_valid[0] <== 1;
    for (var i = 0; i < dim; i++) {
        all_valid[i + 1] <== all_valid[i] * bound_checks[i].out;
    }

    valid <== all_valid[dim];
    valid * (1 - valid) === 0;
}

template VerifyAggregation(num_participants, dim) {
    signal input contributions[num_participants][dim];
    signal input participant_commitments[num_participants];
    signal input aggregated_result[dim];
    signal input min_threshold;
    signal output valid;

    component commit_checks[num_participants];
    for (var p = 0; p < num_participants; p++) {
        commit_checks[p] = PoseidonChain(dim);
        for (var i = 0; i < dim; i++) {
            commit_checks[p].in[i] <== contributions[p][i];
        }
    }

    signal commit_valid[num_participants];
    component commit_eq[num_participants];
    for (var p = 0; p < num_participants; p++) {
        commit_eq[p] = SafeIsEqual();
        commit_eq[p].a <== commit_checks[p].out;
        commit_eq[p].b <== participant_commitments[p];
        commit_valid[p] <== commit_eq[p].out;
    }

    signal sums[dim];
    for (var i = 0; i < dim; i++) {
        signal partial[num_participants + 1];
        partial[0] <== 0;
        for (var p = 0; p < num_participants; p++) {
            partial[p + 1] <== partial[p] + contributions[p][i];
        }
        sums[i] <== partial[num_participants];
    }

    signal result_valid[dim];
    component result_checks[dim];
    for (var i = 0; i < dim; i++) {
        result_checks[i] = SafeIsEqual();
        result_checks[i].a <== sums[i];
        result_checks[i].b <== aggregated_result[i] * num_participants;
        result_valid[i] <== result_checks[i].out;
    }

    component threshold_check = LessThan(SHA256_DIGEST_SIZE());
    threshold_check.in[0] <== min_threshold - 1 + 1;
    threshold_check.in[1] <== num_participants + 1;

    signal all_commits[num_participants + 1];
    all_commits[0] <== 1;
    for (var p = 0; p < num_participants; p++) {
        all_commits[p + 1] <== all_commits[p] * commit_valid[p];
    }

    signal all_results[dim + 1];
    all_results[0] <== 1;
    for (var i = 0; i < dim; i++) {
        all_results[i + 1] <== all_results[i] * result_valid[i];
    }

    valid <== all_commits[num_participants] * all_results[dim] * threshold_check.out;
    valid * (1 - valid) === 0;
}

template DifferentialPrivacyProof(dim) {
    signal input original[dim];
    signal input noisy[dim];
    signal input epsilon;
    signal input sensitivity;
    signal input noise_commitments[dim];
    signal output valid;

    signal noise[dim];
    signal abs_noise[dim];
    component neg_checks[dim];
    component bound_checks[dim];
    component commit_checks[dim];

    signal epsilon_scaled;
    epsilon_scaled <== epsilon;

    signal sensitivity_scaled;
    sensitivity_scaled <== sensitivity * 1000;

    signal max_noise_quotient;
    signal max_noise_remainder;
    max_noise_quotient <-- sensitivity_scaled \ epsilon_scaled;
    max_noise_remainder <-- sensitivity_scaled % epsilon_scaled;

    max_noise_quotient * epsilon_scaled + max_noise_remainder === sensitivity_scaled;

    component lt_eps = LessThan(DEFAULT_BATCH_SIZE());
    lt_eps.in[0] <== max_noise_remainder;
    lt_eps.in[1] <== epsilon_scaled;
    lt_eps.out === 1;

    signal max_allowed;
    max_allowed <== max_noise_quotient;

    for (var i = 0; i < dim; i++) {
        noise[i] <== noisy[i] - original[i];

        component noise_abs = Num2Bits(DEFAULT_BATCH_SIZE());
        noise_abs.in <== noise[i];

        abs_noise[i] <== noise[i];

        bound_checks[i] = LessThan(DEFAULT_BATCH_SIZE());
        bound_checks[i].in[0] <== abs_noise[i];
        bound_checks[i].in[1] <== max_allowed;

        commit_checks[i] = PedersenCommit();
        commit_checks[i].value <== noise[i];
        commit_checks[i].blinding <== 0;
    }

    signal commit_eq[dim];
    component commit_eq_check[dim];
    for (var i = 0; i < dim; i++) {
        commit_eq_check[i] = SafeIsEqual();
        commit_eq_check[i].a <== commit_checks[i].commitment;
        commit_eq_check[i].b <== noise_commitments[i];
        commit_eq[i] <== commit_eq_check[i].out;
    }

    signal all_valid[dim + 1];
    all_valid[0] <== 1;
    for (var i = 0; i < dim; i++) {
        all_valid[i + 1] <== all_valid[i] * bound_checks[i].out * commit_eq[i];
    }

    valid <== all_valid[dim];
    valid * (1 - valid) === 0;
}

template SecureAggregationProof(num_participants, dim) {
    signal input contributions[num_participants][dim];
    signal input participant_commitments[num_participants];
    signal input aggregated_result[dim];
    signal input threshold;
    signal output valid;

    component threshold_check = LessThan(SHA256_DIGEST_SIZE());
    threshold_check.in[0] <== threshold - 1 + 1;
    threshold_check.in[1] <== num_participants + 1;

    component commit_hashes[num_participants];
    component commit_eq[num_participants];
    signal commit_valid[num_participants];

    for (var p = 0; p < num_participants; p++) {
        commit_hashes[p] = PoseidonChain(dim);
        for (var i = 0; i < dim; i++) {
            commit_hashes[p].in[i] <== contributions[p][i];
        }

        commit_eq[p] = SafeIsEqual();
        commit_eq[p].a <== commit_hashes[p].out;
        commit_eq[p].b <== participant_commitments[p];
        commit_valid[p] <== commit_eq[p].out;
    }

    signal sums[dim];
    for (var i = 0; i < dim; i++) {
        signal partial[num_participants + 1];
        partial[0] <== 0;
        for (var p = 0; p < num_participants; p++) {
            partial[p + 1] <== partial[p] + contributions[p][i];
        }
        sums[i] <== partial[num_participants];
    }

    signal avg[dim];
    signal avg_remainder[dim];
    component avg_lt[dim];

    for (var i = 0; i < dim; i++) {
        avg[i] <-- sums[i] \ num_participants;
        avg_remainder[i] <-- sums[i] % num_participants;

        avg[i] * num_participants + avg_remainder[i] === sums[i];

        avg_lt[i] = LessThan(SHA256_DIGEST_SIZE());
        avg_lt[i].in[0] <== avg_remainder[i];
        avg_lt[i].in[1] <== num_participants;
        avg_lt[i].out === 1;
    }

    signal result_valid[dim];
    component result_eq[dim];
    for (var i = 0; i < dim; i++) {
        result_eq[i] = SafeIsEqual();
        result_eq[i].a <== avg[i];
        result_eq[i].b <== aggregated_result[i];
        result_valid[i] <== result_eq[i].out;
    }

    signal all_commits[num_participants + 1];
    all_commits[0] <== 1;
    for (var p = 0; p < num_participants; p++) {
        all_commits[p + 1] <== all_commits[p] * commit_valid[p];
    }

    signal all_results[dim + 1];
    all_results[0] <== 1;
    for (var i = 0; i < dim; i++) {
        all_results[i + 1] <== all_results[i] * result_valid[i];
    }

    valid <== all_commits[num_participants] * all_results[dim] * threshold_check.out;
    valid * (1 - valid) === 0;
}

template FullInferenceProof(num_layers, dim, precision_bits) {
    signal input tokens[dim];
    signal input layer_weights_s[num_layers][dim / 2][dim / 2];
    signal input layer_weights_t[num_layers][dim / 2][dim / 2];
    signal input layer_s_bias[num_layers][dim / 2];
    signal input layer_t_bias[num_layers][dim / 2];
    signal input expected_output[dim];
    signal input input_commitment;
    signal input output_commitment;
    signal input layer_commitments[num_layers];
    signal input max_error_squared;
    signal output is_valid;

    signal layer_outputs[num_layers + 1][dim];
    for (var i = 0; i < dim; i++) {
        layer_outputs[0][i] <== tokens[i];
    }

    component input_hash = PoseidonChain(dim);
    for (var i = 0; i < dim; i++) {
        input_hash.in[i] <== tokens[i];
    }

    component input_check = SafeIsEqual();
    input_check.a <== input_hash.out;
    input_check.b <== input_commitment;

    component rsf_layers[num_layers];
    signal layer_valid[num_layers];

    for (var layer = 0; layer < num_layers; layer++) {
        rsf_layers[layer] = RSFLayerComputation(dim);

        for (var i = 0; i < dim / 2; i++) {
            for (var j = 0; j < dim / 2; j++) {
                rsf_layers[layer].weights_s[i][j] <== layer_weights_s[layer][i][j];
                rsf_layers[layer].weights_t[i][j] <== layer_weights_t[layer][i][j];
            }
            rsf_layers[layer].s_bias[i] <== layer_s_bias[layer][i];
            rsf_layers[layer].t_bias[i] <== layer_t_bias[layer][i];
        }

        for (var i = 0; i < dim; i++) {
            rsf_layers[layer].x[i] <== layer_outputs[layer][i];
        }

        rsf_layers[layer].expected_commitment <== layer_commitments[layer];

        for (var i = 0; i < dim; i++) {
            layer_outputs[layer + 1][i] <== rsf_layers[layer].y[i];
        }

        layer_valid[layer] <== rsf_layers[layer].valid_commitment;
    }

    component output_hash = PoseidonChain(dim);
    for (var i = 0; i < dim; i++) {
        output_hash.in[i] <== layer_outputs[num_layers][i];
    }

    component output_check = SafeIsEqual();
    output_check.a <== output_hash.out;
    output_check.b <== output_commitment;

    signal diff_squared[dim];
    for (var i = 0; i < dim; i++) {
        signal diff;
        diff <== layer_outputs[num_layers][i] - expected_output[i];
        diff_squared[i] <== diff * diff;
    }

    signal error_sum[dim + 1];
    error_sum[0] <== 0;
    for (var i = 0; i < dim; i++) {
        error_sum[i + 1] <== error_sum[i] + diff_squared[i];
    }

    component error_check = LessThan(precision_bits);
    error_check.in[0] <== error_sum[dim];
    error_check.in[1] <== max_error_squared;

    signal all_layers_valid[num_layers + 1];
    all_layers_valid[0] <== 1;
    for (var layer = 0; layer < num_layers; layer++) {
        all_layers_valid[layer + 1] <== all_layers_valid[layer] * layer_valid[layer];
    }

    signal v1;
    v1 <== input_check.out * output_check.out;

    signal v2;
    v2 <== v1 * error_check.out;

    is_valid <== v2 * all_layers_valid[num_layers];
    is_valid * (1 - is_valid) === 0;
}

template InferenceTraceWithBatch(num_layers, dim, batch_size, precision_bits) {
    signal input tokens[batch_size][dim];
    signal input layer_weights_s[num_layers][dim / 2][dim / 2];
    signal input layer_weights_t[num_layers][dim / 2][dim / 2];
    signal input layer_s_bias[num_layers][dim / 2];
    signal input layer_t_bias[num_layers][dim / 2];
    signal input expected_outputs[batch_size][dim];
    signal input input_commitments[batch_size];
    signal input output_commitments[batch_size];
    signal input layer_commitments[num_layers];
    signal input max_error_squared;
    signal input batch_root;
    signal output is_valid;

    component inference_proofs[batch_size];
    signal batch_valid[batch_size];

    for (var b = 0; b < batch_size; b++) {
        inference_proofs[b] = FullInferenceProof(num_layers, dim, precision_bits);

        for (var i = 0; i < dim; i++) {
            inference_proofs[b].tokens[i] <== tokens[b][i];
            inference_proofs[b].expected_output[i] <== expected_outputs[b][i];
        }

        for (var layer = 0; layer < num_layers; layer++) {
            for (var i = 0; i < dim / 2; i++) {
                for (var j = 0; j < dim / 2; j++) {
                    inference_proofs[b].layer_weights_s[layer][i][j] <== layer_weights_s[layer][i][j];
                    inference_proofs[b].layer_weights_t[layer][i][j] <== layer_weights_t[layer][i][j];
                }
                inference_proofs[b].layer_s_bias[layer][i] <== layer_s_bias[layer][i];
                inference_proofs[b].layer_t_bias[layer][i] <== layer_t_bias[layer][i];
            }
            inference_proofs[b].layer_commitments[layer] <== layer_commitments[layer];
        }

        inference_proofs[b].input_commitment <== input_commitments[b];
        inference_proofs[b].output_commitment <== output_commitments[b];
        inference_proofs[b].max_error_squared <== max_error_squared;

        batch_valid[b] <== inference_proofs[b].is_valid;
    }

    component batch_verify = VerifyBatchInference(batch_size, dim);
    for (var b = 0; b < batch_size; b++) {
        for (var i = 0; i < dim; i++) {
            batch_verify.outputs[b][i] <== inference_proofs[b].y[i];
        }
        batch_verify.commitments[b] <== output_commitments[b];
    }
    batch_verify.expected_root <== batch_root;

    signal all_batch_valid[batch_size + 1];
    all_batch_valid[0] <== 1;
    for (var b = 0; b < batch_size; b++) {
        all_batch_valid[b + 1] <== all_batch_valid[b] * batch_valid[b];
    }

    is_valid <== all_batch_valid[batch_size] * batch_verify.valid;
    is_valid * (1 - is_valid) === 0;
}

component main {public [tokens, expected_output, input_commitment, output_commitment]} = FullInferenceProof(8, 32, 64);
