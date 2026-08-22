function initstacks_transformer_b(dff, dk, h, n, n_layers)
    d = h * dk
    n_d = n * d
    n_dff = n * dff
    s_stack = Vector{Float64}(undef, ((((((((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1)
    q_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1))
    k_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1))
    v_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1))
    scores_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1))
    row_max_stack = Vector{Float64}(undef, (((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1)
    probs_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1))
    row_sum_stack = Vector{Float64}(undef, (((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1)
    ctx_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1))
    resid1_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1))
    row_mean_stack = Vector{Float64}(undef, (((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1)
    s2_stack = Vector{Float64}(undef, (((((((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1)
    diff_stack = Vector{Float64}(undef, (((((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1)
    row_var_stack = Vector{Float64}(undef, (((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1)
    denom_stack = Vector{Float64}(undef, (((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1)
    normed1_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1))
    ff_hidden_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1))
    resid2_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1))
    return (s_stack, q_stack, k_stack, v_stack, scores_stack, row_max_stack, probs_stack, row_sum_stack, ctx_stack, resid1_stack, row_mean_stack, s2_stack, diff_stack, row_var_stack, denom_stack, normed1_stack, ff_hidden_stack, resid2_stack, x_stack)
end

function transformer_b(x, xb, wq, wqb, bq, bqb, wk, wkb, bk, bkb, wv, wvb, bv, bvb, wo, wob, bo, bob, ln1_gain, ln1_gainb, ln1_bias, ln1_biasb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, ln2_gain, ln2_gainb, ln2_bias, ln2_biasb, q, qb, k, kb, v, vb, scores, scoresb, probs, probsb, ctx, ctxb, attn_out, attn_outb, resid1, resid1b, normed1, normed1b, ff_hidden, ff_hiddenb, ff_out, ff_outb, resid2, resid2b, x_next, x_nextb, n, dk, h, dff, n_layers, eps, epsb, s_stack, q_stack, k_stack, v_stack, scores_stack, row_max_stack, probs_stack, row_sum_stack, ctx_stack, resid1_stack, row_mean_stack, s2_stack, diff_stack, row_var_stack, denom_stack, normed1_stack, ff_hidden_stack, resid2_stack, x_stack)
    denom = 0.0
    diff = 0.0
    row_max = 0.0
    row_mean = 0.0
    row_sum = 0.0
    row_var = 0.0
    s = 0.0
    s2 = 0.0
    denomb = 0.0
    diffb = 0.0
    row_maxb = 0.0
    row_meanb = 0.0
    row_sumb = 0.0
    row_varb = 0.0
    sb = 0.0
    s2b = 0.0
    d = h * dk
    inv_sqrt_dk = 1.0 / sqrt(dk)
    n_d = n * d
    n_dff = n * dff
    for i_seq_l = 1:n_layers
        w_offset = (i_seq_l - 1) * d * d
        b_offset = (i_seq_l - 1) * d
        ln_offset = (i_seq_l - 1) * d
        w1_offset = (i_seq_l - 1) * d * dff
        b1_offset = (i_seq_l - 1) * dff
        w2_offset = (i_seq_l - 1) * dff * d
        b2_offset = (i_seq_l - 1) * d
        ln2_offset = (i_seq_l - 1) * d
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1] = s
            s = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wq[w_offset + (i_seq_p - 1) * d + j]
            end
            q_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1] = q[(i - 1) * d + j]
            q[(i - 1) * d + j] = s + bq[b_offset + j]
            s_stack[((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)] = s
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s_stack[(((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)] = s
            s = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wk[w_offset + (i_seq_p - 1) * d + j]
            end
            k_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1] = k[(i - 1) * d + j]
            k[(i - 1) * d + j] = s + bk[b_offset + j]
            s_stack[(((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)] = s
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s_stack[((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)] = s
            s = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wv[w_offset + (i_seq_p - 1) * d + j]
            end
            v_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1] = v[(i - 1) * d + j]
            v[(i - 1) * d + j] = s + bv[b_offset + j]
            s_stack[((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)] = s
        end
        for hh = 1:h
            head_offset = (hh - 1) * dk
            score_off = (hh - 1) * n * n
            for idx2 = 1:n * n
                i = div(idx2 - 1, n) + 1
                j = mod(idx2 - 1, n) + 1
                s_stack[(((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1)] = s
                s = 0.0
                for i_seq_p = 1:dk
                    s = s + q[(i - 1) * d + head_offset + i_seq_p] * k[(j - 1) * d + head_offset + i_seq_p]
                end
                scores_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1] = scores[score_off + (i - 1) * n + j]
                scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
                s_stack[(((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1)] = s
            end
            for i = 1:n
                row_max_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1] = row_max
                row_max = scores[score_off + (i - 1) * n + 1]
                for i_seq_j = 2:n
                    row_max_stack[(div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (i - 1) * (div(n - 2, 1) + 1) + (i_seq_j - 2)) + 1)] = row_max
                    row_max = max(row_max, scores[score_off + (i - 1) * n + i_seq_j])
                end
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    probs_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (i - 1) * (div(n - 1, 1) + 1) + (j - 1)) + 1] = probs[kk]
                    probs[kk] = exp(scores[kk] - row_max)
                end
                row_sum_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1] = row_sum
                row_sum = 0.0
                for i_seq_j = 1:n
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_seq_j]
                end
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    probs_stack[(div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (i - 1) * (div(n - 1, 1) + 1) + (j - 1)) + 1)] = probs[kk]
                    probs[kk] = probs[kk] / row_sum
                end
                row_max_stack[((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = row_max
                row_sum_stack[((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = row_sum
            end
            for idx3 = 1:n * dk
                i = div(idx3 - 1, dk) + 1
                p = mod(idx3 - 1, dk) + 1
                s_stack[((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1)] = s
                s = 0.0
                for i_seq_j = 1:n
                    s = s + probs[score_off + (i - 1) * n + i_seq_j] * v[(i_seq_j - 1) * d + head_offset + p]
                end
                ctx_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1] = ctx[(i - 1) * d + head_offset + p]
                ctx[(i - 1) * d + head_offset + p] = s
                s_stack[((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1)] = s
            end
            row_max_stack[(((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * (div(h - 1, 1) + 1) + (hh - 1)) + 1)] = row_max
            row_sum_stack[(((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * (div(h - 1, 1) + 1) + (hh - 1)) + 1)] = row_sum
            s_stack[(((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (((i_seq_l - 1) * (div(h - 1, 1) + 1) + (hh - 1)) + 1)] = s
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s_stack[((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)] = s
            s = 0.0
            for i_seq_p = 1:d
                s = s + ctx[(i - 1) * d + i_seq_p] * wo[w_offset + (i_seq_p - 1) * d + j]
            end
            attn_out[(i - 1) * d + j] = s + bo[b_offset + j]
            s_stack[((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)] = s
        end
        for idx = 1:n_d
            resid1_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1] = resid1[idx]
            resid1[idx] = x[idx] + attn_out[idx]
        end
        for i = 1:n
            s_stack[(((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = s
            s = 0.0
            for i_seq_j = 1:d
                s = s + resid1[(i - 1) * d + i_seq_j]
            end
            row_mean_stack[((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1] = row_mean
            row_mean = s / d
            s2_stack[((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1] = s2
            s2 = 0.0
            for i_seq_j = 1:d
                diff = resid1[(i - 1) * d + i_seq_j] - row_mean
                s2 = s2 + diff * diff
            end
            row_var_stack[((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1] = row_var
            row_var = s2 / d
            denom_stack[((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1] = denom
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                normed1_stack[((i_seq_l - 1) * ((div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (i - 1) * (div(d - 1, 1) + 1) + (j - 1)) + 1] = normed1[kk]
                normed1[kk] = ((resid1[kk] - row_mean) / denom) * ln1_gain[ln_offset + j] + ln1_bias[ln_offset + j]
            end
            s_stack[(((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = s
            s2_stack[((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = s2
        end
        for idx = 1:n_dff
            i = div(idx - 1, dff) + 1
            j = mod(idx - 1, dff) + 1
            s_stack[((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1)] = s
            s = 0.0
            for i_seq_p = 1:d
                s = s + normed1[(i - 1) * d + i_seq_p] * w1[w1_offset + (i_seq_p - 1) * dff + j]
            end
            ff_hidden_stack[((i_seq_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1] = ff_hidden[(i - 1) * dff + j]
            ff_hidden[(i - 1) * dff + j] = max(s + b1[b1_offset + j], 0.0)
            s_stack[((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1)] = s
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s_stack[(((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)] = s
            s = 0.0
            for i_seq_p = 1:dff
                s = s + ff_hidden[(i - 1) * dff + i_seq_p] * w2[w2_offset + (i_seq_p - 1) * d + j]
            end
            ff_out[(i - 1) * d + j] = s + b2[b2_offset + j]
            s_stack[(((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)] = s
        end
        for idx = 1:n_d
            resid2_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1] = resid2[idx]
            resid2[idx] = normed1[idx] + ff_out[idx]
        end
        for i = 1:n
            s_stack[((((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = s
            s = 0.0
            for i_seq_j = 1:d
                s = s + resid2[(i - 1) * d + i_seq_j]
            end
            row_mean_stack[(div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = row_mean
            row_mean = s / d
            s2_stack[(((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = s2
            s2 = 0.0
            for i_seq_j = 1:d
                diff = resid2[(i - 1) * d + i_seq_j] - row_mean
                s2 = s2 + diff * diff
            end
            row_var_stack[(div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = row_var
            row_var = s2 / d
            denom_stack[(div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = denom
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                x_next[kk] = ((resid2[kk] - row_mean) / denom) * ln2_gain[ln2_offset + j] + ln2_bias[ln2_offset + j]
            end
            s_stack[((((((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = s
            s2_stack[(((((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)] = s2
        end
        for idx = 1:n_d
            x_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1] = x[idx]
            x[idx] = x_next[idx]
        end
        denom_stack[((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)] = denom
        row_max_stack[((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + ((i_seq_l - 1) + 1)] = row_max
        row_mean_stack[((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)] = row_mean
        row_sum_stack[((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + ((i_seq_l - 1) + 1)] = row_sum
        row_var_stack[((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)] = row_var
        s_stack[(((((((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)] = s
        s2_stack[((((((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)] = s2
    end
    denom_stack[(((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1] = denom
    row_max_stack[(((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1] = row_max
    row_mean_stack[(((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1] = row_mean
    row_sum_stack[(((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1] = row_sum
    row_var_stack[(((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1] = row_var
    s_stack[((((((((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1] = s
    s2_stack[(((((((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1] = s2
    denom = denom_stack[(((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1]
    row_max = row_max_stack[(((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1]
    row_mean = row_mean_stack[(((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1]
    row_sum = row_sum_stack[(((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1]
    row_var = row_var_stack[(((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1]
    s = s_stack[((((((((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1]
    s2 = s2_stack[(((((((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1)) + 1]
    d = h * dk
    n_d = n * d
    n_dff = n * dff
    for i_seq_l = n_layers:-1:1
        denom = denom_stack[((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)]
        row_max = row_max_stack[((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + ((i_seq_l - 1) + 1)]
        row_mean = row_mean_stack[((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)]
        row_sum = row_sum_stack[((((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + ((i_seq_l - 1) + 1)]
        row_var = row_var_stack[((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)]
        s = s_stack[(((((((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)]
        s2 = s2_stack[((((((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + ((i_seq_l - 1) + 1)]
        w_offset = (i_seq_l - 1) * d * d
        b_offset = (i_seq_l - 1) * d
        ln_offset = (i_seq_l - 1) * d
        w1_offset = (i_seq_l - 1) * d * dff
        b1_offset = (i_seq_l - 1) * dff
        w2_offset = (i_seq_l - 1) * dff * d
        b2_offset = (i_seq_l - 1) * d
        ln2_offset = (i_seq_l - 1) * d
        for idx = n_d:-1:1
            x[idx] = x_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1]
            x_nextb[idx] = x_nextb[idx] + xb[idx]
            xb[idx] = 0.0
        end
        for i = n:-1:1
            s = s_stack[((((((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            s2 = s2_stack[(((((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            for j = 1:d
                kk = (i - 1) * d + j
                resid2b[kk] = resid2b[kk] + (1.0 / denom) * (ln2_gain[ln2_offset + j] * x_nextb[kk])
                row_meanb = row_meanb + -((1.0 / denom) * (ln2_gain[ln2_offset + j] * x_nextb[kk]))
                denomb = denomb + -((resid2[kk] - row_mean) / denom ^ 2) * (ln2_gain[ln2_offset + j] * x_nextb[kk])
                ln2_gainb[ln2_offset + j] = ln2_gainb[ln2_offset + j] + ((resid2[kk] - row_mean) / denom) * x_nextb[kk]
                ln2_biasb[ln2_offset + j] = ln2_biasb[ln2_offset + j] + x_nextb[kk]
                x_nextb[kk] = 0.0
            end
            denom = denom_stack[(div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            row_varb = row_varb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            epsb = epsb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            denomb = 0.0
            row_var = row_var_stack[(div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            s2b = s2b + (1.0 / d) * row_varb
            row_varb = 0.0
            for i_seq_j = 1:d
                diff = resid2[(i - 1) * d + i_seq_j] - row_mean
                s2 = s2 + diff * diff
                diffb = diffb + diff * s2b
                diffb = diffb + diff * s2b
                resid2b[(i - 1) * d + i_seq_j] = resid2b[(i - 1) * d + i_seq_j] + diffb
                row_meanb = row_meanb + -diffb
                diffb = 0.0
            end
            s2 = s2_stack[(((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            s2b = 0.0
            row_mean = row_mean_stack[(div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            sb = sb + (1.0 / d) * row_meanb
            row_meanb = 0.0
            for i_seq_j = 1:d
                s = s + resid2[(i - 1) * d + i_seq_j]
                resid2b[(i - 1) * d + i_seq_j] = resid2b[(i - 1) * d + i_seq_j] + sb
            end
            s = s_stack[((((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            sb = 0.0
        end
        for idx = n_d:-1:1
            resid2[idx] = resid2_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1]
            normed1b[idx] = normed1b[idx] + resid2b[idx]
            ff_outb[idx] = ff_outb[idx] + resid2b[idx]
            resid2b[idx] = 0.0
        end
        for idx = n_d:-1:1
            s = s_stack[(((((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(dff - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sb = sb + ff_outb[(i - 1) * d + j]
            b2b[b2_offset + j] = b2b[b2_offset + j] + ff_outb[(i - 1) * d + j]
            ff_outb[(i - 1) * d + j] = 0.0
            for i_seq_p = 1:dff
                s = s + ff_hidden[(i - 1) * dff + i_seq_p] * w2[w2_offset + (i_seq_p - 1) * d + j]
                ff_hiddenb[(i - 1) * dff + i_seq_p] = ff_hiddenb[(i - 1) * dff + i_seq_p] + w2[w2_offset + (i_seq_p - 1) * d + j] * sb
                w2b[w2_offset + (i_seq_p - 1) * d + j] = w2b[w2_offset + (i_seq_p - 1) * d + j] + ff_hidden[(i - 1) * dff + i_seq_p] * sb
            end
            s = s_stack[(((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)]
            sb = 0.0
        end
        for idx = n_dff:-1:1
            s = s_stack[((((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_dff - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1)]
            i = div(idx - 1, dff) + 1
            j = mod(idx - 1, dff) + 1
            ff_hidden[(i - 1) * dff + j] = ff_hidden_stack[((i_seq_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1]
            sb = sb + (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * ff_hiddenb[(i - 1) * dff + j]
            b1b[b1_offset + j] = b1b[b1_offset + j] + (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * ff_hiddenb[(i - 1) * dff + j]
            ff_hiddenb[(i - 1) * dff + j] = 0.0
            for i_seq_p = 1:d
                s = s + normed1[(i - 1) * d + i_seq_p] * w1[w1_offset + (i_seq_p - 1) * dff + j]
                normed1b[(i - 1) * d + i_seq_p] = normed1b[(i - 1) * d + i_seq_p] + w1[w1_offset + (i_seq_p - 1) * dff + j] * sb
                w1b[w1_offset + (i_seq_p - 1) * dff + j] = w1b[w1_offset + (i_seq_p - 1) * dff + j] + normed1[(i - 1) * d + i_seq_p] * sb
            end
            s = s_stack[((((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1)]
            sb = 0.0
        end
        for i = n:-1:1
            s = s_stack[(((((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            s2 = s2_stack[((div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            for j = d:-1:1
                kk = (i - 1) * d + j
                normed1[kk] = normed1_stack[((i_seq_l - 1) * ((div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (i - 1) * (div(d - 1, 1) + 1) + (j - 1)) + 1]
                resid1b[kk] = resid1b[kk] + (1.0 / denom) * (ln1_gain[ln_offset + j] * normed1b[kk])
                row_meanb = row_meanb + -((1.0 / denom) * (ln1_gain[ln_offset + j] * normed1b[kk]))
                denomb = denomb + -((resid1[kk] - row_mean) / denom ^ 2) * (ln1_gain[ln_offset + j] * normed1b[kk])
                ln1_gainb[ln_offset + j] = ln1_gainb[ln_offset + j] + ((resid1[kk] - row_mean) / denom) * normed1b[kk]
                ln1_biasb[ln_offset + j] = ln1_biasb[ln_offset + j] + normed1b[kk]
                normed1b[kk] = 0.0
            end
            denom = denom_stack[((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1]
            row_varb = row_varb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            epsb = epsb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            denomb = 0.0
            row_var = row_var_stack[((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1]
            s2b = s2b + (1.0 / d) * row_varb
            row_varb = 0.0
            for i_seq_j = 1:d
                diff = resid1[(i - 1) * d + i_seq_j] - row_mean
                s2 = s2 + diff * diff
                diffb = diffb + diff * s2b
                diffb = diffb + diff * s2b
                resid1b[(i - 1) * d + i_seq_j] = resid1b[(i - 1) * d + i_seq_j] + diffb
                row_meanb = row_meanb + -diffb
                diffb = 0.0
            end
            s2 = s2_stack[((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1]
            s2b = 0.0
            row_mean = row_mean_stack[((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1]
            sb = sb + (1.0 / d) * row_meanb
            row_meanb = 0.0
            for i_seq_j = 1:d
                s = s + resid1[(i - 1) * d + i_seq_j]
                resid1b[(i - 1) * d + i_seq_j] = resid1b[(i - 1) * d + i_seq_j] + sb
            end
            s = s_stack[(((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
            sb = 0.0
        end
        for idx = n_d:-1:1
            resid1[idx] = resid1_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1]
            xb[idx] = xb[idx] + resid1b[idx]
            attn_outb[idx] = attn_outb[idx] + resid1b[idx]
            resid1b[idx] = 0.0
        end
        for idx = n_d:-1:1
            s = s_stack[((((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sb = sb + attn_outb[(i - 1) * d + j]
            bob[b_offset + j] = bob[b_offset + j] + attn_outb[(i - 1) * d + j]
            attn_outb[(i - 1) * d + j] = 0.0
            for i_seq_p = 1:d
                s = s + ctx[(i - 1) * d + i_seq_p] * wo[w_offset + (i_seq_p - 1) * d + j]
                ctxb[(i - 1) * d + i_seq_p] = ctxb[(i - 1) * d + i_seq_p] + wo[w_offset + (i_seq_p - 1) * d + j] * sb
                wob[w_offset + (i_seq_p - 1) * d + j] = wob[w_offset + (i_seq_p - 1) * d + j] + ctx[(i - 1) * d + i_seq_p] * sb
            end
            s = s_stack[((((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)]
            sb = 0.0
        end
        for hh = h:-1:1
            row_max = row_max_stack[(((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * (div(h - 1, 1) + 1) + (hh - 1)) + 1)]
            row_sum = row_sum_stack[(((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * (div(h - 1, 1) + 1) + (hh - 1)) + 1)]
            s = s_stack[(((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (((i_seq_l - 1) * (div(h - 1, 1) + 1) + (hh - 1)) + 1)]
            head_offset = (hh - 1) * dk
            score_off = (hh - 1) * n * n
            for idx3 = n * dk:-1:1
                s = s_stack[((((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1)]
                i = div(idx3 - 1, dk) + 1
                p = mod(idx3 - 1, dk) + 1
                ctx[(i - 1) * d + head_offset + p] = ctx_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1]
                sb = sb + ctxb[(i - 1) * d + head_offset + p]
                ctxb[(i - 1) * d + head_offset + p] = 0.0
                for i_seq_j = 1:n
                    s = s + probs[score_off + (i - 1) * n + i_seq_j] * v[(i_seq_j - 1) * d + head_offset + p]
                    probsb[score_off + (i - 1) * n + i_seq_j] = probsb[score_off + (i - 1) * n + i_seq_j] + v[(i_seq_j - 1) * d + head_offset + p] * sb
                    vb[(i_seq_j - 1) * d + head_offset + p] = vb[(i_seq_j - 1) * d + head_offset + p] + probs[score_off + (i - 1) * n + i_seq_j] * sb
                end
                s = s_stack[((((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1)]
                sb = 0.0
            end
            for i = n:-1:1
                row_max = row_max_stack[((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
                row_sum = row_sum_stack[((div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)]
                for j = n:-1:1
                    kk = score_off + (i - 1) * n + j
                    probs[kk] = probs_stack[(div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (i - 1) * (div(n - 1, 1) + 1) + (j - 1)) + 1)]
                    row_sumb = row_sumb + -(probs[kk] / row_sum ^ 2) * probsb[kk]
                    probsb[kk] = (1.0 / row_sum) * probsb[kk]
                end
                for i_seq_j = 1:n
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_seq_j]
                    probsb[score_off + (i - 1) * n + i_seq_j] = probsb[score_off + (i - 1) * n + i_seq_j] + row_sumb
                end
                row_sum = row_sum_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1]
                row_sumb = 0.0
                for j = n:-1:1
                    kk = score_off + (i - 1) * n + j
                    probs[kk] = probs_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (i - 1) * (div(n - 1, 1) + 1) + (j - 1)) + 1]
                    scoresb[kk] = scoresb[kk] + exp(scores[kk] - row_max) * probsb[kk]
                    row_maxb = row_maxb + -(exp(scores[kk] - row_max) * probsb[kk])
                    probsb[kk] = 0.0
                end
                for i_seq_j = n:-1:2
                    row_max = row_max_stack[(div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (i - 1) * (div(n - 2, 1) + 1) + (i_seq_j - 2)) + 1)]
                    scoresb[score_off + (i - 1) * n + i_seq_j] = scoresb[score_off + (i - 1) * n + i_seq_j] + (0.5 * (1.0 + sign(scores[score_off + (i - 1) * n + i_seq_j] - row_max))) * row_maxb
                    row_maxb = (0.5 * (1.0 + sign(row_max - scores[score_off + (i - 1) * n + i_seq_j]))) * row_maxb
                end
                row_max = row_max_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1]
                scoresb[score_off + (i - 1) * n + 1] = scoresb[score_off + (i - 1) * n + 1] + row_maxb
                row_maxb = 0.0
            end
            for idx2 = n * n:-1:1
                s = s_stack[(((((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1) * (div(dk - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1)]
                i = div(idx2 - 1, n) + 1
                j = mod(idx2 - 1, n) + 1
                scores[score_off + (i - 1) * n + j] = scores_stack[((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1]
                sb = sb + inv_sqrt_dk * scoresb[score_off + (i - 1) * n + j]
                scoresb[score_off + (i - 1) * n + j] = 0.0
                for i_seq_p = 1:dk
                    s = s + q[(i - 1) * d + head_offset + i_seq_p] * k[(j - 1) * d + head_offset + i_seq_p]
                    qb[(i - 1) * d + head_offset + i_seq_p] = qb[(i - 1) * d + head_offset + i_seq_p] + k[(j - 1) * d + head_offset + i_seq_p] * sb
                    kb[(j - 1) * d + head_offset + i_seq_p] = kb[(j - 1) * d + head_offset + i_seq_p] + q[(i - 1) * d + head_offset + i_seq_p] * sb
                end
                s = s_stack[(((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1)]
                sb = 0.0
            end
        end
        for idx = n_d:-1:1
            s = s_stack[((((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            v[(i - 1) * d + j] = v_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1]
            sb = sb + vb[(i - 1) * d + j]
            bvb[b_offset + j] = bvb[b_offset + j] + vb[(i - 1) * d + j]
            vb[(i - 1) * d + j] = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wv[w_offset + (i_seq_p - 1) * d + j]
                xb[(i - 1) * d + i_seq_p] = xb[(i - 1) * d + i_seq_p] + wv[w_offset + (i_seq_p - 1) * d + j] * sb
                wvb[w_offset + (i_seq_p - 1) * d + j] = wvb[w_offset + (i_seq_p - 1) * d + j] + x[(i - 1) * d + i_seq_p] * sb
            end
            s = s_stack[((((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)]
            sb = 0.0
        end
        for idx = n_d:-1:1
            s = s_stack[(((((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            k[(i - 1) * d + j] = k_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1]
            sb = sb + kb[(i - 1) * d + j]
            bkb[b_offset + j] = bkb[b_offset + j] + kb[(i - 1) * d + j]
            kb[(i - 1) * d + j] = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wk[w_offset + (i_seq_p - 1) * d + j]
                xb[(i - 1) * d + i_seq_p] = xb[(i - 1) * d + i_seq_p] + wk[w_offset + (i_seq_p - 1) * d + j] * sb
                wkb[w_offset + (i_seq_p - 1) * d + j] = wkb[w_offset + (i_seq_p - 1) * d + j] + x[(i - 1) * d + i_seq_p] * sb
            end
            s = s_stack[(((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)]
            sb = 0.0
        end
        for idx = n_d:-1:1
            s = s_stack[((div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) + (div(n_layers - 1, 1) + 1) * (div(n_d - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            q[(i - 1) * d + j] = q_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1]
            sb = sb + qb[(i - 1) * d + j]
            bqb[b_offset + j] = bqb[b_offset + j] + qb[(i - 1) * d + j]
            qb[(i - 1) * d + j] = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wq[w_offset + (i_seq_p - 1) * d + j]
                xb[(i - 1) * d + i_seq_p] = xb[(i - 1) * d + i_seq_p] + wq[w_offset + (i_seq_p - 1) * d + j] * sb
                wqb[w_offset + (i_seq_p - 1) * d + j] = wqb[w_offset + (i_seq_p - 1) * d + j] + x[(i - 1) * d + i_seq_p] * sb
            end
            s = s_stack[((i_seq_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1]
            sb = 0.0
        end
    end
    inv_sqrt_dkb = 0.0
    return epsb
end

function transformer(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias, w1, b1, w2, b2, ln2_gain, ln2_bias, q, k, v, scores, probs, ctx, attn_out, resid1, normed1, ff_hidden, ff_out, resid2, x_next, n, dk, h, dff, n_layers, eps)
    d = h * dk
    inv_sqrt_dk = 1.0 / sqrt(dk)
    n_d = n * d
    n_dff = n * dff
    for i_seq_l = 1:n_layers
        w_offset = (i_seq_l - 1) * d * d
        b_offset = (i_seq_l - 1) * d
        ln_offset = (i_seq_l - 1) * d
        w1_offset = (i_seq_l - 1) * d * dff
        b1_offset = (i_seq_l - 1) * dff
        w2_offset = (i_seq_l - 1) * dff * d
        b2_offset = (i_seq_l - 1) * d
        ln2_offset = (i_seq_l - 1) * d
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wq[w_offset + (i_seq_p - 1) * d + j]
            end
            q[(i - 1) * d + j] = s + bq[b_offset + j]
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wk[w_offset + (i_seq_p - 1) * d + j]
            end
            k[(i - 1) * d + j] = s + bk[b_offset + j]
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wv[w_offset + (i_seq_p - 1) * d + j]
            end
            v[(i - 1) * d + j] = s + bv[b_offset + j]
        end
        for hh = 1:h
            head_offset = (hh - 1) * dk
            score_off = (hh - 1) * n * n
            for idx2 = 1:n * n
                i = div(idx2 - 1, n) + 1
                j = mod(idx2 - 1, n) + 1
                s = 0.0
                for i_seq_p = 1:dk
                    s = s + q[(i - 1) * d + head_offset + i_seq_p] * k[(j - 1) * d + head_offset + i_seq_p]
                end
                scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
            end
            for i = 1:n
                row_max = scores[score_off + (i - 1) * n + 1]
                for i_seq_j = 2:n
                    row_max = max(row_max, scores[score_off + (i - 1) * n + i_seq_j])
                end
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    probs[kk] = exp(scores[kk] - row_max)
                end
                row_sum = 0.0
                for i_seq_j = 1:n
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_seq_j]
                end
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    probs[kk] = probs[kk] / row_sum
                end
            end
            for idx3 = 1:n * dk
                i = div(idx3 - 1, dk) + 1
                p = mod(idx3 - 1, dk) + 1
                s = 0.0
                for i_seq_j = 1:n
                    s = s + probs[score_off + (i - 1) * n + i_seq_j] * v[(i_seq_j - 1) * d + head_offset + p]
                end
                ctx[(i - 1) * d + head_offset + p] = s
            end
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_seq_p = 1:d
                s = s + ctx[(i - 1) * d + i_seq_p] * wo[w_offset + (i_seq_p - 1) * d + j]
            end
            attn_out[(i - 1) * d + j] = s + bo[b_offset + j]
        end
        for idx = 1:n_d
            resid1[idx] = x[idx] + attn_out[idx]
        end
        for i = 1:n
            s = 0.0
            for i_seq_j = 1:d
                s = s + resid1[(i - 1) * d + i_seq_j]
            end
            row_mean = s / d
            s2 = 0.0
            for i_seq_j = 1:d
                diff = resid1[(i - 1) * d + i_seq_j] - row_mean
                s2 = s2 + diff * diff
            end
            row_var = s2 / d
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                normed1[kk] = ((resid1[kk] - row_mean) / denom) * ln1_gain[ln_offset + j] + ln1_bias[ln_offset + j]
            end
        end
        for idx = 1:n_dff
            i = div(idx - 1, dff) + 1
            j = mod(idx - 1, dff) + 1
            s = 0.0
            for i_seq_p = 1:d
                s = s + normed1[(i - 1) * d + i_seq_p] * w1[w1_offset + (i_seq_p - 1) * dff + j]
            end
            ff_hidden[(i - 1) * dff + j] = max(s + b1[b1_offset + j], 0.0)
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_seq_p = 1:dff
                s = s + ff_hidden[(i - 1) * dff + i_seq_p] * w2[w2_offset + (i_seq_p - 1) * d + j]
            end
            ff_out[(i - 1) * d + j] = s + b2[b2_offset + j]
        end
        for idx = 1:n_d
            resid2[idx] = normed1[idx] + ff_out[idx]
        end
        for i = 1:n
            s = 0.0
            for i_seq_j = 1:d
                s = s + resid2[(i - 1) * d + i_seq_j]
            end
            row_mean = s / d
            s2 = 0.0
            for i_seq_j = 1:d
                diff = resid2[(i - 1) * d + i_seq_j] - row_mean
                s2 = s2 + diff * diff
            end
            row_var = s2 / d
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                x_next[kk] = ((resid2[kk] - row_mean) / denom) * ln2_gain[ln2_offset + j] + ln2_bias[ln2_offset + j]
            end
        end
        for idx = 1:n_d
            x[idx] = x_next[idx]
        end
    end
end
