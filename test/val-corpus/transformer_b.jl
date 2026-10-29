function initstacks_transformer_b(dff, dk, h, n, n_layers)
    d = h * dk
    n_d = n * d
    n_dff = n * dff
    q_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1))
    k_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1))
    v_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1))
    scores_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1))
    row_max_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 2, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    probs_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    ctx_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1))
    resid1_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1))
    row_mean_stack = Vector{Float64}(undef, ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1)
    row_var_stack = Vector{Float64}(undef, ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1)
    denom_stack = Vector{Float64}(undef, ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1)
    normed1_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(d - 1, 1) + 1))
    ff_hidden_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_dff - 1, 1) + 1))
    resid2_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1))
    row_sum_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    s_stack = Vector{Float64}(undef, ((((((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_dff - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    s2_stack = Vector{Float64}(undef, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    return (q_stack, k_stack, v_stack, scores_stack, row_max_stack, probs_stack, ctx_stack, resid1_stack, row_mean_stack, row_var_stack, denom_stack, normed1_stack, ff_hidden_stack, resid2_stack, x_stack, row_sum_stack, s_stack, s2_stack)
end

function transformer_b(x, xb, wq, wqb, bq, bqb, wk, wkb, bk, bkb, wv, wvb, bv, bvb, wo, wob, bo, bob, ln1_gain, ln1_gainb, ln1_bias, ln1_biasb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, ln2_gain, ln2_gainb, ln2_bias, ln2_biasb, q, qb, k, kb, v, vb, scores, scoresb, probs, probsb, ctx, ctxb, attn_out, attn_outb, resid1, resid1b, normed1, normed1b, ff_hidden, ff_hiddenb, ff_out, ff_outb, resid2, resid2b, x_next, x_nextb, n, dk, h, dff, n_layers, eps, epsb, q_stack, k_stack, v_stack, scores_stack, row_max_stack, probs_stack, ctx_stack, resid1_stack, row_mean_stack, row_var_stack, denom_stack, normed1_stack, ff_hidden_stack, resid2_stack, x_stack, row_sum_stack, s_stack, s2_stack)
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
    for i_l = 1:n_layers
        __icse_0 = i_l - 1
        w_offset = __icse_0 * d * d
        __icse_1 = __icse_0 * d
        b_offset = __icse_1
        ln_offset = __icse_1
        w1_offset = __icse_0 * d * dff
        b1_offset = __icse_0 * dff
        w2_offset = __icse_0 * dff * d
        b2_offset = __icse_1
        ln2_offset = __icse_1
        for idx = 1:n_d
            __icse_2 = idx - 1
            i = div(__icse_2, d) + 1
            j = mod(__icse_2, d) + 1
            s = 0.0
            for i_p = 1:d
                s = s + x[(i - 1) * d + i_p] * wq[w_offset + (i_p - 1) * d + j]
            end
            __icse_3 = ((i_l - 1) + (idx - 1) * (div(n_layers - 1, 1) + 1)) + 1
            __idx_q_stack_4 = __icse_3
            q_stack[__idx_q_stack_4] = q[(i - 1) * d + j]
            q[(i - 1) * d + j] = s + bq[b_offset + j]
            __idx_s_stack_7 = __icse_3
            s_stack[__idx_s_stack_7] = s
        end
        for idx = 1:n_d
            __icse_4 = idx - 1
            i = div(__icse_4, d) + 1
            j = mod(__icse_4, d) + 1
            s = 0.0
            for i_p = 1:d
                s = s + x[(i - 1) * d + i_p] * wk[w_offset + (i_p - 1) * d + j]
            end
            __icse_5 = div(n_layers - 1, 1) + 1
            __icse_6 = ((i_l - 1) + (idx - 1) * __icse_5) + 1
            __idx_k_stack_4 = __icse_6
            k_stack[__idx_k_stack_4] = k[(i - 1) * d + j]
            k[(i - 1) * d + j] = s + bk[b_offset + j]
            __idx_s_stack_7 = max(0, __icse_5) * max(0, div(n_d - 1, 1) + 1) + __icse_6
            s_stack[__idx_s_stack_7] = s
        end
        for idx = 1:n_d
            __icse_7 = idx - 1
            i = div(__icse_7, d) + 1
            j = mod(__icse_7, d) + 1
            s = 0.0
            for i_p = 1:d
                s = s + x[(i - 1) * d + i_p] * wv[w_offset + (i_p - 1) * d + j]
            end
            __icse_8 = div(n_layers - 1, 1) + 1
            __icse_9 = ((i_l - 1) + (idx - 1) * __icse_8) + 1
            __idx_v_stack_4 = __icse_9
            v_stack[__idx_v_stack_4] = v[(i - 1) * d + j]
            v[(i - 1) * d + j] = s + bv[b_offset + j]
            __icse_10 = max(0, __icse_8) * max(0, div(n_d - 1, 1) + 1)
            __idx_s_stack_7 = (__icse_10 + __icse_10) + __icse_9
            s_stack[__idx_s_stack_7] = s
        end
        for hh = 1:h
            __icse_11 = hh - 1
            head_offset = __icse_11 * dk
            score_off = __icse_11 * n * n
            for idx2 = 1:n * n
                __icse_12 = idx2 - 1
                i = div(__icse_12, n) + 1
                j = mod(__icse_12, n) + 1
                s = 0.0
                for i_p = 1:dk
                    s = s + q[(i - 1) * d + head_offset + i_p] * k[(j - 1) * d + head_offset + i_p]
                end
                __icse_13 = div(n_layers - 1, 1) + 1
                __icse_14 = ((i_l - 1) + ((hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) * __icse_13) + 1
                __idx_scores_stack_4 = __icse_14
                scores_stack[__idx_scores_stack_4] = scores[score_off + (i - 1) * n + j]
                scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
                __icse_15 = max(0, __icse_13) * max(0, div(n_d - 1, 1) + 1)
                __idx_s_stack_7 = ((__icse_15 + __icse_15) + __icse_15) + __icse_14
                s_stack[__idx_s_stack_7] = s
            end
            for i = 1:n
                row_max = scores[score_off + (i - 1) * n + 1]
                for i_j = 2:n
                    __icse_16 = div(n - 2, 1) + 1
                    __idx_row_max_stack_0 = ((i_l - 1) + ((hh - 1) * ((div(n - 1, 1) + 1) * __icse_16) + (i - 1) * __icse_16 + (i_j - 2)) * (div(n_layers - 1, 1) + 1)) + 1
                    row_max_stack[__idx_row_max_stack_0] = row_max
                    row_max = max(row_max, scores[score_off + (i - 1) * n + i_j])
                end
                for j = 1:n
                    __icse_17 = i - 1
                    kk = score_off + __icse_17 * n + j
                    __icse_18 = div(n - 1, 1) + 1
                    __idx_probs_stack_1 = ((i_l - 1) + ((hh - 1) * (__icse_18 * __icse_18) + __icse_17 * __icse_18 + (j - 1)) * (div(n_layers - 1, 1) + 1)) + 1
                    probs_stack[__idx_probs_stack_1] = probs[kk]
                    probs[kk] = exp(scores[kk] - row_max)
                end
                row_sum = 0.0
                for i_j = 1:n
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_j]
                end
                for j = 1:n
                    __icse_19 = i - 1
                    kk = score_off + __icse_19 * n + j
                    __icse_20 = div(n_layers - 1, 1) + 1
                    __icse_21 = div(n - 1, 1) + 1
                    __icse_22 = max(0, __icse_21)
                    __idx_probs_stack_1 = max(0, __icse_20) * max(0, div(h - 1, 1) + 1) * __icse_22 * __icse_22 + (((i_l - 1) + ((hh - 1) * (__icse_21 * __icse_21) + __icse_19 * __icse_21 + (j - 1)) * __icse_20) + 1)
                    __cse_23 = probs[kk]
                    probs_stack[__idx_probs_stack_1] = __cse_23
                    probs[kk] = __cse_23 / row_sum
                end
                __icse_24 = div(n_layers - 1, 1) + 1
                __icse_25 = div(n - 1, 1) + 1
                __icse_26 = ((i_l - 1) + ((hh - 1) * __icse_25 + (i - 1)) * __icse_24) + 1
                __idx_row_max_stack_6 = max(0, __icse_24) * max(0, div(h - 1, 1) + 1) * max(0, __icse_25) * max(0, div(n - 2, 1) + 1) + __icse_26
                row_max_stack[__idx_row_max_stack_6] = row_max
                __idx_row_sum_stack_8 = __icse_26
                row_sum_stack[__idx_row_sum_stack_8] = row_sum
            end
            for idx3 = 1:n * dk
                __icse_27 = idx3 - 1
                i = div(__icse_27, dk) + 1
                p = mod(__icse_27, dk) + 1
                s = 0.0
                for i_j = 1:n
                    s = s + probs[score_off + (i - 1) * n + i_j] * v[(i_j - 1) * d + head_offset + p]
                end
                __icse_28 = div(n_layers - 1, 1) + 1
                __icse_29 = ((i_l - 1) + ((hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) * __icse_28) + 1
                __idx_ctx_stack_4 = __icse_29
                ctx_stack[__idx_ctx_stack_4] = ctx[(i - 1) * d + head_offset + p]
                ctx[(i - 1) * d + head_offset + p] = s
                __icse_30 = max(0, __icse_28)
                __icse_31 = __icse_30 * max(0, div(n_d - 1, 1) + 1)
                __idx_s_stack_7 = (((__icse_31 + __icse_31) + __icse_31) + __icse_30 * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + __icse_29
                s_stack[__idx_s_stack_7] = s
            end
        end
        for idx = 1:n_d
            __icse_32 = idx - 1
            i = div(__icse_32, d) + 1
            j = mod(__icse_32, d) + 1
            s = 0.0
            for i_p = 1:d
                s = s + ctx[(i - 1) * d + i_p] * wo[w_offset + (i_p - 1) * d + j]
            end
            attn_out[(i - 1) * d + j] = s + bo[b_offset + j]
            __icse_33 = div(n_layers - 1, 1) + 1
            __icse_34 = max(0, __icse_33)
            __icse_35 = __icse_34 * max(0, div(n_d - 1, 1) + 1)
            __icse_36 = max(0, div(h - 1, 1) + 1)
            __idx_s_stack_5 = ((((__icse_35 + __icse_35) + __icse_35) + __icse_34 * __icse_36 * max(0, div(n * n - 1, 1) + 1)) + __icse_34 * __icse_36 * max(0, div(n * dk - 1, 1) + 1)) + (((i_l - 1) + (idx - 1) * __icse_33) + 1)
            s_stack[__idx_s_stack_5] = s
        end
        for idx = 1:n_d
            __idx_resid1_stack_0 = ((i_l - 1) + (idx - 1) * (div(n_layers - 1, 1) + 1)) + 1
            resid1_stack[__idx_resid1_stack_0] = resid1[idx]
            resid1[idx] = x[idx] + attn_out[idx]
        end
        for i = 1:n
            s = 0.0
            for i_j = 1:d
                s = s + resid1[(i - 1) * d + i_j]
            end
            __idx_row_mean_stack_2 = ((i_l - 1) + (i - 1) * (div(n_layers - 1, 1) + 1)) + 1
            row_mean_stack[__idx_row_mean_stack_2] = row_mean
            row_mean = s / d
            s2 = 0.0
            for i_j = 1:d
                diff = resid1[(i - 1) * d + i_j] - row_mean
                s2 = s2 + diff * diff
            end
            __icse_37 = ((i_l - 1) + (i - 1) * (div(n_layers - 1, 1) + 1)) + 1
            __idx_row_var_stack_7 = __icse_37
            row_var_stack[__idx_row_var_stack_7] = row_var
            row_var = s2 / d
            __idx_denom_stack_10 = __icse_37
            denom_stack[__idx_denom_stack_10] = denom
            denom = sqrt(row_var + eps)
            for j = 1:d
                __icse_38 = i - 1
                kk = __icse_38 * d + j
                __idx_normed1_stack_1 = ((i_l - 1) + (__icse_38 * (div(d - 1, 1) + 1) + (j - 1)) * (div(n_layers - 1, 1) + 1)) + 1
                normed1_stack[__idx_normed1_stack_1] = normed1[kk]
                normed1[kk] = ((resid1[kk] - row_mean) / denom) * ln1_gain[ln_offset + j] + ln1_bias[ln_offset + j]
            end
            __icse_39 = div(n_layers - 1, 1) + 1
            __icse_40 = max(0, __icse_39)
            __icse_41 = __icse_40 * max(0, div(n_d - 1, 1) + 1)
            __icse_42 = max(0, div(h - 1, 1) + 1)
            __icse_43 = ((i_l - 1) + (i - 1) * __icse_39) + 1
            __idx_s_stack_14 = (((((__icse_41 + __icse_41) + __icse_41) + __icse_40 * __icse_42 * max(0, div(n * n - 1, 1) + 1)) + __icse_40 * __icse_42 * max(0, div(n * dk - 1, 1) + 1)) + __icse_41) + __icse_43
            s_stack[__idx_s_stack_14] = s
            __idx_s2_stack_16 = __icse_43
            s2_stack[__idx_s2_stack_16] = s2
        end
        for idx = 1:n_dff
            __icse_44 = idx - 1
            i = div(__icse_44, dff) + 1
            j = mod(__icse_44, dff) + 1
            s = 0.0
            for i_p = 1:d
                s = s + normed1[(i - 1) * d + i_p] * w1[w1_offset + (i_p - 1) * dff + j]
            end
            __icse_45 = div(n_layers - 1, 1) + 1
            __icse_46 = ((i_l - 1) + (idx - 1) * __icse_45) + 1
            __idx_ff_hidden_stack_4 = __icse_46
            ff_hidden_stack[__idx_ff_hidden_stack_4] = ff_hidden[(i - 1) * dff + j]
            ff_hidden[(i - 1) * dff + j] = max(s + b1[b1_offset + j], 0.0)
            __icse_47 = max(0, __icse_45)
            __icse_48 = __icse_47 * max(0, div(n_d - 1, 1) + 1)
            __icse_49 = max(0, div(h - 1, 1) + 1)
            __idx_s_stack_7 = ((((((__icse_48 + __icse_48) + __icse_48) + __icse_47 * __icse_49 * max(0, div(n * n - 1, 1) + 1)) + __icse_47 * __icse_49 * max(0, div(n * dk - 1, 1) + 1)) + __icse_48) + __icse_47 * max(0, div(n - 1, 1) + 1)) + __icse_46
            s_stack[__idx_s_stack_7] = s
        end
        for idx = 1:n_d
            __icse_50 = idx - 1
            i = div(__icse_50, d) + 1
            j = mod(__icse_50, d) + 1
            s = 0.0
            for i_p = 1:dff
                s = s + ff_hidden[(i - 1) * dff + i_p] * w2[w2_offset + (i_p - 1) * d + j]
            end
            ff_out[(i - 1) * d + j] = s + b2[b2_offset + j]
            __icse_51 = div(n_layers - 1, 1) + 1
            __icse_52 = max(0, __icse_51)
            __icse_53 = __icse_52 * max(0, div(n_d - 1, 1) + 1)
            __icse_54 = max(0, div(h - 1, 1) + 1)
            __idx_s_stack_5 = (((((((__icse_53 + __icse_53) + __icse_53) + __icse_52 * __icse_54 * max(0, div(n * n - 1, 1) + 1)) + __icse_52 * __icse_54 * max(0, div(n * dk - 1, 1) + 1)) + __icse_53) + __icse_52 * max(0, div(n - 1, 1) + 1)) + __icse_52 * max(0, div(n_dff - 1, 1) + 1)) + (((i_l - 1) + (idx - 1) * __icse_51) + 1)
            s_stack[__idx_s_stack_5] = s
        end
        for idx = 1:n_d
            __idx_resid2_stack_0 = ((i_l - 1) + (idx - 1) * (div(n_layers - 1, 1) + 1)) + 1
            resid2_stack[__idx_resid2_stack_0] = resid2[idx]
            resid2[idx] = normed1[idx] + ff_out[idx]
        end
        for i = 1:n
            s = 0.0
            for i_j = 1:d
                s = s + resid2[(i - 1) * d + i_j]
            end
            __icse_55 = div(n_layers - 1, 1) + 1
            __idx_row_mean_stack_2 = max(0, __icse_55) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) + (i - 1) * __icse_55) + 1)
            row_mean_stack[__idx_row_mean_stack_2] = row_mean
            row_mean = s / d
            s2 = 0.0
            for i_j = 1:d
                diff = resid2[(i - 1) * d + i_j] - row_mean
                s2 = s2 + diff * diff
            end
            __icse_56 = div(n_layers - 1, 1) + 1
            __icse_57 = max(0, __icse_56) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) + (i - 1) * __icse_56) + 1)
            __idx_row_var_stack_7 = __icse_57
            row_var_stack[__idx_row_var_stack_7] = row_var
            row_var = s2 / d
            __idx_denom_stack_10 = __icse_57
            denom_stack[__idx_denom_stack_10] = denom
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                x_next[kk] = ((resid2[kk] - row_mean) / denom) * ln2_gain[ln2_offset + j] + ln2_bias[ln2_offset + j]
            end
            __icse_58 = div(n_layers - 1, 1) + 1
            __icse_59 = max(0, __icse_58)
            __icse_60 = __icse_59 * max(0, div(n_d - 1, 1) + 1)
            __icse_61 = max(0, div(h - 1, 1) + 1)
            __icse_62 = __icse_59 * max(0, div(n - 1, 1) + 1)
            __icse_63 = ((i_l - 1) + (i - 1) * __icse_58) + 1
            __idx_s_stack_14 = ((((((((__icse_60 + __icse_60) + __icse_60) + __icse_59 * __icse_61 * max(0, div(n * n - 1, 1) + 1)) + __icse_59 * __icse_61 * max(0, div(n * dk - 1, 1) + 1)) + __icse_60) + __icse_62) + __icse_59 * max(0, div(n_dff - 1, 1) + 1)) + __icse_60) + __icse_63
            s_stack[__idx_s_stack_14] = s
            __idx_s2_stack_16 = __icse_62 + __icse_63
            s2_stack[__idx_s2_stack_16] = s2
        end
        for idx = 1:n_d
            __idx_x_stack_0 = ((i_l - 1) + (idx - 1) * (div(n_layers - 1, 1) + 1)) + 1
            x_stack[__idx_x_stack_0] = x[idx]
            x[idx] = x_next[idx]
        end
        __icse_64 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)
        __icse_65 = (__icse_64 + __icse_64) + ((i_l - 1) + 1)
        __idx_denom_stack_20 = __icse_65
        denom_stack[__idx_denom_stack_20] = denom
        __idx_row_mean_stack_22 = __icse_65
        row_mean_stack[__idx_row_mean_stack_22] = row_mean
        __idx_row_var_stack_24 = __icse_65
        row_var_stack[__idx_row_var_stack_24] = row_var
    end
    __icse_66 = max(0, div(n_layers - 1, 1) + 1)
    __icse_67 = __icse_66 * max(0, div(n - 1, 1) + 1)
    __icse_68 = ((__icse_67 + __icse_67) + __icse_66) + 1
    __idx_denom_stack_5 = __icse_68
    denom_stack[__idx_denom_stack_5] = denom
    __idx_row_mean_stack_7 = __icse_68
    row_mean_stack[__idx_row_mean_stack_7] = row_mean
    __idx_row_var_stack_9 = __icse_68
    row_var_stack[__idx_row_var_stack_9] = row_var
    __idx_denom_stack_0 = __icse_68
    denom = denom_stack[__idx_denom_stack_0]
    __idx_row_mean_stack_2 = __icse_68
    row_mean = row_mean_stack[__idx_row_mean_stack_2]
    __idx_row_var_stack_4 = __icse_68
    row_var = row_var_stack[__idx_row_var_stack_4]
    d = h * dk
    n_d = n * d
    n_dff = n * dff
    for i_l = n_layers:-1:1
        __icse_69 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)
        __icse_70 = i_l - 1
        __icse_71 = (__icse_69 + __icse_69) + (__icse_70 + 1)
        __idx_denom_stack_0 = __icse_71
        denom = denom_stack[__idx_denom_stack_0]
        __idx_row_mean_stack_2 = __icse_71
        row_mean = row_mean_stack[__idx_row_mean_stack_2]
        __idx_row_var_stack_4 = __icse_71
        row_var = row_var_stack[__idx_row_var_stack_4]
        w_offset = __icse_70 * d * d
        __icse_72 = __icse_70 * d
        b_offset = __icse_72
        ln_offset = __icse_72
        w1_offset = __icse_70 * d * dff
        b1_offset = __icse_70 * dff
        w2_offset = __icse_70 * dff * d
        b2_offset = __icse_72
        ln2_offset = __icse_72
        for idx = n_d:-1:1
            __idx_x_stack_0 = ((i_l - 1) + (idx - 1) * (div(n_layers - 1, 1) + 1)) + 1
            x[idx] = x_stack[__idx_x_stack_0]
            __oldb_2 = xb[idx]
            xb[idx] = 0.0
            x_nextb[idx] = x_nextb[idx] + __oldb_2
        end
        for i = n:-1:1
            __icse_73 = div(n_layers - 1, 1) + 1
            __icse_74 = max(0, __icse_73)
            __icse_75 = __icse_74 * max(0, div(n_d - 1, 1) + 1)
            __icse_76 = max(0, div(h - 1, 1) + 1)
            __icse_77 = __icse_74 * max(0, div(n - 1, 1) + 1)
            __icse_78 = ((i_l - 1) + (i - 1) * __icse_73) + 1
            __idx_s_stack_0 = ((((((((__icse_75 + __icse_75) + __icse_75) + __icse_74 * __icse_76 * max(0, div(n * n - 1, 1) + 1)) + __icse_74 * __icse_76 * max(0, div(n * dk - 1, 1) + 1)) + __icse_75) + __icse_77) + __icse_74 * max(0, div(n_dff - 1, 1) + 1)) + __icse_75) + __icse_78
            s = s_stack[__idx_s_stack_0]
            __idx_s2_stack_2 = __icse_77 + __icse_78
            s2 = s2_stack[__idx_s2_stack_2]
            for j = d:-1:1
                kk = (i - 1) * d + j
                __oldb_0 = x_nextb[kk]
                x_nextb[kk] = 0.0
                __cse_79 = ln2_gain[ln2_offset + j] * __oldb_0
                __cse_80 = (1.0 / denom) * __cse_79
                resid2b[kk] = resid2b[kk] + __cse_80
                row_meanb = row_meanb + -__cse_80
                __cse_81 = resid2[kk] - row_mean
                denomb = denomb + -(__cse_81 / denom ^ 2) * __cse_79
                ln2_gainb[ln2_offset + j] = ln2_gainb[ln2_offset + j] + (__cse_81 / denom) * __oldb_0
                ln2_biasb[ln2_offset + j] = ln2_biasb[ln2_offset + j] + __oldb_0
            end
            __icse_82 = div(n_layers - 1, 1) + 1
            __icse_83 = max(0, __icse_82) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) + (i - 1) * __icse_82) + 1)
            __idx_denom_stack_0 = __icse_83
            denom = denom_stack[__idx_denom_stack_0]
            __oldb_2 = denomb
            denomb = 0.0
            __cse_84 = (1.0 / (2.0 * sqrt(row_var + eps))) * __oldb_2
            row_varb = row_varb + __cse_84
            epsb = epsb + __cse_84
            __idx_row_var_stack_0 = __icse_83
            row_var = row_var_stack[__idx_row_var_stack_0]
            __oldb_2 = row_varb
            row_varb = 0.0
            s2b = s2b + (1.0 / d) * __oldb_2
            for i_j = 1:d
                diff = resid2[(i - 1) * d + i_j] - row_mean
                s2 = s2 + diff * diff
                __cse_85 = diff * s2b
                diffb = diffb + __cse_85
                diffb = diffb + __cse_85
                __oldb_0 = diffb
                diffb = 0.0
                resid2b[(i - 1) * d + i_j] = resid2b[(i - 1) * d + i_j] + __oldb_0
                row_meanb = row_meanb + -__oldb_0
            end
            s2b = 0.0
            __icse_86 = div(n_layers - 1, 1) + 1
            __idx_row_mean_stack_0 = max(0, __icse_86) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) + (i - 1) * __icse_86) + 1)
            row_mean = row_mean_stack[__idx_row_mean_stack_0]
            __oldb_2 = row_meanb
            row_meanb = 0.0
            sb = sb + (1.0 / d) * __oldb_2
            for i_j = 1:d
                s = s + resid2[(i - 1) * d + i_j]
                resid2b[(i - 1) * d + i_j] = resid2b[(i - 1) * d + i_j] + sb
            end
            sb = 0.0
        end
        for idx = n_d:-1:1
            __idx_resid2_stack_0 = ((i_l - 1) + (idx - 1) * (div(n_layers - 1, 1) + 1)) + 1
            resid2[idx] = resid2_stack[__idx_resid2_stack_0]
            __oldb_2 = resid2b[idx]
            resid2b[idx] = 0.0
            normed1b[idx] = normed1b[idx] + __oldb_2
            ff_outb[idx] = ff_outb[idx] + __oldb_2
        end
        for idx = n_d:-1:1
            __icse_87 = div(n_layers - 1, 1) + 1
            __icse_88 = max(0, __icse_87)
            __icse_89 = __icse_88 * max(0, div(n_d - 1, 1) + 1)
            __icse_90 = max(0, div(h - 1, 1) + 1)
            __icse_91 = idx - 1
            __idx_s_stack_0 = (((((((__icse_89 + __icse_89) + __icse_89) + __icse_88 * __icse_90 * max(0, div(n * n - 1, 1) + 1)) + __icse_88 * __icse_90 * max(0, div(n * dk - 1, 1) + 1)) + __icse_89) + __icse_88 * max(0, div(n - 1, 1) + 1)) + __icse_88 * max(0, div(n_dff - 1, 1) + 1)) + (((i_l - 1) + __icse_91 * __icse_87) + 1)
            s = s_stack[__idx_s_stack_0]
            i = div(__icse_91, d) + 1
            j = mod(__icse_91, d) + 1
            __oldb_0 = ff_outb[(i - 1) * d + j]
            ff_outb[(i - 1) * d + j] = 0.0
            sb = sb + __oldb_0
            b2b[b2_offset + j] = b2b[b2_offset + j] + __oldb_0
            for i_p = 1:dff
                __cse_92 = ff_hidden[(i - 1) * dff + i_p]
                __cse_93 = w2[w2_offset + (i_p - 1) * d + j]
                s = s + __cse_92 * __cse_93
                ff_hiddenb[(i - 1) * dff + i_p] = ff_hiddenb[(i - 1) * dff + i_p] + __cse_93 * sb
                w2b[w2_offset + (i_p - 1) * d + j] = w2b[w2_offset + (i_p - 1) * d + j] + __cse_92 * sb
            end
            sb = 0.0
        end
        for idx = n_dff:-1:1
            __icse_94 = div(n_layers - 1, 1) + 1
            __icse_95 = max(0, __icse_94)
            __icse_96 = __icse_95 * max(0, div(n_d - 1, 1) + 1)
            __icse_97 = max(0, div(h - 1, 1) + 1)
            __icse_98 = idx - 1
            __icse_99 = ((i_l - 1) + __icse_98 * __icse_94) + 1
            __idx_s_stack_0 = ((((((__icse_96 + __icse_96) + __icse_96) + __icse_95 * __icse_97 * max(0, div(n * n - 1, 1) + 1)) + __icse_95 * __icse_97 * max(0, div(n * dk - 1, 1) + 1)) + __icse_96) + __icse_95 * max(0, div(n - 1, 1) + 1)) + __icse_99
            s = s_stack[__idx_s_stack_0]
            i = div(__icse_98, dff) + 1
            j = mod(__icse_98, dff) + 1
            __idx_ff_hidden_stack_0 = __icse_99
            ff_hidden[(i - 1) * dff + j] = ff_hidden_stack[__idx_ff_hidden_stack_0]
            __oldb_2 = ff_hiddenb[(i - 1) * dff + j]
            ff_hiddenb[(i - 1) * dff + j] = 0.0
            __cse_100 = (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * __oldb_2
            sb = sb + __cse_100
            b1b[b1_offset + j] = b1b[b1_offset + j] + __cse_100
            for i_p = 1:d
                __cse_101 = normed1[(i - 1) * d + i_p]
                __cse_102 = w1[w1_offset + (i_p - 1) * dff + j]
                s = s + __cse_101 * __cse_102
                normed1b[(i - 1) * d + i_p] = normed1b[(i - 1) * d + i_p] + __cse_102 * sb
                w1b[w1_offset + (i_p - 1) * dff + j] = w1b[w1_offset + (i_p - 1) * dff + j] + __cse_101 * sb
            end
            sb = 0.0
        end
        for i = n:-1:1
            __icse_103 = div(n_layers - 1, 1) + 1
            __icse_104 = max(0, __icse_103)
            __icse_105 = __icse_104 * max(0, div(n_d - 1, 1) + 1)
            __icse_106 = max(0, div(h - 1, 1) + 1)
            __icse_107 = ((i_l - 1) + (i - 1) * __icse_103) + 1
            __idx_s_stack_0 = (((((__icse_105 + __icse_105) + __icse_105) + __icse_104 * __icse_106 * max(0, div(n * n - 1, 1) + 1)) + __icse_104 * __icse_106 * max(0, div(n * dk - 1, 1) + 1)) + __icse_105) + __icse_107
            s = s_stack[__idx_s_stack_0]
            __idx_s2_stack_2 = __icse_107
            s2 = s2_stack[__idx_s2_stack_2]
            for j = d:-1:1
                __icse_108 = i - 1
                kk = __icse_108 * d + j
                __idx_normed1_stack_0 = ((i_l - 1) + (__icse_108 * (div(d - 1, 1) + 1) + (j - 1)) * (div(n_layers - 1, 1) + 1)) + 1
                normed1[kk] = normed1_stack[__idx_normed1_stack_0]
                __oldb_2 = normed1b[kk]
                normed1b[kk] = 0.0
                __cse_109 = ln1_gain[ln_offset + j] * __oldb_2
                __cse_110 = (1.0 / denom) * __cse_109
                resid1b[kk] = resid1b[kk] + __cse_110
                row_meanb = row_meanb + -__cse_110
                __cse_111 = resid1[kk] - row_mean
                denomb = denomb + -(__cse_111 / denom ^ 2) * __cse_109
                ln1_gainb[ln_offset + j] = ln1_gainb[ln_offset + j] + (__cse_111 / denom) * __oldb_2
                ln1_biasb[ln_offset + j] = ln1_biasb[ln_offset + j] + __oldb_2
            end
            __icse_112 = ((i_l - 1) + (i - 1) * (div(n_layers - 1, 1) + 1)) + 1
            __idx_denom_stack_0 = __icse_112
            denom = denom_stack[__idx_denom_stack_0]
            __oldb_2 = denomb
            denomb = 0.0
            __cse_113 = (1.0 / (2.0 * sqrt(row_var + eps))) * __oldb_2
            row_varb = row_varb + __cse_113
            epsb = epsb + __cse_113
            __idx_row_var_stack_0 = __icse_112
            row_var = row_var_stack[__idx_row_var_stack_0]
            __oldb_2 = row_varb
            row_varb = 0.0
            s2b = s2b + (1.0 / d) * __oldb_2
            for i_j = 1:d
                diff = resid1[(i - 1) * d + i_j] - row_mean
                s2 = s2 + diff * diff
                __cse_114 = diff * s2b
                diffb = diffb + __cse_114
                diffb = diffb + __cse_114
                __oldb_0 = diffb
                diffb = 0.0
                resid1b[(i - 1) * d + i_j] = resid1b[(i - 1) * d + i_j] + __oldb_0
                row_meanb = row_meanb + -__oldb_0
            end
            s2b = 0.0
            __idx_row_mean_stack_0 = ((i_l - 1) + (i - 1) * (div(n_layers - 1, 1) + 1)) + 1
            row_mean = row_mean_stack[__idx_row_mean_stack_0]
            __oldb_2 = row_meanb
            row_meanb = 0.0
            sb = sb + (1.0 / d) * __oldb_2
            for i_j = 1:d
                s = s + resid1[(i - 1) * d + i_j]
                resid1b[(i - 1) * d + i_j] = resid1b[(i - 1) * d + i_j] + sb
            end
            sb = 0.0
        end
        for idx = n_d:-1:1
            __idx_resid1_stack_0 = ((i_l - 1) + (idx - 1) * (div(n_layers - 1, 1) + 1)) + 1
            resid1[idx] = resid1_stack[__idx_resid1_stack_0]
            __oldb_2 = resid1b[idx]
            resid1b[idx] = 0.0
            xb[idx] = xb[idx] + __oldb_2
            attn_outb[idx] = attn_outb[idx] + __oldb_2
        end
        for idx = n_d:-1:1
            __icse_115 = div(n_layers - 1, 1) + 1
            __icse_116 = max(0, __icse_115)
            __icse_117 = __icse_116 * max(0, div(n_d - 1, 1) + 1)
            __icse_118 = max(0, div(h - 1, 1) + 1)
            __icse_119 = idx - 1
            __idx_s_stack_0 = ((((__icse_117 + __icse_117) + __icse_117) + __icse_116 * __icse_118 * max(0, div(n * n - 1, 1) + 1)) + __icse_116 * __icse_118 * max(0, div(n * dk - 1, 1) + 1)) + (((i_l - 1) + __icse_119 * __icse_115) + 1)
            s = s_stack[__idx_s_stack_0]
            i = div(__icse_119, d) + 1
            j = mod(__icse_119, d) + 1
            __oldb_0 = attn_outb[(i - 1) * d + j]
            attn_outb[(i - 1) * d + j] = 0.0
            sb = sb + __oldb_0
            bob[b_offset + j] = bob[b_offset + j] + __oldb_0
            for i_p = 1:d
                __cse_120 = ctx[(i - 1) * d + i_p]
                __cse_121 = wo[w_offset + (i_p - 1) * d + j]
                s = s + __cse_120 * __cse_121
                ctxb[(i - 1) * d + i_p] = ctxb[(i - 1) * d + i_p] + __cse_121 * sb
                wob[w_offset + (i_p - 1) * d + j] = wob[w_offset + (i_p - 1) * d + j] + __cse_120 * sb
            end
            sb = 0.0
        end
        for hh = h:-1:1
            __icse_122 = hh - 1
            head_offset = __icse_122 * dk
            score_off = __icse_122 * n * n
            for idx3 = n * dk:-1:1
                __icse_123 = div(n_layers - 1, 1) + 1
                __icse_124 = max(0, __icse_123)
                __icse_125 = __icse_124 * max(0, div(n_d - 1, 1) + 1)
                __icse_126 = idx3 - 1
                __icse_127 = ((i_l - 1) + ((hh - 1) * (div(n * dk - 1, 1) + 1) + __icse_126) * __icse_123) + 1
                __idx_s_stack_0 = (((__icse_125 + __icse_125) + __icse_125) + __icse_124 * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + __icse_127
                s = s_stack[__idx_s_stack_0]
                i = div(__icse_126, dk) + 1
                p = mod(__icse_126, dk) + 1
                __idx_ctx_stack_0 = __icse_127
                ctx[(i - 1) * d + head_offset + p] = ctx_stack[__idx_ctx_stack_0]
                __oldb_2 = ctxb[(i - 1) * d + head_offset + p]
                ctxb[(i - 1) * d + head_offset + p] = 0.0
                sb = sb + __oldb_2
                for i_j = 1:n
                    __cse_128 = probs[score_off + (i - 1) * n + i_j]
                    __cse_129 = v[(i_j - 1) * d + head_offset + p]
                    s = s + __cse_128 * __cse_129
                    probsb[score_off + (i - 1) * n + i_j] = probsb[score_off + (i - 1) * n + i_j] + __cse_129 * sb
                    vb[(i_j - 1) * d + head_offset + p] = vb[(i_j - 1) * d + head_offset + p] + __cse_128 * sb
                end
                sb = 0.0
            end
            for i = n:-1:1
                __icse_130 = div(n_layers - 1, 1) + 1
                __icse_131 = div(n - 1, 1) + 1
                __icse_132 = ((i_l - 1) + ((hh - 1) * __icse_131 + (i - 1)) * __icse_130) + 1
                __idx_row_max_stack_0 = max(0, __icse_130) * max(0, div(h - 1, 1) + 1) * max(0, __icse_131) * max(0, div(n - 2, 1) + 1) + __icse_132
                row_max = row_max_stack[__idx_row_max_stack_0]
                __idx_row_sum_stack_2 = __icse_132
                row_sum = row_sum_stack[__idx_row_sum_stack_2]
                for j = n:-1:1
                    __icse_133 = i - 1
                    kk = score_off + __icse_133 * n + j
                    __icse_134 = div(n_layers - 1, 1) + 1
                    __icse_135 = div(n - 1, 1) + 1
                    __icse_136 = max(0, __icse_135)
                    __idx_probs_stack_0 = max(0, __icse_134) * max(0, div(h - 1, 1) + 1) * __icse_136 * __icse_136 + (((i_l - 1) + ((hh - 1) * (__icse_135 * __icse_135) + __icse_133 * __icse_135 + (j - 1)) * __icse_134) + 1)
                    probs[kk] = probs_stack[__idx_probs_stack_0]
                    __oldb_2 = probsb[kk]
                    probsb[kk] = 0.0
                    probsb[kk] = probsb[kk] + (1.0 / row_sum) * __oldb_2
                    row_sumb = row_sumb + -(probs[kk] / row_sum ^ 2) * __oldb_2
                end
                for i_j = 1:n
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_j]
                    probsb[score_off + (i - 1) * n + i_j] = probsb[score_off + (i - 1) * n + i_j] + row_sumb
                end
                row_sumb = 0.0
                for j = n:-1:1
                    __icse_137 = i - 1
                    kk = score_off + __icse_137 * n + j
                    __icse_138 = div(n - 1, 1) + 1
                    __idx_probs_stack_0 = ((i_l - 1) + ((hh - 1) * (__icse_138 * __icse_138) + __icse_137 * __icse_138 + (j - 1)) * (div(n_layers - 1, 1) + 1)) + 1
                    probs[kk] = probs_stack[__idx_probs_stack_0]
                    __oldb_2 = probsb[kk]
                    probsb[kk] = 0.0
                    __cse_139 = exp(scores[kk] - row_max) * __oldb_2
                    scoresb[kk] = scoresb[kk] + __cse_139
                    row_maxb = row_maxb + -__cse_139
                end
                for i_j = n:-1:2
                    __icse_140 = div(n - 2, 1) + 1
                    __idx_row_max_stack_0 = ((i_l - 1) + ((hh - 1) * ((div(n - 1, 1) + 1) * __icse_140) + (i - 1) * __icse_140 + (i_j - 2)) * (div(n_layers - 1, 1) + 1)) + 1
                    row_max = row_max_stack[__idx_row_max_stack_0]
                    __oldb_2 = row_maxb
                    row_maxb = 0.0
                    __cse_141 = scores[score_off + (i - 1) * n + i_j]
                    row_maxb = row_maxb + (0.5 * (1.0 + sign(row_max - __cse_141))) * __oldb_2
                    scoresb[score_off + (i - 1) * n + i_j] = scoresb[score_off + (i - 1) * n + i_j] + (0.5 * (1.0 + sign(__cse_141 - row_max))) * __oldb_2
                end
                __oldb_0 = row_maxb
                row_maxb = 0.0
                scoresb[score_off + (i - 1) * n + 1] = scoresb[score_off + (i - 1) * n + 1] + __oldb_0
            end
            for idx2 = n * n:-1:1
                __icse_142 = div(n_layers - 1, 1) + 1
                __icse_143 = max(0, __icse_142) * max(0, div(n_d - 1, 1) + 1)
                __icse_144 = idx2 - 1
                __icse_145 = ((i_l - 1) + ((hh - 1) * (div(n * n - 1, 1) + 1) + __icse_144) * __icse_142) + 1
                __idx_s_stack_0 = ((__icse_143 + __icse_143) + __icse_143) + __icse_145
                s = s_stack[__idx_s_stack_0]
                i = div(__icse_144, n) + 1
                j = mod(__icse_144, n) + 1
                __idx_scores_stack_0 = __icse_145
                scores[score_off + (i - 1) * n + j] = scores_stack[__idx_scores_stack_0]
                __oldb_2 = scoresb[score_off + (i - 1) * n + j]
                scoresb[score_off + (i - 1) * n + j] = 0.0
                sb = sb + inv_sqrt_dk * __oldb_2
                for i_p = 1:dk
                    __cse_146 = q[(i - 1) * d + head_offset + i_p]
                    __cse_147 = k[(j - 1) * d + head_offset + i_p]
                    s = s + __cse_146 * __cse_147
                    qb[(i - 1) * d + head_offset + i_p] = qb[(i - 1) * d + head_offset + i_p] + __cse_147 * sb
                    kb[(j - 1) * d + head_offset + i_p] = kb[(j - 1) * d + head_offset + i_p] + __cse_146 * sb
                end
                sb = 0.0
            end
        end
        for idx = n_d:-1:1
            __icse_148 = div(n_layers - 1, 1) + 1
            __icse_149 = max(0, __icse_148) * max(0, div(n_d - 1, 1) + 1)
            __icse_150 = idx - 1
            __icse_151 = ((i_l - 1) + __icse_150 * __icse_148) + 1
            __idx_s_stack_0 = (__icse_149 + __icse_149) + __icse_151
            s = s_stack[__idx_s_stack_0]
            i = div(__icse_150, d) + 1
            j = mod(__icse_150, d) + 1
            __idx_v_stack_0 = __icse_151
            v[(i - 1) * d + j] = v_stack[__idx_v_stack_0]
            __oldb_2 = vb[(i - 1) * d + j]
            vb[(i - 1) * d + j] = 0.0
            sb = sb + __oldb_2
            bvb[b_offset + j] = bvb[b_offset + j] + __oldb_2
            for i_p = 1:d
                __cse_152 = x[(i - 1) * d + i_p]
                __cse_153 = wv[w_offset + (i_p - 1) * d + j]
                s = s + __cse_152 * __cse_153
                xb[(i - 1) * d + i_p] = xb[(i - 1) * d + i_p] + __cse_153 * sb
                wvb[w_offset + (i_p - 1) * d + j] = wvb[w_offset + (i_p - 1) * d + j] + __cse_152 * sb
            end
            sb = 0.0
        end
        for idx = n_d:-1:1
            __icse_154 = div(n_layers - 1, 1) + 1
            __icse_155 = idx - 1
            __icse_156 = ((i_l - 1) + __icse_155 * __icse_154) + 1
            __idx_s_stack_0 = max(0, __icse_154) * max(0, div(n_d - 1, 1) + 1) + __icse_156
            s = s_stack[__idx_s_stack_0]
            i = div(__icse_155, d) + 1
            j = mod(__icse_155, d) + 1
            __idx_k_stack_0 = __icse_156
            k[(i - 1) * d + j] = k_stack[__idx_k_stack_0]
            __oldb_2 = kb[(i - 1) * d + j]
            kb[(i - 1) * d + j] = 0.0
            sb = sb + __oldb_2
            bkb[b_offset + j] = bkb[b_offset + j] + __oldb_2
            for i_p = 1:d
                __cse_157 = x[(i - 1) * d + i_p]
                __cse_158 = wk[w_offset + (i_p - 1) * d + j]
                s = s + __cse_157 * __cse_158
                xb[(i - 1) * d + i_p] = xb[(i - 1) * d + i_p] + __cse_158 * sb
                wkb[w_offset + (i_p - 1) * d + j] = wkb[w_offset + (i_p - 1) * d + j] + __cse_157 * sb
            end
            sb = 0.0
        end
        for idx = n_d:-1:1
            __icse_159 = idx - 1
            __icse_160 = ((i_l - 1) + __icse_159 * (div(n_layers - 1, 1) + 1)) + 1
            __idx_s_stack_0 = __icse_160
            s = s_stack[__idx_s_stack_0]
            i = div(__icse_159, d) + 1
            j = mod(__icse_159, d) + 1
            __idx_q_stack_0 = __icse_160
            q[(i - 1) * d + j] = q_stack[__idx_q_stack_0]
            __oldb_2 = qb[(i - 1) * d + j]
            qb[(i - 1) * d + j] = 0.0
            sb = sb + __oldb_2
            bqb[b_offset + j] = bqb[b_offset + j] + __oldb_2
            for i_p = 1:d
                __cse_161 = x[(i - 1) * d + i_p]
                __cse_162 = wq[w_offset + (i_p - 1) * d + j]
                s = s + __cse_161 * __cse_162
                xb[(i - 1) * d + i_p] = xb[(i - 1) * d + i_p] + __cse_162 * sb
                wqb[w_offset + (i_p - 1) * d + j] = wqb[w_offset + (i_p - 1) * d + j] + __cse_161 * sb
            end
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
    for i_l = 1:n_layers
        w_offset = (i_l - 1) * d * d
        b_offset = (i_l - 1) * d
        ln_offset = (i_l - 1) * d
        w1_offset = (i_l - 1) * d * dff
        b1_offset = (i_l - 1) * dff
        w2_offset = (i_l - 1) * dff * d
        b2_offset = (i_l - 1) * d
        ln2_offset = (i_l - 1) * d
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_p = 1:d
                s = s + x[(i - 1) * d + i_p] * wq[w_offset + (i_p - 1) * d + j]
            end
            q[(i - 1) * d + j] = s + bq[b_offset + j]
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_p = 1:d
                s = s + x[(i - 1) * d + i_p] * wk[w_offset + (i_p - 1) * d + j]
            end
            k[(i - 1) * d + j] = s + bk[b_offset + j]
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_p = 1:d
                s = s + x[(i - 1) * d + i_p] * wv[w_offset + (i_p - 1) * d + j]
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
                for i_p = 1:dk
                    s = s + q[(i - 1) * d + head_offset + i_p] * k[(j - 1) * d + head_offset + i_p]
                end
                scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
            end
            for i = 1:n
                row_max = scores[score_off + (i - 1) * n + 1]
                for i_j = 2:n
                    row_max = max(row_max, scores[score_off + (i - 1) * n + i_j])
                end
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    probs[kk] = exp(scores[kk] - row_max)
                end
                row_sum = 0.0
                for i_j = 1:n
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_j]
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
                for i_j = 1:n
                    s = s + probs[score_off + (i - 1) * n + i_j] * v[(i_j - 1) * d + head_offset + p]
                end
                ctx[(i - 1) * d + head_offset + p] = s
            end
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_p = 1:d
                s = s + ctx[(i - 1) * d + i_p] * wo[w_offset + (i_p - 1) * d + j]
            end
            attn_out[(i - 1) * d + j] = s + bo[b_offset + j]
        end
        for idx = 1:n_d
            resid1[idx] = x[idx] + attn_out[idx]
        end
        for i = 1:n
            s = 0.0
            for i_j = 1:d
                s = s + resid1[(i - 1) * d + i_j]
            end
            row_mean = s / d
            s2 = 0.0
            for i_j = 1:d
                diff = resid1[(i - 1) * d + i_j] - row_mean
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
            for i_p = 1:d
                s = s + normed1[(i - 1) * d + i_p] * w1[w1_offset + (i_p - 1) * dff + j]
            end
            ff_hidden[(i - 1) * dff + j] = max(s + b1[b1_offset + j], 0.0)
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_p = 1:dff
                s = s + ff_hidden[(i - 1) * dff + i_p] * w2[w2_offset + (i_p - 1) * d + j]
            end
            ff_out[(i - 1) * d + j] = s + b2[b2_offset + j]
        end
        for idx = 1:n_d
            resid2[idx] = normed1[idx] + ff_out[idx]
        end
        for i = 1:n
            s = 0.0
            for i_j = 1:d
                s = s + resid2[(i - 1) * d + i_j]
            end
            row_mean = s / d
            s2 = 0.0
            for i_j = 1:d
                diff = resid2[(i - 1) * d + i_j] - row_mean
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
