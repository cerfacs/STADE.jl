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

function transformer_hv(x, xb, wq, wqb, bq, bqb, wk, wkb, bk, bkb, wv, wvb, bv, bvb, wo, wob, bo, bob, ln1_gain, ln1_gainb, ln1_bias, ln1_biasb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, ln2_gain, ln2_gainb, ln2_bias, ln2_biasb, q, qb, k, kb, v, vb, scores, scoresb, probs, probsb, ctx, ctxb, attn_out, attn_outb, resid1, resid1b, normed1, normed1b, ff_hidden, ff_hiddenb, ff_out, ff_outb, resid2, resid2b, x_next, x_nextb, n, dk, h, dff, n_layers, eps, epsb, xd, xbd, wqd, wqbd, bqd, bqbd, wkd, wkbd, bkd, bkbd, wvd, wvbd, bvd, bvbd, wod, wobd, bod, bobd, ln1_gaind, ln1_gainbd, ln1_biasd, ln1_biasbd, w1d, w1bd, b1d, b1bd, w2d, w2bd, b2d, b2bd, ln2_gaind, ln2_gainbd, ln2_biasd, ln2_biasbd, qd, qbd, kd, kbd, vd, vbd, scoresd, scoresbd, probsd, probsbd, ctxd, ctxbd, attn_outd, attn_outbd, resid1d, resid1bd, normed1d, normed1bd, ff_hiddend, ff_hiddenbd, ff_outd, ff_outbd, resid2d, resid2bd, x_nextd, x_nextbd, epsd, epsbd, q_stack, k_stack, v_stack, scores_stack, row_max_stack, probs_stack, ctx_stack, resid1_stack, row_mean_stack, row_var_stack, denom_stack, normed1_stack, ff_hidden_stack, resid2_stack, x_stack, row_sum_stack, s_stack, s2_stack)
    q_stack_d = Vector{Float64}(undef, length(q_stack))
    k_stack_d = Vector{Float64}(undef, length(k_stack))
    v_stack_d = Vector{Float64}(undef, length(v_stack))
    scores_stack_d = Vector{Float64}(undef, length(scores_stack))
    row_max_stack_d = Vector{Float64}(undef, length(row_max_stack))
    probs_stack_d = Vector{Float64}(undef, length(probs_stack))
    ctx_stack_d = Vector{Float64}(undef, length(ctx_stack))
    resid1_stack_d = Vector{Float64}(undef, length(resid1_stack))
    row_mean_stack_d = Vector{Float64}(undef, length(row_mean_stack))
    row_var_stack_d = Vector{Float64}(undef, length(row_var_stack))
    denom_stack_d = Vector{Float64}(undef, length(denom_stack))
    normed1_stack_d = Vector{Float64}(undef, length(normed1_stack))
    ff_hidden_stack_d = Vector{Float64}(undef, length(ff_hidden_stack))
    resid2_stack_d = Vector{Float64}(undef, length(resid2_stack))
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    row_sum_stack_d = Vector{Float64}(undef, length(row_sum_stack))
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s2_stack_d = Vector{Float64}(undef, length(s2_stack))
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
    denomd = 0.0
    denombd = 0.0
    diffd = 0.0
    diffbd = 0.0
    inv_sqrt_dkd = 0.0
    inv_sqrt_dkbd = 0.0
    row_maxd = 0.0
    row_maxbd = 0.0
    row_meand = 0.0
    row_meanbd = 0.0
    row_sumd = 0.0
    row_sumbd = 0.0
    row_vard = 0.0
    row_varbd = 0.0
    sd = 0.0
    sbd = 0.0
    s2d = 0.0
    s2bd = 0.0
    d = h * dk
    inv_sqrt_dkd = 0.0
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
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                sd = sd + (wq[w_offset + (i_p - 1) * d + j] * xd[(i - 1) * d + i_p] + x[(i - 1) * d + i_p] * wqd[w_offset + (i_p - 1) * d + j])
                s = s + x[(i - 1) * d + i_p] * wq[w_offset + (i_p - 1) * d + j]
            end
            __idx_q_stack_4 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            q_stack_d[__idx_q_stack_4] = qd[(i - 1) * d + j]
            q_stack[__idx_q_stack_4] = q[(i - 1) * d + j]
            qd[(i - 1) * d + j] = sd + bqd[b_offset + j]
            q[(i - 1) * d + j] = s + bq[b_offset + j]
            __idx_s_stack_7 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            s_stack_d[__idx_s_stack_7] = sd
            s_stack[__idx_s_stack_7] = s
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                sd = sd + (wk[w_offset + (i_p - 1) * d + j] * xd[(i - 1) * d + i_p] + x[(i - 1) * d + i_p] * wkd[w_offset + (i_p - 1) * d + j])
                s = s + x[(i - 1) * d + i_p] * wk[w_offset + (i_p - 1) * d + j]
            end
            __idx_k_stack_4 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            k_stack_d[__idx_k_stack_4] = kd[(i - 1) * d + j]
            k_stack[__idx_k_stack_4] = k[(i - 1) * d + j]
            kd[(i - 1) * d + j] = sd + bkd[b_offset + j]
            k[(i - 1) * d + j] = s + bk[b_offset + j]
            __idx_s_stack_7 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + (((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)
            s_stack_d[__idx_s_stack_7] = sd
            s_stack[__idx_s_stack_7] = s
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                sd = sd + (wv[w_offset + (i_p - 1) * d + j] * xd[(i - 1) * d + i_p] + x[(i - 1) * d + i_p] * wvd[w_offset + (i_p - 1) * d + j])
                s = s + x[(i - 1) * d + i_p] * wv[w_offset + (i_p - 1) * d + j]
            end
            __idx_v_stack_4 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            v_stack_d[__idx_v_stack_4] = vd[(i - 1) * d + j]
            v_stack[__idx_v_stack_4] = v[(i - 1) * d + j]
            vd[(i - 1) * d + j] = sd + bvd[b_offset + j]
            v[(i - 1) * d + j] = s + bv[b_offset + j]
            __idx_s_stack_7 = (max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + (((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)
            s_stack_d[__idx_s_stack_7] = sd
            s_stack[__idx_s_stack_7] = s
        end
        for hh = 1:h
            head_offset = (hh - 1) * dk
            score_off = (hh - 1) * n * n
            for idx2 = 1:n * n
                i = div(idx2 - 1, n) + 1
                j = mod(idx2 - 1, n) + 1
                sd = 0.0
                s = 0.0
                for i_p = 1:dk
                    sd = sd + (k[(j - 1) * d + head_offset + i_p] * qd[(i - 1) * d + head_offset + i_p] + q[(i - 1) * d + head_offset + i_p] * kd[(j - 1) * d + head_offset + i_p])
                    s = s + q[(i - 1) * d + head_offset + i_p] * k[(j - 1) * d + head_offset + i_p]
                end
                __idx_scores_stack_4 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1
                scores_stack_d[__idx_scores_stack_4] = scoresd[score_off + (i - 1) * n + j]
                scores_stack[__idx_scores_stack_4] = scores[score_off + (i - 1) * n + j]
                scoresd[score_off + (i - 1) * n + j] = inv_sqrt_dk * sd + s * inv_sqrt_dkd
                scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
                __idx_s_stack_7 = ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + (((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1)
                s_stack_d[__idx_s_stack_7] = sd
                s_stack[__idx_s_stack_7] = s
            end
            for i = 1:n
                row_maxd = scoresd[score_off + (i - 1) * n + 1]
                row_max = scores[score_off + (i - 1) * n + 1]
                for i_j = 2:n
                    __idx_row_max_stack_0 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (i - 1) * (div(n - 2, 1) + 1) + (i_j - 2)) + 1
                    row_max_stack_d[__idx_row_max_stack_0] = row_maxd
                    row_max_stack[__idx_row_max_stack_0] = row_max
                    row_maxd = (0.5 * (1.0 + sign(row_max - scores[score_off + (i - 1) * n + i_j]))) * row_maxd + (0.5 * (1.0 + sign(scores[score_off + (i - 1) * n + i_j] - row_max))) * scoresd[score_off + (i - 1) * n + i_j]
                    row_max = max(row_max, scores[score_off + (i - 1) * n + i_j])
                end
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    __idx_probs_stack_1 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (i - 1) * (div(n - 1, 1) + 1) + (j - 1)) + 1
                    probs_stack_d[__idx_probs_stack_1] = probsd[kk]
                    probs_stack[__idx_probs_stack_1] = probs[kk]
                    probsd[kk] = exp(scores[kk] - row_max) * (scoresd[kk] + -row_maxd)
                    probs[kk] = exp(scores[kk] - row_max)
                end
                row_sumd = 0.0
                row_sum = 0.0
                for i_j = 1:n
                    row_sumd = row_sumd + probsd[score_off + (i - 1) * n + i_j]
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_j]
                end
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    __idx_probs_stack_1 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (i - 1) * (div(n - 1, 1) + 1) + (j - 1)) + 1)
                    probs_stack_d[__idx_probs_stack_1] = probsd[kk]
                    probs_stack[__idx_probs_stack_1] = probs[kk]
                    probsd[kk] = (1.0 / row_sum) * probsd[kk] + -(probs[kk] / row_sum ^ 2) * row_sumd
                    probs[kk] = probs[kk] / row_sum
                end
                __idx_row_max_stack_6 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 2, 1) + 1) + (((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
                row_max_stack_d[__idx_row_max_stack_6] = row_maxd
                row_max_stack[__idx_row_max_stack_6] = row_max
                __idx_row_sum_stack_8 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
                row_sum_stack_d[__idx_row_sum_stack_8] = row_sumd
                row_sum_stack[__idx_row_sum_stack_8] = row_sum
            end
            for idx3 = 1:n * dk
                i = div(idx3 - 1, dk) + 1
                p = mod(idx3 - 1, dk) + 1
                sd = 0.0
                s = 0.0
                for i_j = 1:n
                    sd = sd + (v[(i_j - 1) * d + head_offset + p] * probsd[score_off + (i - 1) * n + i_j] + probs[score_off + (i - 1) * n + i_j] * vd[(i_j - 1) * d + head_offset + p])
                    s = s + probs[score_off + (i - 1) * n + i_j] * v[(i_j - 1) * d + head_offset + p]
                end
                __idx_ctx_stack_4 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1
                ctx_stack_d[__idx_ctx_stack_4] = ctxd[(i - 1) * d + head_offset + p]
                ctx_stack[__idx_ctx_stack_4] = ctx[(i - 1) * d + head_offset + p]
                ctxd[(i - 1) * d + head_offset + p] = sd
                ctx[(i - 1) * d + head_offset + p] = s
                __idx_s_stack_7 = (((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + (((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1)
                s_stack_d[__idx_s_stack_7] = sd
                s_stack[__idx_s_stack_7] = s
            end
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                sd = sd + (wo[w_offset + (i_p - 1) * d + j] * ctxd[(i - 1) * d + i_p] + ctx[(i - 1) * d + i_p] * wod[w_offset + (i_p - 1) * d + j])
                s = s + ctx[(i - 1) * d + i_p] * wo[w_offset + (i_p - 1) * d + j]
            end
            attn_outd[(i - 1) * d + j] = sd + bod[b_offset + j]
            attn_out[(i - 1) * d + j] = s + bo[b_offset + j]
            __idx_s_stack_5 = ((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + (((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)
            s_stack_d[__idx_s_stack_5] = sd
            s_stack[__idx_s_stack_5] = s
        end
        for idx = 1:n_d
            __idx_resid1_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            resid1_stack_d[__idx_resid1_stack_0] = resid1d[idx]
            resid1_stack[__idx_resid1_stack_0] = resid1[idx]
            resid1d[idx] = xd[idx] + attn_outd[idx]
            resid1[idx] = x[idx] + attn_out[idx]
        end
        for i = 1:n
            sd = 0.0
            s = 0.0
            for i_j = 1:d
                sd = sd + resid1d[(i - 1) * d + i_j]
                s = s + resid1[(i - 1) * d + i_j]
            end
            __idx_row_mean_stack_2 = ((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
            row_mean_stack_d[__idx_row_mean_stack_2] = row_meand
            row_mean_stack[__idx_row_mean_stack_2] = row_mean
            row_meand = (1.0 / d) * sd
            row_mean = s / d
            s2d = 0.0
            s2 = 0.0
            for i_j = 1:d
                diffd = resid1d[(i - 1) * d + i_j] + -row_meand
                diff = resid1[(i - 1) * d + i_j] - row_mean
                s2d = s2d + (diff * diffd + diff * diffd)
                s2 = s2 + diff * diff
            end
            __idx_row_var_stack_7 = ((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
            row_var_stack_d[__idx_row_var_stack_7] = row_vard
            row_var_stack[__idx_row_var_stack_7] = row_var
            row_vard = (1.0 / d) * s2d
            row_var = s2 / d
            __idx_denom_stack_10 = ((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
            denom_stack_d[__idx_denom_stack_10] = denomd
            denom_stack[__idx_denom_stack_10] = denom
            denomd = (1.0 / (2.0 * sqrt(row_var + eps))) * (row_vard + epsd)
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                __idx_normed1_stack_1 = ((i_l - 1) * ((div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (i - 1) * (div(d - 1, 1) + 1) + (j - 1)) + 1
                normed1_stack_d[__idx_normed1_stack_1] = normed1d[kk]
                normed1_stack[__idx_normed1_stack_1] = normed1[kk]
                normed1d[kk] = (ln1_gain[ln_offset + j] * ((1.0 / denom) * (resid1d[kk] + -row_meand) + -((resid1[kk] - row_mean) / denom ^ 2) * denomd) + ((resid1[kk] - row_mean) / denom) * ln1_gaind[ln_offset + j]) + ln1_biasd[ln_offset + j]
                normed1[kk] = ((resid1[kk] - row_mean) / denom) * ln1_gain[ln_offset + j] + ln1_bias[ln_offset + j]
            end
            __idx_s_stack_14 = (((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            s_stack_d[__idx_s_stack_14] = sd
            s_stack[__idx_s_stack_14] = s
            __idx_s2_stack_16 = ((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
            s2_stack_d[__idx_s2_stack_16] = s2d
            s2_stack[__idx_s2_stack_16] = s2
        end
        for idx = 1:n_dff
            i = div(idx - 1, dff) + 1
            j = mod(idx - 1, dff) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                sd = sd + (w1[w1_offset + (i_p - 1) * dff + j] * normed1d[(i - 1) * d + i_p] + normed1[(i - 1) * d + i_p] * w1d[w1_offset + (i_p - 1) * dff + j])
                s = s + normed1[(i - 1) * d + i_p] * w1[w1_offset + (i_p - 1) * dff + j]
            end
            __idx_ff_hidden_stack_4 = ((i_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1
            ff_hidden_stack_d[__idx_ff_hidden_stack_4] = ff_hiddend[(i - 1) * dff + j]
            ff_hidden_stack[__idx_ff_hidden_stack_4] = ff_hidden[(i - 1) * dff + j]
            ff_hiddend[(i - 1) * dff + j] = (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * (sd + b1d[b1_offset + j])
            ff_hidden[(i - 1) * dff + j] = max(s + b1[b1_offset + j], 0.0)
            __idx_s_stack_7 = ((((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + (((i_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1)
            s_stack_d[__idx_s_stack_7] = sd
            s_stack[__idx_s_stack_7] = s
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:dff
                sd = sd + (w2[w2_offset + (i_p - 1) * d + j] * ff_hiddend[(i - 1) * dff + i_p] + ff_hidden[(i - 1) * dff + i_p] * w2d[w2_offset + (i_p - 1) * d + j])
                s = s + ff_hidden[(i - 1) * dff + i_p] * w2[w2_offset + (i_p - 1) * d + j]
            end
            ff_outd[(i - 1) * d + j] = sd + b2d[b2_offset + j]
            ff_out[(i - 1) * d + j] = s + b2[b2_offset + j]
            __idx_s_stack_5 = (((((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_dff - 1, 1) + 1)) + (((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)
            s_stack_d[__idx_s_stack_5] = sd
            s_stack[__idx_s_stack_5] = s
        end
        for idx = 1:n_d
            __idx_resid2_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            resid2_stack_d[__idx_resid2_stack_0] = resid2d[idx]
            resid2_stack[__idx_resid2_stack_0] = resid2[idx]
            resid2d[idx] = normed1d[idx] + ff_outd[idx]
            resid2[idx] = normed1[idx] + ff_out[idx]
        end
        for i = 1:n
            sd = 0.0
            s = 0.0
            for i_j = 1:d
                sd = sd + resid2d[(i - 1) * d + i_j]
                s = s + resid2[(i - 1) * d + i_j]
            end
            __idx_row_mean_stack_2 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            row_mean_stack_d[__idx_row_mean_stack_2] = row_meand
            row_mean_stack[__idx_row_mean_stack_2] = row_mean
            row_meand = (1.0 / d) * sd
            row_mean = s / d
            s2d = 0.0
            s2 = 0.0
            for i_j = 1:d
                diffd = resid2d[(i - 1) * d + i_j] + -row_meand
                diff = resid2[(i - 1) * d + i_j] - row_mean
                s2d = s2d + (diff * diffd + diff * diffd)
                s2 = s2 + diff * diff
            end
            __idx_row_var_stack_7 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            row_var_stack_d[__idx_row_var_stack_7] = row_vard
            row_var_stack[__idx_row_var_stack_7] = row_var
            row_vard = (1.0 / d) * s2d
            row_var = s2 / d
            __idx_denom_stack_10 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            denom_stack_d[__idx_denom_stack_10] = denomd
            denom_stack[__idx_denom_stack_10] = denom
            denomd = (1.0 / (2.0 * sqrt(row_var + eps))) * (row_vard + epsd)
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                x_nextd[kk] = (ln2_gain[ln2_offset + j] * ((1.0 / denom) * (resid2d[kk] + -row_meand) + -((resid2[kk] - row_mean) / denom ^ 2) * denomd) + ((resid2[kk] - row_mean) / denom) * ln2_gaind[ln2_offset + j]) + ln2_biasd[ln2_offset + j]
                x_next[kk] = ((resid2[kk] - row_mean) / denom) * ln2_gain[ln2_offset + j] + ln2_bias[ln2_offset + j]
            end
            __idx_s_stack_14 = ((((((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_dff - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            s_stack_d[__idx_s_stack_14] = sd
            s_stack[__idx_s_stack_14] = s
            __idx_s2_stack_16 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            s2_stack_d[__idx_s2_stack_16] = s2d
            s2_stack[__idx_s2_stack_16] = s2
        end
        for idx = 1:n_d
            __idx_x_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            x_stack_d[__idx_x_stack_0] = xd[idx]
            x_stack[__idx_x_stack_0] = x[idx]
            xd[idx] = x_nextd[idx]
            x[idx] = x_next[idx]
        end
        __idx_denom_stack_20 = (max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + ((i_l - 1) + 1)
        denom_stack_d[__idx_denom_stack_20] = denomd
        denom_stack[__idx_denom_stack_20] = denom
        __idx_row_mean_stack_22 = (max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + ((i_l - 1) + 1)
        row_mean_stack_d[__idx_row_mean_stack_22] = row_meand
        row_mean_stack[__idx_row_mean_stack_22] = row_mean
        __idx_row_var_stack_24 = (max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + ((i_l - 1) + 1)
        row_var_stack_d[__idx_row_var_stack_24] = row_vard
        row_var_stack[__idx_row_var_stack_24] = row_var
    end
    __idx_denom_stack_5 = ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1
    denom_stack_d[__idx_denom_stack_5] = denomd
    denom_stack[__idx_denom_stack_5] = denom
    __idx_row_mean_stack_7 = ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1
    row_mean_stack_d[__idx_row_mean_stack_7] = row_meand
    row_mean_stack[__idx_row_mean_stack_7] = row_mean
    __idx_row_var_stack_9 = ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1
    row_var_stack_d[__idx_row_var_stack_9] = row_vard
    row_var_stack[__idx_row_var_stack_9] = row_var
    __idx_denom_stack_0 = ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1
    denomd = denom_stack_d[__idx_denom_stack_0]
    denom = denom_stack[__idx_denom_stack_0]
    __idx_row_mean_stack_2 = ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1
    row_meand = row_mean_stack_d[__idx_row_mean_stack_2]
    row_mean = row_mean_stack[__idx_row_mean_stack_2]
    __idx_row_var_stack_4 = ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1
    row_vard = row_var_stack_d[__idx_row_var_stack_4]
    row_var = row_var_stack[__idx_row_var_stack_4]
    d = h * dk
    n_d = n * d
    n_dff = n * dff
    for i_l = n_layers:-1:1
        __idx_denom_stack_0 = (max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + ((i_l - 1) + 1)
        denomd = denom_stack_d[__idx_denom_stack_0]
        denom = denom_stack[__idx_denom_stack_0]
        __idx_row_mean_stack_2 = (max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + ((i_l - 1) + 1)
        row_meand = row_mean_stack_d[__idx_row_mean_stack_2]
        row_mean = row_mean_stack[__idx_row_mean_stack_2]
        __idx_row_var_stack_4 = (max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + ((i_l - 1) + 1)
        row_vard = row_var_stack_d[__idx_row_var_stack_4]
        row_var = row_var_stack[__idx_row_var_stack_4]
        w_offset = (i_l - 1) * d * d
        b_offset = (i_l - 1) * d
        ln_offset = (i_l - 1) * d
        w1_offset = (i_l - 1) * d * dff
        b1_offset = (i_l - 1) * dff
        w2_offset = (i_l - 1) * dff * d
        b2_offset = (i_l - 1) * d
        ln2_offset = (i_l - 1) * d
        for idx = n_d:-1:1
            __idx_x_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            xd[idx] = x_stack_d[__idx_x_stack_0]
            x[idx] = x_stack[__idx_x_stack_0]
            x_nextbd[idx] = x_nextbd[idx] + xbd[idx]
            x_nextb[idx] = x_nextb[idx] + xb[idx]
            xbd[idx] = 0.0
            xb[idx] = 0.0
        end
        for i = n:-1:1
            __idx_s_stack_0 = ((((((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_dff - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            __idx_s2_stack_2 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            s2d = s2_stack_d[__idx_s2_stack_2]
            s2 = s2_stack[__idx_s2_stack_2]
            for j = d:-1:1
                kk = (i - 1) * d + j
                resid2bd[kk] = resid2bd[kk] + ((ln2_gain[ln2_offset + j] * x_nextb[kk]) * (-(1.0 / denom ^ 2) * denomd) + (1.0 / denom) * (x_nextb[kk] * ln2_gaind[ln2_offset + j] + ln2_gain[ln2_offset + j] * x_nextbd[kk]))
                resid2b[kk] = resid2b[kk] + (1.0 / denom) * (ln2_gain[ln2_offset + j] * x_nextb[kk])
                row_meanbd = row_meanbd + -(((ln2_gain[ln2_offset + j] * x_nextb[kk]) * (-(1.0 / denom ^ 2) * denomd) + (1.0 / denom) * (x_nextb[kk] * ln2_gaind[ln2_offset + j] + ln2_gain[ln2_offset + j] * x_nextbd[kk])))
                row_meanb = row_meanb + -((1.0 / denom) * (ln2_gain[ln2_offset + j] * x_nextb[kk]))
                denombd = denombd + ((ln2_gain[ln2_offset + j] * x_nextb[kk]) * -(((1.0 / denom ^ 2) * (resid2d[kk] + -row_meand) + -((resid2[kk] - row_mean) / (denom ^ 2) ^ 2) * ((2denom) * denomd))) + -((resid2[kk] - row_mean) / denom ^ 2) * (x_nextb[kk] * ln2_gaind[ln2_offset + j] + ln2_gain[ln2_offset + j] * x_nextbd[kk]))
                denomb = denomb + -((resid2[kk] - row_mean) / denom ^ 2) * (ln2_gain[ln2_offset + j] * x_nextb[kk])
                ln2_gainbd[ln2_offset + j] = ln2_gainbd[ln2_offset + j] + (x_nextb[kk] * ((1.0 / denom) * (resid2d[kk] + -row_meand) + -((resid2[kk] - row_mean) / denom ^ 2) * denomd) + ((resid2[kk] - row_mean) / denom) * x_nextbd[kk])
                ln2_gainb[ln2_offset + j] = ln2_gainb[ln2_offset + j] + ((resid2[kk] - row_mean) / denom) * x_nextb[kk]
                ln2_biasbd[ln2_offset + j] = ln2_biasbd[ln2_offset + j] + x_nextbd[kk]
                ln2_biasb[ln2_offset + j] = ln2_biasb[ln2_offset + j] + x_nextb[kk]
                x_nextbd[kk] = 0.0
                x_nextb[kk] = 0.0
            end
            __idx_denom_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            denomd = denom_stack_d[__idx_denom_stack_0]
            denom = denom_stack[__idx_denom_stack_0]
            row_varbd = row_varbd + (denomb * (-(1.0 / (2.0 * sqrt(row_var + eps)) ^ 2) * (2.0 * ((1.0 / (2.0 * sqrt(row_var + eps))) * (row_vard + epsd)))) + (1.0 / (2.0 * sqrt(row_var + eps))) * denombd)
            row_varb = row_varb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            epsbd = epsbd + (denomb * (-(1.0 / (2.0 * sqrt(row_var + eps)) ^ 2) * (2.0 * ((1.0 / (2.0 * sqrt(row_var + eps))) * (row_vard + epsd)))) + (1.0 / (2.0 * sqrt(row_var + eps))) * denombd)
            epsb = epsb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            denombd = 0.0
            denomb = 0.0
            __idx_row_var_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            row_vard = row_var_stack_d[__idx_row_var_stack_0]
            row_var = row_var_stack[__idx_row_var_stack_0]
            s2bd = s2bd + (1.0 / d) * row_varbd
            s2b = s2b + (1.0 / d) * row_varb
            row_varbd = 0.0
            row_varb = 0.0
            for i_j = 1:d
                diffd = resid2d[(i - 1) * d + i_j] + -row_meand
                diff = resid2[(i - 1) * d + i_j] - row_mean
                s2d = s2d + (diff * diffd + diff * diffd)
                s2 = s2 + diff * diff
                diffbd = diffbd + (s2b * diffd + diff * s2bd)
                diffb = diffb + diff * s2b
                diffbd = diffbd + (s2b * diffd + diff * s2bd)
                diffb = diffb + diff * s2b
                resid2bd[(i - 1) * d + i_j] = resid2bd[(i - 1) * d + i_j] + diffbd
                resid2b[(i - 1) * d + i_j] = resid2b[(i - 1) * d + i_j] + diffb
                row_meanbd = row_meanbd + -diffbd
                row_meanb = row_meanb + -diffb
                diffbd = 0.0
                diffb = 0.0
            end
            s2bd = 0.0
            s2b = 0.0
            __idx_row_mean_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            row_meand = row_mean_stack_d[__idx_row_mean_stack_0]
            row_mean = row_mean_stack[__idx_row_mean_stack_0]
            sbd = sbd + (1.0 / d) * row_meanbd
            sb = sb + (1.0 / d) * row_meanb
            row_meanbd = 0.0
            row_meanb = 0.0
            for i_j = 1:d
                sd = sd + resid2d[(i - 1) * d + i_j]
                s = s + resid2[(i - 1) * d + i_j]
                resid2bd[(i - 1) * d + i_j] = resid2bd[(i - 1) * d + i_j] + sbd
                resid2b[(i - 1) * d + i_j] = resid2b[(i - 1) * d + i_j] + sb
            end
            sbd = 0.0
            sb = 0.0
        end
        for idx = n_d:-1:1
            __idx_resid2_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            resid2d[idx] = resid2_stack_d[__idx_resid2_stack_0]
            resid2[idx] = resid2_stack[__idx_resid2_stack_0]
            normed1bd[idx] = normed1bd[idx] + resid2bd[idx]
            normed1b[idx] = normed1b[idx] + resid2b[idx]
            ff_outbd[idx] = ff_outbd[idx] + resid2bd[idx]
            ff_outb[idx] = ff_outb[idx] + resid2b[idx]
            resid2bd[idx] = 0.0
            resid2b[idx] = 0.0
        end
        for idx = n_d:-1:1
            __idx_s_stack_0 = (((((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_dff - 1, 1) + 1)) + (((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sbd = sbd + ff_outbd[(i - 1) * d + j]
            sb = sb + ff_outb[(i - 1) * d + j]
            b2bd[b2_offset + j] = b2bd[b2_offset + j] + ff_outbd[(i - 1) * d + j]
            b2b[b2_offset + j] = b2b[b2_offset + j] + ff_outb[(i - 1) * d + j]
            ff_outbd[(i - 1) * d + j] = 0.0
            ff_outb[(i - 1) * d + j] = 0.0
            for i_p = 1:dff
                sd = sd + (w2[w2_offset + (i_p - 1) * d + j] * ff_hiddend[(i - 1) * dff + i_p] + ff_hidden[(i - 1) * dff + i_p] * w2d[w2_offset + (i_p - 1) * d + j])
                s = s + ff_hidden[(i - 1) * dff + i_p] * w2[w2_offset + (i_p - 1) * d + j]
                ff_hiddenbd[(i - 1) * dff + i_p] = ff_hiddenbd[(i - 1) * dff + i_p] + (sb * w2d[w2_offset + (i_p - 1) * d + j] + w2[w2_offset + (i_p - 1) * d + j] * sbd)
                ff_hiddenb[(i - 1) * dff + i_p] = ff_hiddenb[(i - 1) * dff + i_p] + w2[w2_offset + (i_p - 1) * d + j] * sb
                w2bd[w2_offset + (i_p - 1) * d + j] = w2bd[w2_offset + (i_p - 1) * d + j] + (sb * ff_hiddend[(i - 1) * dff + i_p] + ff_hidden[(i - 1) * dff + i_p] * sbd)
                w2b[w2_offset + (i_p - 1) * d + j] = w2b[w2_offset + (i_p - 1) * d + j] + ff_hidden[(i - 1) * dff + i_p] * sb
            end
            sbd = 0.0
            sb = 0.0
        end
        for idx = n_dff:-1:1
            __idx_s_stack_0 = ((((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + (((i_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1)
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            i = div(idx - 1, dff) + 1
            j = mod(idx - 1, dff) + 1
            __idx_ff_hidden_stack_0 = ((i_l - 1) * (div(n_dff - 1, 1) + 1) + (idx - 1)) + 1
            ff_hiddend[(i - 1) * dff + j] = ff_hidden_stack_d[__idx_ff_hidden_stack_0]
            ff_hidden[(i - 1) * dff + j] = ff_hidden_stack[__idx_ff_hidden_stack_0]
            sbd = sbd + (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * ff_hiddenbd[(i - 1) * dff + j]
            sb = sb + (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * ff_hiddenb[(i - 1) * dff + j]
            b1bd[b1_offset + j] = b1bd[b1_offset + j] + (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * ff_hiddenbd[(i - 1) * dff + j]
            b1b[b1_offset + j] = b1b[b1_offset + j] + (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * ff_hiddenb[(i - 1) * dff + j]
            ff_hiddenbd[(i - 1) * dff + j] = 0.0
            ff_hiddenb[(i - 1) * dff + j] = 0.0
            for i_p = 1:d
                sd = sd + (w1[w1_offset + (i_p - 1) * dff + j] * normed1d[(i - 1) * d + i_p] + normed1[(i - 1) * d + i_p] * w1d[w1_offset + (i_p - 1) * dff + j])
                s = s + normed1[(i - 1) * d + i_p] * w1[w1_offset + (i_p - 1) * dff + j]
                normed1bd[(i - 1) * d + i_p] = normed1bd[(i - 1) * d + i_p] + (sb * w1d[w1_offset + (i_p - 1) * dff + j] + w1[w1_offset + (i_p - 1) * dff + j] * sbd)
                normed1b[(i - 1) * d + i_p] = normed1b[(i - 1) * d + i_p] + w1[w1_offset + (i_p - 1) * dff + j] * sb
                w1bd[w1_offset + (i_p - 1) * dff + j] = w1bd[w1_offset + (i_p - 1) * dff + j] + (sb * normed1d[(i - 1) * d + i_p] + normed1[(i - 1) * d + i_p] * sbd)
                w1b[w1_offset + (i_p - 1) * dff + j] = w1b[w1_offset + (i_p - 1) * dff + j] + normed1[(i - 1) * d + i_p] * sb
            end
            sbd = 0.0
            sb = 0.0
        end
        for i = n:-1:1
            __idx_s_stack_0 = (((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + (((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            __idx_s2_stack_2 = ((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
            s2d = s2_stack_d[__idx_s2_stack_2]
            s2 = s2_stack[__idx_s2_stack_2]
            for j = d:-1:1
                kk = (i - 1) * d + j
                __idx_normed1_stack_0 = ((i_l - 1) * ((div(n - 1, 1) + 1) * (div(d - 1, 1) + 1)) + (i - 1) * (div(d - 1, 1) + 1) + (j - 1)) + 1
                normed1d[kk] = normed1_stack_d[__idx_normed1_stack_0]
                normed1[kk] = normed1_stack[__idx_normed1_stack_0]
                resid1bd[kk] = resid1bd[kk] + ((ln1_gain[ln_offset + j] * normed1b[kk]) * (-(1.0 / denom ^ 2) * denomd) + (1.0 / denom) * (normed1b[kk] * ln1_gaind[ln_offset + j] + ln1_gain[ln_offset + j] * normed1bd[kk]))
                resid1b[kk] = resid1b[kk] + (1.0 / denom) * (ln1_gain[ln_offset + j] * normed1b[kk])
                row_meanbd = row_meanbd + -(((ln1_gain[ln_offset + j] * normed1b[kk]) * (-(1.0 / denom ^ 2) * denomd) + (1.0 / denom) * (normed1b[kk] * ln1_gaind[ln_offset + j] + ln1_gain[ln_offset + j] * normed1bd[kk])))
                row_meanb = row_meanb + -((1.0 / denom) * (ln1_gain[ln_offset + j] * normed1b[kk]))
                denombd = denombd + ((ln1_gain[ln_offset + j] * normed1b[kk]) * -(((1.0 / denom ^ 2) * (resid1d[kk] + -row_meand) + -((resid1[kk] - row_mean) / (denom ^ 2) ^ 2) * ((2denom) * denomd))) + -((resid1[kk] - row_mean) / denom ^ 2) * (normed1b[kk] * ln1_gaind[ln_offset + j] + ln1_gain[ln_offset + j] * normed1bd[kk]))
                denomb = denomb + -((resid1[kk] - row_mean) / denom ^ 2) * (ln1_gain[ln_offset + j] * normed1b[kk])
                ln1_gainbd[ln_offset + j] = ln1_gainbd[ln_offset + j] + (normed1b[kk] * ((1.0 / denom) * (resid1d[kk] + -row_meand) + -((resid1[kk] - row_mean) / denom ^ 2) * denomd) + ((resid1[kk] - row_mean) / denom) * normed1bd[kk])
                ln1_gainb[ln_offset + j] = ln1_gainb[ln_offset + j] + ((resid1[kk] - row_mean) / denom) * normed1b[kk]
                ln1_biasbd[ln_offset + j] = ln1_biasbd[ln_offset + j] + normed1bd[kk]
                ln1_biasb[ln_offset + j] = ln1_biasb[ln_offset + j] + normed1b[kk]
                normed1bd[kk] = 0.0
                normed1b[kk] = 0.0
            end
            __idx_denom_stack_0 = ((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
            denomd = denom_stack_d[__idx_denom_stack_0]
            denom = denom_stack[__idx_denom_stack_0]
            row_varbd = row_varbd + (denomb * (-(1.0 / (2.0 * sqrt(row_var + eps)) ^ 2) * (2.0 * ((1.0 / (2.0 * sqrt(row_var + eps))) * (row_vard + epsd)))) + (1.0 / (2.0 * sqrt(row_var + eps))) * denombd)
            row_varb = row_varb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            epsbd = epsbd + (denomb * (-(1.0 / (2.0 * sqrt(row_var + eps)) ^ 2) * (2.0 * ((1.0 / (2.0 * sqrt(row_var + eps))) * (row_vard + epsd)))) + (1.0 / (2.0 * sqrt(row_var + eps))) * denombd)
            epsb = epsb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            denombd = 0.0
            denomb = 0.0
            __idx_row_var_stack_0 = ((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
            row_vard = row_var_stack_d[__idx_row_var_stack_0]
            row_var = row_var_stack[__idx_row_var_stack_0]
            s2bd = s2bd + (1.0 / d) * row_varbd
            s2b = s2b + (1.0 / d) * row_varb
            row_varbd = 0.0
            row_varb = 0.0
            for i_j = 1:d
                diffd = resid1d[(i - 1) * d + i_j] + -row_meand
                diff = resid1[(i - 1) * d + i_j] - row_mean
                s2d = s2d + (diff * diffd + diff * diffd)
                s2 = s2 + diff * diff
                diffbd = diffbd + (s2b * diffd + diff * s2bd)
                diffb = diffb + diff * s2b
                diffbd = diffbd + (s2b * diffd + diff * s2bd)
                diffb = diffb + diff * s2b
                resid1bd[(i - 1) * d + i_j] = resid1bd[(i - 1) * d + i_j] + diffbd
                resid1b[(i - 1) * d + i_j] = resid1b[(i - 1) * d + i_j] + diffb
                row_meanbd = row_meanbd + -diffbd
                row_meanb = row_meanb + -diffb
                diffbd = 0.0
                diffb = 0.0
            end
            s2bd = 0.0
            s2b = 0.0
            __idx_row_mean_stack_0 = ((i_l - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
            row_meand = row_mean_stack_d[__idx_row_mean_stack_0]
            row_mean = row_mean_stack[__idx_row_mean_stack_0]
            sbd = sbd + (1.0 / d) * row_meanbd
            sb = sb + (1.0 / d) * row_meanb
            row_meanbd = 0.0
            row_meanb = 0.0
            for i_j = 1:d
                sd = sd + resid1d[(i - 1) * d + i_j]
                s = s + resid1[(i - 1) * d + i_j]
                resid1bd[(i - 1) * d + i_j] = resid1bd[(i - 1) * d + i_j] + sbd
                resid1b[(i - 1) * d + i_j] = resid1b[(i - 1) * d + i_j] + sb
            end
            sbd = 0.0
            sb = 0.0
        end
        for idx = n_d:-1:1
            __idx_resid1_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            resid1d[idx] = resid1_stack_d[__idx_resid1_stack_0]
            resid1[idx] = resid1_stack[__idx_resid1_stack_0]
            xbd[idx] = xbd[idx] + resid1bd[idx]
            xb[idx] = xb[idx] + resid1b[idx]
            attn_outbd[idx] = attn_outbd[idx] + resid1bd[idx]
            attn_outb[idx] = attn_outb[idx] + resid1b[idx]
            resid1bd[idx] = 0.0
            resid1b[idx] = 0.0
        end
        for idx = n_d:-1:1
            __idx_s_stack_0 = ((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + (((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sbd = sbd + attn_outbd[(i - 1) * d + j]
            sb = sb + attn_outb[(i - 1) * d + j]
            bobd[b_offset + j] = bobd[b_offset + j] + attn_outbd[(i - 1) * d + j]
            bob[b_offset + j] = bob[b_offset + j] + attn_outb[(i - 1) * d + j]
            attn_outbd[(i - 1) * d + j] = 0.0
            attn_outb[(i - 1) * d + j] = 0.0
            for i_p = 1:d
                sd = sd + (wo[w_offset + (i_p - 1) * d + j] * ctxd[(i - 1) * d + i_p] + ctx[(i - 1) * d + i_p] * wod[w_offset + (i_p - 1) * d + j])
                s = s + ctx[(i - 1) * d + i_p] * wo[w_offset + (i_p - 1) * d + j]
                ctxbd[(i - 1) * d + i_p] = ctxbd[(i - 1) * d + i_p] + (sb * wod[w_offset + (i_p - 1) * d + j] + wo[w_offset + (i_p - 1) * d + j] * sbd)
                ctxb[(i - 1) * d + i_p] = ctxb[(i - 1) * d + i_p] + wo[w_offset + (i_p - 1) * d + j] * sb
                wobd[w_offset + (i_p - 1) * d + j] = wobd[w_offset + (i_p - 1) * d + j] + (sb * ctxd[(i - 1) * d + i_p] + ctx[(i - 1) * d + i_p] * sbd)
                wob[w_offset + (i_p - 1) * d + j] = wob[w_offset + (i_p - 1) * d + j] + ctx[(i - 1) * d + i_p] * sb
            end
            sbd = 0.0
            sb = 0.0
        end
        for hh = h:-1:1
            head_offset = (hh - 1) * dk
            score_off = (hh - 1) * n * n
            for idx3 = n * dk:-1:1
                __idx_s_stack_0 = (((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + (((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1)
                sd = s_stack_d[__idx_s_stack_0]
                s = s_stack[__idx_s_stack_0]
                i = div(idx3 - 1, dk) + 1
                p = mod(idx3 - 1, dk) + 1
                __idx_ctx_stack_0 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n * dk - 1, 1) + 1)) + (hh - 1) * (div(n * dk - 1, 1) + 1) + (idx3 - 1)) + 1
                ctxd[(i - 1) * d + head_offset + p] = ctx_stack_d[__idx_ctx_stack_0]
                ctx[(i - 1) * d + head_offset + p] = ctx_stack[__idx_ctx_stack_0]
                sbd = sbd + ctxbd[(i - 1) * d + head_offset + p]
                sb = sb + ctxb[(i - 1) * d + head_offset + p]
                ctxbd[(i - 1) * d + head_offset + p] = 0.0
                ctxb[(i - 1) * d + head_offset + p] = 0.0
                for i_j = 1:n
                    sd = sd + (v[(i_j - 1) * d + head_offset + p] * probsd[score_off + (i - 1) * n + i_j] + probs[score_off + (i - 1) * n + i_j] * vd[(i_j - 1) * d + head_offset + p])
                    s = s + probs[score_off + (i - 1) * n + i_j] * v[(i_j - 1) * d + head_offset + p]
                    probsbd[score_off + (i - 1) * n + i_j] = probsbd[score_off + (i - 1) * n + i_j] + (sb * vd[(i_j - 1) * d + head_offset + p] + v[(i_j - 1) * d + head_offset + p] * sbd)
                    probsb[score_off + (i - 1) * n + i_j] = probsb[score_off + (i - 1) * n + i_j] + v[(i_j - 1) * d + head_offset + p] * sb
                    vbd[(i_j - 1) * d + head_offset + p] = vbd[(i_j - 1) * d + head_offset + p] + (sb * probsd[score_off + (i - 1) * n + i_j] + probs[score_off + (i - 1) * n + i_j] * sbd)
                    vb[(i_j - 1) * d + head_offset + p] = vb[(i_j - 1) * d + head_offset + p] + probs[score_off + (i - 1) * n + i_j] * sb
                end
                sbd = 0.0
                sb = 0.0
            end
            for i = n:-1:1
                __idx_row_max_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 2, 1) + 1) + (((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1)
                row_maxd = row_max_stack_d[__idx_row_max_stack_0]
                row_max = row_max_stack[__idx_row_max_stack_0]
                __idx_row_sum_stack_2 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * (div(n - 1, 1) + 1) + (i - 1)) + 1
                row_sumd = row_sum_stack_d[__idx_row_sum_stack_2]
                row_sum = row_sum_stack[__idx_row_sum_stack_2]
                for j = n:-1:1
                    kk = score_off + (i - 1) * n + j
                    __idx_probs_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + (((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (i - 1) * (div(n - 1, 1) + 1) + (j - 1)) + 1)
                    probsd[kk] = probs_stack_d[__idx_probs_stack_0]
                    probs[kk] = probs_stack[__idx_probs_stack_0]
                    row_sumbd = row_sumbd + (probsb[kk] * -(((1.0 / row_sum ^ 2) * probsd[kk] + -(probs[kk] / (row_sum ^ 2) ^ 2) * ((2row_sum) * row_sumd))) + -(probs[kk] / row_sum ^ 2) * probsbd[kk])
                    row_sumb = row_sumb + -(probs[kk] / row_sum ^ 2) * probsb[kk]
                    probsbd[kk] = probsb[kk] * (-(1.0 / row_sum ^ 2) * row_sumd) + (1.0 / row_sum) * probsbd[kk]
                    probsb[kk] = (1.0 / row_sum) * probsb[kk]
                end
                for i_j = 1:n
                    row_sumd = row_sumd + probsd[score_off + (i - 1) * n + i_j]
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_j]
                    probsbd[score_off + (i - 1) * n + i_j] = probsbd[score_off + (i - 1) * n + i_j] + row_sumbd
                    probsb[score_off + (i - 1) * n + i_j] = probsb[score_off + (i - 1) * n + i_j] + row_sumb
                end
                row_sumbd = 0.0
                row_sumb = 0.0
                for j = n:-1:1
                    kk = score_off + (i - 1) * n + j
                    __idx_probs_stack_0 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (i - 1) * (div(n - 1, 1) + 1) + (j - 1)) + 1
                    probsd[kk] = probs_stack_d[__idx_probs_stack_0]
                    probs[kk] = probs_stack[__idx_probs_stack_0]
                    scoresbd[kk] = scoresbd[kk] + (probsb[kk] * (exp(scores[kk] - row_max) * (scoresd[kk] + -row_maxd)) + exp(scores[kk] - row_max) * probsbd[kk])
                    scoresb[kk] = scoresb[kk] + exp(scores[kk] - row_max) * probsb[kk]
                    row_maxbd = row_maxbd + -((probsb[kk] * (exp(scores[kk] - row_max) * (scoresd[kk] + -row_maxd)) + exp(scores[kk] - row_max) * probsbd[kk]))
                    row_maxb = row_maxb + -(exp(scores[kk] - row_max) * probsb[kk])
                    probsbd[kk] = 0.0
                    probsb[kk] = 0.0
                end
                for i_j = n:-1:2
                    __idx_row_max_stack_0 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (hh - 1) * ((div(n - 1, 1) + 1) * (div(n - 2, 1) + 1)) + (i - 1) * (div(n - 2, 1) + 1) + (i_j - 2)) + 1
                    row_maxd = row_max_stack_d[__idx_row_max_stack_0]
                    row_max = row_max_stack[__idx_row_max_stack_0]
                    scoresbd[score_off + (i - 1) * n + i_j] = scoresbd[score_off + (i - 1) * n + i_j] + (0.5 * (1.0 + sign(scores[score_off + (i - 1) * n + i_j] - row_max))) * row_maxbd
                    scoresb[score_off + (i - 1) * n + i_j] = scoresb[score_off + (i - 1) * n + i_j] + (0.5 * (1.0 + sign(scores[score_off + (i - 1) * n + i_j] - row_max))) * row_maxb
                    row_maxbd = (0.5 * (1.0 + sign(row_max - scores[score_off + (i - 1) * n + i_j]))) * row_maxbd
                    row_maxb = (0.5 * (1.0 + sign(row_max - scores[score_off + (i - 1) * n + i_j]))) * row_maxb
                end
                scoresbd[score_off + (i - 1) * n + 1] = scoresbd[score_off + (i - 1) * n + 1] + row_maxbd
                scoresb[score_off + (i - 1) * n + 1] = scoresb[score_off + (i - 1) * n + 1] + row_maxb
                row_maxbd = 0.0
                row_maxb = 0.0
            end
            for idx2 = n * n:-1:1
                __idx_s_stack_0 = ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + (((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1)
                sd = s_stack_d[__idx_s_stack_0]
                s = s_stack[__idx_s_stack_0]
                i = div(idx2 - 1, n) + 1
                j = mod(idx2 - 1, n) + 1
                __idx_scores_stack_0 = ((i_l - 1) * ((div(h - 1, 1) + 1) * (div(n * n - 1, 1) + 1)) + (hh - 1) * (div(n * n - 1, 1) + 1) + (idx2 - 1)) + 1
                scoresd[score_off + (i - 1) * n + j] = scores_stack_d[__idx_scores_stack_0]
                scores[score_off + (i - 1) * n + j] = scores_stack[__idx_scores_stack_0]
                sbd = sbd + (scoresb[score_off + (i - 1) * n + j] * inv_sqrt_dkd + inv_sqrt_dk * scoresbd[score_off + (i - 1) * n + j])
                sb = sb + inv_sqrt_dk * scoresb[score_off + (i - 1) * n + j]
                scoresbd[score_off + (i - 1) * n + j] = 0.0
                scoresb[score_off + (i - 1) * n + j] = 0.0
                for i_p = 1:dk
                    sd = sd + (k[(j - 1) * d + head_offset + i_p] * qd[(i - 1) * d + head_offset + i_p] + q[(i - 1) * d + head_offset + i_p] * kd[(j - 1) * d + head_offset + i_p])
                    s = s + q[(i - 1) * d + head_offset + i_p] * k[(j - 1) * d + head_offset + i_p]
                    qbd[(i - 1) * d + head_offset + i_p] = qbd[(i - 1) * d + head_offset + i_p] + (sb * kd[(j - 1) * d + head_offset + i_p] + k[(j - 1) * d + head_offset + i_p] * sbd)
                    qb[(i - 1) * d + head_offset + i_p] = qb[(i - 1) * d + head_offset + i_p] + k[(j - 1) * d + head_offset + i_p] * sb
                    kbd[(j - 1) * d + head_offset + i_p] = kbd[(j - 1) * d + head_offset + i_p] + (sb * qd[(i - 1) * d + head_offset + i_p] + q[(i - 1) * d + head_offset + i_p] * sbd)
                    kb[(j - 1) * d + head_offset + i_p] = kb[(j - 1) * d + head_offset + i_p] + q[(i - 1) * d + head_offset + i_p] * sb
                end
                sbd = 0.0
                sb = 0.0
            end
        end
        for idx = n_d:-1:1
            __idx_s_stack_0 = (max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1)) + (((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            __idx_v_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            vd[(i - 1) * d + j] = v_stack_d[__idx_v_stack_0]
            v[(i - 1) * d + j] = v_stack[__idx_v_stack_0]
            sbd = sbd + vbd[(i - 1) * d + j]
            sb = sb + vb[(i - 1) * d + j]
            bvbd[b_offset + j] = bvbd[b_offset + j] + vbd[(i - 1) * d + j]
            bvb[b_offset + j] = bvb[b_offset + j] + vb[(i - 1) * d + j]
            vbd[(i - 1) * d + j] = 0.0
            vb[(i - 1) * d + j] = 0.0
            for i_p = 1:d
                sd = sd + (wv[w_offset + (i_p - 1) * d + j] * xd[(i - 1) * d + i_p] + x[(i - 1) * d + i_p] * wvd[w_offset + (i_p - 1) * d + j])
                s = s + x[(i - 1) * d + i_p] * wv[w_offset + (i_p - 1) * d + j]
                xbd[(i - 1) * d + i_p] = xbd[(i - 1) * d + i_p] + (sb * wvd[w_offset + (i_p - 1) * d + j] + wv[w_offset + (i_p - 1) * d + j] * sbd)
                xb[(i - 1) * d + i_p] = xb[(i - 1) * d + i_p] + wv[w_offset + (i_p - 1) * d + j] * sb
                wvbd[w_offset + (i_p - 1) * d + j] = wvbd[w_offset + (i_p - 1) * d + j] + (sb * xd[(i - 1) * d + i_p] + x[(i - 1) * d + i_p] * sbd)
                wvb[w_offset + (i_p - 1) * d + j] = wvb[w_offset + (i_p - 1) * d + j] + x[(i - 1) * d + i_p] * sb
            end
            sbd = 0.0
            sb = 0.0
        end
        for idx = n_d:-1:1
            __idx_s_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d - 1, 1) + 1) + (((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1)
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            __idx_k_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            kd[(i - 1) * d + j] = k_stack_d[__idx_k_stack_0]
            k[(i - 1) * d + j] = k_stack[__idx_k_stack_0]
            sbd = sbd + kbd[(i - 1) * d + j]
            sb = sb + kb[(i - 1) * d + j]
            bkbd[b_offset + j] = bkbd[b_offset + j] + kbd[(i - 1) * d + j]
            bkb[b_offset + j] = bkb[b_offset + j] + kb[(i - 1) * d + j]
            kbd[(i - 1) * d + j] = 0.0
            kb[(i - 1) * d + j] = 0.0
            for i_p = 1:d
                sd = sd + (wk[w_offset + (i_p - 1) * d + j] * xd[(i - 1) * d + i_p] + x[(i - 1) * d + i_p] * wkd[w_offset + (i_p - 1) * d + j])
                s = s + x[(i - 1) * d + i_p] * wk[w_offset + (i_p - 1) * d + j]
                xbd[(i - 1) * d + i_p] = xbd[(i - 1) * d + i_p] + (sb * wkd[w_offset + (i_p - 1) * d + j] + wk[w_offset + (i_p - 1) * d + j] * sbd)
                xb[(i - 1) * d + i_p] = xb[(i - 1) * d + i_p] + wk[w_offset + (i_p - 1) * d + j] * sb
                wkbd[w_offset + (i_p - 1) * d + j] = wkbd[w_offset + (i_p - 1) * d + j] + (sb * xd[(i - 1) * d + i_p] + x[(i - 1) * d + i_p] * sbd)
                wkb[w_offset + (i_p - 1) * d + j] = wkb[w_offset + (i_p - 1) * d + j] + x[(i - 1) * d + i_p] * sb
            end
            sbd = 0.0
            sb = 0.0
        end
        for idx = n_d:-1:1
            __idx_s_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            __idx_q_stack_0 = ((i_l - 1) * (div(n_d - 1, 1) + 1) + (idx - 1)) + 1
            qd[(i - 1) * d + j] = q_stack_d[__idx_q_stack_0]
            q[(i - 1) * d + j] = q_stack[__idx_q_stack_0]
            sbd = sbd + qbd[(i - 1) * d + j]
            sb = sb + qb[(i - 1) * d + j]
            bqbd[b_offset + j] = bqbd[b_offset + j] + qbd[(i - 1) * d + j]
            bqb[b_offset + j] = bqb[b_offset + j] + qb[(i - 1) * d + j]
            qbd[(i - 1) * d + j] = 0.0
            qb[(i - 1) * d + j] = 0.0
            for i_p = 1:d
                sd = sd + (wq[w_offset + (i_p - 1) * d + j] * xd[(i - 1) * d + i_p] + x[(i - 1) * d + i_p] * wqd[w_offset + (i_p - 1) * d + j])
                s = s + x[(i - 1) * d + i_p] * wq[w_offset + (i_p - 1) * d + j]
                xbd[(i - 1) * d + i_p] = xbd[(i - 1) * d + i_p] + (sb * wqd[w_offset + (i_p - 1) * d + j] + wq[w_offset + (i_p - 1) * d + j] * sbd)
                xb[(i - 1) * d + i_p] = xb[(i - 1) * d + i_p] + wq[w_offset + (i_p - 1) * d + j] * sb
                wqbd[w_offset + (i_p - 1) * d + j] = wqbd[w_offset + (i_p - 1) * d + j] + (sb * xd[(i - 1) * d + i_p] + x[(i - 1) * d + i_p] * sbd)
                wqb[w_offset + (i_p - 1) * d + j] = wqb[w_offset + (i_p - 1) * d + j] + x[(i - 1) * d + i_p] * sb
            end
            sbd = 0.0
            sb = 0.0
        end
    end
    inv_sqrt_dkbd = 0.0
    inv_sqrt_dkb = 0.0
    return (epsb, epsbd)
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
