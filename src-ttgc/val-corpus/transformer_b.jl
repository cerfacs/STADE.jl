function initstacks_transformer_b()
    s_stack = Vector{Float64}()
    q_stack = Vector{Float64}()
    k_stack = Vector{Float64}()
    v_stack = Vector{Float64}()
    scores_stack = Vector{Float64}()
    row_max_stack = Vector{Float64}()
    probs_stack = Vector{Float64}()
    row_sum_stack = Vector{Float64}()
    ctx_stack = Vector{Float64}()
    resid1_stack = Vector{Float64}()
    row_mean_stack = Vector{Float64}()
    s2_stack = Vector{Float64}()
    diff_stack = Vector{Float64}()
    row_var_stack = Vector{Float64}()
    denom_stack = Vector{Float64}()
    normed1_stack = Vector{Float64}()
    ff_hidden_stack = Vector{Float64}()
    resid2_stack = Vector{Float64}()
    x_stack = Vector{Float64}()
    return (s_stack, q_stack, k_stack, v_stack, scores_stack, row_max_stack, probs_stack, row_sum_stack, ctx_stack, resid1_stack, row_mean_stack, s2_stack, diff_stack, row_var_stack, denom_stack, normed1_stack, ff_hidden_stack, resid2_stack, x_stack)
end

function transformer_b(x, xb, wq, wqb, bq, bqb, wk, wkb, bk, bkb, wv, wvb, bv, bvb, wo, wob, bo, bob, ln1_gain, ln1_gainb, ln1_bias, ln1_biasb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, ln2_gain, ln2_gainb, ln2_bias, ln2_biasb, q, qb, k, kb, v, vb, scores, scoresb, probs, probsb, ctx, ctxb, attn_out, attn_outb, resid1, resid1b, normed1, normed1b, ff_hidden, ff_hiddenb, ff_out, ff_outb, resid2, resid2b, x_next, x_nextb, n, d, dk, h, dff, n_layers, eps, epsb, s_stack, q_stack, k_stack, v_stack, scores_stack, row_max_stack, probs_stack, row_sum_stack, ctx_stack, resid1_stack, row_mean_stack, s2_stack, diff_stack, row_var_stack, denom_stack, normed1_stack, ff_hidden_stack, resid2_stack, x_stack)
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
            push!(s_stack, s)
            s = 0.0
            for i_seq_p = 1:d
                push!(s_stack, s)
                s = s + x[(i - 1) * d + i_seq_p] * wq[w_offset + (i_seq_p - 1) * d + j]
            end
            push!(q_stack, q[(i - 1) * d + j])
            q[(i - 1) * d + j] = s + bq[b_offset + j]
            push!(s_stack, s)
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            push!(s_stack, s)
            s = 0.0
            for i_seq_p = 1:d
                push!(s_stack, s)
                s = s + x[(i - 1) * d + i_seq_p] * wk[w_offset + (i_seq_p - 1) * d + j]
            end
            push!(k_stack, k[(i - 1) * d + j])
            k[(i - 1) * d + j] = s + bk[b_offset + j]
            push!(s_stack, s)
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            push!(s_stack, s)
            s = 0.0
            for i_seq_p = 1:d
                push!(s_stack, s)
                s = s + x[(i - 1) * d + i_seq_p] * wv[w_offset + (i_seq_p - 1) * d + j]
            end
            push!(v_stack, v[(i - 1) * d + j])
            v[(i - 1) * d + j] = s + bv[b_offset + j]
            push!(s_stack, s)
        end
        for hh = 1:h
            head_offset = (hh - 1) * dk
            score_off = (hh - 1) * n * n
            for idx2 = 1:n * n
                i = div(idx2 - 1, n) + 1
                j = mod(idx2 - 1, n) + 1
                push!(s_stack, s)
                s = 0.0
                for i_seq_p = 1:dk
                    push!(s_stack, s)
                    s = s + q[(i - 1) * d + head_offset + i_seq_p] * k[(j - 1) * d + head_offset + i_seq_p]
                end
                push!(scores_stack, scores[score_off + (i - 1) * n + j])
                scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
                push!(s_stack, s)
            end
            for i = 1:n
                push!(row_max_stack, row_max)
                row_max = scores[score_off + (i - 1) * n + 1]
                for i_seq_j = 2:n
                    push!(row_max_stack, row_max)
                    row_max = max(row_max, scores[score_off + (i - 1) * n + i_seq_j])
                end
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    push!(probs_stack, probs[kk])
                    probs[kk] = exp(scores[kk] - row_max)
                end
                push!(row_sum_stack, row_sum)
                row_sum = 0.0
                for i_seq_j = 1:n
                    push!(row_sum_stack, row_sum)
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_seq_j]
                end
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    push!(probs_stack, probs[kk])
                    probs[kk] = probs[kk] / row_sum
                end
                push!(row_max_stack, row_max)
                push!(row_sum_stack, row_sum)
            end
            for idx3 = 1:n * dk
                i = div(idx3 - 1, dk) + 1
                p = mod(idx3 - 1, dk) + 1
                push!(s_stack, s)
                s = 0.0
                for i_seq_j = 1:n
                    push!(s_stack, s)
                    s = s + probs[score_off + (i - 1) * n + i_seq_j] * v[(i_seq_j - 1) * d + head_offset + p]
                end
                push!(ctx_stack, ctx[(i - 1) * d + head_offset + p])
                ctx[(i - 1) * d + head_offset + p] = s
                push!(s_stack, s)
            end
            push!(row_max_stack, row_max)
            push!(row_sum_stack, row_sum)
            push!(s_stack, s)
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            push!(s_stack, s)
            s = 0.0
            for i_seq_p = 1:d
                push!(s_stack, s)
                s = s + ctx[(i - 1) * d + i_seq_p] * wo[w_offset + (i_seq_p - 1) * d + j]
            end
            attn_out[(i - 1) * d + j] = s + bo[b_offset + j]
            push!(s_stack, s)
        end
        for idx = 1:n_d
            push!(resid1_stack, resid1[idx])
            resid1[idx] = x[idx] + attn_out[idx]
        end
        for i = 1:n
            push!(s_stack, s)
            s = 0.0
            for i_seq_j = 1:d
                push!(s_stack, s)
                s = s + resid1[(i - 1) * d + i_seq_j]
            end
            push!(row_mean_stack, row_mean)
            row_mean = s / d
            push!(s2_stack, s2)
            s2 = 0.0
            for i_seq_j = 1:d
                push!(diff_stack, diff)
                diff = resid1[(i - 1) * d + i_seq_j] - row_mean
                push!(s2_stack, s2)
                s2 = s2 + diff * diff
            end
            push!(row_var_stack, row_var)
            row_var = s2 / d
            push!(denom_stack, denom)
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                push!(normed1_stack, normed1[kk])
                normed1[kk] = ((resid1[kk] - row_mean) / denom) * ln1_gain[ln_offset + j] + ln1_bias[ln_offset + j]
            end
            push!(diff_stack, diff)
            push!(s_stack, s)
            push!(s2_stack, s2)
        end
        for idx = 1:n_dff
            i = div(idx - 1, dff) + 1
            j = mod(idx - 1, dff) + 1
            push!(s_stack, s)
            s = 0.0
            for i_seq_p = 1:d
                push!(s_stack, s)
                s = s + normed1[(i - 1) * d + i_seq_p] * w1[w1_offset + (i_seq_p - 1) * dff + j]
            end
            push!(ff_hidden_stack, ff_hidden[(i - 1) * dff + j])
            ff_hidden[(i - 1) * dff + j] = max(s + b1[b1_offset + j], 0.0)
            push!(s_stack, s)
        end
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            push!(s_stack, s)
            s = 0.0
            for i_seq_p = 1:dff
                push!(s_stack, s)
                s = s + ff_hidden[(i - 1) * dff + i_seq_p] * w2[w2_offset + (i_seq_p - 1) * d + j]
            end
            ff_out[(i - 1) * d + j] = s + b2[b2_offset + j]
            push!(s_stack, s)
        end
        for idx = 1:n_d
            push!(resid2_stack, resid2[idx])
            resid2[idx] = normed1[idx] + ff_out[idx]
        end
        for i = 1:n
            push!(s_stack, s)
            s = 0.0
            for i_seq_j = 1:d
                push!(s_stack, s)
                s = s + resid2[(i - 1) * d + i_seq_j]
            end
            push!(row_mean_stack, row_mean)
            row_mean = s / d
            push!(s2_stack, s2)
            s2 = 0.0
            for i_seq_j = 1:d
                push!(diff_stack, diff)
                diff = resid2[(i - 1) * d + i_seq_j] - row_mean
                push!(s2_stack, s2)
                s2 = s2 + diff * diff
            end
            push!(row_var_stack, row_var)
            row_var = s2 / d
            push!(denom_stack, denom)
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                x_next[kk] = ((resid2[kk] - row_mean) / denom) * ln2_gain[ln2_offset + j] + ln2_bias[ln2_offset + j]
            end
            push!(diff_stack, diff)
            push!(s_stack, s)
            push!(s2_stack, s2)
        end
        for idx = 1:n_d
            push!(x_stack, x[idx])
            x[idx] = x_next[idx]
        end
        push!(denom_stack, denom)
        push!(diff_stack, diff)
        push!(row_max_stack, row_max)
        push!(row_mean_stack, row_mean)
        push!(row_sum_stack, row_sum)
        push!(row_var_stack, row_var)
        push!(s_stack, s)
        push!(s2_stack, s2)
    end
    push!(denom_stack, denom)
    push!(diff_stack, diff)
    push!(row_max_stack, row_max)
    push!(row_mean_stack, row_mean)
    push!(row_sum_stack, row_sum)
    push!(row_var_stack, row_var)
    push!(s_stack, s)
    push!(s2_stack, s2)
    denom = pop!(denom_stack)
    diff = pop!(diff_stack)
    row_max = pop!(row_max_stack)
    row_mean = pop!(row_mean_stack)
    row_sum = pop!(row_sum_stack)
    row_var = pop!(row_var_stack)
    s = pop!(s_stack)
    s2 = pop!(s2_stack)
    n_d = n * d
    n_dff = n * dff
    for i_seq_l = n_layers:-1:1
        denom = pop!(denom_stack)
        diff = pop!(diff_stack)
        row_max = pop!(row_max_stack)
        row_mean = pop!(row_mean_stack)
        row_sum = pop!(row_sum_stack)
        row_var = pop!(row_var_stack)
        s = pop!(s_stack)
        s2 = pop!(s2_stack)
        w_offset = (i_seq_l - 1) * d * d
        b_offset = (i_seq_l - 1) * d
        ln_offset = (i_seq_l - 1) * d
        w1_offset = (i_seq_l - 1) * d * dff
        b1_offset = (i_seq_l - 1) * dff
        w2_offset = (i_seq_l - 1) * dff * d
        b2_offset = (i_seq_l - 1) * d
        ln2_offset = (i_seq_l - 1) * d
        for idx = n_d:-1:1
            x[idx] = pop!(x_stack)
            x_nextb[idx] = x_nextb[idx] + xb[idx]
            xb[idx] = 0.0
        end
        for i = n:-1:1
            diff = pop!(diff_stack)
            s = pop!(s_stack)
            s2 = pop!(s2_stack)
            for j = 1:d
                kk = (i - 1) * d + j
                resid2b[kk] = resid2b[kk] + (1.0 / denom) * (ln2_gain[ln2_offset + j] * x_nextb[kk])
                row_meanb = row_meanb + -((1.0 / denom) * (ln2_gain[ln2_offset + j] * x_nextb[kk]))
                denomb = denomb + -((resid2[kk] - row_mean) / denom ^ 2) * (ln2_gain[ln2_offset + j] * x_nextb[kk])
                ln2_gainb[ln2_offset + j] = ln2_gainb[ln2_offset + j] + ((resid2[kk] - row_mean) / denom) * x_nextb[kk]
                ln2_biasb[ln2_offset + j] = ln2_biasb[ln2_offset + j] + x_nextb[kk]
                x_nextb[kk] = 0.0
            end
            denom = pop!(denom_stack)
            row_varb = row_varb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            epsb = epsb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            denomb = 0.0
            row_var = pop!(row_var_stack)
            s2b = s2b + (1.0 / d) * row_varb
            row_varb = 0.0
            for i_seq_j = d:-1:1
                s2 = pop!(s2_stack)
                diffb = diffb + diff * s2b
                diffb = diffb + diff * s2b
                diff = pop!(diff_stack)
                resid2b[(i - 1) * d + i_seq_j] = resid2b[(i - 1) * d + i_seq_j] + diffb
                row_meanb = row_meanb + -diffb
                diffb = 0.0
            end
            s2 = pop!(s2_stack)
            s2b = 0.0
            row_mean = pop!(row_mean_stack)
            sb = sb + (1.0 / d) * row_meanb
            row_meanb = 0.0
            for i_seq_j = d:-1:1
                s = pop!(s_stack)
                resid2b[(i - 1) * d + i_seq_j] = resid2b[(i - 1) * d + i_seq_j] + sb
            end
            s = pop!(s_stack)
            sb = 0.0
        end
        for idx = n_d:-1:1
            resid2[idx] = pop!(resid2_stack)
            normed1b[idx] = normed1b[idx] + resid2b[idx]
            ff_outb[idx] = ff_outb[idx] + resid2b[idx]
            resid2b[idx] = 0.0
        end
        for idx = n_d:-1:1
            s = pop!(s_stack)
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sb = sb + ff_outb[(i - 1) * d + j]
            b2b[b2_offset + j] = b2b[b2_offset + j] + ff_outb[(i - 1) * d + j]
            ff_outb[(i - 1) * d + j] = 0.0
            for i_seq_p = dff:-1:1
                s = pop!(s_stack)
                ff_hiddenb[(i - 1) * dff + i_seq_p] = ff_hiddenb[(i - 1) * dff + i_seq_p] + w2[w2_offset + (i_seq_p - 1) * d + j] * sb
                w2b[w2_offset + (i_seq_p - 1) * d + j] = w2b[w2_offset + (i_seq_p - 1) * d + j] + ff_hidden[(i - 1) * dff + i_seq_p] * sb
            end
            s = pop!(s_stack)
            sb = 0.0
        end
        for idx = n_dff:-1:1
            s = pop!(s_stack)
            i = div(idx - 1, dff) + 1
            j = mod(idx - 1, dff) + 1
            ff_hidden[(i - 1) * dff + j] = pop!(ff_hidden_stack)
            sb = sb + (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * ff_hiddenb[(i - 1) * dff + j]
            b1b[b1_offset + j] = b1b[b1_offset + j] + (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * ff_hiddenb[(i - 1) * dff + j]
            ff_hiddenb[(i - 1) * dff + j] = 0.0
            for i_seq_p = d:-1:1
                s = pop!(s_stack)
                normed1b[(i - 1) * d + i_seq_p] = normed1b[(i - 1) * d + i_seq_p] + w1[w1_offset + (i_seq_p - 1) * dff + j] * sb
                w1b[w1_offset + (i_seq_p - 1) * dff + j] = w1b[w1_offset + (i_seq_p - 1) * dff + j] + normed1[(i - 1) * d + i_seq_p] * sb
            end
            s = pop!(s_stack)
            sb = 0.0
        end
        for i = n:-1:1
            diff = pop!(diff_stack)
            s = pop!(s_stack)
            s2 = pop!(s2_stack)
            for j = d:-1:1
                kk = (i - 1) * d + j
                normed1[kk] = pop!(normed1_stack)
                resid1b[kk] = resid1b[kk] + (1.0 / denom) * (ln1_gain[ln_offset + j] * normed1b[kk])
                row_meanb = row_meanb + -((1.0 / denom) * (ln1_gain[ln_offset + j] * normed1b[kk]))
                denomb = denomb + -((resid1[kk] - row_mean) / denom ^ 2) * (ln1_gain[ln_offset + j] * normed1b[kk])
                ln1_gainb[ln_offset + j] = ln1_gainb[ln_offset + j] + ((resid1[kk] - row_mean) / denom) * normed1b[kk]
                ln1_biasb[ln_offset + j] = ln1_biasb[ln_offset + j] + normed1b[kk]
                normed1b[kk] = 0.0
            end
            denom = pop!(denom_stack)
            row_varb = row_varb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            epsb = epsb + (1.0 / (2.0 * sqrt(row_var + eps))) * denomb
            denomb = 0.0
            row_var = pop!(row_var_stack)
            s2b = s2b + (1.0 / d) * row_varb
            row_varb = 0.0
            for i_seq_j = d:-1:1
                s2 = pop!(s2_stack)
                diffb = diffb + diff * s2b
                diffb = diffb + diff * s2b
                diff = pop!(diff_stack)
                resid1b[(i - 1) * d + i_seq_j] = resid1b[(i - 1) * d + i_seq_j] + diffb
                row_meanb = row_meanb + -diffb
                diffb = 0.0
            end
            s2 = pop!(s2_stack)
            s2b = 0.0
            row_mean = pop!(row_mean_stack)
            sb = sb + (1.0 / d) * row_meanb
            row_meanb = 0.0
            for i_seq_j = d:-1:1
                s = pop!(s_stack)
                resid1b[(i - 1) * d + i_seq_j] = resid1b[(i - 1) * d + i_seq_j] + sb
            end
            s = pop!(s_stack)
            sb = 0.0
        end
        for idx = n_d:-1:1
            resid1[idx] = pop!(resid1_stack)
            xb[idx] = xb[idx] + resid1b[idx]
            attn_outb[idx] = attn_outb[idx] + resid1b[idx]
            resid1b[idx] = 0.0
        end
        for idx = n_d:-1:1
            s = pop!(s_stack)
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            sb = sb + attn_outb[(i - 1) * d + j]
            bob[b_offset + j] = bob[b_offset + j] + attn_outb[(i - 1) * d + j]
            attn_outb[(i - 1) * d + j] = 0.0
            for i_seq_p = d:-1:1
                s = pop!(s_stack)
                ctxb[(i - 1) * d + i_seq_p] = ctxb[(i - 1) * d + i_seq_p] + wo[w_offset + (i_seq_p - 1) * d + j] * sb
                wob[w_offset + (i_seq_p - 1) * d + j] = wob[w_offset + (i_seq_p - 1) * d + j] + ctx[(i - 1) * d + i_seq_p] * sb
            end
            s = pop!(s_stack)
            sb = 0.0
        end
        for hh = h:-1:1
            row_max = pop!(row_max_stack)
            row_sum = pop!(row_sum_stack)
            s = pop!(s_stack)
            head_offset = (hh - 1) * dk
            score_off = (hh - 1) * n * n
            for idx3 = n * dk:-1:1
                s = pop!(s_stack)
                i = div(idx3 - 1, dk) + 1
                p = mod(idx3 - 1, dk) + 1
                ctx[(i - 1) * d + head_offset + p] = pop!(ctx_stack)
                sb = sb + ctxb[(i - 1) * d + head_offset + p]
                ctxb[(i - 1) * d + head_offset + p] = 0.0
                for i_seq_j = n:-1:1
                    s = pop!(s_stack)
                    probsb[score_off + (i - 1) * n + i_seq_j] = probsb[score_off + (i - 1) * n + i_seq_j] + v[(i_seq_j - 1) * d + head_offset + p] * sb
                    vb[(i_seq_j - 1) * d + head_offset + p] = vb[(i_seq_j - 1) * d + head_offset + p] + probs[score_off + (i - 1) * n + i_seq_j] * sb
                end
                s = pop!(s_stack)
                sb = 0.0
            end
            for i = n:-1:1
                row_max = pop!(row_max_stack)
                row_sum = pop!(row_sum_stack)
                for j = n:-1:1
                    kk = score_off + (i - 1) * n + j
                    probs[kk] = pop!(probs_stack)
                    row_sumb = row_sumb + -(probs[kk] / row_sum ^ 2) * probsb[kk]
                    probsb[kk] = (1.0 / row_sum) * probsb[kk]
                end
                for i_seq_j = n:-1:1
                    row_sum = pop!(row_sum_stack)
                    probsb[score_off + (i - 1) * n + i_seq_j] = probsb[score_off + (i - 1) * n + i_seq_j] + row_sumb
                end
                row_sum = pop!(row_sum_stack)
                row_sumb = 0.0
                for j = n:-1:1
                    kk = score_off + (i - 1) * n + j
                    probs[kk] = pop!(probs_stack)
                    scoresb[kk] = scoresb[kk] + exp(scores[kk] - row_max) * probsb[kk]
                    row_maxb = row_maxb + -(exp(scores[kk] - row_max) * probsb[kk])
                    probsb[kk] = 0.0
                end
                for i_seq_j = n:-1:2
                    row_max = pop!(row_max_stack)
                    scoresb[score_off + (i - 1) * n + i_seq_j] = scoresb[score_off + (i - 1) * n + i_seq_j] + (0.5 * (1.0 + sign(scores[score_off + (i - 1) * n + i_seq_j] - row_max))) * row_maxb
                    row_maxb = (0.5 * (1.0 + sign(row_max - scores[score_off + (i - 1) * n + i_seq_j]))) * row_maxb
                end
                row_max = pop!(row_max_stack)
                scoresb[score_off + (i - 1) * n + 1] = scoresb[score_off + (i - 1) * n + 1] + row_maxb
                row_maxb = 0.0
            end
            for idx2 = n * n:-1:1
                s = pop!(s_stack)
                i = div(idx2 - 1, n) + 1
                j = mod(idx2 - 1, n) + 1
                scores[score_off + (i - 1) * n + j] = pop!(scores_stack)
                sb = sb + inv_sqrt_dk * scoresb[score_off + (i - 1) * n + j]
                scoresb[score_off + (i - 1) * n + j] = 0.0
                for i_seq_p = dk:-1:1
                    s = pop!(s_stack)
                    qb[(i - 1) * d + head_offset + i_seq_p] = qb[(i - 1) * d + head_offset + i_seq_p] + k[(j - 1) * d + head_offset + i_seq_p] * sb
                    kb[(j - 1) * d + head_offset + i_seq_p] = kb[(j - 1) * d + head_offset + i_seq_p] + q[(i - 1) * d + head_offset + i_seq_p] * sb
                end
                s = pop!(s_stack)
                sb = 0.0
            end
        end
        for idx = n_d:-1:1
            s = pop!(s_stack)
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            v[(i - 1) * d + j] = pop!(v_stack)
            sb = sb + vb[(i - 1) * d + j]
            bvb[b_offset + j] = bvb[b_offset + j] + vb[(i - 1) * d + j]
            vb[(i - 1) * d + j] = 0.0
            for i_seq_p = d:-1:1
                s = pop!(s_stack)
                xb[(i - 1) * d + i_seq_p] = xb[(i - 1) * d + i_seq_p] + wv[w_offset + (i_seq_p - 1) * d + j] * sb
                wvb[w_offset + (i_seq_p - 1) * d + j] = wvb[w_offset + (i_seq_p - 1) * d + j] + x[(i - 1) * d + i_seq_p] * sb
            end
            s = pop!(s_stack)
            sb = 0.0
        end
        for idx = n_d:-1:1
            s = pop!(s_stack)
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            k[(i - 1) * d + j] = pop!(k_stack)
            sb = sb + kb[(i - 1) * d + j]
            bkb[b_offset + j] = bkb[b_offset + j] + kb[(i - 1) * d + j]
            kb[(i - 1) * d + j] = 0.0
            for i_seq_p = d:-1:1
                s = pop!(s_stack)
                xb[(i - 1) * d + i_seq_p] = xb[(i - 1) * d + i_seq_p] + wk[w_offset + (i_seq_p - 1) * d + j] * sb
                wkb[w_offset + (i_seq_p - 1) * d + j] = wkb[w_offset + (i_seq_p - 1) * d + j] + x[(i - 1) * d + i_seq_p] * sb
            end
            s = pop!(s_stack)
            sb = 0.0
        end
        for idx = n_d:-1:1
            s = pop!(s_stack)
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            q[(i - 1) * d + j] = pop!(q_stack)
            sb = sb + qb[(i - 1) * d + j]
            bqb[b_offset + j] = bqb[b_offset + j] + qb[(i - 1) * d + j]
            qb[(i - 1) * d + j] = 0.0
            for i_seq_p = d:-1:1
                s = pop!(s_stack)
                xb[(i - 1) * d + i_seq_p] = xb[(i - 1) * d + i_seq_p] + wq[w_offset + (i_seq_p - 1) * d + j] * sb
                wqb[w_offset + (i_seq_p - 1) * d + j] = wqb[w_offset + (i_seq_p - 1) * d + j] + x[(i - 1) * d + i_seq_p] * sb
            end
            s = pop!(s_stack)
            sb = 0.0
        end
    end
    inv_sqrt_dkb = 0.0
    return epsb
end

function transformer(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias, w1, b1, w2, b2, ln2_gain, ln2_bias, q, k, v, scores, probs, ctx, attn_out, resid1, normed1, ff_hidden, ff_out, resid2, x_next, n, d, dk, h, dff, n_layers, eps)
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
