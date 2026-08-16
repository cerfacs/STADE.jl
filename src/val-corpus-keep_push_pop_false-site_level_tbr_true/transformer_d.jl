function transformer_d(x, xd, wq, wqd, bq, bqd, wk, wkd, bk, bkd, wv, wvd, bv, bvd, wo, wod, bo, bod, ln1_gain, ln1_gaind, ln1_bias, ln1_biasd, w1, w1d, b1, b1d, w2, w2d, b2, b2d, ln2_gain, ln2_gaind, ln2_bias, ln2_biasd, q, qd, k, kd, v, vd, scores, scoresd, probs, probsd, ctx, ctxd, attn_out, attn_outd, resid1, resid1d, normed1, normed1d, ff_hidden, ff_hiddend, ff_out, ff_outd, resid2, resid2d, x_next, x_nextd, n, d, dk, h, dff, n_layers, eps, epsd)
    inv_sqrt_dkd = 0.0
    inv_sqrt_dk = 1.0 / sqrt(dk)
    n_dd = 0.0
    n_d = n * d
    n_dffd = 0.0
    n_dff = n * dff
    for i_seq_l = 1:n_layers
        w_offsetd = 0.0
        w_offset = (i_seq_l - 1) * d * d
        b_offsetd = 0.0
        b_offset = (i_seq_l - 1) * d
        ln_offsetd = 0.0
        ln_offset = (i_seq_l - 1) * d
        w1_offsetd = 0.0
        w1_offset = (i_seq_l - 1) * d * dff
        b1_offsetd = 0.0
        b1_offset = (i_seq_l - 1) * dff
        w2_offsetd = 0.0
        w2_offset = (i_seq_l - 1) * dff * d
        b2_offsetd = 0.0
        b2_offset = (i_seq_l - 1) * d
        ln2_offsetd = 0.0
        ln2_offset = (i_seq_l - 1) * d
        for idx = 1:n_d
            id = 0.0
            i = div(idx - 1, d) + 1
            jd = 0.0
            j = mod(idx - 1, d) + 1
            sd = 0.0
            s = 0.0
            for i_seq_p = 1:d
                sd = sd + (wq[w_offset + (i_seq_p - 1) * d + j] * xd[(i - 1) * d + i_seq_p] + x[(i - 1) * d + i_seq_p] * wqd[w_offset + (i_seq_p - 1) * d + j])
                s = s + x[(i - 1) * d + i_seq_p] * wq[w_offset + (i_seq_p - 1) * d + j]
            end
            qd[(i - 1) * d + j] = sd + bqd[b_offset + j]
            q[(i - 1) * d + j] = s + bq[b_offset + j]
        end
        for idx = 1:n_d
            id = 0.0
            i = div(idx - 1, d) + 1
            jd = 0.0
            j = mod(idx - 1, d) + 1
            sd = 0.0
            s = 0.0
            for i_seq_p = 1:d
                sd = sd + (wk[w_offset + (i_seq_p - 1) * d + j] * xd[(i - 1) * d + i_seq_p] + x[(i - 1) * d + i_seq_p] * wkd[w_offset + (i_seq_p - 1) * d + j])
                s = s + x[(i - 1) * d + i_seq_p] * wk[w_offset + (i_seq_p - 1) * d + j]
            end
            kd[(i - 1) * d + j] = sd + bkd[b_offset + j]
            k[(i - 1) * d + j] = s + bk[b_offset + j]
        end
        for idx = 1:n_d
            id = 0.0
            i = div(idx - 1, d) + 1
            jd = 0.0
            j = mod(idx - 1, d) + 1
            sd = 0.0
            s = 0.0
            for i_seq_p = 1:d
                sd = sd + (wv[w_offset + (i_seq_p - 1) * d + j] * xd[(i - 1) * d + i_seq_p] + x[(i - 1) * d + i_seq_p] * wvd[w_offset + (i_seq_p - 1) * d + j])
                s = s + x[(i - 1) * d + i_seq_p] * wv[w_offset + (i_seq_p - 1) * d + j]
            end
            vd[(i - 1) * d + j] = sd + bvd[b_offset + j]
            v[(i - 1) * d + j] = s + bv[b_offset + j]
        end
        for hh = 1:h
            head_offsetd = 0.0
            head_offset = (hh - 1) * dk
            for idx2 = 1:n * n
                id = 0.0
                i = div(idx2 - 1, n) + 1
                jd = 0.0
                j = mod(idx2 - 1, n) + 1
                sd = 0.0
                s = 0.0
                for i_seq_p = 1:dk
                    sd = sd + (k[(j - 1) * d + head_offset + i_seq_p] * qd[(i - 1) * d + head_offset + i_seq_p] + q[(i - 1) * d + head_offset + i_seq_p] * kd[(j - 1) * d + head_offset + i_seq_p])
                    s = s + q[(i - 1) * d + head_offset + i_seq_p] * k[(j - 1) * d + head_offset + i_seq_p]
                end
                scoresd[(i - 1) * n + j] = inv_sqrt_dk * sd
                scores[(i - 1) * n + j] = s * inv_sqrt_dk
            end
            for i = 1:n
                row_maxd = scoresd[(i - 1) * n + 1]
                row_max = scores[(i - 1) * n + 1]
                for i_seq_j = 2:n
                    row_maxd = (0.5 * (1.0 + sign(row_max - scores[(i - 1) * n + i_seq_j]))) * row_maxd + (0.5 * (1.0 + sign(scores[(i - 1) * n + i_seq_j] - row_max))) * scoresd[(i - 1) * n + i_seq_j]
                    row_max = max(row_max, scores[(i - 1) * n + i_seq_j])
                end
                for j = 1:n
                    kkd = 0.0
                    kk = (i - 1) * n + j
                    probsd[kk] = exp(scores[kk] - row_max) * (scoresd[kk] + -row_maxd)
                    probs[kk] = exp(scores[kk] - row_max)
                end
                row_sumd = 0.0
                row_sum = 0.0
                for i_seq_j = 1:n
                    row_sumd = row_sumd + probsd[(i - 1) * n + i_seq_j]
                    row_sum = row_sum + probs[(i - 1) * n + i_seq_j]
                end
                for j = 1:n
                    kkd = 0.0
                    kk = (i - 1) * n + j
                    probsd[kk] = (1.0 / row_sum) * probsd[kk] + -(probs[kk] / row_sum ^ 2) * row_sumd
                    probs[kk] = probs[kk] / row_sum
                end
            end
            for idx3 = 1:n * dk
                id = 0.0
                i = div(idx3 - 1, dk) + 1
                pd = 0.0
                p = mod(idx3 - 1, dk) + 1
                sd = 0.0
                s = 0.0
                for i_seq_j = 1:n
                    sd = sd + (v[(i_seq_j - 1) * d + head_offset + p] * probsd[(i - 1) * n + i_seq_j] + probs[(i - 1) * n + i_seq_j] * vd[(i_seq_j - 1) * d + head_offset + p])
                    s = s + probs[(i - 1) * n + i_seq_j] * v[(i_seq_j - 1) * d + head_offset + p]
                end
                ctxd[(i - 1) * d + head_offset + p] = sd
                ctx[(i - 1) * d + head_offset + p] = s
            end
        end
        for idx = 1:n_d
            id = 0.0
            i = div(idx - 1, d) + 1
            jd = 0.0
            j = mod(idx - 1, d) + 1
            sd = 0.0
            s = 0.0
            for i_seq_p = 1:d
                sd = sd + (wo[w_offset + (i_seq_p - 1) * d + j] * ctxd[(i - 1) * d + i_seq_p] + ctx[(i - 1) * d + i_seq_p] * wod[w_offset + (i_seq_p - 1) * d + j])
                s = s + ctx[(i - 1) * d + i_seq_p] * wo[w_offset + (i_seq_p - 1) * d + j]
            end
            attn_outd[(i - 1) * d + j] = sd + bod[b_offset + j]
            attn_out[(i - 1) * d + j] = s + bo[b_offset + j]
        end
        for idx = 1:n_d
            resid1d[idx] = xd[idx] + attn_outd[idx]
            resid1[idx] = x[idx] + attn_out[idx]
        end
        for i = 1:n
            sd = 0.0
            s = 0.0
            for i_seq_j = 1:d
                sd = sd + resid1d[(i - 1) * d + i_seq_j]
                s = s + resid1[(i - 1) * d + i_seq_j]
            end
            row_meand = (1.0 / d) * sd
            row_mean = s / d
            s2d = 0.0
            s2 = 0.0
            for i_seq_j = 1:d
                diffd = resid1d[(i - 1) * d + i_seq_j] + -row_meand
                diff = resid1[(i - 1) * d + i_seq_j] - row_mean
                s2d = s2d + (diff * diffd + diff * diffd)
                s2 = s2 + diff * diff
            end
            row_vard = (1.0 / d) * s2d
            row_var = s2 / d
            denomd = (1.0 / (2.0 * sqrt(row_var + eps))) * (row_vard + epsd)
            denom = sqrt(row_var + eps)
            for j = 1:d
                kkd = 0.0
                kk = (i - 1) * d + j
                normed1d[kk] = (ln1_gain[ln_offset + j] * ((1.0 / denom) * (resid1d[kk] + -row_meand) + -((resid1[kk] - row_mean) / denom ^ 2) * denomd) + ((resid1[kk] - row_mean) / denom) * ln1_gaind[ln_offset + j]) + ln1_biasd[ln_offset + j]
                normed1[kk] = ((resid1[kk] - row_mean) / denom) * ln1_gain[ln_offset + j] + ln1_bias[ln_offset + j]
            end
        end
        for idx = 1:n_dff
            id = 0.0
            i = div(idx - 1, dff) + 1
            jd = 0.0
            j = mod(idx - 1, dff) + 1
            sd = 0.0
            s = 0.0
            for i_seq_p = 1:d
                sd = sd + (w1[w1_offset + (i_seq_p - 1) * dff + j] * normed1d[(i - 1) * d + i_seq_p] + normed1[(i - 1) * d + i_seq_p] * w1d[w1_offset + (i_seq_p - 1) * dff + j])
                s = s + normed1[(i - 1) * d + i_seq_p] * w1[w1_offset + (i_seq_p - 1) * dff + j]
            end
            ff_hiddend[(i - 1) * dff + j] = (0.5 * (1.0 + sign(s + b1[b1_offset + j]))) * (sd + b1d[b1_offset + j])
            ff_hidden[(i - 1) * dff + j] = max(s + b1[b1_offset + j], 0.0)
        end
        for idx = 1:n_d
            id = 0.0
            i = div(idx - 1, d) + 1
            jd = 0.0
            j = mod(idx - 1, d) + 1
            sd = 0.0
            s = 0.0
            for i_seq_p = 1:dff
                sd = sd + (w2[w2_offset + (i_seq_p - 1) * d + j] * ff_hiddend[(i - 1) * dff + i_seq_p] + ff_hidden[(i - 1) * dff + i_seq_p] * w2d[w2_offset + (i_seq_p - 1) * d + j])
                s = s + ff_hidden[(i - 1) * dff + i_seq_p] * w2[w2_offset + (i_seq_p - 1) * d + j]
            end
            ff_outd[(i - 1) * d + j] = sd + b2d[b2_offset + j]
            ff_out[(i - 1) * d + j] = s + b2[b2_offset + j]
        end
        for idx = 1:n_d
            resid2d[idx] = normed1d[idx] + ff_outd[idx]
            resid2[idx] = normed1[idx] + ff_out[idx]
        end
        for i = 1:n
            sd = 0.0
            s = 0.0
            for i_seq_j = 1:d
                sd = sd + resid2d[(i - 1) * d + i_seq_j]
                s = s + resid2[(i - 1) * d + i_seq_j]
            end
            row_meand = (1.0 / d) * sd
            row_mean = s / d
            s2d = 0.0
            s2 = 0.0
            for i_seq_j = 1:d
                diffd = resid2d[(i - 1) * d + i_seq_j] + -row_meand
                diff = resid2[(i - 1) * d + i_seq_j] - row_mean
                s2d = s2d + (diff * diffd + diff * diffd)
                s2 = s2 + diff * diff
            end
            row_vard = (1.0 / d) * s2d
            row_var = s2 / d
            denomd = (1.0 / (2.0 * sqrt(row_var + eps))) * (row_vard + epsd)
            denom = sqrt(row_var + eps)
            for j = 1:d
                kkd = 0.0
                kk = (i - 1) * d + j
                x_nextd[kk] = (ln2_gain[ln2_offset + j] * ((1.0 / denom) * (resid2d[kk] + -row_meand) + -((resid2[kk] - row_mean) / denom ^ 2) * denomd) + ((resid2[kk] - row_mean) / denom) * ln2_gaind[ln2_offset + j]) + ln2_biasd[ln2_offset + j]
                x_next[kk] = ((resid2[kk] - row_mean) / denom) * ln2_gain[ln2_offset + j] + ln2_bias[ln2_offset + j]
            end
        end
        for idx = 1:n_d
            xd[idx] = x_nextd[idx]
            x[idx] = x_next[idx]
        end
    end
    return nothing
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
            for idx2 = 1:n * n
                i = div(idx2 - 1, n) + 1
                j = mod(idx2 - 1, n) + 1
                s = 0.0
                for i_seq_p = 1:dk
                    s = s + q[(i - 1) * d + head_offset + i_seq_p] * k[(j - 1) * d + head_offset + i_seq_p]
                end
                scores[(i - 1) * n + j] = s * inv_sqrt_dk
            end
            for i = 1:n
                row_max = scores[(i - 1) * n + 1]
                for i_seq_j = 2:n
                    row_max = max(row_max, scores[(i - 1) * n + i_seq_j])
                end
                for j = 1:n
                    kk = (i - 1) * n + j
                    probs[kk] = exp(scores[kk] - row_max)
                end
                row_sum = 0.0
                for i_seq_j = 1:n
                    row_sum = row_sum + probs[(i - 1) * n + i_seq_j]
                end
                for j = 1:n
                    kk = (i - 1) * n + j
                    probs[kk] = probs[kk] / row_sum
                end
            end
            for idx3 = 1:n * dk
                i = div(idx3 - 1, dk) + 1
                p = mod(idx3 - 1, dk) + 1
                s = 0.0
                for i_seq_j = 1:n
                    s = s + probs[(i - 1) * n + i_seq_j] * v[(i_seq_j - 1) * d + head_offset + p]
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
