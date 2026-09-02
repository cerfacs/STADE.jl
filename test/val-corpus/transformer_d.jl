function transformer_d(x, xd, wq, wqd, bq, bqd, wk, wkd, bk, bkd, wv, wvd, bv, bvd, wo, wod, bo, bod, ln1_gain, ln1_gaind, ln1_bias, ln1_biasd, w1, w1d, b1, b1d, w2, w2d, b2, b2d, ln2_gain, ln2_gaind, ln2_bias, ln2_biasd, q, qd, k, kd, v, vd, scores, scoresd, probs, probsd, ctx, ctxd, attn_out, attn_outd, resid1, resid1d, normed1, normed1d, ff_hidden, ff_hiddend, ff_out, ff_outd, resid2, resid2d, x_next, x_nextd, n, dk, h, dff, n_layers, eps, epsd)
    dd = 0.0
    d = h * dk
    inv_sqrt_dkd = 0.0
    inv_sqrt_dk = 1.0 / sqrt(dk)
    n_dd = 0.0
    n_d = n * d
    n_dffd = 0.0
    n_dff = n * dff
    for i_l = 1:n_layers
        w_offsetd = 0.0
        __icse_0 = i_l - 1
        w_offset = __icse_0 * d * d
        b_offsetd = 0.0
        __icse_1 = __icse_0 * d
        b_offset = __icse_1
        ln_offsetd = 0.0
        ln_offset = __icse_1
        w1_offsetd = 0.0
        w1_offset = __icse_0 * d * dff
        b1_offsetd = 0.0
        b1_offset = __icse_0 * dff
        w2_offsetd = 0.0
        w2_offset = __icse_0 * dff * d
        b2_offsetd = 0.0
        b2_offset = __icse_1
        ln2_offsetd = 0.0
        ln2_offset = __icse_1
        for idx = 1:n_d
            id = 0.0
            __icse_2 = idx - 1
            i = div(__icse_2, d) + 1
            jd = 0.0
            j = mod(__icse_2, d) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                __cse_3 = wq[w_offset + (i_p - 1) * d + j]
                __cse_4 = x[(i - 1) * d + i_p]
                sd = sd + (__cse_3 * xd[(i - 1) * d + i_p] + __cse_4 * wqd[w_offset + (i_p - 1) * d + j])
                s = s + __cse_4 * __cse_3
            end
            qd[(i - 1) * d + j] = sd + bqd[b_offset + j]
            q[(i - 1) * d + j] = s + bq[b_offset + j]
        end
        for idx = 1:n_d
            id = 0.0
            __icse_5 = idx - 1
            i = div(__icse_5, d) + 1
            jd = 0.0
            j = mod(__icse_5, d) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                __cse_6 = wk[w_offset + (i_p - 1) * d + j]
                __cse_7 = x[(i - 1) * d + i_p]
                sd = sd + (__cse_6 * xd[(i - 1) * d + i_p] + __cse_7 * wkd[w_offset + (i_p - 1) * d + j])
                s = s + __cse_7 * __cse_6
            end
            kd[(i - 1) * d + j] = sd + bkd[b_offset + j]
            k[(i - 1) * d + j] = s + bk[b_offset + j]
        end
        for idx = 1:n_d
            id = 0.0
            __icse_8 = idx - 1
            i = div(__icse_8, d) + 1
            jd = 0.0
            j = mod(__icse_8, d) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                __cse_9 = wv[w_offset + (i_p - 1) * d + j]
                __cse_10 = x[(i - 1) * d + i_p]
                sd = sd + (__cse_9 * xd[(i - 1) * d + i_p] + __cse_10 * wvd[w_offset + (i_p - 1) * d + j])
                s = s + __cse_10 * __cse_9
            end
            vd[(i - 1) * d + j] = sd + bvd[b_offset + j]
            v[(i - 1) * d + j] = s + bv[b_offset + j]
        end
        for hh = 1:h
            head_offsetd = 0.0
            __icse_11 = hh - 1
            head_offset = __icse_11 * dk
            score_offd = 0.0
            score_off = __icse_11 * n * n
            for idx2 = 1:n * n
                id = 0.0
                __icse_12 = idx2 - 1
                i = div(__icse_12, n) + 1
                jd = 0.0
                j = mod(__icse_12, n) + 1
                sd = 0.0
                s = 0.0
                for i_p = 1:dk
                    __cse_13 = k[(j - 1) * d + head_offset + i_p]
                    __cse_14 = q[(i - 1) * d + head_offset + i_p]
                    sd = sd + (__cse_13 * qd[(i - 1) * d + head_offset + i_p] + __cse_14 * kd[(j - 1) * d + head_offset + i_p])
                    s = s + __cse_14 * __cse_13
                end
                scoresd[score_off + (i - 1) * n + j] = inv_sqrt_dk * sd
                scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
            end
            for i = 1:n
                row_maxd = scoresd[score_off + (i - 1) * n + 1]
                row_max = scores[score_off + (i - 1) * n + 1]
                for i_j = 2:n
                    __cse_15 = scores[score_off + (i - 1) * n + i_j]
                    row_maxd = (0.5 * (1.0 + sign(row_max - __cse_15))) * row_maxd + (0.5 * (1.0 + sign(__cse_15 - row_max))) * scoresd[score_off + (i - 1) * n + i_j]
                    row_max = max(row_max, __cse_15)
                end
                for j = 1:n
                    kkd = 0.0
                    kk = score_off + (i - 1) * n + j
                    __cse_16 = exp(scores[kk] - row_max)
                    probsd[kk] = __cse_16 * (scoresd[kk] + -row_maxd)
                    probs[kk] = __cse_16
                end
                row_sumd = 0.0
                row_sum = 0.0
                for i_j = 1:n
                    row_sumd = row_sumd + probsd[score_off + (i - 1) * n + i_j]
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_j]
                end
                for j = 1:n
                    kkd = 0.0
                    kk = score_off + (i - 1) * n + j
                    __cse_17 = probs[kk]
                    probsd[kk] = (1.0 / row_sum) * probsd[kk] + -(__cse_17 / row_sum ^ 2) * row_sumd
                    probs[kk] = __cse_17 / row_sum
                end
            end
            for idx3 = 1:n * dk
                id = 0.0
                __icse_18 = idx3 - 1
                i = div(__icse_18, dk) + 1
                pd = 0.0
                p = mod(__icse_18, dk) + 1
                sd = 0.0
                s = 0.0
                for i_j = 1:n
                    __cse_19 = v[(i_j - 1) * d + head_offset + p]
                    __cse_20 = probs[score_off + (i - 1) * n + i_j]
                    sd = sd + (__cse_19 * probsd[score_off + (i - 1) * n + i_j] + __cse_20 * vd[(i_j - 1) * d + head_offset + p])
                    s = s + __cse_20 * __cse_19
                end
                ctxd[(i - 1) * d + head_offset + p] = sd
                ctx[(i - 1) * d + head_offset + p] = s
            end
        end
        for idx = 1:n_d
            id = 0.0
            __icse_21 = idx - 1
            i = div(__icse_21, d) + 1
            jd = 0.0
            j = mod(__icse_21, d) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                __cse_22 = wo[w_offset + (i_p - 1) * d + j]
                __cse_23 = ctx[(i - 1) * d + i_p]
                sd = sd + (__cse_22 * ctxd[(i - 1) * d + i_p] + __cse_23 * wod[w_offset + (i_p - 1) * d + j])
                s = s + __cse_23 * __cse_22
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
            for i_j = 1:d
                sd = sd + resid1d[(i - 1) * d + i_j]
                s = s + resid1[(i - 1) * d + i_j]
            end
            row_meand = (1.0 / d) * sd
            row_mean = s / d
            s2d = 0.0
            s2 = 0.0
            for i_j = 1:d
                diffd = resid1d[(i - 1) * d + i_j] + -row_meand
                diff = resid1[(i - 1) * d + i_j] - row_mean
                __cse_24 = diff * diffd
                s2d = s2d + (__cse_24 + __cse_24)
                s2 = s2 + diff * diff
            end
            row_vard = (1.0 / d) * s2d
            row_var = s2 / d
            __cse_25 = sqrt(row_var + eps)
            denomd = (1.0 / (2.0__cse_25)) * (row_vard + epsd)
            denom = __cse_25
            for j = 1:d
                kkd = 0.0
                kk = (i - 1) * d + j
                __cse_26 = ln1_gain[ln_offset + j]
                __cse_27 = resid1[kk] - row_mean
                __cse_28 = __cse_27 / denom
                normed1d[kk] = (__cse_26 * ((1.0 / denom) * (resid1d[kk] + -row_meand) + -(__cse_27 / denom ^ 2) * denomd) + __cse_28 * ln1_gaind[ln_offset + j]) + ln1_biasd[ln_offset + j]
                normed1[kk] = __cse_28 * __cse_26 + ln1_bias[ln_offset + j]
            end
        end
        for idx = 1:n_dff
            id = 0.0
            __icse_29 = idx - 1
            i = div(__icse_29, dff) + 1
            jd = 0.0
            j = mod(__icse_29, dff) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:d
                __cse_30 = w1[w1_offset + (i_p - 1) * dff + j]
                __cse_31 = normed1[(i - 1) * d + i_p]
                sd = sd + (__cse_30 * normed1d[(i - 1) * d + i_p] + __cse_31 * w1d[w1_offset + (i_p - 1) * dff + j])
                s = s + __cse_31 * __cse_30
            end
            __cse_32 = s + b1[b1_offset + j]
            ff_hiddend[(i - 1) * dff + j] = (0.5 * (1.0 + sign(__cse_32))) * (sd + b1d[b1_offset + j])
            ff_hidden[(i - 1) * dff + j] = max(__cse_32, 0.0)
        end
        for idx = 1:n_d
            id = 0.0
            __icse_33 = idx - 1
            i = div(__icse_33, d) + 1
            jd = 0.0
            j = mod(__icse_33, d) + 1
            sd = 0.0
            s = 0.0
            for i_p = 1:dff
                __cse_34 = w2[w2_offset + (i_p - 1) * d + j]
                __cse_35 = ff_hidden[(i - 1) * dff + i_p]
                sd = sd + (__cse_34 * ff_hiddend[(i - 1) * dff + i_p] + __cse_35 * w2d[w2_offset + (i_p - 1) * d + j])
                s = s + __cse_35 * __cse_34
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
            for i_j = 1:d
                sd = sd + resid2d[(i - 1) * d + i_j]
                s = s + resid2[(i - 1) * d + i_j]
            end
            row_meand = (1.0 / d) * sd
            row_mean = s / d
            s2d = 0.0
            s2 = 0.0
            for i_j = 1:d
                diffd = resid2d[(i - 1) * d + i_j] + -row_meand
                diff = resid2[(i - 1) * d + i_j] - row_mean
                __cse_36 = diff * diffd
                s2d = s2d + (__cse_36 + __cse_36)
                s2 = s2 + diff * diff
            end
            row_vard = (1.0 / d) * s2d
            row_var = s2 / d
            __cse_37 = sqrt(row_var + eps)
            denomd = (1.0 / (2.0__cse_37)) * (row_vard + epsd)
            denom = __cse_37
            for j = 1:d
                kkd = 0.0
                kk = (i - 1) * d + j
                __cse_38 = ln2_gain[ln2_offset + j]
                __cse_39 = resid2[kk] - row_mean
                __cse_40 = __cse_39 / denom
                x_nextd[kk] = (__cse_38 * ((1.0 / denom) * (resid2d[kk] + -row_meand) + -(__cse_39 / denom ^ 2) * denomd) + __cse_40 * ln2_gaind[ln2_offset + j]) + ln2_biasd[ln2_offset + j]
                x_next[kk] = __cse_40 * __cse_38 + ln2_bias[ln2_offset + j]
            end
        end
        for idx = 1:n_d
            xd[idx] = x_nextd[idx]
            x[idx] = x_next[idx]
        end
    end
    return nothing
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
